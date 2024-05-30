# Model utilities
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, MptConfig, GenerationConfig
from sklearn.model_selection import ParameterGrid

# Database
from utils.build_database import Database 
from utils.embedding_helpers import CustomEmbeddingFunction
from utils.model_helpers import load_model_and_tokenizer

# General
import math
import torch
import random
import json
import wandb
from tqdm import tqdm
import numpy as np

# System and logging
import os
import sys
import yaml
import torch.cuda as cuda

from utils.data_helpers import evaluate_results

os.environ['CURL_CA_BUNDLE'] = ''

from transformers import StoppingCriteria, StoppingCriteriaList


class Experiment:
    def __init__(self, config, data_path, generator, tokenizer):
        self.config = config
        self.batch_size = self.config['experiment']['batch_size']
        self.data_path = data_path
        self.generator = generator
        self.tokenizer = tokenizer
        self.data, self.question_data, self.question_indices, self.available_indices = self.load_data()

        self.max_length = self.config[self.config['experiment']['type']]['max_length'] 
        self.token_padding = self.config[self.config['experiment']['type']]['token_padding'] 
            
        stop_token_ids = self.tokenizer.convert_tokens_to_ids(config['experiment']['stop_list'])

        # define custom stopping criteria object
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False
        
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])


    def load_data(self):
        print("[INFO]: Loading data...")
        data_filename = os.path.join(self.data_path, 
                                     self.config['data']['subpath'],
                                     self.config['data']['filename'])
        
        data = []
        with open(data_filename, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        if self.config['data']['data_id'] == 'NQ':
             data = data[0]

        n_data = len(data)
        print("[INFO]: Number of data points: ", n_data)
        random.seed(2024)
        question_indices = random.sample(range(n_data), self.config['data']['n_questions'])
        question_data = [data[i] for i in question_indices]

        available_indices = [i for i in range(len(data)) if i not in question_indices]

        return data, question_data, question_indices, available_indices

    def run(self):
        print("[INFO]: Running experiment...")
        run = wandb.init(project=self.config['wandb_project'], 
                         settings=wandb.Settings(code_dir="."), 
                         tags=[self.config['experiment']['type'],
                               self.config['data']['data_id'], 
                               f"n_docs_{self.config['data']['n_retrieval_docs']}",
                               f"gridsearch_{self.config['experiment']['grid_search']}",
                               f"top_k_{self.config['experiment']['top_k']}",
                               f"n_questions_{self.config['data']['n_questions']}",
                               f"max_length_{self.max_length}",
                               f"sampling_strategy_{self.config[self.config['experiment']['type']]['generation']}", 
                               self.config[self.config['experiment']['type']]['embedder_id']] if self.config['experiment']['type'] == 'rag' else [self.config['experiment']['type'], self.config['data']['data_id']])

        experiment_type = self.config['experiment']['type']

        if experiment_type == 'baseline':
            print("[INFO]: Running baseline experiment...")
            self.run_baseline_experiment(self.config, 
                                         self.generator, 
                                         self.tokenizer, 
                                         self.question_data, 
                                         run, 
                                         self.config['data']['data_id'])
        elif experiment_type == 'rag':
            print("[INFO]: Running RAG experiment...")
            self.run_rag_experiment(self.config, 
                                    self.generator, 
                                    self.tokenizer, 
                                    self.data, 
                                    run, 
                                    dataset_id=self.config['data']['data_id'])
            
        elif experiment_type == 'em':
            print("[INFO]: Running EM experiment...")
            self.run_em_experiment(self.config, 
                                   self.generator, 
                                   self.tokenizer, 
                                   self.data, 
                                   run, 
                                   dataset_id=self.config['data']['data_id'])
            
        else:
            raise ValueError(f"Invalid experiment type: {experiment_type}")

        run.finish()


    def run_baseline_experiment(self, config, generator, tokenizer, data, run, dataset_id):
        token_args = {'return_tensors': 'pt'}
        
        # Prediction table for wandb
        columns = ["id", 
                   "max_length",
                   "generation_method",
                   "userprompt", 
                   "predicted", 
                   "gold", 
                   "bleu", 
                   "rouge",
                   "correct"]
        
        prediction_table = wandb.Table(columns=columns)

        for question, question_index in tqdm(zip(self.question_data, self.question_indices), total=len(self.question_indices)):
            if self.config['data']['data_id'] == 'NQ':                    
                userprompt =  f". Question: {question['question']}?"

            elif self.config['data']['data_id'] == 'QuALITY':
                # Add options with numbers to the userprompt
                options = [f'{i}) {j}' for i,j in zip(range(1, len(question['questions'][0]['options'])+1), question['questions'][0]['options'])]
                userprompt = " ".join([f". Question: {question['questions'][0]['question']}", f"{' '.join(options)}. Answer: "])
            
            # Process userprompt
            userpromt_ids = tokenizer(userprompt, **token_args)['input_ids']

            # Generate answer
            answer_ids = generator.generate(userpromt_ids, 
                                            stopping_criteria=self.stopping_criteria,
                                            pad_token_id=tokenizer.pad_token_id, 
                                            eos_token_id=tokenizer.eos_token_id,
                                            max_length=userpromt_ids.size(-1)+self.max_length)
            answer_string = tokenizer.decode(answer_ids[0][userpromt_ids.size(-1) + 1:])

            bleu, rouge, correct, gold_answers = evaluate_results(answer_string, question, config, userprompt)
            
            prediction_table.add_data(question_index, 
                                      self.max_length,
                                      config[config['experiment']['type']]['generation'],                                    
                                      userprompt, 
                                      answer_string, 
                                      gold_answers,
                                      bleu, 
                                      rouge,
                                      correct)

        # Save results
        run.log({"Prediction overview": prediction_table})

    def run_rag_experiment(self, config, generator, tokenizer, data, run, dataset_id):
        # Accuracy counter
        token_args = {'return_tensors': 'pt'}

        correct = 0
        results = {}

        if config['data']['n_questions'] < len(data):        
            doc_indices = random.sample(self.available_indices, config['data']['n_retrieval_docs'])# - 1)
        else:
            doc_indices = random.sample(self.question_indices, config['data']['n_retrieval_docs'])# - 1)

        if self.config['data']['data_id'] == 'NQ':
            docs = [" ".join(data[i]['ctxs']) for i in doc_indices]
        elif self.config['data']['data_id'] == 'QuALITY':
            docs = [data[i]['article'] for i in doc_indices]

        embedding = CustomEmbeddingFunction(config[config['experiment']['type']]['embedder_id'], config[config['experiment']['type']]['embedder_tokenizer_id'], model_args=None)

        # Build database
        print("[INFO]: Building database...")
        self.ChromaDB = Database(docs=docs, 
                            embedding=embedding,
                            ids=doc_indices, 
                            chunk_params={'separators':config['experiment']['separators'],
                                    'chunk_size':config['experiment']['chunk_size'],
                                    'chunk_overlap':config['experiment']['chunk_overlap']})

        # Prediction table for wandb
        columns = ["experiment", 
                   "id", 
                   "seed",
                   "max_length",
                   "n_retrieval_docs",
                   "top_k",
                   "generation_method",
                   "embedding_id", 
                   "userprompt", 
                   "n_tokens", 
                   "docs_ids",
                   "similarities", 
                   "predicted", 
                   "gold", 
                   "bleu",
                   "rouge",
                   "correct"]
        
        prediction_table = wandb.Table(columns=columns)

        # Calculate the number of batches
        num_batches = math.ceil(len(self.question_indices) / self.batch_size)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(self.question_indices))
            batch_question_indices = self.question_indices[start_idx:end_idx]
            
            # We make a list with start_idx:end_idx and index the data with that list
            start_end_list = list(range(start_idx, end_idx))
            print("Batch_question_indices", batch_question_indices)
            batch_question_data = [self.question_data[i] for i in start_end_list]

            for question, question_index in zip(batch_question_data, batch_question_indices):            
                if self.config['data']['data_id'] == 'NQ':                    
                    userprompt =  f". Question: {question['question']}?"
                    text = " ".join(question['ctxs'])

                elif self.config['data']['data_id'] == 'QuALITY':
                    # Add options with numbers to the userprompt
                    options = [f'{i}) {j}' for i,j in zip(range(1, len(question['questions'][0]['options'])+1), question['questions'][0]['options'])]
                    userprompt = " ".join([f". Question: {question['questions'][0]['question']}", f"{' '.join(options)}. Answer: "])
                    text = question['article']

                results[userprompt] = {}

                temp_k = config['experiment']['top_k']
                print("N chunks in Chroma: ", self.ChromaDB.db.count(), "N k in experiment: ", config['experiment']['top_k'])
                self.ChromaDB.check_and_add(question_index, text)

                if self.ChromaDB.db.count() < config['experiment']['top_k']:
                    print(f"[WARNING]: Top k ({config['experiment']['top_k']}) exceeds the number of documents in the database ({self.ChromaDB.db.count()}). Setting top k to {self.ChromaDB.db.count()}")                    
                    temp_k = self.ChromaDB.db.count()

                print("[INFO]: Querying database...")
                result = self.ChromaDB.db.query(query_texts=[userprompt],
                                        n_results=temp_k, 
                                        include=["documents", 'distances'])

                ids = result.get('ids')[0]
                final_docs = result.get('documents')[0]
                sims = result.get('distances')[0]
                print("[INFO]: Similarities: ", sims)

                final_str = f"Context: {' '.join(final_docs)}. {userprompt}"

                final_prompt_ids = tokenizer(final_str, **token_args)['input_ids']
                
                print("[INFO]: Docs character length: ", len(" ".join(final_docs)))
                S = final_prompt_ids.size(-1)
                if S+ self.max_length > config[config['experiment']['type']]['context_window']-1:
                    print(f"[WARNING] Input length {S} exceeds maximum sequence length {config[config['experiment']['type']]['context_window']}. Truncating input.")
                    final_prompt_ids = final_prompt_ids[:, :config[config['experiment']['type']]['context_window']-self.max_length]
                
                print("[INFO]: Length: ", len(final_prompt_ids[0]))

                # Generate answer
                answer_ids = generator.generate(final_prompt_ids, 
                                                stopping_criteria=self.stopping_criteria,
                                                pad_token_id=tokenizer.pad_token_id, 
                                                eos_token_id=tokenizer.eos_token_id,
                                                max_length=final_prompt_ids.size(-1) + self.max_length,
                                                output_scores=True, 
                                                return_dict_in_generate=True
                                                )
                
                print("[INFO]: Answer generated. Decoding answer...")
                answer_string = tokenizer.decode(answer_ids.sequences[0][final_prompt_ids.size(-1) + 1:], skip_special_tokens=True)

                # Collect metrics
                bleu, rouge, correct, gold_answers = evaluate_results(answer_string, question, config, userprompt)

                prediction_table.add_data(config['experiment']['type'], 
                                        question_index, 
                                        config['experiment']['seed'],
                                        self.max_length,
                                        config['data']['n_retrieval_docs'],      
                                        temp_k,                            
                                        config[config['experiment']['type']]['generation'],                                    
                                        config[config['experiment']['type']]['embedder_id'],
                                        userprompt, 
                                        len(final_prompt_ids[0]), 
                                        ids,
                                        sims,
                                        answer_string,
                                        gold_answers,
                                        bleu,
                                        rouge,
                                        correct)
                
        # Save accuracy and results
        run.log({"Prediction overview": prediction_table})

    def run_em_experiment(self, config, generator, tokenizer, data, run, dataset_id):
        print("[INFO]: Setting token args...")
        # Finding the max length of the prompt
        token_args = {'return_tensors': 'pt', 'padding': 'max_length', 'max_length': self.token_padding}

        # Prediction table for wandb
        columns = ["id", 
                   "max_length",
                   "seed",
                   "generation_method",
                    "n_retrieval_docs",
                    "top_k",
                   "userprompt", 
                   "n_tokens", 
                   "doc_indices",
                   "predicted", 
                   "gold", 
                   "bleu", 
                   "rouge",
                   "correct"]
        
        prediction_table = wandb.Table(columns=columns)
        print("[INFO]: Starting loop...")
                
        num_batches = math.ceil(len(self.question_indices) / self.batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(self.question_indices))
            batch_questions = self.question_data[start_idx:end_idx]
            batch_documents = [" ".join(question['ctxs']) if self.config['data']['data_id'] == 'NQ' else question['article'] for question in batch_questions]

            # Sample additional random documents if needed
            if self.batch_size < config['data']['n_retrieval_docs']:
                additional_doc_indices = random.sample(self.available_indices,  config['data']['n_retrieval_docs'] - self.batch_size)
                additional_docs = [" ".join(self.data[idx]['ctxs']) if self.config['data']['data_id'] == 'NQ' else self.data[idx]['article'] for idx in additional_doc_indices]
                batch_documents.extend(additional_docs)

            # Concatenate the documents for this batch
            batch_docs = " ".join(batch_documents)

            print(f"[INFO]: Tokenizing external memories for batch {batch_idx + 1}...")
            memory_ids = tokenizer(batch_docs, return_tensors='pt')['input_ids'] 

            print(f"[INFO]: Generating cache for batch {batch_idx + 1}...")
            generator._memories = memory_ids

            for question, question_index in zip(batch_questions, range(start_idx, end_idx)):

                if self.config['data']['data_id'] == 'NQ':
                    userprompt = f"Answer the question: {question['question']}?" 

                elif self.config['data']['data_id'] == 'QuALITY':
                    # Add options with numbers to the userprompt
                    options = [f'{i}) {j}' for i,j in zip(range(1, len(question['questions'][0]['options'])+1), question['questions'][0]['options'])]
                    userprompt = " ".join([f". {question['questions'][0]['question']}. Answer one of the following options:", f"{' '.join(options)}. Answer: "]) #Answer: 

                print("[INFO]: Tokenizing userprompt...")
                # Process userprompt
                userpromt_ids = tokenizer(userprompt, **token_args)['input_ids'] 
                print("[INFO]: Generating answer...")
                
                # Generate answer
                answer_ids = generator.generate(userpromt_ids, 
                                                stopping_criteria=self.stopping_criteria,
                                                pad_token_id=tokenizer.pad_token_id, 
                                                eos_token_id=tokenizer.eos_token_id,
                                                max_length=userpromt_ids.size(-1) + self.max_length, 
                                                topk=config['experiment']['top_k'],                                            
                                                output_scores=True, 
                                                return_dict_in_generate=True
                                                )
                
                answer_string = tokenizer.decode(answer_ids.sequences[0][userpromt_ids.size(-1) + 1:], skip_special_tokens=True)
                
                # Collect metrics
                bleu, rouge, correct, gold_answers = evaluate_results(answer_string, question, config)

                prediction_table.add_data(question_index, 
                                            self.max_length,    
                                            config['experiment']['seed'],
                                            config[config['experiment']['type']]['generation'],  
                                            config['data']['n_retrieval_docs'],        
                                            config['experiment']['top_k'],                          
                                            userprompt, 
                                            len(userpromt_ids[0]),
                                            self.question_indices[start_idx:end_idx], 
                                            answer_string, 
                                            gold_answers, 
                                            bleu,
                                            rouge,
                                            correct)
                
            
            print("[INFO]: Emptying memories...")
            generator.empty_memories()

        # Save accuracy and results
        run.log({"Prediction overview": prediction_table})


def main(config_path: str, logging_path: str, data_path: str='data/', metadata_directory_path: str=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config['experiment']['multiple_runs'] == True:
        for i in range(config['experiment']['n_runs']):
            run_id = random.randint(0, 100000)
            config['experiment']['seed'] = run_id
            generator, tokenizer = load_model_and_tokenizer(config)
            experiment = Experiment(config, data_path, generator, tokenizer)
            experiment.run()
            if config['experiment']['type'] == 'rag':
                experiment.ChromaDB.client.delete_collection(name="test")


    if config['experiment']['grid_search'] == True:
        # Define the parameter grid
        param_grid = {
            'max_length': [15, 20],
            'generation': ['beam', 'contrastive', 'sampling_top_k', 'sampling_top_p']
        }

        # Create the parameter grid
        grid = ParameterGrid(param_grid)

        # Loop over the parameter combinations
        for params in grid:
            print("[INFO]: Running experiment with parameters: ", params)
            # Update the configuration with the current parameters
            config[config['experiment']['type']]['max_length']  = params['max_length']
            config[config['experiment']['type']]['generation'] = params['generation']
            # Load the model and tokenizer once
            generator, tokenizer = load_model_and_tokenizer(config)

            experiment = Experiment(config, data_path, generator, tokenizer)
            experiment.run()
            # Delete the database collection
            if config['experiment']['type'] == 'rag':
                experiment.ChromaDB.client.delete_collection(name="test")

    else:
        generator, tokenizer = load_model_and_tokenizer(config)
        experiment = Experiment(config, data_path, generator, tokenizer)
        experiment.run()


if __name__ == '__main__':
    main(*sys.argv[1:])



