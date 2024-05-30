# Model utilities
from transformers import AutoTokenizer, AutoModelForCausalLM, MptConfig, GenerationConfig

from utils.model_helpers import load_model_and_tokenizer

# Database
from utils.build_database import Database 
from utils.embedding_helpers import CustomEmbeddingFunction

# General
import math
import torch
import torch.cuda as cuda
import random
import json
import wandb
from tqdm import tqdm
import numpy as np

# System and logging
import os
import sys
import yaml

from utils.data_helpers import save_results
from utils.model_helpers import perplexity

random.seed(2024)
os.environ['CURL_CA_BUNDLE'] = ''

from transformers import StoppingCriteria, StoppingCriteriaList


if cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = "cpu"
print("[INFO]: Device is: ", device)
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
device = "cpu"


class PerplexityCalculator:
    def __init__(self, config, data_path, generator, tokenizer):
        self.config = config
        self.batch_size = self.config['experiment']['batch_size']
        self.data_path = data_path
        self.generator = generator
        self.tokenizer = tokenizer
        self.data, self.question_data, self.question_indices, self.available_indices = self.load_data()

        self.max_length = self.config[self.config['experiment']['type']]['max_length'] 

            # mtp-7b is trained to add "<|endoftext|>" at the end of generations
        stop_token_ids = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

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
        dataset_id = self.config['data']['data_id']
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
        random.seed(self.config['experiment']['seed'])

        question_indices = random.sample(range(n_data), self.config['data']['n_questions'])
        question_data = [data[i] for i in question_indices]

        available_indices = [i for i in range(len(data)) if i not in question_indices]


        return data, question_data, question_indices, available_indices

    def run(self, outputs_path):
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
                                         outputs_path, 
                                         self.config['data']['data_id'])
        elif experiment_type == 'rag':
            print("[INFO]: Running RAG experiment...")
            self.run_rag_experiment(self.config, 
                                    self.generator, 
                                    self.tokenizer, 
                                    self.data, 
                                    run, 
                                    outputs_path, 
                                    dataset_id=self.config['data']['data_id'])
            
        elif experiment_type == 'em':
            print("[INFO]: Running EM experiment...")
            self.run_em_experiment(self.config, 
                                   self.generator, 
                                   self.tokenizer, 
                                   self.data, 
                                   run, 
                                   outputs_path, 
                                   dataset_id=self.config['data']['data_id'])
            
        else:
            raise ValueError(f"Invalid experiment type: {experiment_type}")

        run.finish()


    def run_baseline_experiment(self, config, generator, tokenizer, data, run, outputs_path, dataset_id):
        token_args = {'return_tensors': 'pt'}
        
        # Prediction table for wandb
        columns = ["experiment",
                   "id", 
                   "userprompt",
                   "perplexity"]
        
        prediction_table = wandb.Table(columns=columns)

        for question, question_index in tqdm(zip(self.question_data, self.question_indices), total=len(self.question_indices)):
            if self.config['data']['data_id'] == 'NQ':                    
                userprompt =  f". Question: {question['question']}?"

            elif self.config['data']['data_id'] == 'QuALITY':
                # Add options with numbers to the userprompt
                options = [f'{i}) {j}' for i,j in zip(range(1, len(question['questions'][0]['options'])+1), question['questions'][0]['options'])]
                userprompt = " ".join([f". Question: {question['questions'][0]['question']}", f"{' '.join(options)}. Answer: "])

            
            # Process userprompt
            userpromt_ids = tokenizer(userprompt, **token_args)['input_ids'].to(device)


            print("[INFO]: Tokenizing userprompt...")
            # Process userprompt
            if self.config['data']['data_id'] == 'NQ':
                gold_answers = question['answers']
            elif self.config['data']['data_id'] == 'QuALITY':
                gold_answers = str(question['questions'][0]['gold_label'])
            if isinstance(gold_answers, list):
                gold_answers = gold_answers[0]

            answer_tokens = tokenizer(gold_answers, **token_args)['input_ids'].to(device)
            
            print("[INFO]: Answer tokens")
            print(answer_tokens)
            input_ids = torch.cat((userpromt_ids, answer_tokens), dim=-1)
            print("[INFO]: Calculating perplexity...")

            perp = perplexity(generator, input_ids, len(answer_tokens), max_length=self.max_length, modeltype="baseline")
            
            prediction_table.add_data(config['experiment']['type'], 
                                    question_index, 
                                    userprompt, 
                                    perp)

        # Save accuracy and results
        run.log({"Prediction overview": prediction_table})


    def run_rag_experiment(self, config, generator, tokenizer, data, run, outputs_path, dataset_id):
        # Accuracy counter
        token_args = {'return_tensors': 'pt'}

        print(config['data']['n_retrieval_docs'] < len(data))
        if config['data']['n_questions'] < len(data):        
            doc_indices = random.sample(self.available_indices, config['data']['n_retrieval_docs'])
        else:
            doc_indices = random.sample(self.question_indices, config['data']['n_retrieval_docs'])

        if self.config['data']['data_id'] == 'NQ':
            docs = [" ".join(data[i]['ctxs']) for i in doc_indices]
        elif self.config['data']['data_id'] == 'QuALITY':
            docs = [data[i]['article'] for i in doc_indices]

        # Now we make sure that 1) the amount of retrieved documents is the same as the amount of questions and 2) that we sample docs from the questions

        embedding = CustomEmbeddingFunction(config[config['experiment']['type']]['embedder_id'], config[config['experiment']['type']]['embedder_tokenizer_id'], model_args=None)
        # Build database
        #docs = ["".join(question_data['ctxs']) for question_data in data]
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
                   "max_length",
                   "n_retrieval_docs",
                   "top_k",
                   "generation_method",
                   "embedding_id", 
                   "userprompt", 
                   "n_tokens", 
                   "docs_ids",
                   "similarities", 
                   "perplexity"]
        
        prediction_table = wandb.Table(columns=columns)

        # Calculate the number of batches
        num_batches = math.ceil(len(self.question_indices) / self.batch_size)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(self.question_indices))
            print("Start index: ", start_idx)
            print("End index: ", end_idx)  
            print("Length of question indices: ", len(self.question_indices))
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
                
                self.ChromaDB.check_and_add(question_index, text)
                result = self.ChromaDB.db.query(query_texts=[userprompt],
                                        n_results=config['experiment']['top_k'], 
                                        include=["documents", 'distances'])

                # Extract the first (and only) list inside 'ids'
                ids = result.get('ids')[0]
                final_docs = result.get('documents')[0]
                sims = result.get('distances')[0]
                
                final_str = f"Context: {' '.join(final_docs)}. {userprompt}"

                final_prompt_ids = tokenizer(final_str, **token_args)['input_ids'].to(device)
                
                print("[INFO]: Docs character length: ", len(" ".join(final_docs)))
                S = final_prompt_ids.size(-1)

                print("[INFO]: Prompt length: ", len(final_prompt_ids[0]))
                print("[INFO]: Max lenght: ", S)

                if S+ self.max_length > config[config['experiment']['type']]['context_window']-1:
                    print(f"[WARNING] Input length {S} exceeds maximum sequence length {config[config['experiment']['type']]['context_window']}. Truncating input.")
                    final_prompt_ids = final_prompt_ids[:, :config[config['experiment']['type']]['context_window']-self.max_length]
                
                print("[INFO]: Lenght: ", len(final_prompt_ids[0]))

                print("[INFO]: Tokenizing userprompt...")
                # Process userprompt
                if self.config['data']['data_id'] == 'NQ':
                    gold_answers = question['answers']
                elif self.config['data']['data_id'] == 'QuALITY':
                    gold_answers = str(question['questions'][0]['gold_label'])
                if isinstance(gold_answers, list):
                    gold_answers = gold_answers[0]

                print("[INFO]: final_prompt_ids tokens")
                print(final_prompt_ids)

                answer_tokens = tokenizer(gold_answers, **token_args)['input_ids'].to(device)
                
                print("[INFO]: Anser tokens")
                print(answer_tokens)
                input_ids = torch.cat((final_prompt_ids, answer_tokens), dim=-1)
                print("[INFO]: Calculating perplexity...")
    
                perp = perplexity(generator, input_ids, len(answer_tokens), max_length=self.max_length, stride=1, modeltype="rag")
                
                prediction_table.add_data(config['experiment']['type'], 
                                        question_index, 
                                        self.max_length,
                                        config['data']['n_retrieval_docs'],      
                                        config['experiment']['top_k'],                            
                                        config[config['experiment']['type']]['generation'],                                    
                                        config[config['experiment']['type']]['embedder_id'],
                                        userprompt, 
                                        len(final_prompt_ids[0]), 
                                        ids,
                                        sims,
                                        perp)
                
        # Save accuracy and results
        run.log({"Prediction overview": prediction_table})

    def run_em_experiment(self, config, generator, tokenizer, data, run, outputs_path, dataset_id):
        print("[INFO]: Setting token args...")
        token_args = {'return_tensors': 'pt'}

        # Prediction table for wandb
        columns = ["id", 
                   "max_length",
                   "generation_method",
                    "n_retrieval_docs",
                    "top_k",
                   "userprompt", 
                   "n_tokens", 
                   "doc_indices",
                   "perplexity"]
        
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
            memory_ids = tokenizer(batch_docs, return_tensors='pt')['input_ids'].to(device)

            print(f"[INFO]: Generating cache for batch {batch_idx + 1}...")

            generator._memories = memory_ids

            print(memory_ids.shape)

            for question, question_index in zip(batch_questions, range(start_idx, end_idx)):
                if self.config['data']['data_id'] == 'NQ':
                    userprompt = f"Answer the question: {question['question']}?" 

                elif self.config['data']['data_id'] == 'QuALITY':
                    # Add options with numbers to the userprompt
                    options = [f'{i}) {j}' for i,j in zip(range(1, len(question['questions'][0]['options'])+1), question['questions'][0]['options'])]
                    userprompt = " ".join([f". {question['questions'][0]['question']}. Answer one of the following options:", f"{' '.join(options)}. Answer: "]) 

                print("[INFO]: Tokenizing userprompt...")
                # Process userprompt
                if self.config['data']['data_id'] == 'NQ':
                    gold_answers = question['answers']
                elif self.config['data']['data_id'] == 'QuALITY':
                    gold_answers = str(question['questions'][0]['gold_label'])
                if isinstance(gold_answers, list):
                    gold_answers = gold_answers[0]

                answer_tokens = tokenizer(gold_answers, **token_args)['input_ids'].to(device)
                

                print("[INFO]: Tokenizing userprompt...")
                # Process userprompt
                userpromt_ids = tokenizer(userprompt, **token_args)['input_ids'].to(device)
                
                print("[INFO]: Answer tokens")
                print(answer_tokens)
                input_ids = torch.cat((userpromt_ids, answer_tokens), dim=-1)

                print("[INFO]: Calculating perplexity...")
                perp = perplexity(generator, input_ids, len(answer_tokens), 
                                  max_length=self.max_length, 
                                  stride=1, 
                                  modeltype="em", 
                                  topk=config['experiment']['top_k'])

                prediction_table.add_data(question_index, 
                                            self.max_length,    
                                            config[config['experiment']['type']]['generation'],  
                                            config['data']['n_retrieval_docs'],        
                                            config['experiment']['top_k'],                          
                                            userprompt, 
                                            len(userprompt[0]),
                                            self.question_indices[start_idx:end_idx], 
                                            perp)
                
            
            print("[INFO]: Emptying memories...")
            generator.empty_memories()

        # Save accuracy and results
        run.log({"Prediction overview": prediction_table})


def main(config_path: str, logging_path: str, outputs_path: str, data_path: str='data/', metadata_directory_path: str=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    generator, tokenizer = load_model_and_tokenizer(config, data_path)
    experiment = PerplexityCalculator(config, data_path, generator, tokenizer)
    experiment.run(outputs_path)


if __name__ == '__main__':
    main(*sys.argv[1:])

