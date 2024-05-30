from transformers import AutoTokenizer, AutoConfig, AutoModel, MptConfig, MptModel, AutoModelForCausalLM
import torch
import sys
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
import torch.nn.functional as F
from torch import Tensor
import os
from typing import Dict
import numpy as np

# # Initialize tokenizer and model for embedding
# tokenizer = AutoTokenizer.from_pretrained(token_path)
# configuration = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, 
#                                              config = configuration, 
#                                              trust_remote_code=True)#MptModel(MptConfig())#AutoModelForCausalLM.from_pretrained(model_path, config = configuration, trust_remote_code=True)

class EmbeddingGenerator:
    def __init__(self, model_path, token_path, model_args):
        # Initialize tokenizer and model for embedding
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.configuration = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        if model_path == 'mosaicml/mpt-7b':
            self.model = MptModel(MptConfig())

        else:
            self.model = AutoModel.from_pretrained(model_path, 
                                                        #config = self.configuration, 
                                                        **model_args if model_args is not None else {},
                                                        trust_remote_code=True)

    # The model works really well with cls pooling (default) but also with mean poolin.
    def pooling(self, outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs
    
    
    def generate_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        #print("[INFO] Inputs: ", inputs)
        print("[INFO] Length of inputs: ", len(inputs['input_ids'][0]))
        with torch.no_grad():
            # Check if model has 'encoder' attribute
            #if hasattr(self.model, 'encoder'):
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state
        embeddings = self.pooling(embeddings, inputs, strategy='mean')
        print(np.linalg.norm(embeddings))
        embeddings = F.normalize(embeddings)
        print("Shape: ", embeddings.shape)
        print(np.linalg.norm(embeddings))
        embeddings = embeddings.numpy().tolist()
        return embeddings[0]

# Inherit from the EmbeddingFunction class to implement our custom embedding function
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path, token_path,model_args):
        self.embedding_generator = EmbeddingGenerator(model_path, token_path, model_args)

    def __call__(self, texts: Documents) -> Embeddings:
        return list(map(self.embedding_generator.generate_embeddings, texts))