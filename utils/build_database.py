# Making a specific embedding class for MPTModel
#from llmfoundry.models import MPTModel, MPTConfig

import torch 
from transformers import AutoTokenizer, MptModel, MptConfig

from typing import Any, List, Mapping

#import pysqlite3
import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain.docstore.document import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
import chromadb

import random
import re
# Pseudocode for new database setup:

# Build database with initally N docs with database.from_documents() and select random indices
    # For each question
        # Check if question 1 doc exists in the database:
            # If it does:
                # Use the model and extract the answer
            # If it doesn't:
                # Remove a random document with database.remove_document() and add the question document with database.add_document()

class Database:
    def __init__(self, docs, ids, embedding, chunk_params = {'separators':['.'," ", ","], 'chunk_size':100, 'chunk_overlap':0}):
        self.chunk_params = chunk_params
        self.unique_ids = {}
        self.chunk_ids = {}
        self.embedding = embedding

        #self.db = self.build_db(docs, ids, self.chunk_params)
        self.ids = ids
        # Create a Chroma client
        self.client = chromadb.Client()

        # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
        self.db = self.client.create_collection(name="test", 
                                                embedding_function=embedding, 
                                                metadata={"hnsw:space": "ip", "M": 2048, "ef": 256}) 
        
        self.build_db(docs, ids, self.chunk_params)

    def build_db(self, texts, ids, chunk_params = {'separators':['.'," ", ","], 'chunk_size':100, 'chunk_overlap':0}):
        # adding meta data?
        # Extract embeddings from test examples

        # Create a dictionary to keep track of unique IDs
        # Generate unique IDs
        document_ids = []

        print("[INFO]: Chunking...")

        document_ids = [{'id': id} for id in ids]
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, document_ids)]
        documents = chunkenizer(documents, **self.chunk_params)
        print("Document ids: ", document_ids)
        ids=[str(d.metadata["id"]) for d in documents]
        documents = [d.page_content for d in documents]
        

        # Generate unique IDs
        for original_id in ids:
            if original_id not in self.unique_ids:
                self.unique_ids[original_id] = 1
                self.chunk_ids[original_id]=[f"{original_id}_{self.unique_ids[original_id]}"]

            else:
                self.unique_ids[original_id] += 1
                self.chunk_ids[original_id].append(f"{original_id}_{self.unique_ids[original_id]}")

        
        chunks_list = [item for value in self.chunk_ids.values() for item in value]
        
        # for id, doc in enumerate(documents):
        #     doc.metadata = {"id": chunks_list[id]}

        print("[INFO]: chunks_list: ", chunks_list)
        print("[INFO]: Chunking done, chunks created: ", len(documents), " Parameters: ", self.chunk_params)
        print("[INFO]: Building database...")

        self.db.add(documents=documents, ids=chunks_list)

        #

        # print("[INFO]: Documents: ")
        # print(documents)
        # db = Chroma.from_documents(documents=documents,
        #                            ids=chunks_list,
        #                             embedding=self.embedding,
        #                             collection_metadata={"hnsw:space": "cosine"}) #https://docs.trychroma.com/usage-guide#changing-the-distance-function
        

        # db = Chroma.from_documents(documents=documents,
        #                            ids=chunks_list,
        #                             embedding=MptModelEmbeddings(tokenizer_name='EleutherAI/gpt-neox-20b', 
        #                                                          chunk_size=chunk_params['chunk_size'],
        #                             config=configuration(attn_config={"attn_impl":"torch"})),
        #                             collection_metadata={"hnsw:space": "cosine"}) #https://docs.trychroma.com/usage-guide#changing-the-distance-function
        print("[INFO]: Database built...")
    
    def check_and_add(self, question_index, text):
        print("[INFO]: Unique ids: ", self.unique_ids)
        if  self.unique_ids.get(question_index) is not None and self.unique_ids.get(question_index) != 0:
            print(f"[INFO]: Question index {question_index} already in database...")
            pass

        else:
            print(f"[INFO]: Question index {question_index} not in database. Removing random doc and adding doc to db..")
            # Select a random index among the docs
            #s = str(random.choice(self.document_ids))
            s = str(random.choice(list(self.unique_ids.keys())))
            # Select all the indices starting with s using regex
            document_id = {"id": question_index}
            documents = [Document(page_content=text, metadata=document_id)]
            documents = chunkenizer(documents, **self.chunk_params)
            
            ids=[str(d.metadata["id"]) for d in documents]
            documents = [d.page_content for d in documents]
            # if document_id not in self.unique_ids:
            #     self.unique_ids[document_id] = 1
            # else:
            for original_id in ids:
                if original_id not in self.unique_ids:
                    self.unique_ids[original_id] = 1
                    self.chunk_ids[original_id]=[f"{original_id}_{self.unique_ids[original_id]}"]

                else:
                    self.unique_ids[original_id] += 1
                    self.chunk_ids[original_id].append(f"{original_id}_{self.unique_ids[original_id]}")

            # print(documents)
            # print(self.chunk_ids)
            
            # for id, doc in enumerate(documents):
            #     doc.metadata = {"id": self.chunk_ids[str(question_index)][id]}
            
            self.db.add(documents=documents, ids=self.chunk_ids[str(question_index)])

            pattern = f'{s}_\d*'
            matching_indices = [str(i) for i in self.chunk_ids[s] if re.match(pattern, str(i))]

            print(f"Random index: {s}")
            print(f"Matching indices: {', '.join(matching_indices)}")
            self.db.delete(ids=matching_indices)
            #Remove id from id list
            #self.unique_ids[s]=0
            del self.unique_ids[s]
            del self.chunk_ids[s]


            print(f"[INFO]: Question added to database...")


class CustomTextSplitter(RecursiveCharacterTextSplitter):
    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer, chunk_size=100, chunk_overlap=0, separators=[".", "\n"]):
        def length_function(text):
            return len(tokenizer.tokenize(text))

        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators, length_function=length_function)

def chunkenizer(docs: list, chunk_size: int, chunk_overlap:int, separators:list):
    print("[INFO]: Chunking documents...")
    print("[INFO]: Chunk size: ", chunk_size, " Chunk overlap: ", chunk_overlap, " Separators: ", separators)   
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    splitter = CustomTextSplitter.from_huggingface_tokenizer(tokenizer, separators=separators)

    docs = splitter.split_documents(docs)
    return docs