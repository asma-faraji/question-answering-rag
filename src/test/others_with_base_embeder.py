import os
import openai
from dataclasses import dataclass

import yaml
import pandas as pd
import logging
import sys
from pprint import pprint

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document
)

from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
# from llama_index.text_splitter import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts.prompts import SimpleInputPrompt

from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)


import os
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
import torch



@dataclass
class _Settings:

    system_prompt = """You are a Q&A assistant. 
    Your goal is to answer questions as accurately as possible based on the instructions and context providedin with detail
    and please provide reference for your answers and do not generate information on yourself and do not generate incomplete sentences. """

    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    # embed_model = HuggingFaceEmbedding(model_name=embed_model_name, max_length=512)

    _embed_model_name = None
    _embed_model = None
    _llm = None

    @property
    def llm(self):
        """Get the LLM."""
        if self._llm is None:
            self._llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            system_prompt=self.system_prompt,
            query_wrapper_prompt=self.query_wrapper_prompt,
            tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            device_map="auto",
            # uncomment this if using CUDA to reduce memory usage
            # model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
        )

        # if self._callback_manager is not None:
        #     self._llm.callback_manager = self._callback_manager

        return self._llm

    @property
    def embed_model(self):
        """Get the Sentence Embedder."""
        if self._embed_model is None:
            self._embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, max_length=512)
            
        return self._embed_model

# test inheriting from base retrival
# https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers.html
class Retriever:

        
    def __init__(self,indexer_db, similarity_top_k=100, rerank_top_n=5):
        
        

        self.sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text")        
    
    
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        
        sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=indexer_db),
        llm=Setting.llm, embed_model=Setting.embed_model, callback_manager=callback_manager
        )
    
    
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
    
        self.sentence_window_engine = sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank],llm=Setting.llm
        )
    def query(self, query):
        llm_response =  self.sentence_window_engine.query(query)
        return llm_response
        

with open('./config.yml', 'r') as file:
    config = yaml.safe_load(file)['others-test-with-lenel-finetuned-embedder']

## config 
Setting = _Settings()
Setting.embed_model_name = config['embed_mdoel_name']
test_file = config['test_file']
indexer_db = config['indexer_db']
result_file = config['result_file']

import json
with open(test_file, 'r') as f:
    test_json = json.load(f)


import time
retriever = Retriever(indexer_db)

qa_llm_response_dict = {'query' :[] , 'llm_response' :[] , 'refrence_response' :[]}
for i in test_json.keys():
    file_path = test_json[i]['retrieval_data']['1']['file_path'] + '/'  + test_json[i]['retrieval_data']['1']['file_name']
    if 'Other' in file_path:
        print(i)        
        query = test_json[i]['query']
        # if 'QICL' in query:
        #     query = query.replace('QICL', '')
            
        print(query)
        response = test_json[i]['response']
        start = time.time()
        llm_response = retriever.query(query)
        print(llm_response)
        end = time.time()
        print(end - start)
        qa_llm_response_dict['query'].append(query)
        qa_llm_response_dict['llm_response'].append(llm_response)
        qa_llm_response_dict['refrence_response'].append(response)


df = pd.DataFrame(qa_llm_response_dict, columns=list(qa_llm_response_dict.keys()))
df.to_csv(result_file)
