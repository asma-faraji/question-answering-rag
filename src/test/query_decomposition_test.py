print('start logging')
import os
import openai
from dataclasses import dataclass

import yaml
import pandas as pd
import logging
import sys
from pprint import pprint
import json
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document
)


from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
# from llama_index.text_splitter import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
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

def complete(llm, prompt, formatted= True, **kwargs
    ) -> CompletionResponse:
        """Completion endpoint."""
        full_prompt = prompt
        if not formatted:
            if llm.query_wrapper_prompt:
                full_prompt = llm.query_wrapper_prompt.format(query_str=prompt)
            if llm.system_prompt:
                full_prompt = f"{llm.system_prompt} {full_prompt}"

        inputs = llm._tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(llm._model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in llm.tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        tokens = llm._model.generate(
            **inputs,
            max_new_tokens=llm.max_new_tokens,
            stopping_criteria=llm._stopping_criteria,
            **llm.generate_kwargs,
        )
        completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
        completion = llm._tokenizer.decode(completion_tokens, skip_special_tokens=True)

        return CompletionResponse(text=completion, raw={"model_output": tokens})



@dataclass
class _Settings:

    system_prompt = """You are a Q&A assistant. 
    Your goal is to answer questions as accurately as possible based on the instructions and context providedin with detail
    and please provide reference for your answers and do not generate information on yourself and do not generate incomplete sentences. """

    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    # embed_model = HuggingFaceEmbedding(model_name=embed_model_name, max_length=512)

    _embed_model_name = "sentence-transformers/all-mpnet-base-v2"
    _embed_model = None
    _llm = None

    @property
    def llm(self):
        """Get the LLM."""
        if self._llm is None:
            self._llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=512,
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

class Engine:

        
    def __init__(self,indexer_db, similarity_top_k=100, rerank_top_n=5):
        
        

        self.sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text")        
    
    
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        
        self.sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=indexer_db),
        llm=Setting.llm, embed_model=Setting.embed_model, callback_manager=callback_manager
        )
    
    
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
    
        self.sentence_window_engine = self.sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank],llm=Setting.llm
        )

        self.retriever_engine = self.sentence_index.as_retriever(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank],llm=Setting.llm)
        
    def query(self, query):
        llm_response =  self.sentence_window_engine.query(query)
        return llm_response
        

## config 
Setting = _Settings()
with open('./config.yml', 'r') as file:
    config = yaml.safe_load(file)['others-test-with-base-embedder']

## config 
Setting = _Settings()
Setting.embed_model_name = config['embed_mdoel_name']
test_file = config['test_file']
indexer_db = config['indexer_db']
result_file = config['result_file']

# Setting.embed_model_name = "sentence-transformers/all-mpnet-base-v2"
# indexer_db = './sentence_index_t5120_30'



engines = Engine(indexer_db)
print(engines.sentence_index)
print(engines.sentence_window_engine)
print(engines.retriever_engine)

query = "what are the  grounding requirements for Standard Duty site as per Motorola R56?"
query = "What are the differences between the grounding requirements for Light Duty and Standard Duty type sites as per Motorola R56?"
query = "Summarise grounding requirements for Standard Duty site as per Motorola R56. Explain the requirements for each applicable grounding component"
# query = "As per Motorola R56, how many grounding systems should there be at a communication site and why?"

prompt = f"""Given the question, decompose it into separate questions that 
            would help in understanding and addressing the original query. 
            Provide each decomposed question as a separate query.
            As an example : 
            query : Summarize grounding requirements for Light Duty site as per Motorola R56 and Explain the requirements for each applicable grounding component.
            Answer : 
            what is the Light Duty site?
            what is the grounding requirements for Light Duty site as per Motorola R56?
            Explain the requirements for each applicable grounding component.

            query : {query}
            Answer : 
            """.format(query=query)

prompt = f"""For the following question, split into separate, single-answer questions 
            query : {query}
            please put each question on a new line starting with "-"
            """.format(query=query)

# prompt = f"""query : Provide minimum qty. of LNL-2220 and LNL-1320 required to interface 50 doors/ access readers to  a Lenel Access Control System. Provide justification for your answer.
#             answer : 
#             - Provide minimum qty. of LNL-2220 and LNL-1320 required to interface 50 doors/ access readers to  a Lenel Access Control System.
#             - Provide justification for your answer.

#             query : Summarise grounding requirements for Standard Duty site as per Motorola R56. Explain the requirements for each applicable grounding component.
#             answer :
#             - Summarise grounding requirements for Standard Duty site as per Motorola R56.
#             - Explain the requirements for each applicable grounding component of Standard Duty site as per Motorola R56.

#             query : Is 35mm2 grounding conductor sufficient for tower mounted Access Point? What is the minimum size required as per Motorola R56?
#             answer : 
#             - Is 35mm2 grounding conductor sufficient for tower mounted Access Point?
#             - What is the minimum size of grounding conductor required as per Motorola R56?

#             query : 
#             """.format(query=query)

# prompt = f"""For the following query, if there are multiple questions or instructions , split into separate questions or instructions . Otherwise, return 1: {query}
#             If applicable, please put each question on a new line starting with "-".
#             """.format(query=query)

query_dict = {'original_query' : [], 'decomposed_query' : [], 'llm_response':[], 'reference_response' :[] }

test_file = '/home/commtel/model/test_qa/Commtel_Captive_LLM_Test_QA_V3_12FEB24.json'
with open(test_file, 'r') as f:
    test_json = json.load(f)
for i in test_json.keys():
    file_path = test_json[i]['retrieval_data']['1']['file_path'] + '/'  + test_json[i]['retrieval_data']['1']['file_name']
    if 'Other' in file_path:
        print(i)        
        query = test_json[i]['query'].replace('Provide references to the answer', '')
        response = test_json[i]['response']
        prompt = f"""For the following question, split into separate, single-answer questions 
                    query : {query}
                    please put each question on a new line starting with "-"
                    """.format(query=query)
                
        print(query)
        decomposed_query = complete(Setting.llm, prompt=prompt, formatted=True)
        decomposed_query = str(decomposed_query).replace('-','').split('\n')



        print(len(decomposed_query))
        retrieve_nodes = []
        print(decomposed_query)
        # print(decomposed_query)
        # for q in decomposed_query:
        #     if q != '':
        #         print(q.rstrip().lstrip())
        #         nodes = engines.retriever_engine.retrieve(q.rstrip().lstrip())
        #         print(type(nodes))
        #         retrieve_nodes += nodes
        #         query_dict['original_query'].append(query)
        #         query_dict['decomposed_query'].append(q)
                
        # query_dict['reference_response'] += [response] * len(decomposed_query)
        # print(type(retrieve_nodes))
        # # retrieve_nodes = list(set(retrieve_nodes))
        # response_synthesizer = get_response_synthesizer(llm=Setting.llm)
        # result = response_synthesizer.synthesize(query, retrieve_nodes)
        # query_dict['llm_response'] + =[result] * len(decomposed_query)
        # print(result)

# result_file = './one_shot_other_decomposed_query.csv'
# df = pd.DataFrame(query_dict, columns=list(query_dict.keys()))
df.to_csv(result_file)