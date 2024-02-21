import os
import openai

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

system_prompt = """You are a Q&A assistant. 
    Your goal is to answer questions as accurately as possible based on the instructions and context providedin with detail
    and please provide reference for your answers and do not generate information on yourself and do not generate incomplete sentences. """



# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
sentence_node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

embed_model_name = "sentence-transformers/all-mpnet-base-v2"
indexer_db = f'../indexer/{embed_model_name}_sentence_idx'
embed_model = HuggingFaceEmbedding(model_name=embed_model_name, max_length=512)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)


llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=indexer_db),
llm=llm, embed_model=embed_model, callback_manager=callback_manager
)

similarity_top_k=100
rerank_top_n=5
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
rerank = SentenceTransformerRerank(
    top_n=rerank_top_n, model="BAAI/bge-reranker-base"
)

sentence_window_engine = sentence_index.as_query_engine(
    similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
)

query = "What's the purpose of CCTV and PIDS integration?"
llm_response =  sentence_window_engine.query(query)
