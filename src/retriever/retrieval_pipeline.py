import os
import openai

import logging
import sys
from pprint import pprint

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document
)

from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.schema import MetadataMode
from llama_index.postprocessor import MetadataReplacementPostProcessor
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt

from llama_index.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)


# create the sentence window node parser w/ default settings



system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context providedin with detail"



# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
sentence_node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)


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
ctx_sentence = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=sentence_node_parser, callback_manager=callback_manager)
# index = load_index_from_storage(
#     StorageContext.from_defaults(persist_dir="./storage"))

root = "path/to/folder"
documents = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if os.path.join(path, name).endswith(".pdf"):
            print(os.path.join(path, name))
            documents = documents + SimpleDirectoryReader(input_files=[os.path.join(path, name)]).load_data()


sentence_index = VectorStoreIndex(nodes, service_context=ctx_sentence)
