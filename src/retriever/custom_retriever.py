# import QueryBundle
from llama_index.core import QueryBundle
from dataclasses import dataclass

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List

import os
import openai

import logging
import sys
from pprint import pprint
from dataclasses import dataclass

# from .settings.Setting import Setting
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
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
# from llama_index.text_splitter import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
import os
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)

from llama_index.core import Settings

@dataclass
class _CustomSettings:

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
            self._embed_model = HuggingFaceEmbedding(model_name=self._embed_model_name, max_length=512)
            
        return self._embed_model

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        ## load from indexer
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
            print(retrieve_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


CustomSetting = _CustomSettings()
Settings.llm = CustomSetting.llm
Settings.embed_model = CustomSetting.embed_model

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

print('loading vector indexer')
vector_index = load_index_from_storage(StorageContext.from_defaults(persist_dir='./test_pipeline/all-mpnet-base-v2_sentence_idx_other'), callback_manager=callback_manager
)

print('keyword indexer')
keyword_index = load_index_from_storage(StorageContext.from_defaults(persist_dir='./test_pipeline/other_keyword_indexer'), callback_manager=callback_manager,)

vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)

print('creating custom keyword indexer')
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

# define response synthesizer
response_synthesizer = get_response_synthesizer(llm=Settings.llm)

# assemble query engine
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

print('answer : ')
response = custom_query_engine.query(
"query"
)

print(response)