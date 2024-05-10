
import json
import logging
import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.finetuning import SentenceTransformersFinetuneEngine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class FineTunePipeline:

    embbed_model_name='sentence-transformers/all-mpnet-base-v2'
    llm_model_name='mistralai/Mistral-7B-Instruct-v0.1'
    llm_tokenizer_name='mistralai/Mistral-7B-Instruct-v0.1'
    train_dataset_path='train_dataset.json'
    val_dataset_path='val_dataset.json'

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        tokenizer_name=llm_tokenizer_name,
        model_name=llm_model_name,
        device_map="auto",
        tokenizer_kwargs={"max_length": 4096},
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16}
    )
    
    # def __init__(self, embbed_model_name='sentence-transformers/all-mpnet-base-v2', 
    #             llm_model_name='mistralai/Mistral-7B-Instruct-v0.1', 
    #             llm_tokenizer_name='mistralai/Mistral-7B-Instruct-v0.1',
    #             train_dataset_path='train_dataset.json',
    #             val_dataset_path='val_dataset.json') -> None:

    @classmethod
    def load_corpus(cls, files, verbose=False):
        if verbose:
            print(f"Loading files {files}")

        reader = SimpleDirectoryReader(input_files=files)
        docs = reader.load_data()
        if verbose:
            print(f"Loaded {len(docs)} docs")

        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

        if verbose:
            print(f"Parsed {len(nodes)} nodes")

        return nodes

    @classmethod
    def generate_embedding_pairs(cls, train_nodes, val_nodes):

        train_dataset = generate_qa_embedding_pairs(
        llm=cls.llm, nodes=train_nodes
        )
        val_dataset = generate_qa_embedding_pairs(
            llm=cls.llm, nodes=val_nodes
        )

        train_dataset.save_json(cls.train_dataset_path)
        val_dataset.save_json(cls.val_dataset_path)


    @classmethod
    def fine_tune_embeddings(cls):


        train_dataset = EmbeddingQAFinetuneDataset.from_json(cls.train_dataset_path)
        val_dataset = EmbeddingQAFinetuneDataset.from_json(cls.val_dataset_path)
        finetune_engine = SentenceTransformersFinetuneEngine(
        epochs=10,
        train_dataset,
        model_id=cls.embbed_model_name,
        model_output_path="test_model",
        val_dataset=val_dataset,
        )

        finetune_engine.finetune()
        embed_model = finetune_engine.get_finetuned_model()

        return embed_model



    
fine_tuner = FineTunePipeline()

print('load datasets')

TRAIN_FILES = ['path/to/folder']
VAL_FILES = ['/path/to/folder']

train_nodes = FineTunePipeline.load_corpus(TRAIN_FILES, verbose=True)
val_nodes = FineTunePipeline.load_corpus(VAL_FILES, verbose=True)

print('generate qa')
FineTunePipeline.generate_embedding_pairs(train_nodes, val_nodes)

print('train_model')
embed_model = FineTunePipeline.fine_tune_embeddings()