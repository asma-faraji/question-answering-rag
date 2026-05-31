# Question Answering RAG

A Retrieval-Augmented Generation (RAG) project for building a PDF-based question-answering system using LlamaIndex, Hugging Face LLMs, sentence-transformer embeddings, sentence-window retrieval, reranking, and optional embedding fine-tuning.

## Project Overview

This project loads PDF documents, splits them into sentence-window nodes, builds persistent indexes, retrieves relevant context for a user query, and generates grounded answers using a local Hugging Face language model.

## Main Features

- PDF document loading
- Sentence-window indexing
- Vector search
- Keyword search
- Hybrid retrieval
- Document summary indexing
- Reranking with `BAAI/bge-reranker-base`
- Local Hugging Face LLM inference
- Condense-question chat engine for multi-turn QA
- Optional embedding fine-tuning

## Repository Structure

```text
src/
├── Settings/
│   ├── Setting.py
│   └── __init__.py
├── indexer/
│   ├── indexer.py
│   ├── keyword_indexer.py
│   └── summary_indexer.py
├── retriever/
│   ├── base_retriever.py
│   ├── custom_retriever.py
│   └── retrieval_pipeline.py
├── condense_chat_engine/
│   ├── condense_chat_engine.py
│   └── config.yml
├── finetuning/
│   └── embedding_finetune.py
└── test/
    ├── chat_engine.py
    ├── condense_chat_engine.py
    ├── config.yml
    ├── one_query_test.py
    ├── others_with_base_embeder.py
    └── query_decomposition_test.py
```

## Main Components

### Settings

`src/Settings/Setting.py` defines shared model settings, including the system prompt, query wrapper prompt, Hugging Face LLM, and Hugging Face embedding model.

### Indexing

The `src/indexer/` folder contains scripts for building different document indexes.

- `indexer.py`: builds a vector index over PDF documents.
- `keyword_indexer.py`: builds a keyword-based index.
- `summary_indexer.py`: builds a document summary index.

### Retrieval

The `src/retriever/` folder contains retrieval pipelines.

- `base_retriever.py`: loads a persisted vector index and queries it using sentence-window retrieval, metadata replacement, and reranking.
- `custom_retriever.py`: implements a hybrid retriever that combines vector retrieval and keyword retrieval.

### Condense Chat Engine

The `src/condense_chat_engine/condense_chat_engine.py` script creates an interactive multi-turn chat engine. It rewrites follow-up questions into standalone questions before retrieving context and generating answers.

### Embedding Fine-Tuning

The `src/finetuning/embedding_finetune.py` file contains a pipeline for generating synthetic question-answer pairs and fine-tuning a sentence-transformer embedding model.

## Models Used

The main LLM used in the code is:

```text
mistralai/Mistral-7B-Instruct-v0.1
```

The default embedding model is:

```text
sentence-transformers/all-mpnet-base-v2
```

The reranker is:

```text
BAAI/bge-reranker-base
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the main dependencies:

```bash
pip install llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface llama-index-postprocessor-sbert-rerank sentence-transformers transformers torch pandas pyyaml tqdm nest-asyncio accelerate bitsandbytes
```

## Configuration

Before running the project, replace placeholder paths such as:

```text
path/to/folder
```

with your actual document, embedding, index, and result paths.

Some scripts currently use hard-coded local paths, so update them before execution.

In the config files, note that the key `embed_mdoel_name` is currently misspelled in the code. Keep the same spelling unless you also update the Python script.

## Usage

### Build a Vector Index

Edit the paths in:

```text
src/indexer/indexer.py
```

Then run:

```bash
python src/indexer/indexer.py
```

### Query a Persisted Index

Update the index path in the retriever script and run:

```bash
python src/retriever/base_retriever.py
```

### Run the Condense Chat Engine

Update:

```text
src/condense_chat_engine/config.yml
```

Then run:

```bash
cd src/condense_chat_engine
python condense_chat_engine.py
```

### Fine-Tune the Embedding Model

Edit the training and validation paths in:

```text
src/finetuning/embedding_finetune.py
```

Then run:

```bash
python src/finetuning/embedding_finetune.py
```

## RAG Pipeline

```text
PDF Documents
      ↓
SimpleDirectoryReader
      ↓
SentenceWindowNodeParser
      ↓
VectorStoreIndex / Keyword Index / Summary Index
      ↓
Retriever
      ↓
Reranker
      ↓
LLM
      ↓
Final Answer
```

## Notes

This repository is currently research/prototype code.

Recommended improvements:

- Add a `requirements.txt` file.
- Remove hard-coded paths.
- Add command-line arguments for document paths and index paths.
- Add example data or a small demo PDF.
- Add unit tests.
- Add a Streamlit or FastAPI interface.
- Standardise naming, for example fixing `embed_mdoel_name` to `embed_model_name`.

## Author

Asma Faraji

## License

No license file is currently included. Add a license such as MIT or Apache-2.0 depending on how you want others to use the project.
