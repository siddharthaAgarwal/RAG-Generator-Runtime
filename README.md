# Local Python RAG Generator

A small command-line RAG retriever with no LangChain or hosted vector database. It extracts PDF, TXT, and DOCX files, creates sentence-aware chunks, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and searches a local FAISS HNSW index using cosine similarity.

## Setup

Use Python 3.10–3.12, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the free Hugging Face embedding model and caches it locally.

## Run

Index one or more files (or directories) and ask a question in one command:

```bash
python main.py \
  --documents handbook.pdf notes.txt ./more_documents \
  --query "What does the warranty cover?" \
  --index-dir ./rag_index \
  --chunk-size 900 --overlap 180 --top-k 3
```

The command prints each retrieved chunk with its cosine-similarity score, source document, document/chunk IDs, and an answer-context block.

To query an existing index without re-indexing documents:

```bash
python main.py --query "What does the warranty cover?" --index-dir ./rag_index --top-k 3
```

The saved `rag_index/` directory contains `index.faiss` and `metadata.json`. Keep both files together. Invalid paths, unsupported files, empty queries, and missing indexes produce clear errors without changing the saved index.

## Check

```bash
python -m unittest tests/test_multi_document.py
```
