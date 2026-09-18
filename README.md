# Plain Python RAG Generator

A small local retrieval-augmented generation (RAG) pipeline with no orchestration framework. It accepts PDF, TXT, and DOCX documents; extracts their text; creates sentence-aware overlapping chunks; embeds them with Hugging Face's free `sentence-transformers/all-MiniLM-L6-v2` model; and searches a local FAISS HNSW index using cosine similarity.

## Setup

Use Python 3.10+ and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first indexing or query run downloads the embedding model from Hugging Face and caches it locally.

## Index documents

Pass individual files, directories, or both. Directories are searched recursively for `.pdf`, `.txt`, and `.docx` files.

```bash
python rag_generator.py index ./documents policy.docx notes.txt --index-dir ./rag_index
```

Optional chunk controls:

```bash
python rag_generator.py index ./documents --chunk-size 900 --overlap 180
```

This writes two reusable local files to `rag_index/`:

- `index.faiss` — FAISS HNSW vector index
- `metadata.json` — chunk text, vector-to-chunk mapping, per-document source information, and model/index settings

Re-run `index` with any number of documents to replace the saved index with that document set. The metadata is validated against the FAISS vector count every time a query reloads it.

## Retrieve context

```bash
python rag_generator.py query "What are the cancellation terms?" --index-dir ./rag_index --top-k 3
```

The query command embeds the question with the same saved model setting used at indexing time, retrieves the requested number of chunks by cosine similarity, prints their scores and provenance, then prints an **Answer context** section. That context is deliberately kept separate from generation so it can be passed to any LLM later.

`--chunk-size`, `--overlap`, and `--top-k` are all configurable. Within one Python process, the embedding model is cached and reused for every document and query embedding request.

## End-to-end check

The included regression test indexes two separate text files, reloads the real FAISS index, and verifies that source provenance remains intact. It uses a deterministic local embedding stub so this storage regression test is fast and offline:

```bash
python -m unittest tests/test_multi_document.py
```

## Implementation notes

- Embeddings are L2-normalized, so FAISS inner-product search equals cosine similarity.
- `IndexHNSWFlat` uses HNSW graph search locally (`M=32`, `efConstruction=80`, `efSearch=64`).
- PDF extraction works for PDFs with embedded/selectable text. Scanned PDFs require OCR, which is intentionally outside this first version.
