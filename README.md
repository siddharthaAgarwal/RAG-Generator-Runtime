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

This writes three local files to `rag_index/`:

- `index.faiss` — FAISS HNSW vector index
- `chunks.json` — chunk text and source metadata
- `config.json` — model and indexing settings

Re-run `index` to replace the saved index with the supplied document set.

## Retrieve context

```bash
python rag_generator.py query "What are the cancellation terms?" --index-dir ./rag_index
```

The query command embeds the question, retrieves the top three chunks by cosine similarity, prints their scores and provenance, then prints an **Answer context** section. That context is deliberately kept separate from generation so it can be passed to any LLM later.

## Implementation notes

- Embeddings are L2-normalized, so FAISS inner-product search equals cosine similarity.
- `IndexHNSWFlat` uses HNSW graph search locally (`M=32`, `efConstruction=80`, `efSearch=64`).
- PDF extraction works for PDFs with embedded/selectable text. Scanned PDFs require OCR, which is intentionally outside this first version.
