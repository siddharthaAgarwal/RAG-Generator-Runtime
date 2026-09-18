# Local Python RAG Generator

A small local retrieval pipeline for PDF, TXT, and DOCX files. It extracts text, creates overlapping chunks, embeds them with Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`, and retrieves matching context from a local FAISS HNSW index. It uses plain Python—no LangChain and no hosted vector database.

## Expected project layout

Run commands from the project directory. For example:

```text
RAG-GENERATOR-RUNTIME/
├── main.py
├── rag_generator.py
├── requirements.txt
└── documents/
    ├── handbook.pdf
    └── notes.txt
```

The `documents/` directory is only an example; you can pass any supported file or directory path.

## Setup

Python 3.10–3.12 is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first indexing run downloads and caches the embedding model. Seeing this message is expected:

```text
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
```

## Index documents and query

This command indexes the two files under `documents/`, saves the index under `./rag_index`, and retrieves the top three matches:

```bash
python main.py \
  --documents documents/handbook.pdf documents/notes.txt \
  --query "What does the warranty cover?" \
  --index-dir ./rag_index \
  --chunk-size 900 \
  --overlap 180 \
  --top-k 3
```

You can pass directories as well as files. Directory inputs are searched recursively for `.pdf`, `.txt`, and `.docx` files:

```bash
python main.py \
  --documents ./documents \
  --query "What does the warranty cover?"
```

The command displays each retrieved chunk, its similarity score, source path, document/chunk IDs, and the combined answer context.

The trailing `\` characters only continue a multi-line shell command. Use a single backslash; they are not part of a path.

## Query an existing index

After an index has been created, query it without passing `--documents`:

```bash
python main.py \
  --query "What does the warranty cover?" \
  --index-dir ./rag_index \
  --top-k 3
```

Keep these files together:

```text
rag_index/
├── index.faiss
└── metadata.json
```

`index.faiss` stores the vectors. `metadata.json` stores chunk text, source paths, document IDs, model settings, and index settings.

## Command options

| Option | Required | Default | Purpose |
|---|---:|---:|---|
| `--documents PATH ...` | Only when creating/replacing an index | — | One or more files or directories |
| `--query TEXT` | Yes | — | Question to retrieve context for |
| `--index-dir PATH` | No | `./rag_index` | Saved FAISS index directory |
| `--chunk-size N` | No | `900` | Approximate characters per chunk |
| `--overlap N` | No | `180` | Characters shared between chunks |
| `--top-k N` | No | `3` | Number of results to retrieve |

Run `python main.py --help` for the same options in the terminal.

## Troubleshooting

- `No valid PDF, TXT, or DOCX files were found`: check the paths and file extensions.
- `Index files are missing`: run an indexing command first, or provide the correct `--index-dir`.
- `query cannot be empty`: provide text after `--query`.
- Scanned/image-only PDFs require OCR and may produce no extractable text.

## Tests

Run the offline regression tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
