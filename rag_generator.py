#!/usr/bin/env python3
"""A lightweight local RAG retriever for PDF, TXT, and DOCX files.

Examples:
  python rag_generator.py index handbook.pdf product_notes.txt --index-dir rag_index
  python rag_generator.py query "How long is the warranty?" --index-dir rag_index --top-k 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
SCHEMA_VERSION = 1
_MODELS: dict[str, SentenceTransformer] = {}


def collect_files(inputs: list[str]) -> list[Path]:
    """Accept any number of supported files and/or directories, deduplicated by path."""
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if not path.exists():
            print(f"Warning: path not found, skipping: {path}", file=sys.stderr)
            continue
        candidates: Iterable[Path] = path.rglob("*") if path.is_dir() else (path,)
        files.extend(item.resolve() for item in candidates if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES)
    return sorted(set(files))


def extract_text(path: Path) -> str:
    """Extract text from a single supported document."""
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".docx":
        document = Document(str(path))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        cells = [cell.text for table in document.tables for row in table.rows for cell in row.cells]
        return "\n".join(paragraphs + cells)
    raise ValueError(f"Unsupported file type: {path}")


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if part.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Create sentence-aware, overlapping chunks without an NLP framework."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be from 0 to chunk_size - 1")
    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        for piece in (sentence[offset : offset + chunk_size] for offset in range(0, len(sentence), chunk_size)):
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = f"{current[-overlap:]} {piece}".strip() if overlap else piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load each model only once per process for both indexing and querying."""
    if model_name not in _MODELS:
        print(f"Loading embedding model: {model_name}")
        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


def make_records(file_paths: list[Path], chunk_size: int, overlap: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for document_id, path in enumerate(file_paths):
        try:
            chunks = chunk_text(extract_text(path), chunk_size, overlap)
        except Exception as error:
            print(f"Warning: could not read {path}: {error}", file=sys.stderr)
            continue
        source = str(path)
        documents.append({"document_id": document_id, "source": source, "chunks": len(chunks)})
        records.extend(
            {"vector_id": len(records), "document_id": document_id, "source": source, "chunk_id": chunk_id, "text": text}
            for chunk_id, text in enumerate(chunks)
        )
        print(f"{path.name}: {len(chunks)} chunks")
    return records, documents


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def save_index(index: faiss.Index, metadata: dict[str, Any], index_dir: Path) -> None:
    """Persist index and metadata without leaving partially written target files."""
    index_dir.mkdir(parents=True, exist_ok=True)
    target = index_dir / INDEX_FILE
    temporary = index_dir / f".{INDEX_FILE}.tmp"
    faiss.write_index(index, str(temporary))
    os.replace(temporary, target)
    atomic_json_write(index_dir / METADATA_FILE, metadata)


def load_index(index_dir: Path) -> tuple[faiss.Index, dict[str, Any]]:
    index_path, metadata_path = index_dir / INDEX_FILE, index_dir / METADATA_FILE
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Index files are missing in {index_dir}. Run the 'index' command first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported metadata schema. Rebuild this index with the current program.")
    if metadata.get("model") != MODEL_NAME:
        raise ValueError(f"Index uses {metadata.get('model')!r}; expected {MODEL_NAME!r}.")
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(metadata.get("chunks", [])):
        raise ValueError("Index and metadata disagree on vector count. Rebuild the index.")
    return index, metadata


def build_index(file_paths: list[Path], index_dir: Path, chunk_size: int, overlap: int) -> None:
    records, documents = make_records(file_paths, chunk_size, overlap)
    if not records:
        raise RuntimeError("No text chunks were extracted. Check that documents contain selectable text.")
    embeddings = get_model(MODEL_NAME).encode(
        [record["text"] for record in records], normalize_embeddings=True, show_progress_bar=True
    )
    vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    index.add(vectors)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "chunking": {"chunk_size": chunk_size, "overlap": overlap, "unit": "characters"},
        "index": {"type": "IndexHNSWFlat", "metric": "cosine similarity (normalized inner product)", "dimensions": int(vectors.shape[1]), "hnsw_m": 32},
        "documents": documents,
        "chunks": records,
    }
    save_index(index, metadata, index_dir)
    print(f"Saved {len(records)} chunks from {len(documents)} document(s) to {index_dir}")


def retrieve(question: str, index_dir: Path, top_k: int) -> list[tuple[float, dict[str, Any]]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    index, metadata = load_index(index_dir)
    vector = get_model(metadata["model"]).encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.ascontiguousarray(np.asarray(vector, dtype=np.float32)), min(top_k, index.ntotal))
    return [(float(score), metadata["chunks"][int(vector_id)]) for score, vector_id in zip(scores[0], ids[0]) if vector_id >= 0]


def query_index(question: str, index_dir: Path, top_k: int) -> None:
    results = retrieve(question, index_dir, top_k)
    print("\nRetrieved chunks:")
    for rank, (score, record) in enumerate(results, 1):
        print(f"\n[{rank}] cosine similarity: {score:.4f}")
        print(f"source: {record['source']} (document {record['document_id']}, chunk {record['chunk_id']})")
        print(record["text"])
    context = "\n\n---\n\n".join(
        f"Source: {record['source']} (chunk {record['chunk_id']})\n{record['text']}" for _, record in results
    )
    print("\nAnswer context (pass this to your preferred LLM):\n")
    print(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local, plain-Python RAG retrieval with FAISS HNSW.")
    commands = parser.add_subparsers(dest="command", required=True)
    index_command = commands.add_parser("index", help="Extract, chunk, embed, and persist one or more documents")
    index_command.add_argument("inputs", nargs="+", help="PDF, TXT, DOCX files and/or directories")
    index_command.add_argument("--index-dir", type=Path, default=Path("rag_index"))
    index_command.add_argument("--chunk-size", type=int, default=900, help="Approximate characters per chunk")
    index_command.add_argument("--overlap", type=int, default=180, help="Characters retained in successive chunks")
    query_command = commands.add_parser("query", help="Load a saved index and retrieve matching chunks")
    query_command.add_argument("question")
    query_command.add_argument("--index-dir", type=Path, default=Path("rag_index"))
    query_command.add_argument("--top-k", type=int, default=3, help="Number of relevant chunks to retrieve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "index":
            files = collect_files(args.inputs)
            if not files:
                raise FileNotFoundError("No supported PDF, TXT, or DOCX files were found.")
            build_index(files, args.index_dir, args.chunk_size, args.overlap)
        else:
            query_index(args.question, args.index_dir, args.top_k)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
