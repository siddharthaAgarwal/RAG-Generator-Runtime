#!/usr/bin/env python3
"""Command-line entry point for the local RAG retriever."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_generator import build_index, collect_files, retrieve


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index PDF, TXT, and DOCX documents locally, then retrieve relevant context."
    )
    parser.add_argument(
        "--documents",
        "-d",
        nargs="+",
        metavar="PATH",
        help="One or more document files or directories to index before querying.",
    )
    parser.add_argument("--query", "-q", required=True, help="Question to search for in the index.")
    parser.add_argument("--index-dir", type=Path, default=Path("rag_index"), help="Directory containing the saved index.")
    parser.add_argument("--chunk-size", type=positive_integer, default=900, help="Approximate characters per chunk when indexing.")
    parser.add_argument("--overlap", type=int, default=180, help="Overlapping characters between chunks when indexing.")
    parser.add_argument("--top-k", type=positive_integer, default=3, help="Number of retrieved chunks to display.")
    return parser.parse_args()


def print_results(question: str, results: list[tuple[float, dict]]) -> None:
    print(f"\nQuery: {question}\n")
    print(f"Retrieved {len(results)} result(s):")
    context_parts: list[str] = []
    for rank, (score, record) in enumerate(results, start=1):
        source = record["source"]
        print(f"\n[{rank}] Similarity: {score:.4f}")
        print(f"Source: {source} (document {record['document_id']}, chunk {record['chunk_id']})")
        print(f"Text: {record['text']}")
        context_parts.append(f"Source: {source} (chunk {record['chunk_id']})\n{record['text']}")
    print("\nAnswer context:\n")
    print("\n\n---\n\n".join(context_parts))


def main() -> int:
    args = parse_args()
    question = args.query.strip()
    if not question:
        print("Error: query cannot be empty.", file=sys.stderr)
        return 2
    try:
        if args.documents:
            files = collect_files(args.documents)
            if not files:
                raise FileNotFoundError("No valid PDF, TXT, or DOCX files were found in --documents.")
            metadata = build_index(files, args.index_dir, args.chunk_size, args.overlap)
            print(f"Indexed {len(metadata['chunks'])} chunks from {len(metadata['documents'])} document(s).")
            print(f"Saved index: {args.index_dir.resolve()}")
        results = retrieve(question, args.index_dir, args.top_k)
        print_results(question, results)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: unable to complete RAG request: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
