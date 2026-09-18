"""Command-line behavior tests that avoid downloading the embedding model."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main


class CommandLineTest(unittest.TestCase):
    def test_documents_stop_before_the_next_option_and_results_show_provenance(self) -> None:
        source_one = Path("handbook.pdf").resolve()
        source_two = Path("notes.txt").resolve()
        result = [(0.8765, {"source": str(source_two), "document_id": 1, "chunk_id": 2, "text": "The warranty covers defects."})]
        metadata = {"chunks": [{}, {}], "documents": [{}, {}]}
        stdout = io.StringIO()
        with (
            patch("main.collect_files", return_value=[source_one, source_two]) as collect,
            patch("main.build_index", return_value=metadata) as build,
            patch("main.retrieve", return_value=result) as retrieve,
            redirect_stdout(stdout),
        ):
            code = main.main([
                "--documents", "handbook.pdf", "notes.txt", "./more_documents",
                "--query", "What does the warranty cover?",
                "--index-dir", "./rag_index",
                "--chunk-size", "900", "--overlap", "180", "--top-k", "3",
            ])

        self.assertEqual(code, 0)
        self.assertEqual(collect.call_args.args[0], ["handbook.pdf", "notes.txt", "./more_documents"])
        self.assertEqual(build.call_args.args[1:], (Path("rag_index"), 900, 180))
        self.assertEqual(retrieve.call_args.args, ("What does the warranty cover?", Path("rag_index"), 3))
        self.assertIn("Similarity: 0.8765", stdout.getvalue())
        self.assertIn(f"Source: {source_two}", stdout.getvalue())

    def test_existing_index_query_does_not_attempt_to_index_documents(self) -> None:
        with patch("main.build_index") as build, patch("main.collect_files") as collect, patch("main.retrieve", return_value=[]):
            self.assertEqual(main.main(["--query", "existing index query"]), 0)
        build.assert_not_called()
        collect.assert_not_called()

    def test_empty_query_returns_a_clear_error(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main.main(["--query", "   "]), 2)
        self.assertIn("query cannot be empty", stderr.getvalue())

    def test_invalid_overlap_is_rejected_by_argument_validation(self) -> None:
        with self.assertRaises(SystemExit) as exit_error, redirect_stderr(io.StringIO()):
            main.parse_args(["--query", "test", "--chunk-size", "10", "--overlap", "10"])
        self.assertEqual(exit_error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
