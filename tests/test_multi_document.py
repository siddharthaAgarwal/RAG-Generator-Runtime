"""Regression test for indexing and reloading more than one document.

The embedding stub keeps this test deterministic and offline; FAISS persistence is real.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import rag_generator


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class DeterministicEmbedder:
    """Maps warranty and benefits language to distinct normalized vectors."""

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if "warranty" in text.lower() or "manufacturing defects" in text.lower() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class MultiDocumentRagTest(unittest.TestCase):
    def test_model_loader_caches_the_same_model_instance(self) -> None:
        rag_generator._MODELS.clear()
        with patch("rag_generator.SentenceTransformer", return_value=object()) as constructor:
            first = rag_generator.get_model(rag_generator.MODEL_NAME)
            second = rag_generator.get_model(rag_generator.MODEL_NAME)
        self.assertIs(first, second)
        constructor.assert_called_once_with(rag_generator.MODEL_NAME)
        rag_generator._MODELS.clear()

    def test_sources_survive_saved_index_reload(self) -> None:
        embedder = DeterministicEmbedder()
        source_files = [FIXTURES / "benefits.txt", FIXTURES / "support.txt"]
        with tempfile.TemporaryDirectory() as directory, patch("rag_generator.get_model", return_value=embedder) as get_model:
            index_dir = Path(directory) / "index"
            rag_generator.build_index(source_files, index_dir, chunk_size=120, overlap=20)

            metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((index_dir / "index.faiss").exists())
            self.assertEqual(len(metadata["documents"]), 2)
            self.assertEqual({Path(chunk["source"]).name for chunk in metadata["chunks"]}, {"benefits.txt", "support.txt"})

            # retrieve() loads the files afresh, proving vector IDs retain original source mapping.
            results = rag_generator.retrieve("What does the warranty cover?", index_dir, top_k=2)
            self.assertEqual(Path(results[0][1]["source"]).name, "support.txt")
            self.assertEqual(len(results), 2)
            self.assertTrue(all("source" in record for _, record in results))
            self.assertEqual(get_model.call_args_list[0].args, (rag_generator.MODEL_NAME,))
            self.assertEqual(get_model.call_args_list[1].args, (rag_generator.MODEL_NAME,))


if __name__ == "__main__":
    unittest.main()
