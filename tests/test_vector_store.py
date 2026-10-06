"""Native Chroma storage integration tests.

Embeddings are deterministic fakes; these tests verify persistence and cosine
math, not native ONNX inference quality.
"""

from __future__ import annotations

import tempfile
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import vector_store as storage


class FixedEmbeddingWorker:
    def __init__(self, *args, **kwargs):
        self.variant = False

    def embed(self, texts):
        result = []
        for text in texts:
            vector = [0.0] * 384
            axis = 1 if "insurance" in text.lower() else 0
            vector[axis] = 2.0  # Non-unit vectors must still yield true cosine.
            if self.variant:
                vector[2] = 0.1
            result.append(vector)
        return result

    def close(self):
        pass


def chunk(source, text, chunk_id, index=0):
    return {
        "source": source, "text": text, "chunk_id": chunk_id, "chunk_index": index,
        "contract_category": "Test", "char_start": 0, "char_end": len(text),
        "starts_at_boundary": True, "ends_at_boundary": True,
    }


class StorageIntegrationTests(unittest.TestCase):
    def test_open_reader_refreshes_after_another_process_indexes(self):
        self.store.replace_chunks("Contract A", [chunk("Contract A", "A governing law clause.", "one")])
        self.assertEqual(self.store.collection.count(), 1)
        previous = self.store.corpus_fingerprint()
        script = """
import sys
sys.path.insert(0, 'tests')
import vector_store as storage
from test_vector_store import FixedEmbeddingWorker, chunk
storage._EmbeddingWorker = FixedEmbeddingWorker
writer = storage.VectorStore(chroma_dir=sys.argv[1], collection='test_contracts')
try:
    writer.replace_chunks('Contract B', [chunk('Contract B', 'Another governing law clause.', 'two')])
finally:
    writer.close()
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.database)],
            cwd=storage.PROJECT_DIR, env=dict(os.environ), capture_output=True, text=True,
            timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.store.collection.count(), 2)
        self.assertNotEqual(self.store.corpus_fingerprint(), previous)
        self.assertEqual({row["chunk_id"] for row in self.store.query("governing law", top_k=5)},
                         {"one", "two"})

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.database = self.root / "chroma_db"
        self.assets = self.root / "onnx"
        self.assets.mkdir()
        # Worker is replaced, so these files are used for fingerprint tests only.
        (self.assets / "model.onnx").write_bytes(b"test-model-signature-only")
        (self.assets / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.environment = mock.patch.dict(
            "os.environ", {"MEETHAQ_ONNX_MODEL_DIR": str(self.assets)}
        )
        self.environment.start()
        self.model_patch = mock.patch.object(storage, "MODEL_DIR", self.assets)
        self.model_patch.start()
        self.worker_patch = mock.patch.object(storage, "_EmbeddingWorker", FixedEmbeddingWorker)
        self.worker_patch.start()
        self.store = storage.VectorStore(chroma_dir=str(self.database), collection="test_contracts")

    def tearDown(self):
        self.store.close()
        self.worker_patch.stop()
        self.environment.stop()
        self.model_patch.stop()
        self.directory.cleanup()

    def test_top_k_distance_ties_use_ascending_ids_including_prefixes(self):
        self.store.replace_chunks("Contract", [
            chunk("Contract", "A governing law clause.", "aa"),
            chunk("Contract", "B governing law clause.", "a", 1),
            chunk("Contract", "C governing law clause.", "ab", 2),
        ])
        self.assertEqual([row["chunk_id"] for row in self.store.query("governing law", top_k=1)], ["a"])
        self.assertEqual([row["chunk_id"] for row in self.store.query("governing law", top_k=2)], ["a", "aa"])

    def test_annotation_filter_precedes_top_k(self):
        rows = [chunk("Contract", "The supplier shall comply with applicable laws.", "a"),
                chunk("Contract", "This agreement shall be governed by Illinois law.", "z", 1)]
        for row, label in zip(rows, ["Insurance", "Governing Law"]):
            row["annotated_evidence"] = json.dumps([{"clause_type": label, "text": row["text"]}])
        self.store.replace_chunks("Contract", rows)
        self.assertEqual([row["chunk_id"] for row in self.store.query("governing law", top_k=1)], ["z"])

    def test_context_window_cannot_displace_complete_clause_unit(self):
        rows = [chunk("Contract", "A neighboring governing law passage.", "a"),
                chunk("Contract", "This agreement shall be governed by Illinois law.", "z", 1)]
        rows[0]["retrieval_unit"] = "context_window"
        rows[1]["retrieval_unit"] = "complete_clause"
        self.store.replace_chunks("Contract", rows)
        self.assertEqual([row["chunk_id"] for row in self.store.query("governing law", top_k=1)], ["z"])

    def test_embedding_reuse_requires_matching_text_and_inference_provenance(self):
        rows = [chunk("Contract", "A governing law clause.", "one")]
        self.store.replace_chunks("Contract", rows, reuse_embeddings=True)
        with mock.patch.object(self.store.embedding_fn, "embed", wraps=self.store.embedding_fn.embed) as embed:
            self.store.replace_chunks("Contract", rows, reuse_embeddings=True)
            embed.assert_not_called()
            changed = [chunk("Contract", "A changed governing law clause.", "one")]
            self.store.replace_chunks("Contract", changed, reuse_embeddings=True)
            embed.assert_called_once()

    def test_true_cosine_stable_ids_and_boundary_metadata(self):
        self.store.replace_chunks("Contract", [
            chunk("Contract", "A governing law clause.", "z"),
            chunk("Contract", "B governing law clause.", "a", 1),
            chunk("Contract", "Insurance must be maintained.", "insurance", 2),
        ])
        results = self.store.query("governing law", top_k=3)
        self.assertEqual([row["chunk_id"] for row in results], ["a", "z", "insurance"])
        self.assertAlmostEqual(results[0]["cosine_distance"], 0.0)
        self.assertAlmostEqual(results[2]["cosine_distance"], 1.0)
        self.assertAlmostEqual(results[2]["cosine_similarity"], 0.0)
        self.assertAlmostEqual(results[2]["rerank_score"], 0.0)
        self.assertTrue(results[0]["starts_at_boundary"])
        self.assertTrue(results[0]["ends_at_boundary"])

    def test_source_replacement_removes_stale_ids_and_preserves_other_contracts(self):
        self.store.replace_chunks("Contract A", [
            chunk("Contract A", "A governing law clause.", "old-a"),
            chunk("Contract A", "Insurance is required.", "old-b", 1),
        ])
        self.store.replace_chunks("Contract B", [
            chunk("Contract B", "Another governing law clause.", "other"),
        ])
        self.assertEqual(self.store.collection.count(), 3)
        self.store.replace_chunks("Contract A", [
            chunk("Contract A", "New insurance is required.", "new"),
        ])
        self.assertEqual(self.store.collection.count(), 2)
        ids = {row["chunk_id"] for row in self.store.query("governing law", top_k=10)}
        self.assertEqual(ids, {"new", "other"})

    def test_logged_native_upsert_failure_is_not_reported_as_success(self):
        # The real SegmentAPI consumer can swallow a bad-allocation exception.
        with mock.patch.object(self.store.collection.native, "upsert", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "did not persist every native vector"):
                self.store.replace_chunks("Contract", [chunk("Contract", "A governing law clause.", "one")])
        self.assertEqual(self.store.collection.count(), 0)

    def test_native_batch_and_disk_sync_boundaries_survive_reopen(self):
        prepared = [chunk("Contract", f"Governing law clause {index}.", f"native-{index:03}", index)
                    for index in range(145)]
        self.store.replace_chunks("Contract", prepared)
        before = self.store.corpus_fingerprint()
        self.store.close()
        self.store = storage.VectorStore(chroma_dir=str(self.database), collection="test_contracts")
        self.assertEqual(self.store.collection.count(), 145)
        self.assertEqual(self.store.corpus_fingerprint(), before)
        self.assertEqual([row["chunk_id"] for row in self.store.query("governing law", top_k=3)],
                         ["native-000", "native-001", "native-002"])

    def test_reopen_reads_actual_persistent_counts(self):
        self.store.replace_chunks("Contract", [chunk("Contract", "A governing law clause.", "one")])
        before = self.store.corpus_fingerprint()
        self.store.close()
        self.store = storage.VectorStore(chroma_dir=str(self.database), collection="test_contracts")
        self.assertEqual(self.store.collection.count(), 1)
        self.assertEqual(self.store.corpus_fingerprint(), before)

    def test_bounded_embedding_requests_and_reindex_idempotency(self):
        calls = []
        original_embed = self.store.embedding_fn.embed

        def track(texts):
            calls.append(len(texts))
            return original_embed(texts)
        self.store.embedding_fn.embed = track
        chunks = [chunk("Contract", f"Governing law clause {number}.", str(number), number)
                  for number in range(19)]
        with mock.patch.dict("os.environ", {"MEETHAQ_INDEX_BATCH_SIZE": "10"}):
            self.store.replace_chunks("Contract", chunks)
            first = self.store.corpus_fingerprint()
            self.store.replace_chunks("Contract", chunks)
        self.assertEqual(self.store.collection.count(), 19)
        self.assertEqual(self.store.corpus_fingerprint(), first)
        self.assertTrue(calls)
        self.assertLessEqual(max(calls), 8)

    def test_same_count_document_and_embedding_mutation_changes_fingerprint(self):
        self.store.replace_chunks("Contract", [chunk("Contract", "A governing law clause.", "one")])
        before = self.store.corpus_fingerprint()
        self.store.replace_chunks("Contract", [chunk("Contract", "A changed governing law clause.", "one")])
        after_text = self.store.corpus_fingerprint()
        self.assertNotEqual(before, after_text)
        self.assertEqual(self.store.collection.count(), 1)
        self.store.embedding_fn.variant = True
        self.store.replace_chunks("Contract", [chunk("Contract", "A changed governing law clause.", "one")])
        self.assertNotEqual(after_text, self.store.corpus_fingerprint())
        self.assertEqual(self.store.collection.count(), 1)


class StorageConfigurationTests(unittest.TestCase):
    def test_paths_are_anchored_to_project_not_process_cwd(self):
        project = Path(storage.__file__).resolve().parent
        with mock.patch.dict("os.environ", {"MEETHAQ_CHROMA_DIR": "custom_db"}):
            self.assertEqual(Path(storage.resolve_chroma_dir(None)), project / "custom_db")
        self.assertEqual(Path(storage.resolve_chroma_dir("other_db")), project / "other_db")


if __name__ == "__main__":
    unittest.main()

