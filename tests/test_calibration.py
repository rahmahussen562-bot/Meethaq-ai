"""Regression tests for heldout separation and fully supported calibration labels."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from calibrate_rag import calibrate, evaluate, load_cases, matches_expected
from grounding import policy_signature
from test_grounding import FakeStore, candidate, calibration_artifact


def fixture():
    expected = [{"source": "Example contract",
                 "text": "Either party may terminate on 30 days written notice."}]
    return {
        "schema_version": 1, "cases": [
            {"id": "train-positive", "split": "calibration", "query": "What termination applies?",
             "answerable": True, "expected_evidence": expected},
            {"id": "train-negative", "split": "calibration", "query": "Who won the football cup?",
             "answerable": False, "expected_evidence": []},
            {"id": "validation-positive", "split": "validation", "query": "What notice applies?",
             "answerable": True, "expected_evidence": expected},
            {"id": "validation-negative", "split": "validation", "query": "What is the Paris weather?",
             "answerable": False, "expected_evidence": []},
        ]
    }


class CalibrationTests(unittest.TestCase):
    def test_section_prefix_does_not_mask_factual_amounts(self):
        case = {"expected_evidence": [{"source": "Example contract",
                 "text": "Subject to Clause 9.1, liability excludes indirect losses;\n\n(a) lost profit"}]}
        row = {"source": "Example contract", "evidence_sentences": [
               "9.3 Subject to Clause 9.1, liability excludes indirect losses;"]}
        self.assertTrue(matches_expected(case, row))
        case["expected_evidence"][0]["text"] = "Million dollars shall be paid;\n\n(a) in cash"
        row["evidence_sentences"] = ["1.5 Million dollars shall be paid;"]
        self.assertFalse(matches_expected(case, row))

    def test_cuad_omitted_punctuation_preserves_every_factual_word(self):
        case = {"expected_evidence": [{"source": "Example contract",
                 "text": "Insurance shall cover flood. Requires a waiver of subrogation"}]}
        row = {"source": "Example contract", "evidence_sentences": ["Requires a waiver of subrogation."]}
        self.assertTrue(matches_expected(case, row))
        for sentence in ["Requires a waiver of subrogationx.",
                         "Requires a waiver of subrogation without consent.",
                         "Requires no waiver of subrogation."]:
            row["evidence_sentences"] = [sentence]
            self.assertFalse(matches_expected(case, row))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "cases.json"
        self.output = Path(self.directory.name) / "calibration.json"
        self.path.write_text(json.dumps(fixture()), encoding="utf-8")
        self.store = FakeStore()
        self.candidates = {
            "What termination applies?": [candidate(distance=0.2)],
            "Who won the football cup?": [candidate(distance=0.0)],
            "What notice applies?": [candidate(distance=0.1)],
            "What is the Paris weather?": [candidate(distance=0.0)],
        }
        self.pipeline = SimpleNamespace(
            vector_store=self.store,
            retriever=SimpleNamespace(retrieve=lambda query: self.candidates[query]),
            max_sources=3, policy=policy_signature(12, 3),
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_heldout_distance_never_influences_cutoff(self):
        self.candidates["What notice applies?"] = [candidate(distance=0.9)]
        report = calibrate(self.pipeline, self.path, self.output)
        self.assertEqual(report["max_cosine_distance"], math.nextafter(0.2, math.inf))
        self.assertFalse(report["validation_passed"])
        self.assertEqual(report["validation_metrics"]["supported_positive_recall"], 0.0)
        self.assertFalse(json.loads(self.output.read_text())["validation_passed"])

    def test_frozen_holdout_pass_installs_measured_threshold(self):
        report = calibrate(self.pipeline, self.path, self.output)
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["max_cosine_distance"], math.nextafter(0.2, math.inf))
        self.assertEqual(report["validation_metrics"]["false_accept_count"], 0)
        self.assertEqual(report["validation_case_count"], 2)
        self.assertEqual(report["calibration_case_count"], 2)

    def test_one_true_chunk_cannot_hide_an_unsupported_answer(self):
        cases = [fixture()["cases"][0]]
        candidates = {"train-positive": [
            candidate(distance=0.2),
            candidate("The party may terminate on 60 days written notice.", "irrelevant", 0.25),
        ]}
        measured = evaluate(cases, candidates, 0.3, 3)
        self.assertEqual(measured["supported_positive_count"], 0)
        self.assertEqual(measured["unsupported_positive_answer_count"], 1)

    def test_measured_unsafe_training_revokes_older_calibration(self):
        self.output.write_text(json.dumps(calibration_artifact(self.store)), encoding="utf-8")
        payload = fixture()
        payload["cases"][1]["query"] = "What termination is permitted?"
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.candidates["What termination is permitted?"] = [candidate(distance=0.1)]
        with self.assertRaisesRegex(ValueError, "No cutoff"):
            calibrate(self.pipeline, self.path, self.output)
        revoked = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(revoked["validation_passed"])
        self.assertIsNone(revoked["max_cosine_distance"])

    def test_training_failure_does_not_query_heldout_cases(self):
        queried = []
        self.candidates["What termination applies?"] = [
            candidate("Either party may terminate on 60 days written notice.", distance=0.2),
        ]
        def retrieve(query):
            queried.append(query)
            if query in {"What notice applies?", "What is the Paris weather?"}:
                self.fail("Held-out retrieval must remain untouched when training fails")
            return self.candidates[query]
        self.pipeline.retriever.retrieve = retrieve
        with self.assertRaisesRegex(ValueError, "No cutoff"):
            calibrate(self.pipeline, self.path, self.output)
        self.assertEqual(queried, ["What termination applies?", "Who won the football cup?"])
        report = json.loads(self.output.with_suffix(".report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["validation_evaluated"])
        self.assertNotIn("validation_metrics", report)
        diagnostics = report["calibration_diagnostics"]
        self.assertEqual([row["id"] for row in diagnostics["cases"]],
                         ["train-positive", "train-negative"])
        first_chunk = diagnostics["cases"][0]["retrieved_chunks"][0]
        self.assertEqual(first_chunk["cosine_distance"], 0.2)
        self.assertFalse(first_chunk["matches_labeled_evidence"])
        self.assertEqual(diagnostics["threshold_trials"], [])

    def test_rejected_training_trials_report_their_metrics(self):
        payload = fixture()
        payload["cases"][1]["query"] = "What termination is permitted?"
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.candidates["What termination is permitted?"] = [candidate(distance=0.1)]
        with self.assertRaisesRegex(ValueError, "No cutoff"):
            calibrate(self.pipeline, self.path, self.output)
        report = json.loads(self.output.with_suffix(".report.json").read_text(encoding="utf-8"))
        trials = report["calibration_diagnostics"]["threshold_trials"]
        self.assertEqual(len(trials), 1)
        self.assertFalse(trials[0]["acceptable"])
        self.assertEqual(trials[0]["max_cosine_distance"], math.nextafter(0.2, math.inf))
        self.assertEqual(trials[0]["calibration_metrics"]["false_accept_count"], 1)
        self.assertEqual(trials[0]["calibration_metrics"]["supported_positive_count"], 1)
        self.assertNotIn("validation_metrics", report)

    def test_cross_split_duplicate_query_is_rejected(self):
        payload = fixture()
        payload["cases"][2]["query"] = payload["cases"][0]["query"]
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            load_cases(self.path)

    def test_corpus_change_prevents_installation(self):
        def mutate(query):
            self.store.fingerprint = "mutated-corpus"
            return self.candidates[query]
        self.pipeline.retriever.retrieve = mutate
        with self.assertRaisesRegex(RuntimeError, "Corpus changed"):
            calibrate(self.pipeline, self.path, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

