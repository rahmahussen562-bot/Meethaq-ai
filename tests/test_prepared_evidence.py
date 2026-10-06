"""Regression checks for complete CUAD evidence and independent type filtering."""
import json
import unittest

from grounding import applicable_annotations, canonical_embedding_query, evidence_sentences, eligible_chunks
from rag_pipeline import RAGPipeline, TextChunker


def typed(text, label, **changes):
    return {"text": text, "source": "Example contract", "chunk_id": "typed",
            "starts_at_boundary": True, "ends_at_boundary": True,
            "cosine_distance": 0.1,
            "annotated_evidence": json.dumps([{"clause_type": label, "text": text}]),
            **changes}


class PreparedEvidenceTests(unittest.TestCase):
    def test_embedding_synonyms_preserve_entities_quantities_and_modifiers(self):
        self.assertEqual(canonical_embedding_query("What liability limitations apply?"),
                         canonical_embedding_query("Show the liability caps."))
        self.assertEqual(canonical_embedding_query("What exclusivity requirements apply?"),
                         canonical_embedding_query("Describe the exclusivity clauses."))
        words = canonical_embedding_query("What unlimited insurance of 777777 dollars applies to Atlantis?").split()
        for word in ["unlimited", "insurance", "777777", "dollars", "atlantis"]:
            self.assertIn(word, words)

    def test_type_gate_rejects_nearby_laws_and_assignment_of_ip(self):
        text = "The supplier must comply with all applicable laws."
        self.assertEqual(applicable_annotations("What governing law applies?", typed(text, "Insurance")), [])
        text = "The developer shall assign intellectual property to the customer."
        self.assertEqual(applicable_annotations("What assignment requirements apply?",
                                               typed(text, "IP Ownership Assignment")), [])

    def test_headings_do_not_establish_contractual_requirements(self):
        self.assertEqual(evidence_sentences("What liability limitations apply?",
                                           typed("Limitations on Liability.", "Cap On Liability")), [])

    def test_labels_cannot_supply_unsupported_entities_amounts_or_modifiers(self):
        clause = "The supplier shall maintain insurance of 500 dollars."
        for question in ["Does insurance require 777777 dollars?",
                         "What insurance applies to Atlantis?",
                         "What unlimited insurance requirements apply?"]:
            self.assertEqual(eligible_chunks(question, [typed(clause, "Insurance")], 0.5), [])

    def test_classification_cannot_supply_unlimited_or_perpetual_modifiers(self):
        clause = "The supplier shall grant a license for one year."
        for label, query in [("Unlimited/All-You-Can-Eat-License", "What unlimited license applies?"),
                             ("Irrevocable Or Perpetual License", "What perpetual license applies?")]:
            self.assertEqual(evidence_sentences(query, typed(clause, label)), [])

    def test_full_clause_preserves_late_exception_and_fragments(self):
        sentences = ["The supplier shall maintain insurance."]
        sentences.extend(f"Insurance shall cover category {index}." for index in range(8))
        sentences.append("Except for claims arising before the effective date.")
        body = " ".join(sentences)
        self.assertEqual(evidence_sentences("What insurance requirements apply?",
                                           typed(body, "Insurance")), sentences)

    def test_invalid_or_forged_annotation_fails_closed(self):
        body = "The supplier shall maintain insurance."
        for encoded in ["invalid", "{}", json.dumps([{"clause_type": "Insurance", "text": "Invented."}])]:
            self.assertEqual(evidence_sentences("What insurance applies?",
                                               typed(body, "Insurance", annotated_evidence=encoded)), [])

    def test_long_partial_label_keeps_entire_original_sentence(self):
        condition = "provided the customer has paid all outstanding fees."
        body = "The supplier shall maintain insurance covering " + "specified risks " * 90 + condition
        pipeline = object.__new__(RAGPipeline)
        pipeline.chunker = TextChunker(chunk_size=180, overlap=40)
        doc = {"contract_name": "Example contract", "text": body,
               "annotations": [{"clause_type": "Insurance", "text": "maintain insurance", "start": 25}]}
        first = pipeline._prepared_chunks(doc)
        second = pipeline._prepared_chunks(doc)
        self.assertEqual(first, second)
        full = [row for row in first if row["text"] == body]
        self.assertEqual(len(full), 1)
        self.assertEqual(evidence_sentences("What insurance requirements apply?", full[0]), [body])
        self.assertIn(condition, full[0]["text"])


if __name__ == "__main__":
    unittest.main()
