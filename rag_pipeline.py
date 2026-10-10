"""Calibrated retrieval with validated local or cloud LLM synthesis."""

from __future__ import annotations

import bisect
import json
import hashlib
import logging
import os
import re
from pathlib import Path

from grounding import (
    ABSTENTION, MAX_QUESTION_LENGTH, eligible_chunks, in_contract_scope,
    load_calibration, normalize_question, policy_signature, render_evidence,
)
from llm_synthesis import LLMUnavailable, SynthesisRejected, build_synthesizer

EMBED_MODEL_NAME = "ONNX-all-MiniLM-L6-v2 (Native)"
RERANK_MODEL_NAME = "Exact cosine similarity"
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 12
PROJECT_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


class TextChunker:
    """Bounded original-text windows; IDs hash all text rather than its prefix."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        if chunk_size < 32 or not 0 <= overlap < chunk_size:
            raise ValueError("Require chunk_size >= 32 and 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _boundaries(text: str) -> list[int]:
        return [match.end() for match in re.finditer(
            r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n\s*\n", text
        )]

    def chunk(self, doc: dict) -> list[dict]:
        source, text = doc.get("source"), doc.get("text")
        if not isinstance(source, str) or not source.strip() or not isinstance(text, str):
            raise ValueError("Each document needs a nonempty source and string text")
        boundaries = self._boundaries(text)
        boundary_set = set(boundaries) | {0, len(text)}
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                # Prefer whole sentences/paragraphs; a single long clause remains
                # bounded and its incomplete edges will not be used as evidence.
                options = [point for point in boundaries
                           if start + self.chunk_size // 3 <= point <= end]
                if options:
                    end = options[-1]
                else:
                    space = text.rfind(" ", start + self.chunk_size // 3, end)
                    if space > start:
                        end = space + 1
            raw = text[start:end]
            body = raw.strip()
            if body:
                left = start + len(raw) - len(raw.lstrip())
                right = end - len(raw) + len(raw.rstrip())
                chunk_id = hashlib.sha256(
                    f"{source}\0{len(chunks)}\0{body}".encode("utf-8")
                ).hexdigest()
                chunks.append({
                    "text": body, "source": source, "chunk_index": len(chunks),
                    "chunk_id": chunk_id, "char_start": left, "char_end": right,
                    "starts_at_boundary": start in boundary_set,
                    "ends_at_boundary": end in boundary_set,
                })
            if end == len(text):
                break
            desired = end - self.overlap
            previous = [point for point in boundaries if start < point <= desired]
            following = [point for point in boundaries if desired < point <= end]
            next_start = previous[-1] if previous else following[0] if following else end
            start = max(start + 1, next_start)
        return chunks


class QueryHandler:
    normalize = staticmethod(normalize_question)

    @staticmethod
    def expand_with_llm(query: str, model: str = LLM_MODEL) -> str:
        """Compatibility only: generative expansion is disabled for grounding."""
        return normalize_question(query)


class Retriever:
    def __init__(self, vector_store, top_k: int = RETRIEVAL_TOP_K, **kwargs):
        self.store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        return self.store.query(query, top_k=self.top_k)


class ContextAssembler:
    @staticmethod
    def assemble(chunks: list[dict]) -> tuple[str, list[dict]]:
        return render_evidence(chunks)


class RAGPipeline:
    def __init__(
        self, embed_model: str = EMBED_MODEL_NAME,
        rerank_model: str = RERANK_MODEL_NAME, llm_model: str = LLM_MODEL,
        chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
        retrieval_top_k: int = RETRIEVAL_TOP_K, rerank_top_n: int = 3,
        chroma_dir: str | None = None, collection_name: str | None = None,
        calibration_path: str | Path | None = None, vector_store=None,
        synthesizer=None,
    ):
        if retrieval_top_k < 1 or rerank_top_n < 1:
            raise ValueError("Retrieval limits must be positive")
        if vector_store is None:
            # Native dependencies initialize only when a pipeline is requested.
            from vector_store import VectorStore
            vector_store = VectorStore(chroma_dir=chroma_dir, collection=collection_name)
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.vector_store = vector_store
        self.retriever = Retriever(vector_store, top_k=retrieval_top_k)
        self.query_handler = QueryHandler()
        self.assembler = ContextAssembler()
        self.max_sources = rerank_top_n
        self.synthesizer = synthesizer or build_synthesizer(llm_model)
        self.llm_model = getattr(self.synthesizer, "model", llm_model)
        self.policy = policy_signature(retrieval_top_k, rerank_top_n)
        selected_path = Path(calibration_path or os.getenv(
            "MEETHAQ_CALIBRATION_PATH", "data/rag_calibration.json"
        ))
        self.calibration_path = selected_path if selected_path.is_absolute() else PROJECT_DIR / selected_path

    def _prepared_chunks(self, doc: dict) -> list[dict]:
        """Preserve independently labeled complete clauses as evidence boundaries."""
        source, text = doc["contract_name"], doc["text"]
        chunks = self.chunker.chunk({"source": source, "text": text})
        boundaries = sorted(set([0, len(text), *self.chunker._boundaries(text)]))
        spans = []
        seen = set()
        skipped = 0
        for annotation in doc.get("annotations", []):
            label, annotated = annotation.get("clause_type"), annotation.get("text")
            if not isinstance(label, str) or not isinstance(annotated, str) or not annotated.strip():
                skipped += 1
                continue
            start = annotation.get("start", -1)
            if not isinstance(start, int) or start < 0 or text[start:start + len(annotated)] != annotated:
                start = text.find(annotated)
            if start < 0:
                skipped += 1
                continue
            end = start + len(annotated)
            left = boundaries[max(0, bisect.bisect_right(boundaries, start) - 1)]
            right = boundaries[min(len(boundaries) - 1, bisect.bisect_left(boundaries, end))]
            original = text[left:right]
            body = original.strip()
            left += len(original) - len(original.lstrip())
            right -= len(original) - len(original.rstrip())
            # Do not truncate a clause to make it fit an evidence window.
            if not body or len(body) > 12000:
                skipped += 1
                continue
            key = (label, left, right)
            if key not in seen:
                seen.add(key)
                spans.append({"clause_type": label, "text": body, "start": left, "end": right})
        extra_ranges = set()
        for span in spans:
            if any(chunk["char_start"] == span["start"] and span["end"] == chunk["char_end"]
                   for chunk in chunks):
                continue
            key = (span["start"], span["end"])
            if key in extra_ranges:
                continue
            extra_ranges.add(key)
            identifier = hashlib.sha256(
                f"{source}\0complete-clause\0{key[0]}\0{key[1]}\0{span['text']}".encode("utf-8")
            ).hexdigest()
            chunks.append({"source": source, "text": span["text"], "chunk_id": identifier,
                           "chunk_index": len(chunks), "char_start": key[0], "char_end": key[1],
                           "starts_at_boundary": True, "ends_at_boundary": True})
        for chunk in chunks:
            supported = [{"clause_type": span["clause_type"], "text": span["text"]}
                         for span in spans if chunk["char_start"] <= span["start"]
                         and span["end"] <= chunk["char_end"] and span["text"] in chunk["text"]]
            chunk["annotated_evidence"] = json.dumps(supported, ensure_ascii=False, sort_keys=True)
            chunk["retrieval_unit"] = "complete_clause" if any(
                chunk["char_start"] == span["start"] and chunk["char_end"] == span["end"]
                for span in spans) else "context_window"
            chunk["contract_category"] = str(doc.get("contract_category", "Unknown"))
        print(f"[Indexing] Complete labeled evidence: {len(spans)} spans; "
              f"{len(extra_ranges)} additional complete-clause chunks; {skipped} labels excluded.", flush=True)
        return chunks

    def index_prepared(self, docs: list[dict]) -> int:
        """Validate first, then index one document at a time with visible progress."""
        seen_sources = set()
        for doc in docs:
            name, text = doc.get("contract_name"), doc.get("text")
            if not isinstance(name, str) or not name.strip() or not isinstance(text, str) or not text.strip():
                raise ValueError("Prepared documents require contract_name and nonempty text")
            if name in seen_sources:
                raise ValueError(f"Duplicate prepared contract source: {name}")
            seen_sources.add(name)
        print(f"[Indexing] Ingesting {len(docs)} structured contract records...", flush=True)
        indexed = 0
        for number, doc in enumerate(docs, 1):
            chunks = self._prepared_chunks(doc)
            print(f"[Indexing] Contract {number}/{len(docs)}: {len(chunks)} chunks", flush=True)
            self.vector_store.replace_chunks(doc["contract_name"], chunks, reuse_embeddings=True)
            indexed += len(chunks)
            print(f"[Indexing] Persisted count: {self.vector_store.collection.count()}", flush=True)
        print(f"[Indexing] Complete. ChromaDB count: {self.vector_store.collection.count()} chunks.", flush=True)
        return indexed

    def _state(self) -> dict:
        fingerprint = self.vector_store.corpus_fingerprint()
        status, threshold = load_calibration(
            self.calibration_path, fingerprint, self.vector_store.embedding_signature, self.policy
        )
        return {
            "collection": self.vector_store.collection.name,
            "indexed_chunks": self.vector_store.collection.count(),
            "chroma_dir": self.vector_store.chroma_dir,
            "corpus_fingerprint": fingerprint,
            "embedding_signature": self.vector_store.embedding_signature,
            "embed_model": EMBED_MODEL_NAME, "rerank_model": RERANK_MODEL_NAME,
            "llm_model": getattr(self.synthesizer, "model", self.llm_model),
            "answer_mode": "grounded_llm_synthesis",
            "llm_invoked": False, "calibration_status": status,
            "max_cosine_distance": threshold,
            "llm_status": getattr(self.synthesizer, "last_status", "unchecked"),
        }

    def stats(self) -> dict:
        return self._state()

    def llm_health(self) -> dict:
        return self.synthesizer.health()

    def audit(self, question: str, expand_query: bool = False) -> dict:
        normalized = normalize_question(question)
        if not normalized or len(normalized) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question must contain 1 to {MAX_QUESTION_LENGTH} characters")
        stats = self._state()
        stats["query_expansion"] = "disabled"
        response = {"answer": ABSTENTION, "sources": [], "stats": stats}
        if not in_contract_scope(normalized):
            stats["abstention_reason"] = "out_of_scope"
            return response
        if stats["indexed_chunks"] == 0:
            stats["abstention_reason"] = "empty_index"
            return response
        if stats["calibration_status"] != "calibrated":
            stats["abstention_reason"] = "calibration_required"
            return response
        chunks = self.retriever.retrieve(normalized)
        evidence = eligible_chunks(normalized, chunks, stats["max_cosine_distance"])[:self.max_sources]
        # A concurrent ingestion must not reuse calibration from the old corpus.
        if self.vector_store.corpus_fingerprint() != stats["corpus_fingerprint"]:
            stats["calibration_status"] = "stale"
            stats["max_cosine_distance"] = None
            stats["abstention_reason"] = "corpus_changed"
            stats["indexed_chunks"] = self.vector_store.collection.count()
            return response
        _, sources = self.assembler.assemble(evidence)
        if not sources:
            stats["abstention_reason"] = "insufficient_evidence"
            return response
        try:
            synthesis = self.synthesizer.synthesize(normalized, sources)
        except LLMUnavailable as exc:
            logger.error("Grounded synthesis unavailable: %s", exc)
            stats["llm_status"] = "unavailable"
            stats["abstention_reason"] = "llm_unavailable"
            return response
        except SynthesisRejected as exc:
            logger.warning("Rejected ungrounded LLM output: %s", exc)
            stats["llm_status"] = "invalid_output"
            stats["abstention_reason"] = "invalid_llm_output"
            return response
        if synthesis is None:
            stats["llm_status"] = "ready"
            stats["abstention_reason"] = "llm_abstained"
            return response
        stats["llm_invoked"] = True
        stats["llm_status"] = "ready"
        cited = set(synthesis.cited_labels)
        cited_sources = [source for source in sources if source["label"] in cited]
        return {"answer": synthesis.answer, "sources": cited_sources, "stats": stats}

    def close(self) -> None:
        close_store = getattr(self.vector_store, "close", None)
        if callable(close_store):
            close_store()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def query(self, question: str, stream: bool = False, expand_query: bool = False) -> str:
        return self.audit(question, expand_query=expand_query)["answer"]
