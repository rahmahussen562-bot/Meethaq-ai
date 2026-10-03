"""
rag_pipeline.py
================
Meethaq AI - PyTorch-Free, ONNX-accelerated local RAG pipeline.
Uses native ChromaDB ONNX embeddings and cosine similarity to bypass Windows torch/c10.dll issues.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import ollama

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION & DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

EMBED_MODEL_NAME = "ONNX-all-MiniLM-L6-v2 (Native)"
RERANK_MODEL_NAME = "HNSW Cosine Similarity"
LLM_MODEL = "llama3.2:3b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 5

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "meethaq_contracts"

SYSTEM_PROMPT = """\
You are Meethaq AI, a deterministic, uncompromising legal auditor and contract compliance assistant.

STRICT OPERATIONAL RULES:
1. Rely EXCLUSIVELY and ENTIRELY on the provided context passages below. Never extrapolate, interpolate, or draw on external legal doctrine.
2. If the context does not contain direct, explicit textual evidence to answer the question, output EXACTLY this sentence and nothing else:
   "I could not find an answer to this question in the provided documents."
3. Every factual claim, number, percentage, or deadline must be followed immediately by its citation tag (e.g., [Source 1], [Source 2]).
4. Maintain an objective, structured legal audit tone. Highlight identified contractual discrepancies or conflicts clearly.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEXT CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.overlap)]

        sep = separators[0]
        parts = text.split(sep)
        chunks: list[str] = []
        buffer = ""

        for part in parts:
            candidate = (buffer + sep + part).strip() if buffer else part.strip()
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, separators[1:]))
                    buffer = ""
                else:
                    buffer = part.strip()

        if buffer:
            chunks.append(buffer)

        return [c for c in chunks if c.strip()]

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.overlap:]
            result.append((tail + " " + chunks[i]).strip())
        return result

    def chunk(self, doc: dict) -> list[dict]:
        separators = ["\n\n", "\n", ". ", "; ", " "]
        raw_chunks = self._split(doc["text"], separators)
        overlapping = self._add_overlap(raw_chunks)

        result = []
        for idx, text in enumerate(overlapping):
            if not text.strip():
                continue
            chunk_id = hashlib.md5(
                f"{doc['source']}::{idx}::{text[:50]}".encode()
            ).hexdigest()
            result.append({
                "text": text,
                "source": doc["source"],
                "chunk_index": idx,
                "chunk_id": chunk_id,
            })
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. VECTOR STORE (Native ONNX ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    def __init__(self, chroma_dir: str = CHROMA_DIR, collection: str = COLLECTION_NAME):
        print("[VectorStore] Initializing ChromaDB with Native ONNX Embedding Function...")
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        batch_size = 200
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [
                {
                    "source": str(c.get("source", "unknown")),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "contract_category": str(c.get("contract_category", "Unknown"))
                }
                for c in batch
            ]
            self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, query_text: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        total_count = self.collection.count()
        if total_count == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, total_count),
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        if not results["documents"] or not results["documents"][0]:
            return candidates

        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            candidates.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "contract_category": meta.get("contract_category", "Unknown"),
                "rerank_score": round(similarity_score, 4)
            })
        return candidates


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUERY HANDLER & RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class QueryHandler:
    @staticmethod
    def normalize(query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    @staticmethod
    def expand_with_llm(query: str, model: str = LLM_MODEL) -> str:
        prompt = (
            "Rewrite the search query to improve contract retrieval accuracy. "
            "Output ONLY the expanded query, no explanations.\n\n"
            f"Query: {query}"
        )
        try:
            res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            expanded = res["message"]["content"].strip()
            return expanded if len(expanded) < 300 else query
        except Exception:
            return query


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int = RETRIEVAL_TOP_K, **kwargs):
        self.store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        # Fast native vector retrieval with ranked cosine similarity scores
        return self.store.query(query, top_k=self.top_k)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTEXT & GENERATION
# ══════════════════════════════════════════════════════════════════════════════

class ContextAssembler:
    @staticmethod
    def assemble(chunks: list[dict]) -> tuple[str, list[dict]]:
        lines = []
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            label = f"[Source {i}]"
            lines.append(
                f"{label}\n"
                f"Document: {chunk['source']} (Category: {chunk.get('contract_category', 'N/A')})\n"
                f"Confidence Score: {chunk.get('rerank_score', 0):.4f}\n\n"
                f"{chunk['text']}"
            )
            sources.append({
                "label": label,
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "rerank_score": chunk.get("rerank_score", 0),
                "text": chunk["text"],
            })
        return "\n\n" + ("─" * 50 + "\n\n").join(lines), sources


class LLMGenerator:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    def generate(self, question: str, context: str, stream: bool = False) -> str:
        user_message = (
            f"Context passages from verified contracts:\n{context}\n\n"
            f"Audit Question: {question}\n\n"
            "Audit Assessment (Citing explicit [Source N] tags):"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. ORCHESTRATOR (RAGPipeline)
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    def __init__(
        self,
        embed_model: str = EMBED_MODEL_NAME,
        rerank_model: str = RERANK_MODEL_NAME,
        llm_model: str = LLM_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        rerank_top_n: int = 3,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.vector_store = VectorStore(chroma_dir=chroma_dir, collection=collection_name)
        self.retriever = Retriever(vector_store=self.vector_store, top_k=rerank_top_n)
        self.query_handler = QueryHandler()
        self.assembler = ContextAssembler()
        self.generator = LLMGenerator(model=llm_model)

    def index_prepared(self, docs: list[dict]) -> int:
        """Indexes pre-processed contract records directly from ingest_cuad.py."""
        print(f"\n[Indexing] Ingesting {len(docs)} structured contract records...")
        all_chunks: list[dict] = []
        for doc in docs:
            chunks = self.chunker.chunk({"source": doc["contract_name"], "text": doc["text"]})
            for c in chunks:
                c["contract_category"] = doc.get("contract_category", "Unknown")
            all_chunks.extend(chunks)

        self.vector_store.add_chunks(all_chunks)
        print(f"✓ Indexing complete. ChromaDB count: {self.vector_store.collection.count()} chunks.\n")
        return len(all_chunks)

    def stats(self) -> dict:
        return {
            "collection": self.vector_store.collection.name,
            "indexed_chunks": self.vector_store.collection.count(),
            "embed_model": EMBED_MODEL_NAME,
            "rerank_model": RERANK_MODEL_NAME,
            "llm_model": LLM_MODEL,
        }

    def query(
        self,
        question: str,
        stream: bool = False,
        expand_query: bool = False,
    ) -> str:
        """Run the full RAG pipeline for a user question."""
        normalized = self.query_handler.normalize(question)
        if expand_query:
            normalized = self.query_handler.expand_with_llm(normalized)
        chunks = self.retriever.retrieve(normalized)
        if not chunks:
            return "I could not find an answer to this question in the provided documents."
        context, _ = self.assembler.assemble(chunks)
        return self.generator.generate(question=normalized, context=context, stream=stream)
