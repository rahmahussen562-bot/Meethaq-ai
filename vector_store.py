"""Project-anchored, serialized Chroma persistence with isolated local ONNX embeddings."""
from __future__ import annotations

import atexit
from contextlib import nullcontext
import heapq
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time

PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(PROJECT_DIR / "chroma_db")
COLLECTION_NAME = "meethaq_contracts"
_PROJECT_MODEL_DIR = PROJECT_DIR / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx"
_LEGACY_MODEL_DIR = Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx"
_DEFAULT_MODEL_DIR = _LEGACY_MODEL_DIR if (_LEGACY_MODEL_DIR / "model.onnx").is_file() else _PROJECT_MODEL_DIR
MODEL_DIR = Path(os.environ.get("MEETHAQ_ONNX_MODEL_DIR", str(_DEFAULT_MODEL_DIR))).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = PROJECT_DIR / MODEL_DIR
MODEL_DIR = MODEL_DIR.resolve()
_MIN_FREE_BYTES = 64 * 1024 * 1024
_SYSTEM_COUNTS = {}
_SYSTEM_COUNTS_LOCK = threading.RLock()


def resolve_chroma_dir(directory=None):
    value = Path(directory or os.environ.get("MEETHAQ_CHROMA_DIR") or CHROMA_DIR).expanduser()
    return str((value if value.is_absolute() else PROJECT_DIR / value).resolve())


def ensure_disk_space(directory, required_bytes=_MIN_FREE_BYTES):
    path = Path(directory)
    while not path.exists():
        path = path.parent
    available = shutil.disk_usage(path).free
    if available < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space for indexing: {available:,} bytes free at {path}; "
            f"need at least {required_bytes:,}. Free disk space before retrying."
        )
    return available


class _DatabaseLock:
    """Serialize embedded Chroma clients across API and indexing processes."""
    def __init__(self, directory):
        self.path = Path(directory) / ".meethaq.lock"
        self.thread_lock = threading.RLock()
        self.handle = None

    def __enter__(self):
        self.thread_lock.acquire()
        try:
            self.handle = open(self.path, "a+b")
            self.handle.seek(0, os.SEEK_END)
            if not self.handle.tell():
                self.handle.write(b"0")
                self.handle.flush()
            deadline = time.monotonic() + 30
            while True:
                try:
                    self.handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Timed out waiting for the Chroma index lock; another operation is still running.")
                    time.sleep(0.05)
        except BaseException:
            if self.handle:
                self.handle.close()
                self.handle = None
            self.thread_lock.release()
            raise

    def __exit__(self, *_):
        try:
            if self.handle:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.handle.close()
                self.handle = None
        finally:
            self.thread_lock.release()


class _EmbeddingWorker:
    """Keep native ONNX failures outside the API and supervise every request."""
    def __init__(self):
        self.process = None
        self.responses = queue.Queue()
        self.diagnostics = []
        self.lock = threading.Lock()

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.responses = queue.Queue()
        self.diagnostics = []
        responses = self.responses
        diagnostics = self.diagnostics
        environment = dict(os.environ)
        environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        process = subprocess.Popen(
            [sys.executable, "-B", "-u", str(PROJECT_DIR / "embedding_worker.py"), str(MODEL_DIR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=environment, cwd=str(PROJECT_DIR),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.process = process
        def read_responses():
            for line in process.stdout:
                responses.put(line)
            responses.put(None)
        def read_diagnostics():
            for line in process.stderr:
                diagnostics.append(line.rstrip())
                diagnostics[:] = diagnostics[-20:]
                print(line.rstrip(), file=sys.stderr, flush=True)
        threading.Thread(target=read_responses, daemon=True).start()
        threading.Thread(target=read_diagnostics, daemon=True).start()

    def embed(self, texts):
        if not texts:
            return []
        with self.lock:
            self._start()
            process = self.process
            try:
                process.stdin.write(json.dumps({"texts": texts}, ensure_ascii=False) + "\n")
                process.stdin.flush()
                timeout = float(os.environ.get("MEETHAQ_EMBED_TIMEOUT_SECONDS", "120"))
                response = self.responses.get(timeout=timeout)
                if response is None:
                    process.wait(timeout=5)
                    code = process.returncode
                    raise RuntimeError(
                        f"Native embedding worker exited with code {code} (0x{code & 0xffffffff:08X}). "
                        + " ".join(self.diagnostics[-6:])
                    )
                payload = json.loads(response)
                if payload.get("error"):
                    raise RuntimeError("Local embedding failed: " + payload["error"] + " " + " ".join(self.diagnostics[-6:]))
                embeddings = payload["embeddings"]
                if len(embeddings) != len(texts) or any(
                    len(vector) != 384 or not all(math.isfinite(value) for value in vector)
                    or not any(vector) for vector in embeddings
                ):
                    raise RuntimeError("Embedding worker returned invalid MiniLM vectors.")
                return embeddings
            except queue.Empty as exc:
                self.close()
                raise RuntimeError("Local embedding worker timed out; verify model cache, memory and available disk space.") from exc
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise RuntimeError("Local embedding process stopped unexpectedly; check native runtime and disk space.") from exc

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                # A dead native worker may already have closed the pipe.
                # Cleanup must preserve the original failure and its traceback.
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        for pipe in (process.stdout, process.stderr):
            if pipe:
                pipe.close()


class _Collection:
    """Preserve familiar name/count/get APIs while locking native operations."""
    def __init__(self, store, native):
        self.store = store
        self.native = native
        self.name = native.name
    def count(self):
        with self.store._lock:
            self.store._refresh()
            return self.native.count()
    def get(self, **kwargs):
        with self.store._lock:
            self.store._refresh()
            return self.native.get(**kwargs)
    def delete(self, **kwargs):
        with self.store._lock:
            self.store._refresh()
            result = self.native.delete(**kwargs)
            self.store._revision = self.store._database_revision()
            return result


class VectorStore:
    def __init__(self, chroma_dir=None, collection=None):
        from native_runtime import configure_native_runtime
        self.native_runtime_signature = configure_native_runtime()
        import chromadb
        from chromadb.config import Settings
        from chromadb.errors import NotFoundError
        self.chroma_dir = resolve_chroma_dir(chroma_dir)
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        self._lock = _DatabaseLock(self.chroma_dir)
        self.embedding_fn = _EmbeddingWorker()
        with self._lock:
            # The installed Windows Rust API fatally crashes on a one-vector upsert.
            # Use Chroma's Python SegmentAPI consistently; embeddings remain native ONNX.
            self._settings = Settings(anonymized_telemetry=False, chroma_api_impl="chromadb.api.segment.SegmentAPI")
            self.client = chromadb.PersistentClient(path=self.chroma_dir, settings=self._settings)
            name = collection or os.environ.get("MEETHAQ_COLLECTION") or COLLECTION_NAME
            try:
                native = self.client.get_collection(name=name, embedding_function=None)
            except NotFoundError:
                native = self.client.create_collection(name=name, embedding_function=None, metadata={"hnsw:space": "cosine", "hnsw:num_threads": 1, "hnsw:batch_size": 64, "hnsw:sync_threshold": 128})
            metadata = native.metadata or {}
            configuration = getattr(native, "configuration", {}) or {}
            metric = metadata.get("hnsw:space") or configuration.get("hnsw", {}).get("space")
            if metric != "cosine":
                with _SYSTEM_COUNTS_LOCK:
                    if not _SYSTEM_COUNTS.get(self.chroma_dir):
                        self.client._system.stop()
                        cache = getattr(type(self.client), "_identifier_to_system", {})
                        cache.pop(getattr(self.client, "_identifier", None), None)
                raise RuntimeError(f"Collection {name!r} uses {metric or 'unknown'} distance; create/reindex a cosine collection before auditing.")
            self.collection = _Collection(self, native)
            with _SYSTEM_COUNTS_LOCK:
                _SYSTEM_COUNTS[self.chroma_dir] = _SYSTEM_COUNTS.get(self.chroma_dir, 0) + 1
        self._revision = self._database_revision()
        self._signature = None
        atexit.register(self.close)
        print(f"[VectorStore] Persistent cosine collection {self.collection.name!r} at {self.chroma_dir}", flush=True)

    @property
    def embedding_signature(self):
        if self._signature is None:
            model = MODEL_DIR / "model.onnx"
            digest = hashlib.sha256()
            if model.is_file():
                with model.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                model_digest = digest.hexdigest()
            else:
                model_digest = "missing"
            versions = {}
            for package in ("onnxruntime", "numpy", "tokenizers"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = "missing"
            self._signature = {
                "model": "all-MiniLM-L6-v2", "model_sha256": model_digest,
                "worker_sha256": hashlib.sha256((PROJECT_DIR / "embedding_worker.py").read_bytes()).hexdigest(),
                "storage_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256((MODEL_DIR / "tokenizer.json").read_bytes()).hexdigest() if (MODEL_DIR / "tokenizer.json").is_file() else "missing",
                "query_algorithm": "exact-cosine-canonical-query-v2", "runtime_versions": versions,
                "native_runtime": self.native_runtime_signature,
                "native_runtime_source_sha256": hashlib.sha256((PROJECT_DIR / "native_runtime.py").read_bytes()).hexdigest(),
            }
        return self._signature

    def _database_revision(self):
        revisions = []
        for filename in ("chroma.sqlite3", "chroma.sqlite3-wal"):
            database = Path(self.chroma_dir) / filename
            information = database.stat() if database.exists() else None
            revisions.append((information.st_mtime_ns, information.st_size) if information else None)
        return tuple(revisions)

    def _refresh(self):
        revision = self._database_revision()
        if revision == self._revision:
            return
        # Chroma embedded readers cache vector segments. Reopen on writes from
        # another process so an already-running API sees the newly indexed corpus.
        import chromadb
        with _SYSTEM_COUNTS_LOCK:
            if _SYSTEM_COUNTS.get(self.chroma_dir, 1) > 1:
                raise RuntimeError("The index changed while multiple local clients were open; close and reopen those clients.")
            name = self.collection.name
            client = self.client
            client._system.stop()
            cache = getattr(type(client), "_identifier_to_system", {})
            cache.pop(getattr(client, "_identifier", None), None)
            self.client = chromadb.PersistentClient(path=self.chroma_dir, settings=self._settings)
            self.collection.native = self.client.get_collection(name=name, embedding_function=None)
            self._revision = self._database_revision()

    def _records(self):
        self._refresh()
        identifiers = sorted(self.collection.native.get(include=[])["ids"])
        for start in range(0, len(identifiers), 128):
            result = self.collection.native.get(
                ids=identifiers[start:start + 128],
                include=["documents", "metadatas", "embeddings"],
            )
            embeddings = result["embeddings"]
            if embeddings is None or len(embeddings) != len(result["ids"]):
                raise RuntimeError("Chroma metadata and native vectors disagree; indexing failed. Reindex before auditing.")
            records = []
            for index, identifier in enumerate(result["ids"]):
                vector = embeddings[index]
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                records.append((identifier, result["documents"][index], result["metadatas"][index] or {}, vector))
            yield from sorted(records, key=lambda record: record[0])

    def corpus_fingerprint(self):
        digest = hashlib.sha256()
        with self._lock:
            for identifier, document, metadata, vector in self._records():
                digest.update(json.dumps([identifier, document, metadata], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                digest.update(struct.pack(f"<{len(vector)}f", *vector))
        return digest.hexdigest()

    def add_chunks(self, chunks):
        return self._add_chunks(chunks)

    def _add_chunks(self, chunks, lock_held=False, reuse_embeddings=False):
        if not chunks:
            return
        ensure_disk_space(self.chroma_dir)
        inference = {key: value for key, value in self.embedding_signature.items()
                     if key not in {"storage_sha256", "query_algorithm"}}
        embedding_fingerprint = hashlib.sha256(json.dumps(inference, sort_keys=True).encode("utf-8")).hexdigest()
        batch_size = min(100, max(1, int(os.environ.get("MEETHAQ_INDEX_BATCH_SIZE", "50"))))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [chunk["text"] for chunk in batch]
            embeddings = []
            cached = {}
            with (nullcontext() if lock_held else self._lock):
                self._refresh()
                existing = self.collection.native.get(ids=[chunk["chunk_id"] for chunk in batch], include=["documents", "embeddings", "metadatas"])
                vectors = existing["embeddings"]
                if vectors is not None and len(vectors) == len(existing["ids"]):
                    for position, identifier in enumerate(existing["ids"]):
                        vector = [float(value) for value in vectors[position]]
                        metadata = existing["metadatas"][position] or {}
                        if (reuse_embeddings and metadata.get("embedding_fingerprint") == embedding_fingerprint
                                and len(vector) == 384 and all(math.isfinite(value) for value in vector) and any(vector)):
                            cached[identifier] = (existing["documents"][position], vector)
            missing = [index for index, chunk in enumerate(batch)
                       if chunk["chunk_id"] not in cached or cached[chunk["chunk_id"]][0] != chunk["text"]]
            fresh = []
            print(f"[Indexing] Embedding chunks {start + 1}-{start + len(batch)}/{len(chunks)} (CPU, subbatch 8)...", flush=True)
            for offset in range(0, len(missing), 8):
                fresh.extend(self.embedding_fn.embed([texts[index] for index in missing[offset:offset + 8]]))
            fresh_by_index = dict(zip(missing, fresh))
            embeddings = [fresh_by_index[index] if index in fresh_by_index else cached[chunk["chunk_id"]][1]
                          for index, chunk in enumerate(batch)]
            if len(missing) < len(batch):
                print(f"[Indexing] Reused {len(batch) - len(missing)} validated stored embeddings.", flush=True)
            metadatas = []
            for chunk in batch:
                metadata = {
                    "source": str(chunk.get("source", "unknown")),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "contract_category": str(chunk.get("contract_category", "Unknown")),
                    "embedding_fingerprint": embedding_fingerprint,
                }
                for key in ("char_start", "char_end", "starts_at_boundary", "ends_at_boundary", "annotated_evidence", "retrieval_unit"):
                    if key in chunk:
                        metadata[key] = chunk[key]
                metadatas.append(metadata)
            ensure_disk_space(self.chroma_dir)
            with (nullcontext() if lock_held else self._lock):
                self._refresh()
                identifiers = [chunk["chunk_id"] for chunk in batch]
                self.collection.native.upsert(ids=identifiers, documents=texts, embeddings=embeddings, metadatas=metadatas)
                self._revision = self._database_revision()
                # SegmentAPI can log a failed native consumer without raising to upsert.
                # Verify the actual stored vectors and text before claiming persistence.
                stored = self.collection.native.get(ids=identifiers, include=["documents", "embeddings", "metadatas"])
                vectors = stored["embeddings"]
                if set(stored["ids"]) != set(identifiers) or vectors is None or len(vectors) != len(identifiers):
                    raise RuntimeError("Chroma upsert did not persist every native vector; check memory and disk, then reindex.")
                expected = {identifier: (text, metadata, vector) for identifier, text, metadata, vector in zip(identifiers, texts, metadatas, embeddings)}
                for position, identifier in enumerate(stored["ids"]):
                    text, metadata, vector = expected[identifier]
                    actual = vectors[position]
                    expected_norm = math.sqrt(sum(value * value for value in vector))
                    actual_norm = math.sqrt(sum(float(value) ** 2 for value in actual))
                    if (stored["documents"][position] != text or stored["metadatas"][position] != metadata
                            or len(actual) != 384 or not actual_norm
                            or any(not math.isfinite(float(value)) for value in actual)
                            or any(abs(float(left) / actual_norm - right / expected_norm) > 1e-5 for left, right in zip(actual, vector))):
                        raise RuntimeError("Chroma upsert verification failed; no completed indexing was reported. Reindex after resolving native storage errors.")
            count = self.collection.native.count() if lock_held else self.collection.count()
            print(f"[Indexing] Persisted {start + len(batch)}/{len(chunks)} chunks; collection count={count}", flush=True)

    def replace_chunks(self, source, chunks, reuse_embeddings=False):
        active = {chunk["chunk_id"] for chunk in chunks}
        # A document replacement is one serialized operation: concurrent writers
        # cannot delete each other's new versions during obsolete-ID cleanup.
        with self._lock:
            self._refresh()
            self._add_chunks(chunks, lock_held=True, reuse_embeddings=reuse_embeddings)
            previous = self.collection.native.get(where={"source": source}, include=[])["ids"]
            obsolete = [identifier for identifier in previous if identifier not in active]
            if obsolete:
                self.collection.native.delete(ids=obsolete)
                self._revision = self._database_revision()

    def query(self, query_text, top_k=5):
        from grounding import applicable_annotations, canonical_embedding_query

        if top_k < 1 or not query_text.strip():
            return []
        if self.collection.count() == 0:
            return []
        query_vector = self.embedding_fn.embed([canonical_embedding_query(query_text)])[0]
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        candidates = []
        sequence = 0
        with self._lock:
            for identifier, document, metadata, vector in self._records():
                if not document:
                    continue
                if metadata.get("retrieval_unit") == "context_window":
                    continue
                if applicable_annotations(query_text, {**metadata, "text": document}) == []:
                    continue
                norm = math.sqrt(sum(value * value for value in vector))
                if len(vector) != len(query_vector) or not norm or not all(math.isfinite(value) for value in vector):
                    raise RuntimeError("Persistent collection contains invalid embeddings; reindex before auditing.")
                similarity = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * norm)
                similarity = max(-1.0, min(1.0, similarity))
                candidate = {
                    **metadata, "chunk_id": identifier, "text": document,
                    "source": metadata.get("source", "unknown"), "chunk_index": metadata.get("chunk_index", -1),
                    "contract_category": metadata.get("contract_category", "Unknown"),
                    "cosine_distance": 1.0 - similarity, "cosine_similarity": similarity, "rerank_score": similarity,
                }
                # Keep only top_k records in memory while scanning the corpus.
                # Larger distance/ID is the heap's least desirable record.
                inverse_id = tuple(-ord(character) for character in identifier) + (1,)
                entry = (-candidate["cosine_distance"], inverse_id, sequence, candidate)
                sequence += 1
                if len(candidates) < top_k:
                    heapq.heappush(candidates, entry)
                elif entry[:2] > candidates[0][:2]:
                    heapq.heapreplace(candidates, entry)
        ranked = [entry[3] for entry in candidates]
        ranked.sort(key=lambda item: (item["cosine_distance"], item["chunk_id"]))
        return ranked

    def close(self):
        self.embedding_fn.close()
        # Chroma persists every committed operation; _system is its embedded-runtime
        # lifecycle hook. Do not stop a shared system while another local client uses it.
        client = getattr(self, "client", None)
        if client is not None and hasattr(client, "_system"):
            with self._lock:
                with _SYSTEM_COUNTS_LOCK:
                    remaining = _SYSTEM_COUNTS.get(self.chroma_dir, 1) - 1
                    if remaining > 0:
                        _SYSTEM_COUNTS[self.chroma_dir] = remaining
                    else:
                        _SYSTEM_COUNTS.pop(self.chroma_dir, None)
                        client._system.stop()
                        identifier = getattr(client, "_identifier", None)
                        cache = getattr(type(client), "_identifier_to_system", {})
                        if identifier is not None:
                            cache.pop(identifier, None)
            self.client = None
            self.collection.native = None
        atexit.unregister(self.close)

