# Meethaq AI audit and completion report

Audit date: 2026-10-06. Backend ready at http://127.0.0.1:8000.

## Completed state

- Applied native storage, ingestion, deterministic grounding, frontend state/API and deployment patches.
- Indexed all 20 prepared contracts into the project-root `chroma_db` directory and `meethaq_contracts` collection: **1,863 persisted records**, with 384-dimensional vectors.
- Of those records, 350 are complete clause retrieval units. Context windows remain persisted and counted; prepared-corpus retrieval selects complete clauses before semantic top-k selection.
- Reopened the real collection from a separate process: 1,863 records, 20 distinct sources, calibrated policy and exact annotation membership. SQLite `PRAGMA quick_check` returned `ok`.
- Installed strict raw cosine distance cutoff **0.5760104796196565**. Equality is rejected: eligible distance must be less than the cutoff.
- Final actual C: headroom was 202 MiB; indexing retains its 64 MiB reserve guard.
- Started the backend using the project Python 3.11 environment. Live `/api/health` returns `ready=true`; telemetry reports identical `indexed_chunks` and `total_chunks`.

## Bugs, root causes and applied patches

| Finding | Root cause / evidence | Applied correction |
| --- | --- | --- |
| Native ingestion crash | Reproduced access violations in native indexing; Windows event 1000 identified `MSVCP140.dll` 14.31.31005.0 with exception 0xC0000005. The large native-index probe succeeded after preloading signed Microsoft 14.44 DLLs. | `native_runtime.py` validates the project runtime manifest and DLL hashes, then preloads the runtime before native imports. `provision_native_runtime.ps1` imports only signed Microsoft x64 runtime files from an explicitly supplied CAB. No Windows DLL or service was changed. |
| Runtime installer unavailable | The official redistributable installer failed with exit 1601; the Windows Installer service was absent. | Provisioned and verified the project-local runtime instead. The importer passed against the official CAB, including existing-cache reuse. |
| Unstable native backend/startup | Rust-backed Chroma writes and ONNX imports crashed during diagnosis. Changing batch size or NumPy alone did not resolve the HNSW crash. | Use Chroma 1.5.9 Python SegmentAPI with chroma-hnswlib 0.7.6; verified ONNX Runtime 1.20.1 and NumPy 1.26.4. Pins describe the tested stack rather than an unproven single-package root cause. |
| Disk exhaustion | C: repeatedly reached zero free bytes. This independently prevented writes and caused failed setup/indexing. | Check for a 64 MiB reserve before index batches; bounded model setup and cleanup. No legal corpus was deleted. |
| Silent Python/native failure | A native access violation cannot be caught by Python try/except. Large implicit embedding/upsert calls gave little progress information. | Supervised ingestion and embedding workers, faulthandler, native exit codes, flushed progress, explicit CPU embeddings, 50-record write batches and at most 8 texts per embedding request. Windows fault dialogs are disabled for these workers. |
| Logged write failure mistaken for success | Segment consumers can log a write exception without propagating it to the caller. | Verify every batch's stored IDs, documents, metadata and native vectors before reporting completed persistence. |
| Stale or conflicting database readers | Embedded vector segments are cached; a live API can retain an older collection view after another process indexes. | Serialize native access with a cross-process file lock and thread lock; reopen the native reader on database revision changes; stop/release native clients cleanly. A real two-process regression confirms immediate refreshed counts and retrieval. |
| Working-directory dependent storage | Relative `./chroma_db` could identify different databases depending on how a process started. | Shared project-anchored database and model paths, with explicit environment overrides. |
| Fake 729 count / misleading zero | Frontend defaults and fallback paths fabricated 729. The initial actual collection was empty and healthy; no collection-name mismatch was found in that database. | Initial count is unknown until successful telemetry; display actual persistent counts. Storage failures return HTTP 503 rather than fabricated zero. Both indexer and API use `meethaq_contracts`. |
| Incorrect similarity / no relevance cutoff | Earlier conversion used `1-distance/2`, and top-k candidates went directly to generation. | Exact cosine scan over native vectors, `similarity=1-distance`, stable ID tie breaking, strict unrounded cutoff and finite/dimension validation. |
| Nearby text and incomplete clauses | General 1,000-character windows included headings or adjacent material; clipped edges could omit qualifications. | Preserve complete original sentences around independently annotated CUAD spans; index full clause units, never truncate annotated evidence to six sentences, and retain exceptions, list fragments and negations. |
| Incorrect topic association | Governing-law vocabulary collapsed to generic law references; assignment could match IP ownership assignment; renewal could select a notice-specific classification. | Independent CUAD type filtering before top-k, dominant topic cues, and exact original-text annotation validation. Headings alone cannot establish an obligation. |
| Synonym-dependent geometry | Question verbs and aliases produced different embeddings for the same grounded intent, including liability caps versus liability limitations. | Canonicalize query vocabulary deterministically before embedding. Entity, quantity and modifier terms remain present; no generative query expansion is used. |
| Topic labels supplied unsupported modifiers | A classification could otherwise imply unlimited or perpetual terms absent from the excerpt. | Labels supply only explicit legal topic vocabulary. Other modifiers, entities and quantities must appear in emitted evidence; numeric filenames cannot establish quantities. |
| Ungrounded answer generation | An LLM prompt does not establish that every generated claim is entailed by retrieved text. Query expansion could invoke Ollama before relevance checks. | Strict mode emits exact retrieved excerpts and never calls Ollama. Each paragraph has a sequential `[Source N]` citation mapped to a unique retrieved chunk. |
| Forged / mismapped citations | Generated tags could be invented; UI lookup previously fell back to source-array position. | Reject reserved citation tags in document text; verify exact excerpt membership, sequential labels, unique IDs and citation mappings on both backend and client. |
| False offline alert | HTTP, parse, timeout, routing and configuration errors became a hardcoded port-8000 offline message. A remote fallback selected the visitor's loopback. | Resolve the active API URL from deployment context and VITE_API_URL; use typed failures and the active API for telemetry/reachability polling. Preserve backend error detail. |
| Stale telemetry / audit state | One-time telemetry and overlapping requests could overwrite newer responses; failures retained previous evidence. | Poll without overlap, abort superseded requests, version state updates, clear old evidence and synchronize count from actual audit stats. |
| Strict Mode loading hang | Effect replay cancelled an initial audit while another guard prevented restart. | Defer actual-unmount cancellation and check lifecycle generation. |
| URL question overwrote follow-up | An effect watched answer/loading state and replayed its original URL question after unrelated state changes. | Audit the URL question only when that question changes. |
| Fabricated comparison / export | Inspector comparisons and export used static penalty facts rather than live evidence. | Remove invented comparisons and export actual answer/source text. |
| Misleading samples / file attachments | Sample counts looked live; selecting a file sent only its filename. | Label sample workspaces and disclose that selected files are not indexed. |
| Overbroad CORS | Wildcard origins were combined with credentialed requests. | Anchored localhost, 127.0.0.1, IPv6 loopback, pages.dev, workers.dev and trycloudflare.com matching; explicit optional custom origins; credentials disabled. |
| Duplicate initialization / missing shutdown | Import-time native initialization and development reload could duplicate clients; resources lacked a shutdown path. | Lazy locked singleton, one app instance, explicit API lifespan cleanup and worker/client close. |
| Missing deployment routing | SPA refresh routing and same-origin API forwarding were not fully configured. | Scoped Pages redirects and tested Workers asset/API routing with an explicitly configured HTTPS tunnel origin. |
| Online fonts / obsolete dependencies and docs | Google Fonts required networking; obsolete Torch/cross-encoder dependencies consumed disk; docs described earlier APIs. | System fonts, native ONNX requirements, explicit offline-capable provisioning and current setup/deployment instructions. |
| Evaluation rejected valid label boundaries | CUAD spans often omit final punctuation or start after a clause number. Strict raw substring comparison falsely rejected complete original clauses. | Evaluation ignores punctuation, a trailing list marker and narrowly recognized section prefixes. Factual words, quantities, negations and conditions remain checked; regression tests cover these distinctions. |

## Calibration evidence and limits

The installed policy and complete measurement report are in `data/rag_calibration.json` and `data/rag_calibration.report.json`. The fixed fixture is `data/retrieval_evaluation_v2.json`.

| Measurement | Calibration/development | Validation query set |
| --- | --- | --- |
| Positive queries with supporting evidence | 8 / 9 | 8 / 9 |
| Negative queries that abstained | 13 / 13 | 9 / 9 |
| Unsupported positive answers | 0 | 0 |
| False accepts | 0 | 0 |

Exclusivity questions in this fixture abstain conservatively. The cutoff was selected from calibration/development distances, then frozen before validation evaluation; the acceptance target remained 0.8 positive recall, zero accepted negatives and zero unsupported positive answers.

The initial held-out evaluation failed. Its report is preserved in `data/rag_calibration.initial-validation.report.json`. Earlier clause-unit/evaluation runs are also retained. After observing the original holdout, those cases became development data and 18 new queries were frozen before their first retrieval. Those queries were subsequently reused while correcting evaluation boundaries and implementing canonical query normalization. Synonymous queries can normalize to the same intent across splits. **The final numbers are finite regression results, not pristine independent holdout accuracy or a guarantee for new topics/corpora.** No cutoff was forced on after a failed run.

Exact excerpts prevent newly generated factual prose. Semantic relevance and complete legal interpretation cannot be proved for every possible question by this fixture. Unsupported questions, missing/stale calibration and insufficient evidence return exactly:

> I could not find an answer to this question in the provided documents.

Calibration binds corpus text, metadata, stored float32 vectors, model/tokenizer hashes, native runtime/settings and grounding/retrieval code. Corpus or policy changes invalidate it. Restart after backend/model changes; reindex and recalibrate as appropriate.

Installed corpus fingerprint: `fd8072a5a3fe1f912307149e6c91464e849da9cf46a3dd79297de72798a72a8e`.

Fixture SHA256: `36551ea67a6e4f331a807349d17f1b5002f259dbbe190a2d1a16dbcf07b980ec`.

## Verification actually executed

- **65 backend tests passed**: original 31 plus 34 added regressions. Breakdown: API 9, grounding/pipeline/chunker 16, calibration 10, prepared evidence 8, model provisioning 9, native storage/configuration 13.
- **19 client/proxy tests passed**: frontend contract 7, API client 6, Cloudflare Worker 6.
- Frontend TypeScript and standalone Worker TypeScript checks passed.
- Production Vite build passed. Python syntax checks and `git diff --check` passed. `pip check` found no broken requirements.
- Actual full-corpus ingestion completed with exit code 0. The final real collection contains 1,863 records across 20 sources; all inspected native vectors have 384 dimensions and every stored annotated span is an exact substring of its chunk.
- A native 145-vector integration test crossed both batching and disk synchronization boundaries and retained count, fingerprint and cosine results after reopening.
- Two-process native test kept a reader open while a writer added another contract, then verified refreshed count, fingerprint and retrieval.
- Live backend: 3 supported queries, 3 identical-repeat comparisons, exact citation/distance checks, and 4 exact negative abstentions, all without LLM invocation.
- Live CORS: localhost, 127.0.0.1, Pages, Workers and trycloudflare origins accepted; a crafted pages.dev.evil.example suffix rejected.
- The real frontend response decoder accepted all 7 captured live audit responses and verified citation mappings.

Local evidence: `data/live_verification.json`, `data/live_audit_responses.json`, `logs/api.stdout.log` and `logs/api.stderr.log`.

## Operation and review artifacts

The backend is left running at http://127.0.0.1:8000 using the verified project environment. Server PID at completion: 13008; Windows virtual-environment launcher PID: 20324. Logs contain successful startup and no backend traceback from the completed live checks.

For this installed corpus, rerun calibration with:

```powershell
.venv\Scripts\python.exe -B calibrate_rag.py --cases data/retrieval_evaluation_v2.json
```

Frontend local URL resolution and remote deployment configuration are tested. No live Cloudflare deployment/tunnel or browser interaction was performed in this continuation; an actual deployment still needs its HTTPS tunnel origin configured. The existing Vite development server was not manually started.

Applied source/configuration/test changes are also packaged in `MEETHAQ_AUDIT_PATCH.diff`. The 34-file patch passed `git apply --reverse --check` against the applied workspace. Legal data, database files, model/runtime binaries, logs and real environment secrets are excluded from that portable patch; they remain local. Full source files in the workspace are the applied implementation.

The cosine definition follows [Chroma's metric documentation](https://docs.trychroma.com/docs/collections/configure). CORS behavior follows [FastAPI's middleware documentation](https://fastapi.tiangolo.com/tutorial/cors/). Vite build-time environment behavior follows [Vite's environment documentation](https://vite.dev/guide/env-and-mode). The verified Windows runtime setup is consistent with [ONNX Runtime's Windows installation requirements](https://onnxruntime.ai/docs/install/).
