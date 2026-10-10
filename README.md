---
title: Meethaq AI Backend
emoji: ⚖️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: 3.11
app_file: app.py
pinned: false
---

# Meethaq AI

React 19/Vite client and a local FastAPI contract evidence service. ChromaDB 1.5.9 uses its Python SegmentAPI with chroma-hnswlib 0.7.6 to persist CPU all-MiniLM-L6-v2 ONNX embeddings in the project-root chroma_db directory. The default collection is meethaq_contracts. Retrieval computes exact cosine distances over stored vectors. The Python SegmentAPI is used consistently with explicit persistence verification. Native index crashes on this Windows installation were traced to the older system MSVCP140.dll 14.31 runtime; the optional project runtime below supplies verified 14.44 DLLs before native imports.

## Answer contract

Audits first apply the calibrated cosine cutoff and deterministic clause/evidence gates. Eligible evidence is sent to the selected synthesis provider: local Ollama, Groq, or another OpenAI-compatible endpoint. Cloud deployment uses Groq `openai/gpt-oss-20b`; local deployment can continue to use Ollama `llama3.2:3b`. Both paths request temperature zero and structured JSON. Accepted answers contain one concise assertion with immediate clickable `[Source N]` citations. A server-side validator rejects unknown citations, unsupported meaningful vocabulary, changed legal modality/negation/timing and invented entities or amounts. Query expansion remains disabled, including for legacy clients that send `expand_query=true`.

Questions outside contract scope, missing evidence, distances at or above the calibrated cutoff, missing/stale calibration, model abstention, unavailable Ollama, and invalid model output return exactly:

    I could not find an answer to this question in the provided documents.

A finite relevance evaluation cannot prove perfect relevance for every possible question, and deterministic validation cannot prove semantic entailment. Conservative scope, evidence coverage, citation, vocabulary and modality checks fail closed and can abstain on valid paraphrases, compound questions, or unsupported languages. The complete retrieved chunks remain available in the inspector for legal review.

## Local setup

Use Node 22.18+ or 24 and Python 3.11, tested here with Python 3.11.9. Python 3.14 has not been validated for this backend. Install Python requirements in a virtual environment and frontend dependencies with pnpm. The existing Vite server uses port 8443.

    py -3.11 -m venv .venv
    .venv\Scripts\python.exe -B -m pip install -r requirements.txt
    pnpm install

For local frontend pages on localhost/127.0.0.1, an unset VITE_API_URL selects http://127.0.0.1:8000. Database paths resolve against this project's directory rather than the terminal's working directory.

On Windows, use the Microsoft Visual C++ x64 runtime 14.44 or later for these native dependencies. When installing the current official redistributable is unavailable, import its vcRuntimeMinimum_amd64 CAB into the project cache:

    powershell -NoProfile -File .\provision_native_runtime.ps1 -CabPath C:\path\to\vcRuntimeMinimum_amd64\cab1.cab

The importer accepts exactly twelve x64 DLLs with valid Microsoft signatures and a consistent version of at least 14.44. It checks sizes and free space, verifies hashes, and writes .cache/native-runtime/dll/provisioned.json. Existing DLLs are reused only when they match the supplied CAB. This setup performs no download or installer action and changes no Windows DLLs or services.

When this verified project cache is present, the backend preloads its runtime before importing Chroma, NumPy, or ONNX. Otherwise it uses the system runtime. The system MSVCP140.dll on the audited machine remains at 14.31; the backend's process uses the project-local 14.44 copy. Restart the backend after provisioning a native runtime.

Provision the official ONNX model explicitly while online, before indexing or air-gapped operation:

    .venv\Scripts\python.exe -B provision_onnx.py --download

For an air-gapped installation, transfer the official Chroma onnx.tar.gz archive and import it locally:

    .venv\Scripts\python.exe -B provision_onnx.py --archive C:\path\to\onnx.tar.gz

Setup verifies the official SHA-256 before extracting files, checks available disk space throughout, and prints the absolute model directory. The project cache is .cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx. The backend uses an existing legacy cache at ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx when model.onnx is present; otherwise it defaults to the project cache. Set MEETHAQ_ONNX_MODEL_DIR to select a cache explicitly. Relative values resolve against the project root:

    $env:MEETHAQ_ONNX_MODEL_DIR = ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx"

Prestage the CUAD input before going offline. Embedding and audit execution use local CPU inference and never download models. Ollama remains an optional local service; it is unnecessary for extractive audits. A successful model setup alone does not populate the collection or install calibration; perform the indexing and calibration steps below.

## Index and calibrate

Index the processed CUAD JSONL already present in data. Reprocessing raw CUAD is optional and --download is an explicit online operation.

    .venv\Scripts\python.exe -B ingest_cuad.py --processed data/cuad_processed.jsonl --index
    .venv\Scripts\python.exe -B calibrate_rag.py --create-cuad-cases data/cuad_processed.jsonl
    .venv\Scripts\python.exe -B calibrate_rag.py --cases data/retrieval_evaluation.json

Start the backend with the project virtual environment after setup, indexing, and calibration. Restart it after changing model files or backend code:

    .venv\Scripts\python.exe -B api.py

For the complete production startup, including an Ollama/model check, model warm-up with a 24-hour keep-alive, and an HTTP/2 Cloudflare quick tunnel, run:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\run_production.ps1

Add `-DeployWorker` to rebuild the same-origin frontend, update the Worker's `MEETHAQ_API_ORIGIN` secret to the new tunnel, and deploy it. The script starts `ollama serve` when the local binary exists but its API is unavailable; it never downloads a missing model automatically and prints the exact `ollama pull` diagnostic instead.

Calibration fixes clause-topic and negative-query splits before retrieving distances. The cutoff is selected on calibration cases, then frozen for held-out validation. Failed validation does not enable retrieval. Review and expand the independently labeled fixture to cover your actual contracts and questions.

The local data/rag_calibration.report.json records raw distances, sample sizes, supported positive recall, false accepts and threshold selection. The installed data/rag_calibration.json is bound to corpus content/metadata, embedding settings, metric and gate/retrieval code. Reindexing or changing the policy requires recalibration. These artifacts and legal data are intentionally ignored by Git.

The completed local corpus has 1,863 persisted records across 20 contracts, including 350 complete clause retrieval units. The installed cutoff is distance < 0.5760104796196565. For this installation, use the fixed data/retrieval_evaluation_v2.json fixture when recalibrating:

    .venv\Scripts\python.exe -B calibrate_rag.py --cases data/retrieval_evaluation_v2.json

The final fixture has 8/9 supported positives and 9/9 negative abstentions. Queries were reused during debugging and synonymous queries can normalize to the same intent, so these are regression measurements rather than independent generalization accuracy. AUDIT_REPORT.md preserves the failed evaluations and explains the limits.

GET /api/telemetry reports the real persistent indexed_chunks and total_chunks, plus calibration_status. GET /api/health returns ready=true only for a nonempty, calibrated index. A reachable but unready backend is distinct from an offline backend. Storage errors return HTTP 503, never a fake zero.

## Cloudflare

For the deployed Worker, leave `VITE_API_URL` unset so the browser uses same-origin `/api` requests. For a separately hosted frontend, set it to the persistent HTTPS backend origin before building. `/api` and trailing slash suffixes are normalized.

For Workers, `wrangler.toml` and `cloudflare/worker.ts` serve `dist` assets and proxy `/api` requests. Set `MEETHAQ_API_ORIGIN` in the Worker environment to the persistent HTTPS Docker backend. See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for the Groq key, Hugging Face/Render push, Worker update, verification commands, and free-tier availability limits. `run_production.ps1` remains available for temporary local/tunnel operation.

Remote sites never silently select local loopback. HTTPS sites reject HTTP API configuration. FastAPI permits anchored localhost, 127.0.0.1, pages.dev, workers.dev and trycloudflare.com origins; MEETHAQ_CORS_ORIGINS adds comma-separated explicit origins for custom domains. Cookies are unused and credentialed CORS is disabled.

Cloudflare access transmits queries and returned excerpts through Cloudflare. Fully air-gapped use means running the frontend/backend locally without a tunnel.

## Validation and audit results

    pnpm run typecheck
    pnpm test
    pnpm run build
    .venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py"

On Windows, direct Node entrypoints can be used when the pnpm command shim is unavailable:

    node node_modules/typescript/bin/tsc --noEmit
    node node_modules/typescript/bin/tsc --noEmit --strict --target ES2022 --module ESNext --moduleResolution bundler --lib ES2022,DOM --skipLibCheck cloudflare/worker.ts
    node tests/frontend.test.mjs
    node tests/api-client.test.mjs
    node cloudflare/test_worker.mjs
    node node_modules/vite/bin/vite.js build

See AUDIT_REPORT.md for identified bugs, evidence, applied fixes and the actual validation results. The historical PROJECT_DOCUMENTATION.md describes earlier design goals and must not be used as the current implementation specification.
