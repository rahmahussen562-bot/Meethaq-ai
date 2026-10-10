# Meethaq AI cloud deployment

The production path is:

`Cloudflare Worker UI and /api proxy -> Gradio Space with mounted FastAPI -> packaged ChromaDB + ONNX embeddings -> Groq`

The browser never receives the Groq key. The FastAPI container reads it from the host's secret store. The Worker keeps using same-origin `/api` requests, so no backend URL is compiled into the frontend bundle.

## 1. Create a Groq API key

1. Open <https://console.groq.com> and sign in.
2. Create or select a project named `meethaq-ai-prod`.
3. Open **API Keys**, select **Create API Key**, name it `meethaq-ai-space`, and copy it once.
4. Store it as the `GROQ_API_KEY` secret in the cloud host. Never put it in `.env`, Git, Vite variables, or Cloudflare frontend code.

The backend defaults to `openai/gpt-oss-20b`, a Groq model available to the deployment account with JSON Object Mode. To change it, set `MEETHAQ_LLM_MODEL` without changing source code.

## 2. Deploy the backend to a Hugging Face Gradio Space

Create a Space at <https://huggingface.co/new-space> with these values:

- Name: `meethaq-ai-backend`
- SDK: **Gradio**
- Application file: `app.py`
- Port: `7860` (the entrypoint runs the mounted FastAPI/Gradio application)
- Hardware: CPU Basic for testing, or upgraded CPU for strict always-on service

In **Space Settings -> Variables and secrets**, add:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `GROQ_API_KEY` | the Groq key |
| Variable | `MEETHAQ_LLM_PROVIDER` | `groq` |
| Variable | `MEETHAQ_LLM_MODEL` | `openai/gpt-oss-20b` |
| Variable | `MEETHAQ_CORS_ORIGINS` | `https://meethaq-ai.rahmahussen562.workers.dev` |

Authenticate and push the prepared bundle. Replace `HF_USER` with the Space owner:

```powershell
python -m pip install --upgrade huggingface_hub
hf auth login
git xet install
git remote add hf https://huggingface.co/spaces/HF_USER/meethaq-ai-backend
git add app.py requirements.txt README.md CLOUD_DEPLOYMENT.md
git add api.py grounding.py llm_synthesis.py native_runtime.py rag_pipeline.py vector_store.py embedding_worker.py
git add chroma_db data/rag_calibration.json .cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx
git commit -m "deploy: run Meethaq backend independently in the cloud"
git push hf HEAD:main
```

The persistent backend origin is `https://HF_USER-meethaq-ai-backend.hf.space`.

Verify it before changing the Worker:

```powershell
$backend = "https://HF_USER-meethaq-ai-backend.hf.space"
Invoke-RestMethod "$backend/api/health"
Invoke-RestMethod "$backend/api/telemetry"
```

Both responses must report `indexed_chunks: 1863`, `calibration_status: calibrated`, and `llm_ready: true`.

## 3. Point the Cloudflare Worker at the persistent backend

Keep `VITE_API_URL` blank. The browser calls the Worker's same-origin `/api` routes and the Worker proxies them to the Space.

```powershell
$backend = "https://HF_USER-meethaq-ai-backend.hf.space"
$backend | pnpm dlx wrangler@3.114.15 secret put MEETHAQ_API_ORIGIN --name meethaq-ai
pnpm run build
pnpm dlx wrangler@3.114.15 deploy --name meethaq-ai
```

Then verify the public deployment:

```powershell
Invoke-RestMethod "https://meethaq-ai.rahmahussen562.workers.dev/api/health"
Invoke-RestMethod "https://meethaq-ai.rahmahussen562.workers.dev/api/telemetry"
```

The permanent user-facing URL remains <https://meethaq-ai.rahmahussen562.workers.dev/audit>.

## Render alternative

`render.yaml` defines the same Docker backend. Push the branch to GitHub, create a Render Blueprint from the repository, and enter `GROQ_API_KEY` when prompted:

```powershell
git add Dockerfile .dockerignore render.yaml requirements.txt
git add api.py grounding.py llm_synthesis.py native_runtime.py rag_pipeline.py vector_store.py embedding_worker.py
git add chroma_db data/rag_calibration.json .cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx
git commit -m "deploy: add independent cloud backend"
git push origin HEAD
```

After Render assigns `https://SERVICE.onrender.com`, use that origin in the same Wrangler secret command above.

## Availability limits

The deployment removes all laptop, Ollama, and quick-tunnel dependencies. Hugging Face CPU Basic currently sleeps after 48 hours without traffic, and current Hub policy requires a paid plan to create a new compute Space even though CPU Basic has no hourly charge. Render Free sleeps after 15 minutes. A request wakes a sleeping service, but may experience a cold start. Strict always-on availability requires upgraded Hugging Face hardware or a paid Render instance. The packaged database is read-only application data, so an ephemeral filesystem does not lose the 1,863 indexed chunks on restarts; each Space rebuild restores them.

## Generic OpenAI-compatible or Workers AI endpoint

To use another OpenAI-compatible service instead of Groq, set:

```text
MEETHAQ_LLM_PROVIDER=openai-compatible
MEETHAQ_LLM_BASE_URL=https://provider.example/v1
MEETHAQ_LLM_API_KEY=<host secret>
MEETHAQ_LLM_MODEL=<provider model id>
```

Cloudflare Workers AI can be used through its OpenAI-compatible account endpoint with the same variables. The server continues to request temperature zero and JSON output, then applies the same citation, vocabulary, modality, and abstention validator before returning any answer.
