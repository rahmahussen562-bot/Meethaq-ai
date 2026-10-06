"""FastAPI entrypoint for the persistent, grounded local audit service."""

from __future__ import annotations

from native_runtime import configure_native_runtime
configure_native_runtime()

import logging
import os
import threading
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from grounding import MAX_QUESTION_LENGTH
from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _pipeline
    if _pipeline is not None:
        _pipeline.close()
        _pipeline = None


app = FastAPI(title="Meethaq AI Local Server", lifespan=lifespan)

# Anchored host matching prevents suffix tricks such as pages.dev.evil.example.
# Cookies are not used, so credentials are disabled.
ORIGIN_REGEX = (
    r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?$"
    r"|^https?://(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:pages\.dev|workers\.dev|trycloudflare\.com)(?::[0-9]{1,5})?$"
)


def additional_origins() -> list[str]:
    origins = []
    for value in os.getenv("MEETHAQ_CORS_ORIGINS", "").split(","):
        value = value.strip()
        if not value:
            continue
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or "*" in value):
            raise ValueError("MEETHAQ_CORS_ORIGINS must contain explicit HTTP(S) origins")
        origins.append(f"{parsed.scheme}://{parsed.netloc}")
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=additional_origins(),
    allow_origin_regex=ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

def get_rag() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                try:
                    _pipeline = RAGPipeline()
                except Exception as exc:
                    logger.exception("Persistent contract index initialization failed")
                    raise HTTPException(
                        status_code=503,
                        detail="Contract index unavailable. Check the backend traceback and storage configuration.",
                    ) from exc
    return _pipeline


class AuditRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    # Kept for existing clients; no LLM query expansion is performed.
    expand_query: bool = False


class AuditResponse(BaseModel):
    answer: str
    sources: list[dict]
    stats: dict


@app.post("/api/audit", response_model=AuditResponse)
def run_audit(req: AuditRequest, rag: RAGPipeline = Depends(get_rag)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        return rag.audit(req.query, expand_query=req.expand_query)
    except Exception as exc:
        logger.exception("Contract audit failed")
        raise HTTPException(
            status_code=503,
            detail="Contract retrieval unavailable. Check the backend traceback.",
        ) from exc


@app.get("/api/telemetry")
def get_telemetry(rag: RAGPipeline = Depends(get_rag)):
    try:
        stats = rag.stats()
        return {**stats, "total_chunks": stats["indexed_chunks"],
                "status": "Air-Gapped Local Host"}
    except Exception as exc:
        logger.exception("Persistent contract telemetry failed")
        raise HTTPException(
            status_code=503,
            detail="Contract index telemetry unavailable. Check the backend traceback.",
        ) from exc


@app.get("/api/health")
def get_health(rag: RAGPipeline = Depends(get_rag)):
    telemetry = get_telemetry(rag)
    ready = telemetry["indexed_chunks"] > 0 and telemetry["calibration_status"] == "calibrated"
    return {**telemetry, "status": "ready" if ready else "not_ready", "ready": ready}


if __name__ == "__main__":
    # Passing the app directly avoids a second api module initialization.
    # Restart explicitly after backend edits; no production reloader process.
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
