"""Hugging Face Gradio Space entrypoint for the Meethaq FastAPI backend."""

from __future__ import annotations

import os

try:
    import spaces
except ImportError:  # Available only on Hugging Face ZeroGPU runtimes.
    spaces = None

import gradio as gr
import uvicorn
from fastapi.responses import RedirectResponse

from deployment_assets import ensure_packaged_assets

ensure_packaged_assets()

from api import app as fastapi_app, get_rag


if spaces is not None:

    @spaces.GPU(duration=1)
    def _zerogpu_startup_probe() -> None:
        """Register this CPU-only service with the ZeroGPU scheduler."""
        return None


def system_snapshot() -> dict:
    """Return a small, non-sensitive readiness view for the Space dashboard."""
    try:
        rag = get_rag()
        stats = rag.stats()
        llm = rag.llm_health()
        ready = (
            stats.get("indexed_chunks", 0) > 0
            and stats.get("calibration_status") == "calibrated"
            and llm.get("llm_ready") is True
        )
        return {
            "status": "ready" if ready else "not_ready",
            "indexed_chunks": stats.get("indexed_chunks", 0),
            "collection": stats.get("collection", ""),
            "calibration_status": stats.get("calibration_status", "unknown"),
            "llm_ready": llm.get("llm_ready", False),
            "llm_provider": llm.get("llm_provider", "ollama"),
            "llm_model": llm.get("llm_model", stats.get("llm_model", "")),
        }
    except Exception:
        return {
            "status": "unavailable",
            "indexed_chunks": 0,
            "calibration_status": "unavailable",
            "llm_ready": False,
        }


with gr.Blocks(title="Meethaq AI Backend") as dashboard:
    gr.Markdown(
        "# Meethaq AI Backend\n"
        "Grounded contract retrieval service. The production interface is served by "
        "[Meethaq AI](https://meethaq-ai.rahmahussen562.workers.dev/audit)."
    )
    telemetry = gr.JSON(label="System health", value=system_snapshot)
    refresh = gr.Button("Refresh health", variant="primary")
    refresh.click(fn=system_snapshot, outputs=telemetry, api_name=False)
    dashboard.load(fn=system_snapshot, outputs=telemetry, api_name=False)
    if spaces is not None:
        zerogpu_probe = gr.Button(visible=False)
        zerogpu_probe.click(fn=_zerogpu_startup_probe, api_name=False)


@fastapi_app.get("/", include_in_schema=False)
def dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/")


app = gr.mount_gradio_app(
    fastapi_app,
    dashboard,
    path="/dashboard",
    ssr_mode=False,
    show_error=False,
    footer_links=[],
)


if __name__ == "__main__":
    if spaces is not None:
        from spaces.zero import startup as zerogpu_startup

        zerogpu_startup()
    port = int(os.getenv("APP_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
