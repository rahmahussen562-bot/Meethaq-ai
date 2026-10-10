FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    MEETHAQ_CHROMA_DIR=/app/chroma_db \
    MEETHAQ_COLLECTION=meethaq_contracts \
    MEETHAQ_EXPECTED_CHUNKS=1863 \
    MEETHAQ_CALIBRATION_PATH=/app/data/rag_calibration.json \
    MEETHAQ_ONNX_MODEL_DIR=/app/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx \
    MEETHAQ_LLM_PROVIDER=groq \
    MEETHAQ_LLM_MODEL=openai/gpt-oss-20b \
    MEETHAQ_STATUS="Cloud Backend" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=appuser:appuser api.py grounding.py llm_synthesis.py native_runtime.py ./
COPY --chown=appuser:appuser rag_pipeline.py vector_store.py embedding_worker.py ./
COPY --chown=appuser:appuser chroma_db ./chroma_db
COPY --chown=appuser:appuser data/rag_calibration.json ./data/rag_calibration.json
COPY --chown=appuser:appuser .cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx \
    ./.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','7860')+'/api/health', timeout=10)" || exit 1

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
