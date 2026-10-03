from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_pipeline import RAGPipeline

app = FastAPI(title="Meethaq AI Local Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline(collection_name="meethaq_contracts", chunk_size=1000, chunk_overlap=150)

class AuditRequest(BaseModel):
    query: str
    expand_query: bool = False

class AuditResponse(BaseModel):
    answer: str
    sources: List[dict]
    stats: dict

@app.post("/api/audit", response_model=AuditResponse)
def run_audit(req: AuditRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    normalized = rag.query_handler.normalize(req.query)
    if req.expand_query:
        normalized = rag.query_handler.expand_with_llm(normalized)

    top_chunks = rag.retriever.retrieve(normalized)
    if not top_chunks:
        return {
            "answer": "I could not find an answer to this question in the provided documents.",
            "sources": [],
            "stats": rag.stats()
        }

    context, sources = rag.assembler.assemble(top_chunks)
    answer = rag.generator.generate(question=normalized, context=context, stream=False)

    return {
        "answer": answer,
        "sources": sources,
        "stats": rag.stats()
    }

@app.get("/api/telemetry")
def get_telemetry():
    stats = rag.stats()
    return {
        "total_chunks": stats.get("indexed_chunks", 0),
        "embed_model": stats.get("embed_model", ""),
        "rerank_model": stats.get("rerank_model", ""),
        "llm_model": stats.get("llm_model", ""),
        "status": "Air-Gapped Local Host"
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
