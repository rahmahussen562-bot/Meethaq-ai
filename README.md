# Local RAG Pipeline

End-to-end Retrieval-Augmented Generation with a two-stage retrieval
architecture, running entirely on your machine — no cloud APIs, no API keys.

```
Documents → Clean → Chunk → Embed → ChromaDB
                                        │
User Query → Normalize ─────► Dense Retrieval (Top-12)
                                        │
                              Cross-Encoder Rerank (Top-3)
                                        │
                               Context Assembly
                                        │
                          Ollama llama3.2:3b → Answer
```

---

## Stack

| Component | Library | Model |
|---|---|---|
| Dense embeddings | `sentence-transformers` | `all-MiniLM-L6-v2` |
| Vector store | `chromadb` | — |
| Cross-encoder reranker | `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `ollama` | `llama3.2:3b` |
| PDF loading | `pymupdf` | — |

---

## Setup

### 1. Virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB) and the cross-encoder
> (~80 MB) from HuggingFace automatically.

### 3. Pull the LLM

Make sure Ollama is running, then:

```bash
ollama pull llama3.2:3b
```

---

## Quickstart

### As a library

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Index a folder of PDFs, text files, and/or markdown files
rag.index_documents("./my_docs")

# Ask a question — answer streams to the terminal, full string returned
answer = rag.query("What are the main findings of the report?")
```

### Index a single file

```python
rag.index_documents("./contract.pdf")
answer = rag.query("What is the termination clause?")
```

### Interactive CLI

```bash
# Start the REPL
python rag_pipeline.py

# Or index a folder on startup
python rag_pipeline.py ./my_docs
```

CLI commands:

| Command | Action |
|---|---|
| `/index <path>` | Index a file or directory |
| `/stats` | Show collection statistics |
| `/clear` | Wipe the entire index |
| `exit` / `quit` | Exit |

---

## Configuration

All defaults live at the top of `rag_pipeline.py`:

```python
EMBED_MODEL      = "all-MiniLM-L6-v2"
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL        = "llama3.2:3b"

CHUNK_SIZE       = 500    # characters
CHUNK_OVERLAP    = 75     # characters
RETRIEVAL_TOP_K  = 12     # candidates from dense retrieval
RERANK_TOP_N     = 3      # final chunks after reranking

CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "rag_docs"
```

Override per-instance:

```python
rag = RAGPipeline(
    chunk_size=800,
    rerank_top_n=5,
    llm_model="llama3.1:8b",   # swap in a better model if your RAM allows
)
```

---

## Adding Document Formats

Open `DocumentLoader` in `rag_pipeline.py` and add a loader method plus a
dispatch entry:

```python
@staticmethod
def _load_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

# Register it:
_LOADERS = {
    ".txt":  _load_txt.__func__,
    ".md":   _load_md.__func__,
    ".pdf":  _load_pdf.__func__,
    ".docx": _load_docx.__func__,   # ← new
}
```

---

## Pipeline Module Map

| Step | Class | Key method |
|---|---|---|
| 1. Load | `DocumentLoader` | `.load(source)` |
| 2. Clean | `TextCleaner` | `.clean(text)` |
| 3. Chunk | `TextChunker` | `.chunk(doc)` |
| 4–5. Embed + Store | `VectorStore` | `.add_chunks(chunks)` |
| 6. Query rewrite | `QueryHandler` | `.normalize(q)` / `.expand_with_llm(q)` |
| 7. Query embed | `VectorStore` | `.embed([text])` |
| 8. Dense retrieval | `VectorStore` | `.query(q, top_k)` |
| 9. Reranking | `Retriever` | `.retrieve(q)` |
| 10. Context | `ContextAssembler` | `.assemble(chunks)` |
| 11. Generation | `LLMGenerator` | `.generate(q, context)` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` | Ollama not running | Start Ollama or run `ollama serve` |
| `model not found` | Model not pulled | `ollama pull llama3.2:3b` |
| Empty retrieval results | No documents indexed | Call `rag.index_documents(path)` first |
| Very slow first query | Models loading into memory | Normal — subsequent queries are faster |
| Out of memory | Embedding too many chunks | Reduce `RETRIEVAL_TOP_K` or use a smaller embed model |
| `No supported documents found` | Wrong path or unsupported format | Check the path; add a custom loader for the format |
