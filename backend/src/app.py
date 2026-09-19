"""
FastAPI REST API Gateway for LexiRAG (ContractAdvisor-AI).
Exposes endpoints for contract querying, streaming token generation, PDF ingestion, and system health checks.
"""

import logging
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag.rag_system import RAGSystem
from .core.config import config
from .core.pdf_loader import PDFLoader

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LexiRAG API",
    description="Autonomous Contract Legal Intelligence & Verification Engine",
    version=config.VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG System singleton
rag_system = RAGSystem()


# Request / Response Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language contract question", min_length=2)
    top_k: Optional[int] = Field(default=4, description="Number of relevant clauses to retrieve")


class CitationItem(BaseModel):
    id: int
    source: str
    page: Any
    snippet: str
    chunk_id: Any


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationItem] = []
    query: str
    model: Optional[str] = None
    retrieval_mode: Optional[str] = None
    faithfulness: Optional[Dict[str, Any]] = None
    telemetry: Optional[Dict[str, Any]] = None
    status: str = "success"


class DocumentItem(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class AddDocumentsRequest(BaseModel):
    documents: List[DocumentItem]


# --- API Endpoints ---

@app.get("/")
async def root():
    """Returns service information and capabilities."""
    return {
        "service": config.PROJECT_NAME,
        "version": config.VERSION,
        "status": "online",
        "features": [
            "Hybrid Retrieval (Dense + BM25 RRF)",
            "Groundedness & Hallucination Guardrails",
            "Granular Clause Citation Attribution",
            "Streaming Token Synthesis",
            "Real-time PDF Ingestion"
        ],
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Healthcheck endpoint for monitoring and container probes."""
    return {
        "status": "healthy",
        "has_openai_key": bool(config.OPENAI_API_KEY),
        "vector_store_path": config.CHROMA_PERSIST_DIR,
        "hybrid_retrieval_ready": True
    }


@app.post("/query/", response_model=QueryResponse)
async def query_contract(request: QueryRequest):
    """Answers a contract legal question with grounded clause citations and faithfulness metrics."""
    try:
        result = rag_system.answer_query(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/query/stream")
async def query_contract_stream(request: QueryRequest):
    """
    Streams the legal answer token-by-token via Server-Sent Events (SSE).
    """
    result = rag_system.answer_query(request.question, top_k=request.top_k)
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    telemetry = result.get("telemetry", {})
    faithfulness = result.get("faithfulness", {})

    async def event_generator():
        # Stream citations first
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        await asyncio.sleep(0.02)

        # Stream answer chunks
        words = answer.split(" ")
        for word in words:
            yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
            await asyncio.sleep(0.015)

        # Stream final metrics
        yield f"data: {json.dumps({'type': 'done', 'faithfulness': faithfulness, 'telemetry': telemetry})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload/")
async def upload_contract_pdf(file: UploadFile = File(...)):
    """Uploads and indexes a contract PDF file directly into the vector store and BM25 index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported.")

    try:
        content = await file.read()
        docs = PDFLoader.extract_text_from_bytes(content, filename=file.filename)
        
        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        chunks_added = rag_system.add_documents(docs)
        return {
            "message": f"Contract '{file.filename}' processed successfully.",
            "pages_extracted": len(docs),
            "chunks_indexed": chunks_added,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Error uploading contract PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest contract: {str(e)}")


@app.post("/add-documents/")
async def add_documents(request: List[DocumentItem]):
    """Ingest raw text documents."""
    try:
        texts = [doc.content for doc in request if doc.content.strip()]
        metas = [doc.metadata or {"source": "api_upload"} for doc in request if doc.content.strip()]

        if not texts:
            raise HTTPException(status_code=400, detail="No non-empty documents provided.")

        count = rag_system.add_documents(texts, metadatas=metas)
        return {"message": "Documents added successfully.", "chunks_indexed": count}
    except Exception as e:
        logger.error(f"Error adding text documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Optional Unified Static Files Serving (Single-Container / Space Mode)
# --------------------------------------------------------------------------
frontend_build_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "build"
if frontend_build_dir.exists() and (frontend_build_dir / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    static_assets = frontend_build_dir / "static"
    if static_assets.exists():
        app.mount("/static", StaticFiles(directory=str(static_assets)), name="static")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa_frontend(full_path: str):
        api_prefixes = ("query", "upload", "health", "add-documents", "docs", "openapi.json", "redoc")
        if full_path.startswith(api_prefixes):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        file_candidate = frontend_build_dir / full_path
        if file_candidate.is_file():
            return FileResponse(file_candidate)
        return FileResponse(frontend_build_dir / "index.html")
