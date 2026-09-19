"""
FastAPI REST API Gateway for LexiRAG (ContractAdvisor-AI).
Exposes endpoints for contract querying, PDF ingestion, and system health checks.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
    status: str = "success"


class DocumentItem(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class AddDocumentsRequest(BaseModel):
    documents: List[DocumentItem]


# --- API Endpoints ---

@app.get("/")
async def root():
    """Returns service information."""
    return {
        "service": config.PROJECT_NAME,
        "version": config.VERSION,
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Healthcheck endpoint for monitoring and container probes."""
    return {
        "status": "healthy",
        "has_openai_key": bool(config.OPENAI_API_KEY),
        "vector_store_path": config.CHROMA_PERSIST_DIR
    }


@app.post("/query/", response_model=QueryResponse)
async def query_contract(request: QueryRequest):
    """Answers a contract legal question with grounded clause citations."""
    try:
        result = rag_system.answer_query(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/upload/")
async def upload_contract_pdf(file: UploadFile = File(...)):
    """Uploads and indexes a contract PDF file directly into the vector store."""
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
    """Backward-compatible endpoint to ingest raw text documents."""
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
