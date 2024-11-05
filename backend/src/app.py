# src/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag.rag_system import RAGSystem
from src.core.config import config

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG System
rag_system = RAGSystem(openai_api_key=config.OPENAI_API_KEY)

# Define request models
class Document(BaseModel):
    content: str

class Query(BaseModel):
    question: str

# Endpoint to add documents
@app.post("/add-documents/")
async def add_documents(documents: list[Document]):
    """Endpoint to add documents to the RAG system."""
    try:
        rag_system.add_documents([doc.content for doc in documents])
        return {"message": "Documents added successfully."}
    except Exception as e:
        print(f"Error adding documents: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to answer a query
@app.post("/query/")
async def answer_query(query: Query):
    """Endpoint to answer a user's query."""
    try:
        response = rag_system.answer_query(query.question)
        return {"answer": response}
    except Exception as e:
        print(f"Error answering query: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail="Error answering query.")
