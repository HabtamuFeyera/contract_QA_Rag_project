"""
Application Configuration Module for LexiRAG (ContractAdvisor-AI).
Handles environment variables, API credentials, and persistence paths with safe fallbacks.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate project base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()  # Fallback to local environment


class Config:
    """Central configuration for LexiRAG system."""

    PROJECT_NAME: str = "LexiRAG - Autonomous Contract Legal Intelligence"
    VERSION: str = "2.0.0"
    
    # API Keys (Google Gemini primary with legacy OpenAI fallback)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Storage & Persistence Paths
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: str = os.getenv(
        "VECTOR_STORE_PATH", 
        str(BASE_DIR / "data" / "chroma_db")
    )
    PDF_DIRECTORY: str = os.getenv(
        "PDF_DIRECTORY", 
        str(BASE_DIR / "data" / "contracts")
    )

    # LLM Settings (Google Gemini)
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "gemini-1.5-pro")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1500"))

    # RAG & Ingestion Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))

    # CORS Settings
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]

    @classmethod
    def validate(cls) -> bool:
        """Validates that critical credentials exist for production runtime."""
        return bool(cls.GEMINI_API_KEY or cls.OPENAI_API_KEY)

    @classmethod
    def ensure_directories(cls):
        """Ensures required data directories exist on disk."""
        os.makedirs(cls.CHROMA_PERSIST_DIR, exist_ok=True)
        os.makedirs(cls.PDF_DIRECTORY, exist_ok=True)


config = Config()
config.ensure_directories()
