"""
Google Gemini Embeddings Wrapper module.
Provides embedding vector generation using Google's text-embedding-004 model.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_gemini_embeddings_class():
    """Dynamically imports GoogleGenerativeAIEmbeddings from langchain_google_genai."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings
    except ImportError:
        return None


class GeminiEmbeddingsWrapper:
    """Wrapper around Google Gemini Embeddings with offline fallback."""

    def __init__(
        self, 
        gemini_api_key: Optional[str] = None, 
        model: str = "models/text-embedding-004",
        **kwargs
    ):
        self.api_key = (
            gemini_api_key 
            or kwargs.get("openai_api_key") 
            or os.getenv("GEMINI_API_KEY") 
            or os.getenv("GOOGLE_API_KEY") 
            or ""
        )
        self.model = model
        self._embeddings = None

        if self.api_key:
            EmbeddingsClass = get_gemini_embeddings_class()
            if EmbeddingsClass is not None:
                try:
                    self._embeddings = EmbeddingsClass(
                        model=self.model,
                        google_api_key=self.api_key
                    )
                    logger.info(f"Initialized GoogleGenerativeAIEmbeddings with {self.model}")
                except Exception as e:
                    logger.warning(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}.")

    @property
    def embeddings(self):
        """Returns the underlying LangChain Embeddings instance."""
        if self._embeddings is not None:
            return self._embeddings

        # Fallback fake embeddings for offline testing / development
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=768)

    def encode(self, document: str) -> List[float]:
        """Generates a 1D vector embedding for a single text document."""
        if not document or not document.strip():
            return [0.0] * 768
        return self.embeddings.embed_query(document)

    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """Generates a list of 1D vector embeddings for multiple documents in batch."""
        if not documents:
            return []
        cleaned_docs = [doc if doc.strip() else " " for doc in documents]
        return self.embeddings.embed_documents(cleaned_docs)
