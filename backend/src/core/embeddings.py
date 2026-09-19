"""
Embeddings Handler module.
Handles batching, validation, and encoding workflows for ingested documents.
"""

import logging
from typing import List, Optional
from ..models.gemini_embeddings import GeminiEmbeddingsWrapper

logger = logging.getLogger(__name__)


class EmbeddingsHandler:
    """Orchestrates embedding creation with validation and error resilience using Google Gemini."""

    def __init__(self, gemini_api_key: Optional[str] = None, **kwargs):
        self.embeddings_wrapper = GeminiEmbeddingsWrapper(gemini_api_key=gemini_api_key, **kwargs)

    def get_embedding(self, document: str) -> List[float]:
        """
        Generates an embedding vector for a single document string.
        """
        try:
            return self.embeddings_wrapper.encode(document)
        except Exception as e:
            logger.error(f"Error generating embedding for document: {str(e)}")
            raise

    def get_embeddings(self, documents: List[str]) -> List[List[float]]:
        """
        Generates batch embeddings for a list of document strings.
        """
        if not documents:
            logger.warning("No documents provided for embedding.")
            return []

        try:
            return self.embeddings_wrapper.encode_documents(documents)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
