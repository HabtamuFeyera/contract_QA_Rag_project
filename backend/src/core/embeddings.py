"""
Embeddings Handler module.
Handles batching, validation, and encoding workflows for ingested documents.
"""

import logging
from typing import List, Optional
from ..models.openai_embeddings import OpenAIEmbeddingsWrapper

logger = logging.getLogger(__name__)


class EmbeddingsHandler:
    """Orchestrates embedding creation with validation and error resilience."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.embeddings_wrapper = OpenAIEmbeddingsWrapper(openai_api_key)

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
