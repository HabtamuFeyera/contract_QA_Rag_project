"""
OpenAI Embeddings Wrapper module.
Provides consistent embedding vector generation with offline testing fallback.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_openai_embeddings_class():
    """Dynamically imports OpenAIEmbeddings from langchain_openai or langchain_community."""
    try:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings
        except ImportError:
            return None


class OpenAIEmbeddingsWrapper:
    """Wrapper around LangChain OpenAIEmbeddings with fallback support."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or ""
        self._embeddings = None

        if self.api_key:
            EmbeddingsClass = get_openai_embeddings_class()
            if EmbeddingsClass is not None:
                try:
                    self._embeddings = EmbeddingsClass(
                        openai_api_key=self.api_key
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAIEmbeddings: {e}.")

    @property
    def embeddings(self):
        """Returns the underlying LangChain Embeddings instance."""
        if self._embeddings is not None:
            return self._embeddings
        
        # Fallback fake embeddings for offline testing / development
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=1536)

    def encode(self, document: str) -> List[float]:
        """Generates a 1D vector embedding for a single text document."""
        if not document or not document.strip():
            return [0.0] * 1536
        return self.embeddings.embed_query(document)

    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """Generates a list of 1D vector embeddings for multiple documents in batch."""
        if not documents:
            return []
        cleaned_docs = [doc if doc.strip() else " " for doc in documents]
        return self.embeddings.embed_documents(cleaned_docs)
