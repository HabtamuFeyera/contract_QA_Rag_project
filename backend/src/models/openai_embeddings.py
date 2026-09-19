"""
Embeddings compatibility module.
Bridges legacy OpenAI embeddings references to Google Gemini embeddings.
"""

from .gemini_embeddings import GeminiEmbeddingsWrapper

# Alias for backward compatibility
OpenAIEmbeddingsWrapper = GeminiEmbeddingsWrapper
