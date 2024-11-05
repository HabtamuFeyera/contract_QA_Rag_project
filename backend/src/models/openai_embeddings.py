# src/core/openai_embeddings.py

from langchain.embeddings.openai import OpenAIEmbeddings

class OpenAIEmbeddingsWrapper:
    def __init__(self, openai_api_key):
        """Initializes the OpenAIEmbeddingsWrapper with the OpenAI API key."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def encode(self, document):
        """
        Generates embeddings for the provided document.

        Args:
            document (str): The document or text to encode.

        Returns:
            list: The generated embeddings.
        """
        return self.embeddings.encode(document)
