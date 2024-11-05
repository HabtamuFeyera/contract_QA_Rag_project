# src/core/embeddings.py

from src.core.openai_embeddings import OpenAIEmbeddingsWrapper

class EmbeddingsHandler:
    def __init__(self, openai_api_key):
        """Initializes the EmbeddingsHandler with the OpenAI API key."""
        self.embeddings_wrapper = OpenAIEmbeddingsWrapper(openai_api_key)

    def get_embedding(self, document):
        """
        Generates an embedding for a single document.

        Args:
            document (str): The document to embed.

        Returns:
            list: The generated embedding.
        """
        return self.embeddings_wrapper.encode(document)

    def get_embeddings(self, documents):
        """
        Generates embeddings for a list of documents.

        Args:
            documents (list): A list of documents to embed.

        Returns:
            list: A list of generated embeddings.
        """
        return [self.embeddings_wrapper.encode(doc) for doc in documents]
