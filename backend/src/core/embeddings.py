# src/core/embeddings.py

from backend.src.core.config import config
from backend.src.models.openai_embeddings import OpenAIEmbeddingsWrapper

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
        try:
            embedding = self.embeddings_wrapper.encode(document)
            return embedding
        except Exception as e:
            print(f"Error generating embedding for document: {str(e)}")
            raise

    def get_embeddings(self, documents):
        """
        Generates embeddings for a list of documents.

        Args:
            documents (list): A list of documents to embed.

        Returns:
            list: A list of generated embeddings.
        """
        embeddings = []
        for doc in documents:
            try:
                embedding = self.get_embedding(doc)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for document '{doc}': {str(e)}")
        return embeddings
