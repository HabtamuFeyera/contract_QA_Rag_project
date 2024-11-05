import logging
from ..models.openai_embeddings import OpenAIEmbeddingsWrapper
from ..core.config import config

# Set up logging for better tracking of events
logging.basicConfig(level=logging.INFO)

class EmbeddingsHandler:
    def __init__(self, openai_api_key: str):
        """Initializes the EmbeddingsHandler with the OpenAI API key."""
        self.embeddings_wrapper = OpenAIEmbeddingsWrapper(openai_api_key)

    def get_embedding(self, document: str):
        """
        Generates an embedding for a single document.

        Args:
            document (str): The document to embed.

        Returns:
            list: The generated embedding.
        """
        try:
            embedding = self.embeddings_wrapper.encode(document)
            logging.info(f"Successfully generated embedding for document.")
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding for document: {str(e)}")
            raise

    def get_embeddings(self, documents: list):
        """
        Generates embeddings for a list of documents.

        Args:
            documents (list): A list of documents to embed.

        Returns:
            list: A list of generated embeddings.
        """
        embeddings = []
        
        if not documents:
            logging.warning("No documents provided for embedding.")
            return embeddings

        for i, doc in enumerate(documents):
            if not doc.strip():  # Skip empty documents
                logging.warning(f"Skipping empty document at index {i}.")
                continue

            try:
                embedding = self.get_embedding(doc)
                embeddings.append(embedding)
            except Exception as e:
                logging.error(f"Error generating embedding for document '{doc}': {str(e)}")

        logging.info(f"Generated embeddings for {len(embeddings)} documents.")
        return embeddings
