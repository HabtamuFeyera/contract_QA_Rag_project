# src/models/vector_store.py

from langchain.vectorstores import Chroma
from src.models.openai_embeddings import OpenAIEmbeddingsWrapper

class VectorStore:
    def __init__(self, openai_api_key):
        """Initializes the VectorStore with OpenAI API key and sets up embeddings."""
        self.embeddings_wrapper = OpenAIEmbeddingsWrapper(openai_api_key=openai_api_key)
        self.vector_store = Chroma(embedding_function=self.embeddings_wrapper.encode)

    def add_documents(self, documents):
        """
        Adds documents to the vector store by generating embeddings.

        Args:
            documents (list): A list of documents (strings) to add to the store.
        """
        embeddings = [self.embeddings_wrapper.encode(doc) for doc in documents]
        self.vector_store.add_texts(texts=documents, embeddings=embeddings)

    def query(self, query_text, k=5):
        """
        Queries the vector store for similar documents.

        Args:
            query_text (str): The query text to find similar documents.
            k (int): The number of similar documents to retrieve.

        Returns:
            list: A list of the top k similar documents.
        """
        query_embedding = self.embeddings_wrapper.encode(query_text)
        return self.vector_store.similarity_search(query_embedding, k=k)

