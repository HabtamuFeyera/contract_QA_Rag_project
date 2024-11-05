from langchain_chroma import Chroma
from openai_embeddings import OpenAIEmbeddingsWrapper  
import logging
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))


logging.basicConfig(level=logging.INFO)

class VectorStore:
    def __init__(self, openai_api_key):
        """Initializes the VectorStore with OpenAI API key and sets up embeddings."""
        self.embeddings_wrapper = OpenAIEmbeddingsWrapper(openai_api_key=openai_api_key)
        self.vector_store = Chroma(embedding_function=self.embeddings_wrapper.embeddings)

    def add_documents(self, documents):
        """Adds documents to the vector store by generating embeddings."""
        try:
            
            documents = [str(doc) if not isinstance(doc, str) else doc for doc in documents]
            
           
            documents = self.chunk_documents(documents)

            embeddings = [self.embeddings_wrapper.encode(doc) for doc in documents]

            
            self.vector_store.add_texts(texts=documents, embeddings=embeddings)
            logging.info(f"Successfully added {len(documents)} documents to the vector store.")
        except Exception as e:
            logging.error(f"Error while adding documents: {str(e)}")
            raise

    def query(self, query_text, k=5):
        """
        Queries the vector store for similar documents based on the query text.
        Ensures the query text is a string before querying.
        """
        try:
            
            if isinstance(query_text, list):
                query_text = query_text[0] 
            if isinstance(query_text, (int, float)):
                query_text = str(query_text) 

            if not isinstance(query_text, str):
                raise ValueError("The query_text must be a string or convertible to a string.")

            
            query_embedding = self.embeddings_wrapper.encode(query_text)

            
            if isinstance(query_embedding, list):
                query_embedding = query_embedding[0]  

            logging.info(f"Performing similarity search for the query: {query_text}")
            return self.vector_store.similarity_search(query_embedding, k=k)

        except Exception as e:
            logging.error(f"Error in querying the vector store: {str(e)}")
            raise

    def as_retriever(self):
        """Converts the vector store into a retriever format."""
        try:
            return self.vector_store.as_retriever()
        except Exception as e:
            logging.error(f"Error in converting to retriever: {str(e)}")
            raise

    def chunk_documents(self, documents, chunk_size=1000):
        """Chunks the documents into smaller parts to avoid large text issues."""
        chunks = []
        for doc in documents:
            while len(doc) > chunk_size:
                chunk = doc[:chunk_size]
                chunks.append(chunk)
                doc = doc[chunk_size:]  
            if doc: 
                chunks.append(doc)
        return chunks
