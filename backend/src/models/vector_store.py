"""
VectorStore Manager for LexiRAG.
Handles persistent and in-memory Chroma storage, legal-aware text chunking, and similarity search.
"""

import logging
import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .gemini_embeddings import GeminiEmbeddingsWrapper

logger = logging.getLogger(__name__)


def get_chroma_class():
    """Dynamically imports Chroma from langchain_chroma or langchain_community."""
    try:
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
        return Chroma


class VectorStore:
    """Manages document chunking, indexing, and vector similarity retrieval."""

    def __init__(
        self, 
        gemini_api_key: Optional[str] = None, 
        persist_directory: Optional[str] = None,
        collection_name: str = "legal_contracts_gemini",
        **kwargs
    ):
        api_key = (
            gemini_api_key 
            or kwargs.get("openai_api_key") 
            or os.getenv("GEMINI_API_KEY") 
            or os.getenv("GOOGLE_API_KEY") 
            or os.getenv("OPENAI_API_KEY")
        )
        self.embeddings_wrapper = GeminiEmbeddingsWrapper(gemini_api_key=api_key)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._vector_store = None
        self._init_vector_store()

        # Legal-aware recursive chunker
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=[
                "\n\nSection ",
                "\n\nClause ",
                "\n\nARTICLE ",
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def _init_vector_store(self):
        """Initializes ChromaDB vector store safely."""
        Chroma = get_chroma_class()
        try:
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings_wrapper.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"Initialized persistent ChromaDB at {self.persist_directory}")
            else:
                self._vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings_wrapper.embeddings
                )
                logger.info("Initialized in-memory ChromaDB")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma vector store: {e}")
            raise

    @property
    def vector_store(self):
        """Returns the underlying Chroma instance."""
        return self._vector_store

    def chunk_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        Chunks text strings into LangChain Documents with legal-aware boundaries and metadata.
        """
        docs_to_chunk = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {"source": f"doc_{i+1}"}
            docs_to_chunk.append(Document(page_content=text, metadata=meta))

        chunked_docs = self.text_splitter.split_documents(docs_to_chunk)
        for idx, chunk in enumerate(chunked_docs):
            chunk.metadata["chunk_id"] = idx
        return chunked_docs

    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Splits and adds raw text strings into the vector store.
        """
        try:
            chunks = self.chunk_texts(texts, metadatas)
            if not chunks:
                logger.warning("No valid text chunks generated to index.")
                return []
            ids = self._vector_store.add_documents(chunks)
            logger.info(f"Successfully indexed {len(chunks)} chunks into vector store.")
            return ids
        except Exception as e:
            logger.error(f"Error indexing texts into vector store: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Splits and indexes LangChain Document objects (preserving pages and source paths).
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = idx
            ids = self._vector_store.add_documents(chunks)
            logger.info(f"Indexed {len(chunks)} document chunks.")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def query(self, query_text: str, k: int = 4) -> List[Document]:
        """
        Queries the vector store for top-k semantically relevant chunks.
        """
        if not query_text or not isinstance(query_text, str):
            raise ValueError("query_text must be a non-empty string.")

        try:
            logger.info(f"Executing semantic similarity search for: '{query_text}' (top_k={k})")
            results = self._vector_store.similarity_search(query_text, k=k)
            return results
        except Exception as e:
            logger.error(f"Error executing similarity search: {str(e)}")
            raise

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Returns a LangChain retriever interface for chaining."""
        kwargs = search_kwargs or {"k": 4}
        return self._vector_store.as_retriever(search_kwargs=kwargs)
