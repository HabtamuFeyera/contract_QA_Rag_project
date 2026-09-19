"""
RAG System Orchestration for LexiRAG.
Coordinates hybrid retrieval, context synthesis, citation extraction, and LLM inference.
"""

import logging
from typing import List, Dict, Any, Union, Optional
from langchain_core.documents import Document

from ..core.config import config
from ..models.vector_store import VectorStore
from ..models.chat_model import ChatModel
from ..utils.helpers import format_context_for_prompt, extract_citations_from_docs

logger = logging.getLogger(__name__)


class RAGSystem:
    """Enterprise RAG pipeline tailored for high-precision legal contract QA."""

    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = openai_api_key or config.OPENAI_API_KEY
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        self.model_name = model_name or config.DEFAULT_MODEL

        logger.info(f"Initializing LexiRAG System with model: {self.model_name}")
        self.vector_store = VectorStore(
            openai_api_key=self.api_key,
            persist_directory=self.persist_directory
        )
        self.chat_model = ChatModel(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=config.TEMPERATURE
        )

    def add_documents(
        self, 
        documents: Union[List[str], List[Document]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Ingests and chunks contract documents into the vector store.
        """
        if not documents:
            logger.warning("No documents supplied for ingestion.")
            return 0

        try:
            if isinstance(documents[0], str):
                ids = self.vector_store.add_texts(documents, metadatas)
            else:
                ids = self.vector_store.add_documents(documents)
            
            count = len(ids)
            logger.info(f"Successfully ingested and indexed {count} document chunks.")
            return count
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def add_pdf(self, file_path: str) -> int:
        """Loads, chunks, and indexes a PDF contract from file path."""
        from ..core.pdf_loader import PDFLoader
        loader = PDFLoader(file_path)
        docs = loader.load_documents()
        return self.add_documents(docs)

    def answer_query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query:
        1. Retrieves top-k semantically relevant contract clauses.
        2. Compiles grounded prompt with exact page/section metadata.
        3. Generates legal synthesis.
        4. Returns answer alongside structured citations.
        """
        if not query or not query.strip():
            return {
                "answer": "Please provide a valid question regarding the contract.",
                "citations": [],
                "query": query,
                "status": "empty_query"
            }

        k = top_k or config.TOP_K_RESULTS
        logger.info(f"Processing query: '{query}' (top_k={k})")

        try:
            # 1. Retrieve relevant chunks
            retrieved_docs = self.vector_store.query(query, k=k)

            # If vector store is empty or no relevant chunks
            if not retrieved_docs:
                return {
                    "answer": "No contract documents have been indexed yet, or no relevant clauses were found matching your query.",
                    "citations": [],
                    "query": query,
                    "status": "no_context"
                }

            # 2. Extract structured citations
            citations = extract_citations_from_docs(retrieved_docs)

            # 3. Format context block with citations for LLM
            context_block = format_context_for_prompt(retrieved_docs)

            # 4. Generate answer via domain-specific prompt
            answer = self.chat_model.generate_answer(query=query, context_text=context_block)

            return {
                "answer": answer,
                "citations": citations,
                "query": query,
                "model": self.model_name,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error answering query '{query}': {str(e)}")
            return {
                "answer": f"An error occurred while analyzing the contract: {str(e)}",
                "citations": [],
                "query": query,
                "status": "error"
            }
