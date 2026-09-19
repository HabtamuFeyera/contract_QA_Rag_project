"""
RAG System Orchestration for LexiRAG.
Coordinates Hybrid Retrieval (Dense + BM25 RRF), context synthesis,
citation extraction, groundedness guardrails, and telemetry profiling.
"""

import time
import logging
from typing import List, Dict, Any, Union, Optional
from langchain_core.documents import Document

from ..core.config import config
from ..models.vector_store import VectorStore
from ..models.chat_model import ChatModel
from ..rag.hybrid_retriever import HybridRetriever
from ..rag.guardrails import GroundednessGuardrail
from ..utils.helpers import format_context_for_prompt, extract_citations_from_docs

logger = logging.getLogger(__name__)


class RAGSystem:
    """Senior-grade Autonomous RAG pipeline tailored for high-precision legal contract QA."""

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
        # Hybrid Retriever: Dense Chroma + Sparse BM25 via Reciprocal Rank Fusion (RRF)
        self.hybrid_retriever = HybridRetriever(self.vector_store)

    def add_documents(
        self, 
        documents: Union[List[str], List[Document]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Ingests and chunks contract documents into both the Dense Vector Store and BM25 index.
        """
        if not documents:
            logger.warning("No documents supplied for ingestion.")
            return 0

        try:
            if isinstance(documents[0], str):
                chunks = self.vector_store.chunk_texts(documents, metadatas)
                ids = self.vector_store._vector_store.add_documents(chunks)
            else:
                chunks = self.vector_store.text_splitter.split_documents(documents)
                ids = self.vector_store._vector_store.add_documents(chunks)
            
            # Synchronize BM25 index with new documents
            self.hybrid_retriever.update_index(chunks)

            count = len(ids)
            logger.info(f"Successfully ingested and hybrid-indexed {count} document chunks.")
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
        Executes end-to-end Hybrid RAG query:
        1. Hybrid retrieval (Dense Vector + BM25 RRF fusion).
        2. Compiles grounded prompt with exact page/section metadata.
        3. Generates legal synthesis.
        4. Runs Groundedness Guardrail (faithfulness & financial figures validation).
        5. Computes execution latency metrics.
        """
        start_time = time.time()
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
            # 1. Hybrid Retrieval (Dense Vector + BM25 RRF)
            retrieval_start = time.time()
            retrieved_docs = self.hybrid_retriever.retrieve(query, top_k=k)
            retrieval_time_ms = round((time.time() - retrieval_start) * 1000, 1)

            # If no relevant chunks
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
            gen_start = time.time()
            answer = self.chat_model.generate_answer(query=query, context_text=context_block)
            gen_time_ms = round((time.time() - gen_start) * 1000, 1)

            # 5. Hallucination & Groundedness Guardrail
            guardrail_start = time.time()
            guardrail_result = GroundednessGuardrail.compute_faithfulness_score(answer, retrieved_docs)
            guardrail_time_ms = round((time.time() - guardrail_start) * 1000, 1)

            total_time_ms = round((time.time() - start_time) * 1000, 1)

            return {
                "answer": answer,
                "citations": citations,
                "query": query,
                "model": self.model_name,
                "retrieval_mode": "Hybrid (Dense Vector + BM25 RRF)",
                "faithfulness": guardrail_result,
                "telemetry": {
                    "retrieval_ms": retrieval_time_ms,
                    "generation_ms": gen_time_ms,
                    "guardrail_ms": guardrail_time_ms,
                    "total_ms": total_time_ms
                },
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
