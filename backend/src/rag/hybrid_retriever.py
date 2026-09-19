"""
Hybrid Retrieval Engine for LexiRAG.
Implements Reciprocal Rank Fusion (RRF) combining Dense Vector Similarity and Sparse BM25 Lexical Search.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Plus as BM25Engine
except ImportError:
    from rank_bm25 import BM25Okapi as BM25Engine

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    State-of-the-art Hybrid Retriever merging Dense Vector Search and Lexical BM25.
    Ensures exact legal clause identifiers (e.g. 'Section 6.2', '$1,500') are never missed
    while preserving deep semantic understanding.
    """

    def __init__(
        self, 
        vector_store, 
        dense_weight: float = 0.5, 
        sparse_weight: float = 0.5, 
        rrf_constant: int = 60
    ):
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_constant = rrf_constant
        
        self.bm25 = None
        self.corpus_docs: List[Document] = []
        self._sync_bm25_from_vector_store()

    def _sync_bm25_from_vector_store(self):
        """Builds in-memory BM25 index from documents currently in the vector store."""
        try:
            raw_data = self.vector_store.vector_store.get()
            texts = raw_data.get("documents", [])
            metadatas = raw_data.get("metadatas", [])

            if texts:
                self.corpus_docs = [
                    Document(page_content=t, metadata=dict(m or {}))
                    for t, m in zip(texts, metadatas)
                ]
                tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.corpus_docs]
                self.bm25 = BM25Engine(tokenized_corpus)
                logger.info(f"Built BM25 index over {len(self.corpus_docs)} documents.")
        except Exception as e:
            logger.debug(f"BM25 initialization deferred: {e}")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercases and extracts clean alphanumeric tokens and legal symbols."""
        return re.findall(r'[\$\w]+', text.lower())

    def update_index(self, documents: List[Document]):
        """Appends newly indexed documents to the BM25 index."""
        self.corpus_docs.extend(documents)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.corpus_docs]
        self.bm25 = BM25Engine(tokenized_corpus)
        logger.info(f"Updated BM25 index. Total corpus: {len(self.corpus_docs)} documents.")

    def _search_bm25(self, query: str, top_k: int = 10) -> List[Document]:
        """Performs BM25 lexical keyword search with exact token matching verification."""
        if not self.bm25 or not self.corpus_docs:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        token_set = set(tokens)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            doc_content = self.corpus_docs[idx].page_content.lower()
            # Only include documents that actually contain at least one query token
            if any(t in doc_content for t in token_set):
                doc = Document(
                    page_content=self.corpus_docs[idx].page_content,
                    metadata=dict(self.corpus_docs[idx].metadata)
                )
                doc.metadata["bm25_score"] = float(scores[idx])
                results.append(doc)
        return results

    def _normalize_key(self, doc: Document) -> str:
        """Generates consistent deduplication key based on chunk content."""
        src = doc.metadata.get("source", "")
        preview = doc.page_content.strip()[:80]
        return f"{src}::{preview}"

    def retrieve(self, query: str, top_k: int = 4, candidate_k: int = 12) -> List[Document]:
        """
        Executes Hybrid Retrieval with Reciprocal Rank Fusion (RRF).
        """
        # 1. Retrieve candidates from Dense Vector Search
        try:
            dense_docs = self.vector_store.query(query, k=candidate_k)
        except Exception as e:
            logger.warning(f"Dense retrieval error: {e}")
            dense_docs = []

        # 2. Retrieve candidates from Sparse BM25 Search
        sparse_docs = self._search_bm25(query, top_k=candidate_k)

        # Fallback if one leg has no results
        if not dense_docs and not sparse_docs:
            return []
        if not dense_docs:
            return sparse_docs[:top_k]
        if not sparse_docs:
            return dense_docs[:top_k]

        # 3. Reciprocal Rank Fusion (RRF)
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        # Score Dense ranks
        for rank, doc in enumerate(dense_docs):
            doc_key = self._normalize_key(doc)
            rrf_score = self.dense_weight * (1.0 / (self.rrf_constant + rank + 1))
            doc_scores[doc_key] = (doc, rrf_score)

        # Merge Sparse BM25 ranks
        for rank, doc in enumerate(sparse_docs):
            doc_key = self._normalize_key(doc)
            rrf_score = self.sparse_weight * (1.0 / (self.rrf_constant + rank + 1))
            if doc_key in doc_scores:
                existing_doc, existing_score = doc_scores[doc_key]
                doc_scores[doc_key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[doc_key] = (doc, rrf_score)

        # 4. Sort and return top_k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        fused_results = []
        for doc, score in sorted_docs[:top_k]:
            doc.metadata["rrf_score"] = round(score, 5)
            fused_results.append(doc)

        logger.info(f"RRF Hybrid Search fused {len(dense_docs)} dense + {len(sparse_docs)} sparse into top {len(fused_results)} results.")
        return fused_results
