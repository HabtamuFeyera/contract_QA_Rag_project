"""
Helper Utilities for LexiRAG.
Provides text sanitization, citation formatting, and metadata helpers.
"""

import re
from typing import List, Dict, Any
from langchain_core.documents import Document


def clean_legal_text(text: str) -> str:
    """Removes excessive whitespace, header/footer artifacts, and form-feed characters."""
    if not text:
        return ""
    # Replace multiple newlines with at most two
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip leading/trailing whitespaces
    return text.strip()


def extract_citations_from_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Transforms retrieved document chunks into clean citation objects.
    """
    citations = []
    for idx, doc in enumerate(docs):
        metadata = doc.metadata or {}
        source = metadata.get("source", f"Document {idx+1}")
        page = metadata.get("page_number") or metadata.get("page", 1)
        
        # Take an excerpt snippet
        content_preview = doc.page_content.strip()
        if len(content_preview) > 280:
            content_preview = content_preview[:280] + "..."

        citations.append({
            "id": idx + 1,
            "source": source,
            "page": page,
            "snippet": content_preview,
            "chunk_id": metadata.get("chunk_id", idx)
        })
    return citations


def format_context_for_prompt(docs: List[Document]) -> str:
    """Formats retrieved document chunks into an explicit numbered context block for LLM prompting."""
    formatted_chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Contract")
        page = doc.metadata.get("page_number", doc.metadata.get("page", "?"))
        formatted_chunks.append(
            f"--- [EXCERPT {i+1} | Source: {source}, Page: {page}] ---\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)
