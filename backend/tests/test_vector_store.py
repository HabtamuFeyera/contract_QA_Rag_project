import pytest
from src.models.vector_store import VectorStore


def test_vector_store_chunking():
    store = VectorStore(openai_api_key=None, persist_directory=None)
    sample_text = (
        "Section 1. Agreement Terms.\n"
        "The Advisor shall perform advisory services.\n\n"
        "Section 2. Compensation.\n"
        "The Company will pay $1,500 per month for services rendered."
    )
    chunks = store.chunk_texts([sample_text], metadatas=[{"source": "test_contract.pdf"}])
    assert len(chunks) > 0
    assert chunks[0].metadata["source"] == "test_contract.pdf"
    assert "chunk_id" in chunks[0].metadata


def test_vector_store_indexing_and_query():
    store = VectorStore(openai_api_key=None, persist_directory=None)
    texts = [
        "Payment to the Advisor shall be $1,500 monthly.",
        "Non-compete obligation remains in effect for 12 months after termination."
    ]
    ids = store.add_texts(texts)
    assert len(ids) == 2

    # Query with natural language
    results = store.query("What are the payments to the Advisor?", k=1)
    assert len(results) == 1
    assert len(results[0].page_content) > 0
