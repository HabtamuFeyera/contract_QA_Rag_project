import pytest
from src.rag.rag_system import RAGSystem


def test_rag_system_query_empty():
    rag = RAGSystem(openai_api_key=None, persist_directory=None)
    response = rag.answer_query("")
    assert response["status"] == "empty_query"


def test_rag_system_flow():
    rag = RAGSystem(openai_api_key=None, persist_directory=None)
    rag.add_documents([
        "Section 6. Payments. Fees of $9 per hour up to a monthly limit of $1,500. Workspace expense of $100 per month."
    ])
    response = rag.answer_query("What are the payments to the Advisor?")
    assert "answer" in response
    assert response["status"] == "success"
    assert len(response["citations"]) > 0
