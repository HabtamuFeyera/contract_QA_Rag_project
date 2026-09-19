import pytest
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.guardrails import GroundednessGuardrail
from src.models.vector_store import VectorStore
from langchain_core.documents import Document


def test_hybrid_retriever_rrf():
    store = VectorStore(openai_api_key=None, persist_directory=None)
    store.add_texts([
        "Section 6. Payments to Advisor shall be $1,500 monthly.",
        "Section 9. Governing Law of the State of Delaware."
    ])
    hybrid = HybridRetriever(store)
    results = hybrid.retrieve("payments $1,500", top_k=1)
    assert len(results) == 1
    assert "payments" in results[0].page_content.lower()
    assert "rrf_score" in results[0].metadata


def test_guardrails_financial_figures():
    context = "Fees of $1,500 per month and $100 workspace allowance."
    
    # Safe answer
    safe_ans = "The fees are $1,500 per month."
    res = GroundednessGuardrail.verify_financial_figures(safe_ans, context)
    assert res["figures_safe"] is True

    # Hallucinated answer with figure not in context
    hallucinated_ans = "The advisor is paid $50,000 upfront."
    res_bad = GroundednessGuardrail.verify_financial_figures(hallucinated_ans, context)
    assert res_bad["figures_safe"] is False
    assert "$50,000" in res_bad["unverified_figures"]


def test_guardrails_faithfulness_score():
    docs = [Document(page_content="The agreement is effective for 12 months.")]
    answer = "The agreement is effective for 12 months."
    score_info = GroundednessGuardrail.compute_faithfulness_score(answer, docs)
    assert score_info["faithfulness_score"] >= 0.7
    assert score_info["status"] == "VERIFIED"
