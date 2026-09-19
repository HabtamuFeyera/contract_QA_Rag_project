import pytest
from src.utils.evaluation_metrics import RAGSystemEvaluator
from src.rag.rag_system import RAGSystem


def test_evaluator_bleu_and_scoring():
    rag = RAGSystem(openai_api_key=None, persist_directory=None)
    rag.add_documents([
        "Payments to Advisor: $1,500 monthly workspace and retainer fee."
    ])
    evaluator = RAGSystemEvaluator(rag)
    
    test_set = [
        ("What are the payments?", "$1,500 monthly workspace and retainer fee.")
    ]
    summary = evaluator.evaluate_test_set(test_set)
    assert summary["total_questions"] == 1
    assert "average_bleu_score" in summary
    assert len(summary["results"]) == 1
