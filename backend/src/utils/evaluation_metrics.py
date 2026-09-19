"""
Evaluation Metrics & Benchmarking Suite for LexiRAG.
Implements RAGSystemEvaluator for empirical assessment of legal answer accuracy,
groundedness, and retrieval context relevance.
"""

import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class RAGSystemEvaluator:
    """
    Evaluator to benchmark RAG responses against ground-truth contract answers.
    Supports BLEU score calculation, lexical recall, and discrepancy detection.
    """

    def __init__(self, rag_instance):
        """
        Initializes the evaluator with either a RAGSystem instance or a callable QA chain.
        """
        self.rag_instance = rag_instance

    def _query(self, question: str) -> str:
        """Helper to invoke RAG instance polymorphically."""
        if hasattr(self.rag_instance, "answer_query"):
            res = self.rag_instance.answer_query(question)
            return res.get("answer", "")
        elif callable(self.rag_instance):
            res = self.rag_instance({"question": question, "chat_history": []})
            if isinstance(res, dict):
                return res.get("answer", "")
            return str(res)
        raise ValueError("Unsupported RAG instance provided to evaluator.")

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Computes sentence-level BLEU score with smoothing."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            ref_tokens = [reference.lower().split()]
            cand_tokens = candidate.lower().split()
            smooth = SmoothingFunction().method1
            return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth)
        except Exception:
            # Fallback token Jaccard similarity if nltk is not downloaded
            ref_set = set(reference.lower().split())
            cand_set = set(candidate.lower().split())
            if not ref_set or not cand_set:
                return 0.0
            return len(ref_set & cand_set) / len(ref_set | cand_set)

    def evaluate_test_set(
        self, 
        test_pairs: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluates a list of (Question, Ground_Truth_Answer) pairs.
        
        Args:
            test_pairs: List of tuples containing (question, expected_answer).
            
        Returns:
            Dict containing individual results, average scores, and discrepancy flags.
        """
        results = []
        bleu_scores = []

        for q, expected in test_pairs:
            logger.info(f"Evaluating Question: {q}")
            actual = self._query(q)
            score = self.calculate_bleu(expected, actual)
            bleu_scores.append(score)

            results.append({
                "question": q,
                "expected": expected,
                "actual": actual,
                "bleu_score": round(score, 4),
                "grounded": "not explicitly mentioned" not in actual.lower()
            })

        avg_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        return {
            "total_questions": len(test_pairs),
            "average_bleu_score": round(avg_score, 4),
            "results": results
        }

    def print_report(self, evaluation_summary: Dict[str, Any]):
        """Prints a human-readable benchmark report."""
        print("\n" + "=" * 60)
        print("          LEXIRAG BENCHMARK EVALUATION REPORT          ")
        print("=" * 60)
        print(f"Total Test Cases Evaluated : {evaluation_summary['total_questions']}")
        print(f"Average BLEU Similarity    : {evaluation_summary['average_bleu_score']:.4f}\n")
        
        for idx, item in enumerate(evaluation_summary["results"], 1):
            print(f"[{idx}] Question: {item['question']}")
            print(f"    Expected : {item['expected']}")
            print(f"    Actual   : {item['actual']}")
            print(f"    BLEU     : {item['bleu_score']}")
            print("-" * 60)
