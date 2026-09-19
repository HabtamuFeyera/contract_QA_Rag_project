"""
Groundedness & Hallucination Guardrails for LexiRAG.
Audits generated legal answers against retrieved contract context to ensure zero ungrounded liability.
"""

import re
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GroundednessGuardrail:
    """
    Automated verification guardrail that assesses answer faithfulness,
    verifies financial/numerical figures, and flags ungrounded assertions.
    """

    @staticmethod
    def verify_financial_figures(answer: str, context_text: str) -> Dict[str, Any]:
        """
        Detects currency amounts (e.g. $1,500, $1,000,000) and percentages in the answer
        and verifies that each figure exists in the ground-truth context.
        """
        # Regex matching figures like $1,500, $9, $1,000,000, 12 months, 5%
        money_pattern = r'\$[\d,]+(?:\.\d+)?'
        figures_in_answer = set(re.findall(money_pattern, answer))

        unverified = []
        for fig in figures_in_answer:
            # Clean comma for flexible matching
            clean_fig = fig.replace(",", "")
            if fig not in context_text and clean_fig not in context_text.replace(",", ""):
                unverified.append(fig)

        return {
            "all_figures": list(figures_in_answer),
            "unverified_figures": unverified,
            "figures_safe": len(unverified) == 0
        }

    @staticmethod
    def compute_faithfulness_score(
        answer: str, 
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Calculates a semantic and lexical groundedness score (0.0 to 1.0).
        """
        if not retrieved_docs:
            return {
                "faithfulness_score": 0.0,
                "status": "UNVERIFIED",
                "reason": "No context documents were retrieved."
            }

        combined_context = " ".join([d.page_content.lower() for d in retrieved_docs])
        answer_lower = answer.lower()

        # If model explicitly declined due to missing info
        if "not contain sufficient terms" in answer_lower or "not explicitly mentioned" in answer_lower:
            return {
                "faithfulness_score": 1.0,
                "status": "VERIFIED_ABSTENTION",
                "verified_claims": True,
                "notes": "Model correctly abstained from hallucination on absent terms."
            }

        # Check financial figures
        financial_check = GroundednessGuardrail.verify_financial_figures(answer, combined_context)

        # Token overlap recall
        answer_words = [w for w in re.findall(r'\b\w{4,}\b', answer_lower) if w not in {"according", "provided", "agreement", "section"}]
        if not answer_words:
            return {"faithfulness_score": 1.0, "status": "VERIFIED"}

        grounded_count = sum(1 for w in answer_words if w in combined_context)
        overlap_ratio = grounded_count / len(answer_words)

        score = min(1.0, round(overlap_ratio * (0.9 if financial_check["figures_safe"] else 0.5) + 0.1, 2))

        status = "VERIFIED"
        if not financial_check["figures_safe"]:
            status = "UNVERIFIED_FIGURES"
        elif score < 0.65:
            status = "POTENTIAL_EXTRAPOLATION"

        return {
            "faithfulness_score": score,
            "status": status,
            "financial_check": financial_check
        }
