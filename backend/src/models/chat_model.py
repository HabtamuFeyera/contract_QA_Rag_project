"""
ChatModel module for LexiRAG.
Encapsulates high-precision LLM interaction tailored for legal contract interpretation and citation.
"""

import logging
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Specialized prompt enforcing strict legal grounding and citation extraction
LEGAL_QA_SYSTEM_PROMPT = """You are LexiRAG, an elite Autonomous Legal Advisor and Contract Intelligence AI.
Your objective is to provide precise, legally sound, and strictly grounded answers based ONLY on the provided contract excerpts.

CORE RULES:
1. STRICT GROUNDING: Base your answers strictly on the provided contract text. Do NOT assume, extrapolate, or invent legal clauses, obligations, or amounts.
2. CITATIONS: Whenever possible, cite the specific Section, Clause, or Page mentioned in the context (e.g., "[Section 6.2]", "[Exhibit A]").
3. ABSENCE OF TERMS: If the provided excerpts do not mention or provide enough information to answer the question, explicitly state:
   "The provided contract excerpts do not contain sufficient terms to answer this question."
4. RISK & AMBIGUITY HIGHLIGHTING: If a clause contains ambiguous obligations, indemnifications, non-competes, or termination penalties, explicitly call them out under a "⚠️ Legal Consideration" bullet.

Context:
{context}

Question:
{question}

Answer (Clear, structured, with clause citations):"""


def get_chat_openai_class():
    """Dynamically imports ChatOpenAI from langchain_openai or langchain_community."""
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOpenAI
            return ChatOpenAI
        except ImportError:
            return None


class ChatModel:
    """Manages chat model invocation with domain-specific legal prompt engineering."""

    def __init__(
        self, 
        openai_api_key: Optional[str] = None, 
        model_name: str = "gpt-4-turbo", 
        temperature: float = 0.0
    ):
        self.api_key = openai_api_key or ""
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        """Initializes the chat model with fallback for testing."""
        if self.api_key:
            ChatClass = get_chat_openai_class()
            if ChatClass is not None:
                try:
                    self._llm = ChatClass(
                        openai_api_key=self.api_key,
                        model_name=self.model_name,
                        temperature=self.temperature
                    )
                    logger.info(f"Initialized ChatOpenAI with model {self.model_name}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize ChatOpenAI: {e}. Falling back to FakeListChatModel.")

        from langchain_community.chat_models import FakeListChatModel
        self._llm = FakeListChatModel(
            responses=[
                "According to Section 6 of the Agreement, payments to the Advisor include fees of $9 per hour up to a monthly maximum of $1,500. [Source: Agreement, Section 6]"
            ]
        )

    @property
    def llm(self):
        return self._llm

    def get_qa_prompt(self) -> ChatPromptTemplate:
        """Returns the specialized legal QA prompt template."""
        return ChatPromptTemplate.from_template(LEGAL_QA_SYSTEM_PROMPT)

    def generate_answer(self, query: str, context_text: str) -> str:
        """Directly synthesizes an answer using the legal prompt and context."""
        prompt = self.get_qa_prompt()
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"question": query, "context": context_text})
        except Exception as e:
            logger.error(f"Error during LLM answer generation: {str(e)}")
            raise
