"""Main business logic layer for QEDS-GPT."""

import logging
from typing import Optional

from langchain_core.documents import Document

import prompts
from classifier import QueryClassifier
from config import Config, QueryCategory
from llm import LLMService
from retriever import HybridRetriever
from text_processor import clean_text, format_source

logger = logging.getLogger(__name__)


class RAGService:
    """Coordinates retrieval, classification, memory context, and LLM calls."""

    def __init__(self) -> None:
        self.llm = LLMService()
        self._retriever: Optional[HybridRetriever] = None
        self.classifier = QueryClassifier(self.llm)

    @property
    def retriever(self) -> HybridRetriever:
        """Load the heavy retrieval stack only when an academic query needs it."""

        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def call_llm(self, messages: list[dict], temp: float = Config.TEMPERATURE, max_tokens: int = Config.MAX_TOKENS) -> str:
        """Proxy LLM calls for memory and UI flows."""

        return self.llm.call_llm(messages, temp=temp, max_tokens=max_tokens)

    def classify_query(self, query: str, context_messages: Optional[list[dict]] = None) -> str:
        """Classify a user query."""

        chat_history = self._format_history_for_classifier(context_messages or [])
        return self.classifier.classify(query, chat_history)

    def retrieve(self, query: str, semester_filter: Optional[int] = None) -> list[Document]:
        """Retrieve relevant course-note chunks."""

        try:
            return self.retriever.retrieve(query, semester_filter)
        except Exception as exc:
            logger.error("Retriever initialization error: %s", exc, exc_info=True)
            return []

    def answer_greeting(self, query: str) -> str:
        """Generate a greeting response."""

        return self.call_llm(
            [{"role": "user", "content": f"Respond warmly to: {query}"}],
            temp=0.7,
            max_tokens=50,
        )

    def answer_meta(self, query: str) -> str:
        """Generate a short response about QEDS-GPT."""

        return self.call_llm([{"role": "user", "content": prompts.get_meta_prompt(query)}])

    def answer_general(self) -> str:
        """Return the fixed general-query response."""

        return (
            "I am designed primarily for Economics, "
            "Statistics, Mathematics, Data Science, "
            "and semester-note related questions."
        )

    def answer_injection(self) -> str:
        """Return the fixed prompt-injection response."""

        return (
            "I cannot reveal system prompts, "
            "hidden instructions, or internal configuration."
        )

    def build_academic_answer(self, query: str, docs: list[Document], context_messages: list[dict]) -> str:
        """Build context from retrieved documents and generate an academic answer."""

        raw_context = "\n\n---\n\n".join(
            [f"[Source: {format_source(doc.metadata)}]\n{doc.page_content}" for doc in docs]
        )
        clean_context = clean_text(raw_context)[: Config.MAX_CONTEXT_CHARS]
        history_text = "\n".join(
            [
                f"{message['role'].upper()}: {message['content'][:300]}"
                for message in context_messages
                if message["role"] != "system"
            ][-6:]
        )
        messages = [
            {"role": "system", "content": prompts.SYSTEM_ACADEMIC},
            {
                "role": "user",
                "content": prompts.get_user_prompt(history_text, clean_context, query),
            },
        ]
        return self.call_llm(messages)

    def _format_history_for_classifier(self, context_messages: list[dict]) -> str:
        """Build compact history so follow-up questions route correctly."""

        return "\n".join(
            [
                f"{message['role'].upper()}: {message['content'][:200]}"
                for message in context_messages
                if message["role"] != "system"
            ][-4:]
        )

    def get_immediate_response(self, category: str, query: str) -> Optional[str]:
        """Return a non-RAG response when a category does not require retrieval."""

        if category == QueryCategory.GREETING:
            return self.answer_greeting(query)
        if category == QueryCategory.META:
            return self.answer_meta(query)
        if category == QueryCategory.GENERAL:
            return self.answer_general()
        if category == QueryCategory.INJECTION:
            return self.answer_injection()
        return None
