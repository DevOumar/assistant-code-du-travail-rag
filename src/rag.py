"""RAG orchestration layer.

The pipeline wires retrieval and generation through protocols so the core RAG
logic stays independent from ChromaDB and Groq implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from config import LEGAL_DISCLAIMER
from prompting import PromptMessages, build_no_context_answer, build_prompt_messages, build_small_talk_answer
from question_agents import QuestionFormatter


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass(frozen=True)
class RagResponse:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    scores: list[float | None] = field(default_factory=list)
    top_k: int | None = None
    used_context: bool = False


class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return relevant chunks for a question."""


class AnswerGenerator(Protocol):
    def generate(self, messages: PromptMessages) -> str:
        """Generate a final answer from prepared prompt messages."""


@dataclass
class RagPipeline:
    retriever: Retriever
    generator: AnswerGenerator
    top_k: int = 5
    corpus_date: str | None = None
    legal_disclaimer: str = LEGAL_DISCLAIMER

    def answer(self, question: str) -> RagResponse:
        normalized_question = _require_non_empty(question, "question")
        routing = QuestionFormatter().format(normalized_question)

        if routing.should_skip_retrieval:
            return RagResponse(
                question=normalized_question,
                answer=build_small_talk_answer(self.legal_disclaimer),
                sources=[],
                scores=[],
                top_k=self.top_k,
                used_context=False,
            )

        retrieved_chunks = self.retriever.retrieve(routing.cleaned_question, top_k=self.top_k)

        if not retrieved_chunks:
            return RagResponse(
                question=normalized_question,
                answer=build_no_context_answer(self.legal_disclaimer),
                sources=[],
                scores=[],
                top_k=self.top_k,
                used_context=False,
            )

        messages = build_prompt_messages(
            question=normalized_question,
            context_items=[_to_prompt_context(chunk) for chunk in retrieved_chunks],
            corpus_date=self.corpus_date,
            legal_disclaimer=self.legal_disclaimer,
        )
        generated_answer = self.generator.generate(messages)
        answer = ensure_legal_disclaimer(generated_answer, self.legal_disclaimer)

        return RagResponse(
            question=normalized_question,
            answer=answer,
            sources=retrieved_chunks,
            scores=[chunk.score for chunk in retrieved_chunks],
            top_k=self.top_k,
            used_context=True,
        )


def ensure_legal_disclaimer(answer: str, legal_disclaimer: str = LEGAL_DISCLAIMER) -> str:
    """Guarantee that the legal disclaimer is present in the final answer."""

    normalized_answer = _require_non_empty(answer, "answer")
    if legal_disclaimer in normalized_answer:
        return normalized_answer

    return f"{normalized_answer.rstrip()}\n\n{legal_disclaimer}"


def _to_prompt_context(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "metadata": chunk.metadata,
        "score": chunk.score,
    }


def _require_non_empty(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")

    return normalized
