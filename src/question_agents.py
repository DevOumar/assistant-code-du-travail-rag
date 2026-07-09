"""Dedicated question-handling agents for the RAG assistant.

The project keeps the architecture explicit for the professor's workflow:
- a question formatter cleans and routes incoming questions;
- a reference retriever delegates vector search to the retrieval layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from config import AppConfig, load_config
from moderator import is_small_talk
from question_processing import QuestionPreparation, normalize_question, prepare_question, split_atomic_questions

if TYPE_CHECKING:
    from rag import RetrievedChunk


@dataclass(frozen=True)
class QuestionRoutingResult:
    original_question: str
    cleaned_question: str
    atomic_questions: list[str]
    removed_fragments: tuple[str, ...]
    is_small_talk: bool = False

    @property
    def should_skip_retrieval(self) -> bool:
        return self.is_small_talk or not self.cleaned_question


@dataclass(frozen=True)
class QuestionFormatter:
    """Clean and route user questions before retrieval."""

    max_atomic_questions: int = 4
    enable_reformulation: bool = True

    def format(self, question: str) -> QuestionRoutingResult:
        if self.enable_reformulation:
            preparation: QuestionPreparation = prepare_question(
                question,
                max_atomic_questions=self.max_atomic_questions,
            )
            original_question = preparation.original_question
            normalized_question = preparation.cleaned_question
            atomic_questions = preparation.atomic_questions
            removed_fragments = preparation.removed_fragments
        else:
            original_question = normalize_question(question)
            if not original_question:
                return QuestionRoutingResult(
                    original_question="",
                    cleaned_question="",
                    atomic_questions=[],
                    removed_fragments=(),
                    is_small_talk=False,
                )

            normalized_question = original_question
            atomic_questions = split_atomic_questions(
                original_question,
                max_atomic_questions=self.max_atomic_questions,
            )
            removed_fragments = ()

        return QuestionRoutingResult(
            original_question=original_question,
            cleaned_question=normalized_question,
            atomic_questions=atomic_questions,
            removed_fragments=removed_fragments,
            is_small_talk=is_small_talk(original_question),
        )


@dataclass(frozen=True)
class ReferenceRetriever:
    """Reference retrieval agent backed by the vector store."""

    config: AppConfig | None = None
    decompose: bool = True
    enable_reformulation: bool = True
    enable_hyde: bool | None = None
    enable_hybrid_search: bool | None = None

    def retrieve(self, question: str, top_k: int | None = None) -> list[Any]:
        from retrieval import VectorStoreRetriever

        active_config = self.config or load_config()
        return VectorStoreRetriever(
            config=active_config,
            decompose=self.decompose,
            enable_reformulation=self.enable_reformulation,
            enable_hyde=self.enable_hyde,
            enable_hybrid_search=self.enable_hybrid_search,
        ).retrieve(
            question, top_k=top_k
        )

    def retrieve_references(self, question: str, top_k: int | None = None) -> list[Any]:
        return self.retrieve(question, top_k=top_k)
