from dataclasses import dataclass

import pytest

from config import LEGAL_DISCLAIMER
from prompting import PromptMessages
from rag import RagPipeline, RagResponse, RetrievedChunk, ensure_legal_disclaimer


@dataclass
class FakeRetriever:
    chunks: list[RetrievedChunk]
    last_question: str | None = None
    last_top_k: int | None = None

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.last_question = question
        self.last_top_k = top_k
        return self.chunks


@dataclass
class FakeGenerator:
    answer: str
    last_messages: PromptMessages | None = None

    def generate(self, messages: PromptMessages) -> str:
        self.last_messages = messages
        return self.answer


def test_rag_pipeline_returns_fallback_when_no_context_is_found() -> None:
    retriever = FakeRetriever(chunks=[])
    generator = FakeGenerator(answer="Cette reponse ne doit pas etre utilisee.")
    pipeline = RagPipeline(retriever=retriever, generator=generator, top_k=3)

    response = pipeline.answer("Que dit le Code du travail sur ce sujet ?")

    assert response == RagResponse(
        question="Que dit le Code du travail sur ce sujet ?",
        answer=(
            "Je suis un assistant spécialisé dans le droit du travail français, "
            "mais je ne trouve pas d'information pertinente dans la base de connaissances pour cette question.\n\n"
            f"{LEGAL_DISCLAIMER}"
        ),
        sources=[],
        scores=[],
        top_k=3,
        used_context=False,
    )
    assert retriever.last_top_k == 3
    assert generator.last_messages is None


def test_rag_pipeline_builds_prompt_and_returns_sources() -> None:
    chunk = RetrievedChunk(
        text="La duree legale du travail effectif est fixee a trente-cinq heures.",
        metadata={"article": "L3121-27", "source": "legi"},
        score=0.92,
    )
    retriever = FakeRetriever(chunks=[chunk])
    generator = FakeGenerator(answer="La duree legale est de trente-cinq heures [L3121-27].")
    pipeline = RagPipeline(
        retriever=retriever,
        generator=generator,
        top_k=5,
        corpus_date="2026-07-08",
    )

    response = pipeline.answer("Quelle est la duree legale du travail ?")

    assert response.used_context is True
    assert response.sources == [chunk]
    assert "trente-cinq heures" in response.answer
    assert LEGAL_DISCLAIMER in response.answer
    assert generator.last_messages is not None
    assert "Article : L3121-27" in generator.last_messages.system
    assert "2026-07-08" in generator.last_messages.system


def test_ensure_legal_disclaimer_does_not_duplicate_existing_disclaimer() -> None:
    answer = f"Reponse sourcee.\n\n{LEGAL_DISCLAIMER}"

    assert ensure_legal_disclaimer(answer) == answer


def test_rag_pipeline_rejects_empty_question() -> None:
    pipeline = RagPipeline(retriever=FakeRetriever(chunks=[]), generator=FakeGenerator(answer="x"))

    with pytest.raises(ValueError, match="question"):
        pipeline.answer(" ")
