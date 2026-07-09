"""Application wiring for the concrete RAG pipeline."""

from __future__ import annotations

from config import AppConfig, load_config
from llm import GroqAnswerGenerator
from rag import RagPipeline
from retrieval import VectorStoreRetriever


class PipelineBuildError(Exception):
    """Raised when the concrete pipeline cannot be created."""


def build_rag_pipeline(config: AppConfig | None = None) -> RagPipeline:
    """Build the concrete application pipeline from configuration."""

    active_config = config or load_config()
    if active_config.llm.provider.casefold() != "groq":
        raise PipelineBuildError(f"Unsupported LLM provider: {active_config.llm.provider}")

    return RagPipeline(
        retriever=VectorStoreRetriever(config=active_config),
        generator=GroqAnswerGenerator(config=active_config),
        top_k=active_config.retrieval.top_k,
        corpus_date=active_config.corpus.date,
        legal_disclaimer=active_config.legal_disclaimer,
    )
