"""Application wiring for the concrete RAG pipeline."""

from __future__ import annotations

from config import AppConfig, load_config
from llm import GroqAnswerGenerator
from question_agents import QuestionFormatter, ReferenceRetriever
from rag import RagPipeline


class PipelineBuildError(Exception):
    """Raised when the concrete pipeline cannot be created."""


def build_rag_pipeline(
    config: AppConfig | None = None,
    *,
    decompose: bool | None = None,
    enable_hyde: bool | None = None,
    enable_hybrid_search: bool | None = None,
    enable_reformulation: bool | None = None,
) -> RagPipeline:
    """Build the concrete application pipeline from configuration."""

    active_config = config or load_config()
    if active_config.llm.provider.casefold() != "groq":
        raise PipelineBuildError(f"Unsupported LLM provider: {active_config.llm.provider}")

    active_decompose = True if decompose is None else decompose
    active_hyde = active_config.retrieval.enable_hyde if enable_hyde is None else enable_hyde
    active_hybrid = (
        active_config.retrieval.enable_hybrid_search
        if enable_hybrid_search is None
        else enable_hybrid_search
    )
    active_reformulation = True if enable_reformulation is None else enable_reformulation

    return RagPipeline(
        retriever=ReferenceRetriever(
            config=active_config,
            decompose=active_decompose,
            enable_reformulation=active_reformulation,
            enable_hyde=active_hyde,
            enable_hybrid_search=active_hybrid,
        ),
        generator=GroqAnswerGenerator(config=active_config),
        question_formatter=QuestionFormatter(enable_reformulation=active_reformulation),
        top_k=active_config.retrieval.top_k,
        corpus_date=active_config.corpus.date,
        legal_disclaimer=active_config.legal_disclaimer,
    )
