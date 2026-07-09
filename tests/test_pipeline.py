import pytest

from config import load_config
from llm import GroqAnswerGenerator
from pipeline import PipelineBuildError, build_rag_pipeline
from rag import RagPipeline
from retrieval import VectorStoreRetriever


def test_build_rag_pipeline_wires_concrete_components() -> None:
    config = load_config(
        env_file=None,
        environ={
            "GROQ_API_KEY": "test-key",
            "RETRIEVAL_TOP_K": "7",
            "CORPUS_DATE": "2026-07-08",
        },
    )

    pipeline = build_rag_pipeline(config)

    assert isinstance(pipeline, RagPipeline)
    assert isinstance(pipeline.retriever, VectorStoreRetriever)
    assert isinstance(pipeline.generator, GroqAnswerGenerator)
    assert pipeline.top_k == 7
    assert pipeline.corpus_date == "2026-07-08"


def test_build_rag_pipeline_rejects_unsupported_llm_provider() -> None:
    config = load_config(
        env_file=None,
        environ={
            "LLM_PROVIDER": "openai",
            "GROQ_API_KEY": "test-key",
        },
    )

    with pytest.raises(PipelineBuildError, match="Unsupported LLM provider"):
        build_rag_pipeline(config)
