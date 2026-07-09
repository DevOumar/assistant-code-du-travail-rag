from pathlib import Path
from unittest.mock import patch

import pytest

from config import load_config
from retrieval import (
    RetrievalError,
    RetrievedChunk,
    VectorStoreRetriever,
    decompose_question,
    format_retrieved_chunks,
    retrieve,
    retrieve_decomposed,
    retrieve_as_context,
)
from vector_store import VectorStoreError


def _config(tmp_path: Path):
    return load_config(
        env_file=None,
        environ={
            "CHROMA_DB_DIR": str(tmp_path),
            "RETRIEVAL_TOP_K": "3",
        },
    )


def _raw_result() -> list[dict]:
    return [
        {
            "id": "article-L3121-27::chunk-001",
            "text": "La durée légale du travail effectif est fixée à trente-cinq heures.",
            "metadata": {
                "article": "L3121-27",
                "section": "Durée légale",
                "source": "Légifrance",
            },
            "distance": 0.13,
            "similarity": 0.87,
        }
    ]


def test_retrieve_maps_vector_results_to_retrieved_chunks(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with patch("retrieval.query_vector_database", return_value=_raw_result()) as query:
        chunks = retrieve("Quelle est la durée légale du travail ?", config=config)

    query.assert_called_once_with(
        question="Quelle est la durée légale du travail ?",
        top_k=5,
        config=config,
    )
    assert chunks == [
        RetrievedChunk(
            chunk_id="article-L3121-27::chunk-001",
            text="La durée légale du travail effectif est fixée à trente-cinq heures.",
            metadata={
                "article": "L3121-27",
                "section": "Durée légale",
                "source": "Légifrance",
            },
            distance=0.13,
            similarity=0.87,
        )
    ]


def test_retrieve_rejects_empty_question(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="Question"):
        retrieve("   ", config=_config(tmp_path))


def test_retrieve_wraps_vector_store_errors(tmp_path: Path) -> None:
    with patch("retrieval.query_vector_database", side_effect=VectorStoreError("missing")):
        with pytest.raises(RetrievalError, match="missing"):
            retrieve("Question droit du travail", config=_config(tmp_path))


def test_vector_store_retriever_matches_rag_pipeline_contract(tmp_path: Path) -> None:
    with patch("retrieval.query_vector_database", return_value=_raw_result()):
        chunks = VectorStoreRetriever(config=_config(tmp_path)).retrieve("durée travail", top_k=1)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("La durée légale")
    assert chunks[0].score == 0.87
    assert chunks[0].metadata["article"] == "L3121-27"
    assert chunks[0].metadata["chunk_id"] == "article-L3121-27::chunk-001"


def test_decompose_question_splits_compound_labor_law_question() -> None:
    sub_questions = decompose_question(
        "Quelle est la duree legale du travail et quelles sont les regles des conges payes ?"
    )

    assert sub_questions == [
        "Quelle est la duree legale du travail",
        "les regles des conges payes",
    ]


def test_decompose_question_keeps_simple_question_unchanged() -> None:
    assert decompose_question("Quelle est la duree du preavis ?") == [
        "Quelle est la duree du preavis ?"
    ]


def test_retrieve_decomposed_queries_each_sub_question_and_deduplicates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = [
        {
            "id": "article-L3121-27::chunk-001",
            "text": "Duree legale.",
            "metadata": {"article": "L3121-27"},
            "distance": 0.2,
            "similarity": 0.8,
        }
    ]
    second = [
        {
            "id": "article-L3121-27::chunk-001",
            "text": "Duree legale.",
            "metadata": {"article": "L3121-27"},
            "distance": 0.1,
            "similarity": 0.9,
        },
        {
            "id": "article-L3141-1::chunk-001",
            "text": "Conges payes.",
            "metadata": {"article": "L3141-1"},
            "distance": 0.3,
            "similarity": 0.7,
        },
    ]

    with patch("retrieval.query_vector_database", side_effect=[first, second]) as query:
        chunks = retrieve_decomposed(
            "Quelle est la duree legale du travail et quelles sont les regles des conges payes ?",
            top_k=3,
            config=config,
        )

    assert query.call_count == 2
    assert [chunk.chunk_id for chunk in chunks] == [
        "article-L3121-27::chunk-001",
        "article-L3141-1::chunk-001",
    ]
    assert chunks[0].similarity == 0.9


def test_format_retrieved_chunks_displays_article_source_and_score() -> None:
    rendered = format_retrieved_chunks(
        [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Texte source",
                metadata={"article": "L3121-27", "section": "Durée", "source": "Légifrance"},
                distance=0.13,
                similarity=0.87,
            )
        ]
    )

    assert "L3121-27" in rendered
    assert "Légifrance" in rendered
    assert "0.8700" in rendered


def test_retrieve_as_context_formats_prompt_ready_context(tmp_path: Path) -> None:
    with patch("retrieval.query_vector_database", return_value=_raw_result()):
        context = retrieve_as_context("durée travail", config=_config(tmp_path))

    assert "Document 1" in context
    assert "Article : L3121-27" in context
    assert "Texte : La durée légale" in context
