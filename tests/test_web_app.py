from config import LEGAL_DISCLAIMER, load_config
from rag import RetrievedChunk
from web_app import UnavailablePipeline, build_corpus_status, format_sources_markdown


def test_unavailable_pipeline_returns_clear_placeholder_answer() -> None:
    response = UnavailablePipeline().answer("Quelle est la durée légale du travail ?")

    assert response.used_context is False
    assert response.sources == []
    assert "pipeline RAG complet n'est pas encore connecté" in response.answer
    assert LEGAL_DISCLAIMER in response.answer


def test_format_sources_markdown_renders_articles_and_scores() -> None:
    markdown = format_sources_markdown(
        [
            RetrievedChunk(
                text="Texte source",
                metadata={"article": "L3121-27", "source": "Code du travail"},
                score=0.91234,
            )
        ]
    )

    assert "**Sources utilisées**" in markdown
    assert "`L3121-27`" in markdown
    assert "Code du travail" in markdown
    assert "0.9123" in markdown


def test_format_sources_markdown_returns_empty_string_without_sources() -> None:
    assert format_sources_markdown([]) == ""


def test_build_corpus_status_mentions_source_and_date() -> None:
    config = load_config(
        env_file=None,
        environ={
            "CORPUS_SOURCE": "Code du travail export Legifrance",
            "CORPUS_DATE": "2026-07-08",
        },
    )

    status = build_corpus_status(config)

    assert "Code du travail export Legifrance" in status
    assert "2026-07-08" in status
