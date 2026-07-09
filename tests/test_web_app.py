from config import LEGAL_DISCLAIMER, load_config
from rag import RetrievedChunk
from web_app import (
    UnavailablePipeline,
    _build_pipeline_or_fallback,
    build_corpus_status,
    build_corpus_freshness,
    format_sources_markdown,
    QUESTION_PRESETS,
)


def test_unavailable_pipeline_returns_clear_placeholder_answer() -> None:
    response = UnavailablePipeline().answer("Quelle est la duree legale du travail ?")

    assert response.used_context is False
    assert response.sources == []
    assert "pipeline RAG complet n'est pas disponible" in response.answer
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

    assert "**Sources" in markdown
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


def test_build_corpus_freshness_mentions_age_and_risk() -> None:
    config = load_config(
        env_file=None,
        environ={
            "CORPUS_DATE": "2026-07-08",
        },
    )

    freshness = build_corpus_freshness(config)

    assert "Fraîcheur" in freshness
    assert "Risque" in freshness


def test_build_pipeline_or_fallback_returns_concrete_pipeline(monkeypatch) -> None:
    config = load_config(env_file=None, environ={"GROQ_API_KEY": "test-key"})
    concrete_pipeline = object()

    monkeypatch.setattr("web_app.build_rag_pipeline", lambda config: concrete_pipeline)

    assert _build_pipeline_or_fallback(config) is concrete_pipeline


def test_build_pipeline_or_fallback_keeps_ui_available(monkeypatch) -> None:
    config = load_config(env_file=None, environ={})

    def fail(config):
        raise RuntimeError("not indexed")

    monkeypatch.setattr("web_app.build_rag_pipeline", fail)

    assert isinstance(_build_pipeline_or_fallback(config), UnavailablePipeline)


def test_question_presets_cover_core_topics() -> None:
    assert len(QUESTION_PRESETS) >= 5
    assert any("durée légale du travail" in preset for preset in QUESTION_PRESETS)
    assert any("fusion-acquisition" in preset for preset in QUESTION_PRESETS)
