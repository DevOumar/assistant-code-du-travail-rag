import pytest

from config import LEGAL_DISCLAIMER
from prompting import (
    PromptContextItem,
    build_no_context_answer,
    build_prompt_messages,
    format_context,
    render_rag_system_prompt,
)


def test_rendered_system_prompt_contains_project_constraints() -> None:
    prompt = render_rag_system_prompt(
        context="Article : L3121-27\nTexte : duree legale.",
        corpus_date="2026-07-08",
    )

    assert "uniquement sur le contexte fourni" in prompt
    assert "N'invente jamais" in prompt
    assert "Information not found in the knowledge base." in prompt
    assert "2026-07-08" in prompt
    assert "Article : L3121-27" in prompt
    assert LEGAL_DISCLAIMER in prompt


def test_format_context_includes_article_metadata_and_text() -> None:
    context = format_context(
        [
            PromptContextItem(
                text="La duree legale du travail effectif est fixee a trente-cinq heures.",
                article="L3121-27",
                source="legi",
                section="Duree legale",
                theme="duree_travail",
                score=0.87,
            )
        ]
    )

    assert "[Source 1]" in context
    assert "Article : L3121-27" in context
    assert "Source : legi" in context
    assert "Score : 0.8700" in context
    assert "trente-cinq heures" in context


def test_build_prompt_messages_formats_question_context_and_disclaimer() -> None:
    messages = build_prompt_messages(
        question="Quelle est la duree legale du travail ?",
        corpus_date="2026-07-08",
        context_items=[
            {
                "text": "La duree legale du travail effectif est fixee a trente-cinq heures.",
                "metadata": {"article": "L3121-27", "source": "legi"},
                "score": 0.91,
            }
        ],
    )

    assert messages.user == "Quelle est la duree legale du travail ?"
    assert "2026-07-08" in messages.system
    assert "Article : L3121-27" in messages.system
    assert LEGAL_DISCLAIMER in messages.system


def test_build_no_context_answer_contains_required_fallback_and_disclaimer() -> None:
    answer = build_no_context_answer()

    assert "Information not found in the knowledge base." in answer
    assert LEGAL_DISCLAIMER in answer


def test_prompt_building_rejects_empty_question_or_context_text() -> None:
    with pytest.raises(ValueError, match="question"):
        build_prompt_messages(question=" ", context_items=[])

    with pytest.raises(ValueError, match="context text"):
        format_context([{"text": " ", "metadata": {}}])
