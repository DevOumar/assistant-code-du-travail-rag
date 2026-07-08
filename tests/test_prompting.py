import pytest

from config import LEGAL_DISCLAIMER
from prompting import (
    SYSTEM_PROMPT,
    PromptContextItem,
    build_no_context_answer,
    build_prompt_messages,
    format_context,
)


def test_system_prompt_contains_project_constraints() -> None:
    assert "uniquement a partir du contexte fourni" in SYSTEM_PROMPT
    assert "N'invente jamais" in SYSTEM_PROMPT
    assert "Je ne trouve pas cette information dans ma base." in SYSTEM_PROMPT
    assert LEGAL_DISCLAIMER in SYSTEM_PROMPT


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

    assert messages.system == SYSTEM_PROMPT
    assert "Question utilisateur" in messages.user
    assert "2026-07-08" in messages.user
    assert "Article : L3121-27" in messages.user
    assert LEGAL_DISCLAIMER in messages.user


def test_build_no_context_answer_contains_required_fallback_and_disclaimer() -> None:
    answer = build_no_context_answer()

    assert "Je ne trouve pas cette information dans ma base." in answer
    assert LEGAL_DISCLAIMER in answer


def test_prompt_building_rejects_empty_question_or_context_text() -> None:
    with pytest.raises(ValueError, match="question"):
        build_prompt_messages(question=" ", context_items=[])

    with pytest.raises(ValueError, match="context text"):
        format_context([{"text": " ", "metadata": {}}])
