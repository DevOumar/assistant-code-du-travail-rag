"""Prompt construction for legal RAG answers.

This module prepares messages for a future LLM call. It does not call Groq or
any other model provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from config import LEGAL_DISCLAIMER, PROJECT_ROOT


RAG_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "rag_prompt_system.txt"


@dataclass(frozen=True)
class PromptContextItem:
    text: str
    article: str | None = None
    source: str | None = None
    section: str | None = None
    theme: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class PromptMessages:
    system: str
    user: str


def build_prompt_messages(
    question: str,
    context_items: Iterable[PromptContextItem | Mapping[str, Any]],
    corpus_date: str | None = None,
    legal_disclaimer: str = LEGAL_DISCLAIMER,
    template_path: str | Path = RAG_PROMPT_TEMPLATE_PATH,
) -> PromptMessages:
    """Build system and user messages for the future generation step."""

    normalized_question = _require_non_empty(question, "question")
    normalized_items = [_normalize_context_item(item) for item in context_items]
    system_prompt = render_rag_system_prompt(
        context=format_context(normalized_items),
        corpus_date=corpus_date,
        legal_disclaimer=legal_disclaimer,
        template_path=template_path,
    )

    return PromptMessages(
        system=system_prompt,
        user=normalized_question,
    )


def format_context(context_items: Iterable[PromptContextItem | Mapping[str, Any]]) -> str:
    """Format retrieved chunks as numbered context blocks."""

    normalized_items = [_normalize_context_item(item) for item in context_items]
    if not normalized_items:
        return "Aucun contexte fourni."

    blocks = []
    for index, item in enumerate(normalized_items, start=1):
        metadata = _format_metadata(item)
        blocks.append(f"[Source {index}]\n{metadata}\nTexte : {item.text}")

    return "\n\n".join(blocks)


def build_no_context_answer(legal_disclaimer: str = LEGAL_DISCLAIMER) -> str:
    """Return the required fallback answer when no retrieved context is available."""

    return (
        "Information not found in the knowledge base.\n\n"
        f"{legal_disclaimer}"
    )


def render_rag_system_prompt(
    context: str,
    corpus_date: str | None = None,
    legal_disclaimer: str = LEGAL_DISCLAIMER,
    template_path: str | Path = RAG_PROMPT_TEMPLATE_PATH,
) -> str:
    """Render the file-based RAG system prompt."""

    template = read_prompt_template(template_path)
    return (
        template.replace("{{CONTEXT}}", context)
        .replace("{{CORPUS_DATE}}", corpus_date or "non renseignee")
        .replace("{{LEGAL_DISCLAIMER}}", legal_disclaimer)
    )


def read_prompt_template(template_path: str | Path) -> str:
    """Read a prompt template from disk."""

    return Path(template_path).read_text(encoding="utf-8")


def _normalize_context_item(item: PromptContextItem | Mapping[str, Any]) -> PromptContextItem:
    if isinstance(item, PromptContextItem):
        text = _require_non_empty(item.text, "context text")
        return PromptContextItem(
            text=text,
            article=item.article,
            source=item.source,
            section=item.section,
            theme=item.theme,
            score=item.score,
        )

    text = _require_non_empty(str(item.get("text", "")), "context text")
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("context metadata must be a mapping.")

    return PromptContextItem(
        text=text,
        article=_optional_string(metadata, "article"),
        source=_optional_string(metadata, "source"),
        section=_optional_string(metadata, "section"),
        theme=_optional_string(metadata, "theme"),
        score=_optional_score(item, metadata),
    )


def _format_metadata(item: PromptContextItem) -> str:
    fields = [
        ("Article", item.article),
        ("Source", item.source),
        ("Section", item.section),
        ("Theme", item.theme),
    ]
    if item.score is not None:
        fields.append(("Score", f"{item.score:.4f}"))

    rendered = [f"{label} : {value}" for label, value in fields if value]
    if not rendered:
        return "Metadonnees : non renseignees"

    return "\n".join(rendered)


def _optional_string(source: Mapping[str, Any], key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    return normalized


def _optional_score(item: Mapping[str, Any], metadata: Mapping[str, Any]) -> float | None:
    raw_score = item.get("score", metadata.get("score"))
    if raw_score is None:
        return None

    try:
        return float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("context score must be numeric.") from exc


def _require_non_empty(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")

    return normalized
