"""Question preprocessing utilities for the retrieval stage.

The goal is to clean the user wording before vector search:
- remove parasitic phrases;
- normalize punctuation and whitespace;
- split compound questions into atomic sub-questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


_LEADING_PATTERNS = (
    r"^(?:bonjour|bonsoir|salut|coucou|hello|hi)\s*[,;:.-]*\s*",
    r"^(?:peux[- ]?tu|pouvez[- ]?vous|pourrais[- ]?tu|pourriez[- ]?vous)\s+(?:me\s+)?(?:dire|expliquer|préciser|indiquer)\s*[,;:.-]*\s*",
    r"^(?:j['’]aimerais|je\s+voudrais|je\s+veux)\s+savoir\s*[,;:.-]*\s*",
    r"^(?:dis[- ]?moi|explique[- ]?moi|donne[- ]?moi|dites[- ]?moi)\s*[,;:.-]*\s*",
    r"^(?:est[- ]ce\s+que\s+tu\s+peux|est[- ]ce\s+que\s+vous\s+pouvez)\s+(?:me\s+)?(?:dire|expliquer|préciser|indiquer)\s*[,;:.-]*\s*",
    r"^(?:merci|svp|stp)\s*[,;:.-]*\s*",
)

_FILLER_WORDS = (
    r"\b(euh|ben|du\s+coup|genre|voilà|en\s+fait)\b",
)

_TRAILING_POLITENESS = (
    r"\s*(?:merci|svp|stp|s['’]il\s+te\s+pla[tî]t|s['’]il\s+vous\s+pla[tî]t)\s*$",
)

_QUESTION_SPLIT_PATTERN = re.compile(
    r"\s*(?:\?|;|\bet\b|\bainsi que\b|\bmais\b|\bpuis\b|\bcompare(?:r|z)?\b|"
    r"\bcomparaison entre\b|\bdifference entre\b|\bdifférence entre\b|\bquelles? sont\b)\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionPreparation:
    original_question: str
    cleaned_question: str
    atomic_questions: list[str]
    removed_fragments: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_multiple_atomic_questions(self) -> bool:
        return len(self.atomic_questions) > 1


def prepare_question(question: str, max_atomic_questions: int = 4) -> QuestionPreparation:
    original = normalize_question(question)
    if not original:
        return QuestionPreparation(
            original_question="",
            cleaned_question="",
            atomic_questions=[],
        )

    cleaned, removed_fragments = strip_parasitic_phrases(original)
    atomic_questions = split_atomic_questions(cleaned, max_atomic_questions=max_atomic_questions)

    return QuestionPreparation(
        original_question=original,
        cleaned_question=cleaned,
        atomic_questions=atomic_questions,
        removed_fragments=tuple(removed_fragments),
    )


def strip_parasitic_phrases(question: str) -> tuple[str, list[str]]:
    """Remove polite filler phrases and verbal noise from a user question."""

    cleaned = normalize_question(question)
    removed: list[str] = []

    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        for pattern in _LEADING_PATTERNS:
            new_value = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            if new_value != cleaned:
                removed.append(cleaned[: len(cleaned) - len(new_value)].strip())
                cleaned = new_value.strip()
        for pattern in _TRAILING_POLITENESS:
            new_value = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            if new_value != cleaned:
                removed.append(cleaned[len(new_value) :].strip())
                cleaned = new_value.strip()

    for pattern in _FILLER_WORDS:
        new_value = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        if new_value != cleaned:
            removed.append("fillers")
            cleaned = new_value

    cleaned = normalize_question(cleaned)
    return cleaned, [fragment for fragment in removed if fragment]


def split_atomic_questions(question: str, max_atomic_questions: int = 4) -> list[str]:
    """Split a composite question into atomic retrieval queries."""

    normalized = normalize_question(question)
    if not normalized:
        return []

    parts = [
        _clean_sub_question(part)
        for part in _QUESTION_SPLIT_PATTERN.split(normalized)
        if _clean_sub_question(part)
    ]
    unique = _deduplicate_strings(part for part in parts if _is_informative_sub_question(part))

    if len(unique) < 2:
        return [normalized]

    return unique[:max_atomic_questions]


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def _clean_sub_question(text: str) -> str:
    cleaned = text.strip(" ,.:;?-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _is_informative_sub_question(text: str) -> bool:
    return len(text) >= 12 and len(text.split()) >= 2


def _deduplicate_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique
