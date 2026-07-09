"""Input moderation helpers.

The moderator is deterministic and local. It flags obvious prompt-injection and
out-of-scope requests before the RAG pipeline is called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from config import PROJECT_ROOT


MODERATOR_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "moderator_prompt_system.txt"


class ModerationStatus(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ModerationDecision:
    status: ModerationStatus
    reasons: list[str] = field(default_factory=list)
    sanitized_question: str | None = None

    @property
    def is_allowed(self) -> bool:
        return self.status is ModerationStatus.ALLOWED


PROMPT_INJECTION_PATTERNS = (
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\boublie\s+(toutes\s+)?les\s+instructions\b",
    r"\bignore\s+le\s+prompt\b",
    r"\breveal\s+(the\s+)?(system\s+)?prompt\b",
    r"\bmontre\s+(moi\s+)?(le\s+)?prompt\s+systeme\b",
    r"\bdeveloper\s+message\b",
    r"\bsystem\s+message\b",
    r"\bjailbreak\b",
)

LEGAL_SCOPE_KEYWORDS = (
    "code du travail",
    "travail",
    "salarie",
    "salarié",
    "employeur",
    "contrat",
    "cdi",
    "cdd",
    "licenciement",
    "rupture conventionnelle",
    "conges",
    "heures supplementaires",
    "salaire",
    "smic",
    "harcelement",
    "discrimination",
    "preavis",
    "temps de travail",
)

CORPORATE_FINANCE_KEYWORDS = (
    "fusion acquisition",
    "fusion-acquisition",
    "m&a",
    "merger",
    "acquisition",
    "due diligence",
    "valorisation",
    "actionnaire",
    "actionnaires",
    "parts sociales",
    "cession de titres",
    "code de commerce",
)

LABOR_CONTEXT_KEYWORDS = (
    "code du travail",
    "contrat de travail",
    "salarie",
    "salarié",
    "employeur",
    "licenciement",
    "transfert",
    "l1224",
    "l. 1224",
    "comite social",
    "cse",
    "representants du personnel",
    "représentants du personnel",
)

OUT_OF_SCOPE_MESSAGE = (
    "La question ne semble pas porter sur le droit du travail francais."
)


@dataclass(frozen=True)
class InputModerator:
    scope_keywords: tuple[str, ...] = LEGAL_SCOPE_KEYWORDS
    injection_patterns: tuple[str, ...] = PROMPT_INJECTION_PATTERNS
    adjacent_business_keywords: tuple[str, ...] = CORPORATE_FINANCE_KEYWORDS
    labor_context_keywords: tuple[str, ...] = LABOR_CONTEXT_KEYWORDS
    enforce_scope: bool = True

    def moderate(self, question: str) -> ModerationDecision:
        sanitized_question = _normalize_question(question)
        reasons: list[str] = []

        if not sanitized_question:
            return ModerationDecision(
                status=ModerationStatus.BLOCKED,
                reasons=["La question est vide."],
                sanitized_question=None,
            )

        if _matches_any(sanitized_question, self.injection_patterns):
            reasons.append("Tentative probable de prompt injection.")

        if self.enforce_scope:
            if _is_adjacent_business_question(
                sanitized_question,
                self.adjacent_business_keywords,
                self.labor_context_keywords,
            ):
                reasons.append(OUT_OF_SCOPE_MESSAGE)
            elif not _contains_any_keyword(sanitized_question, self.scope_keywords):
                reasons.append(OUT_OF_SCOPE_MESSAGE)

        if reasons:
            return ModerationDecision(
                status=ModerationStatus.BLOCKED,
                reasons=reasons,
                sanitized_question=sanitized_question,
            )

        return ModerationDecision(
            status=ModerationStatus.ALLOWED,
            reasons=[],
            sanitized_question=sanitized_question,
        )


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def read_moderator_prompt(template_path: str | Path = MODERATOR_PROMPT_TEMPLATE_PATH) -> str:
    """Read the moderation policy prompt from disk."""

    return Path(template_path).read_text(encoding="utf-8")


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    normalized_text = text.casefold()
    return any(keyword.casefold() in normalized_text for keyword in keywords)


def _is_adjacent_business_question(
    text: str,
    business_keywords: Iterable[str],
    labor_keywords: Iterable[str],
) -> bool:
    """Reject corporate-law questions unless they are anchored in labor law."""

    normalized_text = text.casefold()
    has_business_topic = any(keyword.casefold() in normalized_text for keyword in business_keywords)
    has_labor_context = any(keyword.casefold() in normalized_text for keyword in labor_keywords)
    return has_business_topic and not has_labor_context
