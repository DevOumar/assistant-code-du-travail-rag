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
    confidence: float = 1.0

    @property
    def is_allowed(self) -> bool:
        return self.status is ModerationStatus.ALLOWED


UNRELATED_TOPIC_KEYWORDS = (
    "recette",
    "cuisine",
    "football",
    "sport",
    "match",
    "cinema",
    "film",
    "voyage",
    "vacances",
    "restaurant",
    "musique",
    "politique",
    "sante",
    "medecin",
    "voiture",
    "internet",
    "finance personnelle",
)


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


def moderate_query(question: str, enforce_scope: bool = True) -> dict[str, object]:
    """Classify a query for the labor-law RAG assistant.

    Returns a soft decision with confidence and a reason. The function only
    blocks clear prompt-injection attempts or questions manifestly outside the
    labor-law domain.
    """

    normalized_question = _normalize_question(question)
    if not normalized_question:
        return {
            "allowed": False,
            "confidence": 0.0,
            "reason": "Question vide ou uniquement constituée d'espaces.",
        }

    if _matches_any(normalized_question, PROMPT_INJECTION_PATTERNS):
        return {
            "allowed": False,
            "confidence": 0.99,
            "reason": "Tentative probable de prompt injection détectée.",
        }

    if not enforce_scope:
        return {
            "allowed": True,
            "confidence": 0.65,
            "reason": "Le contrôle du périmètre est désactivé : la question est autorisée.",
        }

    if _contains_any_keyword(normalized_question, UNRELATED_TOPIC_KEYWORDS) and not _contains_any_keyword(
        normalized_question, LEGAL_SCOPE_KEYWORDS
    ):
        return {
            "allowed": False,
            "confidence": 0.90,
            "reason": "Question clairement hors périmètre du droit du travail.",
        }

    if _is_adjacent_business_question(
        normalized_question,
        CORPORATE_FINANCE_KEYWORDS,
        LABOR_CONTEXT_KEYWORDS,
    ):
        return {
            "allowed": False,
            "confidence": 0.80,
            "reason": "Question d'entreprise sans lien explicite avec le droit du travail.",
        }

    if _contains_any_keyword(normalized_question, LEGAL_SCOPE_KEYWORDS):
        return {
            "allowed": True,
            "confidence": 0.95,
            "reason": "Question liée au droit du travail.",
        }

    return {
        "allowed": True,
        "confidence": 0.55,
        "reason": "Question ambiguë, mais autorisée par sécurité.",
    }


@dataclass(frozen=True)
class InputModerator:
    scope_keywords: tuple[str, ...] = LEGAL_SCOPE_KEYWORDS
    injection_patterns: tuple[str, ...] = PROMPT_INJECTION_PATTERNS
    adjacent_business_keywords: tuple[str, ...] = CORPORATE_FINANCE_KEYWORDS
    labor_context_keywords: tuple[str, ...] = LABOR_CONTEXT_KEYWORDS
    enforce_scope: bool = True

    def moderate(self, question: str) -> ModerationDecision:
        sanitized_question = _normalize_question(question)
        decision = moderate_query(question, enforce_scope=self.enforce_scope)

        if not sanitized_question:
            return ModerationDecision(
                status=ModerationStatus.BLOCKED,
                reasons=["La question est vide."],
                sanitized_question=None,
                confidence=0.0,
            )

        if decision["allowed"]:
            return ModerationDecision(
                status=ModerationStatus.ALLOWED,
                reasons=[],
                sanitized_question=sanitized_question,
                confidence=float(decision["confidence"]),
            )

        return ModerationDecision(
            status=ModerationStatus.BLOCKED,
            reasons=[str(decision["reason"])],
            sanitized_question=sanitized_question,
            confidence=float(decision["confidence"]),
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
