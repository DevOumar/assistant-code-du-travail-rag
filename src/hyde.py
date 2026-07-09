"""HyDE helpers for retrieval query expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import AppConfig, load_config


class HyDEGenerationError(Exception):
    """Raised when a hypothetical search document cannot be generated."""


@dataclass(frozen=True)
class GroqHyDEGenerator:
    """Generate a hypothetical legal answer used to enrich retrieval."""

    config: AppConfig | None = None

    def generate(self, question: str) -> str:
        active_config = self.config or load_config()
        api_key = active_config.llm.api_key
        if not api_key:
            raise HyDEGenerationError("GROQ_API_KEY must be set before using HyDE.")

        prompt = _build_hyde_messages(question)

        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=active_config.llm.model_name,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as exc:
            raise HyDEGenerationError(f"HyDE generation failed: {exc}") from exc

        content = _extract_message_content(completion)
        if not content:
            raise HyDEGenerationError("HyDE returned an empty hypothetical document.")

        return content.strip()


def build_hyde_search_document(question: str, config: AppConfig | None = None) -> str:
    """Return a hypothetical search document for retrieval expansion."""

    generator = GroqHyDEGenerator(config=config)
    return generator.generate(question)


def _build_hyde_messages(question: str) -> dict[str, str]:
    normalized_question = question.strip()
    return {
        "system": (
            "Tu aides un moteur de recherche juridique. "
            "Redige un paragraphe court, neutre et factuel qui ressemble a un extrait "
            "de documentation utile pour rechercher la reponse a la question. "
            "N'ajoute pas de disclaimer, pas de conclusion, pas de liste."
        ),
        "user": f"Question: {normalized_question}\nHypothetical search document:",
    }


def _extract_message_content(completion: Any) -> str:
    choices = getattr(completion, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""
