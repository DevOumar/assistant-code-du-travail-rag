"""Concrete LLM generators used by the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import AppConfig, load_config
from prompting import PromptMessages


class LlmGenerationError(Exception):
    """Raised when the LLM cannot generate an answer."""


@dataclass(frozen=True)
class GroqAnswerGenerator:
    """Answer generator backed by Groq chat completions."""

    config: AppConfig | None = None

    def generate(self, messages: PromptMessages) -> str:
        active_config = self.config or load_config()
        api_key = active_config.llm.api_key
        if not api_key:
            raise LlmGenerationError("GROQ_API_KEY must be set before using Groq generation.")

        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=active_config.llm.model_name,
                messages=[
                    {"role": "system", "content": messages.system},
                    {"role": "user", "content": messages.user},
                ],
                temperature=active_config.llm.temperature,
                max_tokens=active_config.llm.max_tokens,
            )
        except Exception as exc:
            raise LlmGenerationError(f"Groq generation failed: {exc}") from exc

        content = _extract_message_content(completion)
        if not content:
            raise LlmGenerationError("Groq returned an empty answer.")

        return content.strip()


def _extract_message_content(completion: Any) -> str:
    choices = getattr(completion, "choices", None)
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""
