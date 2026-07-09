"""Command-line interface helpers.

This module owns the interactive loop and keeps it testable by dependency
injection. The production entry point builds the concrete RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from moderator import InputModerator
from pipeline import PipelineBuildError, build_rag_pipeline
from rag import RagResponse, RetrievedChunk


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


class AnsweringPipeline(Protocol):
    def answer(self, question: str) -> RagResponse:
        """Return a RAG response for a moderated question."""


@dataclass(frozen=True)
class CliConfig:
    prompt: str = "Question> "
    exit_commands: tuple[str, ...] = ("exit", "quit", "q")
    max_question_chars: int = 1000


def run_interactive_loop(
    pipeline: AnsweringPipeline,
    moderator: InputModerator | None = None,
    config: CliConfig | None = None,
    input_func: InputFunc = input,
    output_func: OutputFunc = print,
) -> None:
    """Run the interactive question-answer loop."""

    active_moderator = moderator or InputModerator()
    active_config = config or CliConfig()

    output_func("Assistant Code du travail - tapez 'exit' pour quitter.")

    while True:
        raw_question = input_func(active_config.prompt)
        command = raw_question.strip().casefold()

        if command in active_config.exit_commands:
            output_func("Fin de session.")
            return

        if len(raw_question) > active_config.max_question_chars:
            output_func("Question trop longue. Reformulez votre demande.")
            continue

        decision = active_moderator.moderate(raw_question)
        if not decision.is_allowed:
            output_func(_format_moderation_block(decision.reasons))
            continue

        try:
            response = pipeline.answer(decision.sanitized_question or raw_question)
        except Exception as exc:
            output_func(f"Erreur du pipeline RAG : {exc}")
            continue

        output_func(_format_response(response))


def _format_response(response: RagResponse) -> str:
    parts = [response.answer]

    if response.sources:
        parts.append("Sources :")
        parts.extend(_format_source(index, source) for index, source in enumerate(response.sources, 1))

    return "\n".join(parts)


def _format_source(index: int, source: RetrievedChunk) -> str:
    article = source.metadata.get("article") or "article non renseigne"
    origin = source.metadata.get("source") or "source non renseignee"
    score = f", score={source.score:.4f}" if source.score is not None else ""
    return f"- [{index}] {article} ({origin}{score})"


def _format_moderation_block(reasons: list[str]) -> str:
    if not reasons:
        return "Question refusee par la moderation."

    rendered_reasons = "\n".join(f"- {reason}" for reason in reasons)
    return f"Question refusee par la moderation :\n{rendered_reasons}"


def main() -> int:
    """Run the concrete interactive CLI."""

    try:
        pipeline = build_rag_pipeline()
    except PipelineBuildError as exc:
        print(f"Configuration invalide : {exc}")
        return 1

    run_interactive_loop(pipeline=pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
