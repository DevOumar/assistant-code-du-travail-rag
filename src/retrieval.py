"""Retrieval layer for the RAG pipeline.

This module searches the persistent vector database and returns the most
relevant chunks with their metadata. It does not index documents and does not
call the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import AppConfig, load_config
from src.vector_store import VectorStoreError, query_vector_database


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk retrieved from the vector database."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None
    similarity: float | None


class RetrievalError(Exception):
    """Raised when retrieval cannot be performed."""


def retrieve(
    question: str,
    top_k: int | None = None,
    config: AppConfig | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a user question.

    Args:
        question: Natural language question asked by the user.
        top_k: Number of chunks to retrieve. If None, use the config value.
        config: Optional application configuration.

    Returns:
        List of retrieved chunks ordered by relevance.
    """
    config = config or load_config()

    if not question or not question.strip():
        raise RetrievalError("Question cannot be empty.")

    try:
        raw_results = query_vector_database(
            question=question,
            top_k=top_k or config.retrieval.top_k,
            config=config,
        )
    except VectorStoreError as exc:
        raise RetrievalError(str(exc)) from exc

    return [
        RetrievedChunk(
            chunk_id=str(item["id"]),
            text=str(item["text"]),
            metadata=dict(item.get("metadata") or {}),
            distance=item.get("distance"),
            similarity=item.get("similarity"),
        )
        for item in raw_results
    ]


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for console display."""
    if not chunks:
        return "Aucun chunk pertinent trouvé."

    lines: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        article = chunk.metadata.get("article", "Article inconnu")
        section = chunk.metadata.get("section", "Section inconnue")
        source = chunk.metadata.get("source", "Source inconnue")

        score = (
            f"{chunk.similarity:.4f}"
            if isinstance(chunk.similarity, float)
            else "N/A"
        )

        lines.append(
            f"[{index}] {article} — {section}\n"
            f"Source : {source}\n"
            f"Score de similarité : {score}\n"
            f"Texte : {chunk.text}\n"
        )

    return "\n".join(lines)


def retrieve_as_context(
    question: str,
    top_k: int | None = None,
    config: AppConfig | None = None,
) -> str:
    """Return retrieved chunks as a context block for the future RAG prompt."""
    chunks = retrieve(question=question, top_k=top_k, config=config)

    context_blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        article = chunk.metadata.get("article", "Article inconnu")
        section = chunk.metadata.get("section", "Section inconnue")
        source = chunk.metadata.get("source", "Source inconnue")

        context_blocks.append(
            f"Document {index}\n"
            f"Article : {article}\n"
            f"Section : {section}\n"
            f"Source : {source}\n"
            f"Texte : {chunk.text}"
        )

    return "\n\n".join(context_blocks)


def main() -> None:
    """Small CLI to test retrieval without any LLM call."""
    print("Test du retrieval — Assistant Code du travail")
    print("Tapez 'exit' pour quitter.\n")

    while True:
        question = input("Question : ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Fin du test retrieval.")
            break

        try:
            chunks = retrieve(question)
            print()
            print(format_retrieved_chunks(chunks))
            print("-" * 80)
        except RetrievalError as exc:
            print(f"Erreur retrieval : {exc}")


if __name__ == "__main__":
    main()