"""Retrieval layer for the RAG pipeline.

This module queries the persistent vector database and adapts raw ChromaDB
results to the interface expected by ``RagPipeline``. It does not index
documents and does not call the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import AppConfig, load_config
from rag import RetrievedChunk as RagRetrievedChunk
from vector_store import VectorStoreError, query_vector_database


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk retrieved from the vector database."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None
    similarity: float | None

    def to_rag_chunk(self) -> RagRetrievedChunk:
        """Convert the retrieval result to the RAG pipeline chunk contract."""

        return RagRetrievedChunk(
            text=self.text,
            metadata={**self.metadata, "chunk_id": self.chunk_id},
            score=self.similarity,
        )


class RetrievalError(Exception):
    """Raised when retrieval cannot be performed."""


@dataclass(frozen=True)
class VectorStoreRetriever:
    """Retriever adapter compatible with ``RagPipeline``."""

    config: AppConfig | None = None

    def retrieve(self, question: str, top_k: int | None = None) -> list[RagRetrievedChunk]:
        return [
            chunk.to_rag_chunk()
            for chunk in retrieve(question=question, top_k=top_k, config=self.config)
        ]


def retrieve(
    question: str,
    top_k: int | None = None,
    config: AppConfig | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a user question."""

    active_config = config or load_config()

    if not question or not question.strip():
        raise RetrievalError("Question cannot be empty.")

    try:
        raw_results = query_vector_database(
            question=question.strip(),
            top_k=top_k or active_config.retrieval.top_k,
            config=active_config,
        )
    except VectorStoreError as exc:
        raise RetrievalError(str(exc)) from exc

    return [_to_retrieved_chunk(item) for item in raw_results]


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for console display."""

    if not chunks:
        return "Aucun chunk pertinent trouvé."

    lines: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        article = chunk.metadata.get("article", "Article inconnu")
        section = chunk.metadata.get("section", "Section inconnue")
        source = chunk.metadata.get("source", "Source inconnue")
        score = f"{chunk.similarity:.4f}" if isinstance(chunk.similarity, float) else "N/A"

        lines.append(
            f"[{index}] {article} - {section}\n"
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
    """Return retrieved chunks as a context block for manual retrieval tests."""

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


def _to_retrieved_chunk(item: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(item["id"]),
        text=str(item["text"]),
        metadata=dict(item.get("metadata") or {}),
        distance=_optional_float(item.get("distance")),
        similarity=_optional_float(item.get("similarity")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise RetrievalError(f"Invalid numeric retrieval score: {value!r}")


def main() -> None:
    """Small CLI to test retrieval without any LLM call."""

    print("Test du retrieval - Assistant Code du travail")
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
