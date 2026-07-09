"""Retrieval layer for the RAG pipeline.

This module queries the persistent vector database and adapts raw ChromaDB
results to the interface expected by ``RagPipeline``. It does not index
documents and does not call the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import AppConfig, load_config
from rag import RetrievedChunk as RagRetrievedChunk
from vector_store import VectorStoreError, query_vector_database


DEFAULT_TOP_K_BUMP = 2


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
    decompose: bool = True

    def retrieve(self, question: str, top_k: int | None = None) -> list[RagRetrievedChunk]:
        retrieval_func = retrieve_decomposed if self.decompose else retrieve
        return [
            chunk.to_rag_chunk()
            for chunk in retrieval_func(question=question, top_k=top_k, config=self.config)
        ]


def retrieve_documents(
    question: str,
    config: AppConfig | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks and return them with scores and metadata."""
    return retrieve(question=question, top_k=top_k, config=config)


def rerank_documents(
    question: str,
    documents: list[RetrievedChunk],
    config: AppConfig | None = None,
) -> list[RetrievedChunk]:
    """Optionally rerank retrieved chunks using a cross-encoder or a fallback heuristic."""
    active_config = config or load_config()
    if not active_config.retrieval.enable_reranking or not documents:
        return documents

    model_name = active_config.retrieval.reranker_model_name
    if not model_name:
        return documents

    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(model_name)
        pairs = [(question, chunk.text) for chunk in documents]
        scores = reranker.predict(pairs, convert_to_numpy=True).tolist()

        return sorted(
            [
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    distance=chunk.distance,
                    similarity=float(score),
                )
                for chunk, score in zip(documents, scores)
            ],
            key=lambda chunk: chunk.similarity or 0.0,
            reverse=True,
        )
    except ImportError:
        return documents
    except Exception:
        return documents


def retrieve(
    question: str,
    top_k: int | None = None,
    config: AppConfig | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a user question."""

    active_config = config or load_config()

    if not question or not question.strip():
        raise RetrievalError("Question cannot be empty.")

    active_top_k = _effective_top_k(active_config, top_k)
    similarity_threshold = active_config.retrieval.similarity_threshold

    try:
        raw_results = query_vector_database(
            question=question.strip(),
            top_k=active_top_k,
            config=active_config,
        )
    except VectorStoreError as exc:
        raise RetrievalError(str(exc)) from exc

    candidates = [_to_retrieved_chunk(item) for item in raw_results]
    filtered = [
        candidate
        for candidate in candidates
        if candidate.similarity is not None and candidate.similarity >= similarity_threshold
    ]

    ranked = rerank_documents(question=question, documents=filtered, config=active_config)
    return ranked


def _effective_top_k(config: AppConfig, requested_top_k: int | None = None) -> int:
    base_top_k = requested_top_k if requested_top_k is not None else config.retrieval.top_k
    if requested_top_k is None and base_top_k < config.retrieval.max_top_k:
        return min(base_top_k + DEFAULT_TOP_K_BUMP, config.retrieval.max_top_k)
    return min(base_top_k, config.retrieval.max_top_k)


def retrieve_decomposed(
    question: str,
    top_k: int | None = None,
    config: AppConfig | None = None,
    max_sub_questions: int = 4,
) -> list[RetrievedChunk]:
    """Retrieve with deterministic query decomposition and result deduplication."""

    active_config = config or load_config()
    active_top_k = top_k or active_config.retrieval.top_k
    sub_questions = decompose_question(question, max_sub_questions=max_sub_questions)

    candidates: list[RetrievedChunk] = []
    for sub_question in sub_questions:
        candidates.extend(retrieve(sub_question, top_k=active_top_k, config=active_config))

    return _deduplicate_chunks(candidates)[:active_top_k]


def decompose_question(question: str, max_sub_questions: int = 4) -> list[str]:
    """Split a compound user question into atomic retrieval queries.

    This is intentionally deterministic: it improves retrieval recall without adding
    an extra LLM call before every question.
    """

    normalized = _normalize_question(question)
    if not normalized:
        raise RetrievalError("Question cannot be empty.")

    parts = [
        _clean_sub_question(part)
        for part in _QUESTION_SPLIT_PATTERN.split(normalized)
        if _clean_sub_question(part)
    ]

    filtered = _deduplicate_strings(part for part in parts if _is_informative_sub_question(part))
    if len(filtered) < 2:
        return [normalized]

    return filtered[:max_sub_questions]


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


_QUESTION_SPLIT_PATTERN = re.compile(
    r"\s*(?:\?|;|\bet\b|\bainsi que\b|\bmais\b|\bpuis\b|\bcompare(?:r|z)?\b|"
    r"\bcomparaison entre\b|\bdifference entre\b|\bdifférence entre\b|\bquelles? sont\b)\s*",
    flags=re.IGNORECASE,
)


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def _clean_sub_question(text: str) -> str:
    cleaned = text.strip(" ,.:;?-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _is_informative_sub_question(text: str) -> bool:
    return len(text) >= 12 and len(text.split()) >= 2


def _deduplicate_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    by_key: dict[str, RetrievedChunk] = {}

    for chunk in chunks:
        key = chunk.chunk_id or _fallback_chunk_key(chunk)
        current = by_key.get(key)
        if current is None or _score(chunk) > _score(current):
            by_key[key] = chunk

    return sorted(by_key.values(), key=_score, reverse=True)


def _fallback_chunk_key(chunk: RetrievedChunk) -> str:
    article = str(chunk.metadata.get("article") or "")
    return f"{article}:{chunk.text[:120]}"


def _score(chunk: RetrievedChunk) -> float:
    if chunk.similarity is not None:
        return chunk.similarity
    if chunk.distance is not None:
        return 1 - chunk.distance
    return 0.0


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
