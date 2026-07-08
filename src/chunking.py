"""Chunking utilities for already parsed legal documents.

The parser branch owns raw corpus parsing. This module only receives normalized
documents and splits their text into chunks ready for later indexing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    min_chars: int = 120

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be greater than 0.")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must be greater than or equal to 0.")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars.")
        if self.min_chars < 0:
            raise ValueError("min_chars must be greater than or equal to 0.")


@dataclass(frozen=True)
class TextChunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_documents(
    documents: Iterable[Mapping[str, Any]],
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Split normalized documents into text chunks.

    Expected document fields are:
    - id: stable document identifier;
    - text: cleaned text;
    - metadata: optional dictionary produced by the parser branch.
    """

    active_config = config or ChunkingConfig()
    chunks: list[TextChunk] = []

    for document in documents:
        chunks.extend(chunk_document(document, active_config))

    return chunks


def chunk_document(
    document: Mapping[str, Any],
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Split one normalized document while preserving its metadata."""

    active_config = config or ChunkingConfig()
    document_id = _required_string(document, "id")
    text = _normalize_text(_required_string(document, "text"))

    if not text:
        return []

    metadata = dict(document.get("metadata") or {})
    parts = _split_text(text, active_config)
    total = len(parts)

    return [
        TextChunk(
            id=f"{document_id}::chunk-{index + 1:03d}",
            document_id=document_id,
            text=part,
            metadata={
                **metadata,
                "chunk_index": index,
                "chunk_count": total,
                "chunking_strategy": "article-preserving",
            },
        )
        for index, part in enumerate(parts)
    ]


def _split_text(text: str, config: ChunkingConfig) -> list[str]:
    if len(text) <= config.max_chars:
        return [text]

    units = _paragraphs(text)
    if any(len(unit) > config.max_chars for unit in units):
        units = _sentences(text)

    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > config.max_chars:
            chunks.extend(_split_long_unit(unit, config))
            current = ""
            continue

        candidate = _join_text(current, unit)
        if current and len(candidate) > config.max_chars:
            chunks.append(current)
            current = _join_text(_overlap_from(current, config.overlap_chars), unit)
        else:
            current = candidate

    if current:
        chunks.append(current)

    return _merge_tiny_tail(chunks, config)


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _sentences(text: str) -> list[str]:
    normalized = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _split_long_unit(unit: str, config: ChunkingConfig) -> list[str]:
    words = unit.split()
    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = _join_text(current, word)
        if current and len(candidate) > config.max_chars:
            chunks.append(current)
            current = _join_text(_overlap_from(current, config.overlap_chars), word)
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _merge_tiny_tail(chunks: list[str], config: ChunkingConfig) -> list[str]:
    if len(chunks) < 2 or len(chunks[-1]) >= config.min_chars:
        return chunks

    tail = chunks.pop()
    chunks[-1] = _join_text(chunks[-1], tail)
    return chunks


def _overlap_from(text: str, overlap_chars: int) -> str:
    if overlap_chars == 0:
        return ""

    words = text.split()
    selected: list[str] = []
    total_length = 0

    for word in reversed(words):
        projected = total_length + len(word) + (1 if selected else 0)
        if projected > overlap_chars:
            break
        selected.append(word)
        total_length = projected

    return " ".join(reversed(selected))


def _join_text(left: str, right: str) -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    return f"{left.strip()} {right.strip()}"


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _required_string(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise ValueError(f"document {key!r} must be a string.")
    return value
