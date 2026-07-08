"""Central application configuration.

This module only loads and validates settings. It does not initialize external
services such as ChromaDB, Groq, or Legifrance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEGAL_DISCLAIMER = (
    "Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou "
    "l'inspection du travail pour votre situation personnelle."
)


@dataclass(frozen=True)
class PathsConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    chroma_db_dir: Path
    prompts_dir: Path


@dataclass(frozen=True)
class CorpusConfig:
    source: str | None
    date: str | None
    minimum_theme_count: int = 5


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class LegifranceConfig:
    client_id: str | None = field(default=None, repr=False)
    client_secret: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class AppConfig:
    environment: str
    paths: PathsConfig
    corpus: CorpusConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    llm: LlmConfig
    legifrance: LegifranceConfig
    legal_disclaimer: str = LEGAL_DISCLAIMER


def load_config(
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load application settings from an optional .env file and environment variables."""

    if env_file is not None and environ is None:
        load_dotenv(env_file, override=False)

    source = environ if environ is not None else os.environ

    paths = PathsConfig(
        raw_data_dir=_path(source, "RAW_DATA_DIR", "data/raw"),
        processed_data_dir=_path(source, "PROCESSED_DATA_DIR", "data/processed"),
        chroma_db_dir=_path(source, "CHROMA_DB_DIR", "data/chroma"),
        prompts_dir=_path(source, "PROMPTS_DIR", "prompts"),
    )

    corpus = CorpusConfig(
        source=_optional(source, "CORPUS_SOURCE"),
        date=_optional(source, "CORPUS_DATE"),
    )

    embedding = EmbeddingConfig(
        model_name=_string(
            source,
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        collection_name=_string(source, "CHROMA_COLLECTION_NAME", "code_du_travail"),
    )

    retrieval = RetrievalConfig(
        top_k=_integer(source, "RETRIEVAL_TOP_K", 5, minimum=1),
    )

    llm = LlmConfig(
        provider=_string(source, "LLM_PROVIDER", "groq"),
        model_name=_string(source, "GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=_floating(source, "GROQ_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0),
        max_tokens=_integer(source, "GROQ_MAX_TOKENS", 1024, minimum=1),
        api_key=_optional(source, "GROQ_API_KEY"),
    )

    legifrance = LegifranceConfig(
        client_id=_optional(source, "LEGIFRANCE_CLIENT_ID"),
        client_secret=_optional(source, "LEGIFRANCE_CLIENT_SECRET"),
    )

    return AppConfig(
        environment=_string(source, "APP_ENV", "development"),
        paths=paths,
        corpus=corpus,
        embedding=embedding,
        retrieval=retrieval,
        llm=llm,
        legifrance=legifrance,
    )


def _optional(source: Mapping[str, str], key: str) -> str | None:
    value = source.get(key)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _string(source: Mapping[str, str], key: str, default: str) -> str:
    value = _optional(source, key)
    if value is None:
        return default
    return value


def _path(source: Mapping[str, str], key: str, default: str) -> Path:
    raw_value = _string(source, key, default)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _integer(
    source: Mapping[str, str],
    key: str,
    default: int,
    minimum: int | None = None,
) -> int:
    raw_value = _string(source, key, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer.") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be greater than or equal to {minimum}.")

    return value


def _floating(
    source: Mapping[str, str],
    key: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw_value = _string(source, key, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number.") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be greater than or equal to {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be less than or equal to {maximum}.")

    return value
