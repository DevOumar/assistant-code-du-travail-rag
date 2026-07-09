"""Vector database management with ChromaDB.

This module is responsible for:
- loading the embedding model;
- creating or loading a persistent ChromaDB collection;
- indexing chunks with metadata;
- checking whether the vector database already contains documents.

It does not load the corpus, does not chunk documents, and does not call the LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from config import AppConfig, load_config

if TYPE_CHECKING:
    import chromadb
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import SentenceTransformer


class VectorStoreError(Exception):
    """Raised when the vector store receives invalid input."""


def load_embedding_model(config: AppConfig | None = None) -> "SentenceTransformer":
    """Load the embedding model defined in the project configuration."""
    from sentence_transformers import SentenceTransformer

    config = config or load_config()
    return SentenceTransformer(config.embedding.model_name)


def get_chroma_client(config: AppConfig | None = None) -> "chromadb.PersistentClient":
    """Create a persistent ChromaDB client.

    The database is stored on disk in config.paths.chroma_db_dir.
    """
    import chromadb

    config = config or load_config()
    db_path: Path = config.paths.chroma_db_dir
    db_path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(db_path))


def get_or_create_collection(config: AppConfig | None = None) -> "Collection":
    """Load or create the configured ChromaDB collection."""
    config = config or load_config()
    client = get_chroma_client(config)

    return client.get_or_create_collection(
        name=config.embedding.collection_name,
        metadata={
            "embedding_model": config.embedding.model_name,
            "description": "Articles du Code du travail indexés pour le RAG",
        },
    )


def get_collection(config: AppConfig | None = None) -> "Collection":
    """Load an existing ChromaDB collection.

    This function must be used by retrieval code to avoid silently creating
    an empty database when indexing has not been done yet.
    """
    config = config or load_config()
    client = get_chroma_client(config)

    try:
        return client.get_collection(name=config.embedding.collection_name)
    except Exception as exc:
        raise VectorStoreError(
            "Vector database not found. Run the indexing step before retrieval."
        ) from exc


def collection_count(config: AppConfig | None = None) -> int:
    """Return the number of chunks stored in the ChromaDB collection."""
    collection = get_collection(config)
    return collection.count()


def is_vector_database_ready(config: AppConfig | None = None) -> bool:
    """Check whether the vector database exists and contains at least one chunk."""
    try:
        return collection_count(config) > 0
    except VectorStoreError:
        return False


def reset_collection(config: AppConfig | None = None) -> None:
    """Delete and recreate the configured collection.

    This should only be used during explicit reindexing, never automatically
    during retrieval.
    """
    config = config or load_config()
    client = get_chroma_client(config)

    try:
        client.delete_collection(name=config.embedding.collection_name)
    except Exception:
        pass

    client.get_or_create_collection(
        name=config.embedding.collection_name,
        metadata={
            "embedding_model": config.embedding.model_name,
            "description": "Articles du Code du travail indexés pour le RAG",
        },
    )


def index_chunks(
    chunks: list[dict[str, Any]],
    config: AppConfig | None = None,
    recreate: bool = False,
) -> int:
    """Index chunks into ChromaDB.

    Expected chunk format:
    {
        "id": "unique_chunk_id",
        "text": "text to embed",
        "metadata": {
            "article": "L3121-27",
            "section": "Durée du travail",
            "source": "Légifrance",
            ...
        }
    }

    Args:
        chunks: List of prepared chunks.
        config: Optional application configuration.
        recreate: If True, delete the existing collection before indexing.

    Returns:
        Number of indexed chunks.
    """
    config = config or load_config()

    if not chunks:
        raise VectorStoreError("No chunks provided for indexing.")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for position, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("id") or f"chunk_{position}")
        text = str(chunk.get("text") or "").strip()
        metadata = chunk.get("metadata") or {}

        if not text:
            raise VectorStoreError(f"Chunk {chunk_id} has empty text.")

        if not isinstance(metadata, dict):
            raise VectorStoreError(f"Chunk {chunk_id} metadata must be a dictionary.")

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(_sanitize_metadata(metadata))

    if recreate:
        reset_collection(config)

    collection = get_or_create_collection(config)
    model = load_embedding_model(config)

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def query_vector_database(
    question: str,
    top_k: int | None = None,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Search the vector database and return the most relevant chunks.

    This function is useful for testing the vector store before the full
    retrieval branch is implemented.
    """
    config = config or load_config()
    top_k = top_k or config.retrieval.top_k

    if not question.strip():
        raise VectorStoreError("Question cannot be empty.")

    collection = get_collection(config)
    model = load_embedding_model(config)

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return _format_query_results(results)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert metadata values to ChromaDB-compatible scalar values."""
    sanitized: dict[str, str | int | float | bool] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = str(value)

    return sanitized


def _format_query_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert raw ChromaDB query results into a simple list of dictionaries."""
    formatted: list[dict[str, Any]] = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        formatted.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata or {},
                "distance": distance,
                "similarity": 1 - distance if distance is not None else None,
            }
        )

    return formatted
