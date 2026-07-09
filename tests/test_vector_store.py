from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config import load_config
from vector_store import (
    VectorStoreError,
    _format_query_results,
    _sanitize_metadata,
    index_chunks,
    is_vector_database_ready,
)


class FakeEmbeddings(list):
    def tolist(self) -> list[list[float]]:
        return list(self)


def test_sanitize_metadata_keeps_only_chroma_compatible_scalars() -> None:
    metadata = _sanitize_metadata(
        {
            "article": "L3121-27",
            "score": 0.91,
            "rank": 1,
            "active": True,
            "none": None,
            "path": ["Livre", "Titre"],
        }
    )

    assert metadata == {
        "article": "L3121-27",
        "score": 0.91,
        "rank": 1,
        "active": True,
        "path": "['Livre', 'Titre']",
    }


def test_format_query_results_returns_simple_chunk_dicts() -> None:
    results = {
        "ids": [["chunk-1"]],
        "documents": [["Texte source"]],
        "metadatas": [[{"article": "L3121-27"}]],
        "distances": [[0.12]],
    }

    formatted = _format_query_results(results)

    assert formatted == [
        {
            "id": "chunk-1",
            "text": "Texte source",
            "metadata": {"article": "L3121-27"},
            "distance": 0.12,
            "similarity": 0.88,
        }
    ]


def test_index_chunks_validates_non_empty_input(tmp_path: Path) -> None:
    config = load_config(env_file=None, environ={"CHROMA_DB_DIR": str(tmp_path)})

    with pytest.raises(VectorStoreError, match="No chunks"):
        index_chunks([], config=config)


def test_index_chunks_upserts_embeddings_and_metadata(tmp_path: Path) -> None:
    config = load_config(env_file=None, environ={"CHROMA_DB_DIR": str(tmp_path)})
    collection = Mock()
    model = Mock()
    model.encode.return_value = FakeEmbeddings([[0.1, 0.2, 0.3]])

    with patch("vector_store.get_or_create_collection", return_value=collection):
        with patch("vector_store.load_embedding_model", return_value=model):
            count = index_chunks(
                [
                    {
                        "id": "article-L3121-27::chunk-001",
                        "text": "La durée légale du travail est fixée à trente-cinq heures.",
                        "metadata": {"article": "L3121-27", "tags": ["durée"]},
                    }
                ],
                config=config,
            )

    assert count == 1
    collection.upsert.assert_called_once_with(
        ids=["article-L3121-27::chunk-001"],
        documents=["La durée légale du travail est fixée à trente-cinq heures."],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"article": "L3121-27", "tags": "['durée']"}],
    )


def test_index_chunks_rejects_empty_text(tmp_path: Path) -> None:
    config = load_config(env_file=None, environ={"CHROMA_DB_DIR": str(tmp_path)})

    with pytest.raises(VectorStoreError, match="empty text"):
        index_chunks([{"id": "chunk-1", "text": "   ", "metadata": {}}], config=config)


def test_is_vector_database_ready_returns_false_when_collection_missing() -> None:
    with patch("vector_store.collection_count", side_effect=VectorStoreError("missing")):
        assert is_vector_database_ready() is False
