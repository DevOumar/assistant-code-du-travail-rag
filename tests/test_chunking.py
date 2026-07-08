import pytest

from chunking import ChunkingConfig, TextChunk, chunk_document, chunk_documents


def test_chunk_document_keeps_short_article_as_single_chunk() -> None:
    document = {
        "id": "article-L3121-1",
        "text": "Article L3121-1. La duree du travail effectif est le temps pendant lequel...",
        "metadata": {"article": "L3121-1", "source": "legi"},
    }

    chunks = chunk_document(document)

    assert chunks == [
        TextChunk(
            id="article-L3121-1::chunk-001",
            document_id="article-L3121-1",
            text=document["text"],
            metadata={
                "article": "L3121-1",
                "source": "legi",
                "chunk_index": 0,
                "chunk_count": 1,
                "chunking_strategy": "article-preserving",
            },
        )
    ]


def test_chunk_document_splits_long_text_and_preserves_metadata() -> None:
    text = (
        "Premier paragraphe sur la duree du travail. "
        "Il contient une regle importante.\n\n"
        "Deuxieme paragraphe sur les heures supplementaires. "
        "Il precise les conditions d'application.\n\n"
        "Troisieme paragraphe avec une reserve importante."
    )
    document = {
        "id": "article-L3121-28",
        "text": text,
        "metadata": {"article": "L3121-28", "theme": "duree_travail"},
    }

    chunks = chunk_document(document, ChunkingConfig(max_chars=95, overlap_chars=20, min_chars=20))

    assert len(chunks) > 1
    assert all(chunk.document_id == "article-L3121-28" for chunk in chunks)
    assert all(chunk.metadata["article"] == "L3121-28" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["chunk_count"] == len(chunks) for chunk in chunks)
    assert all(len(chunk.text) <= 130 for chunk in chunks)


def test_chunk_documents_skips_empty_documents() -> None:
    chunks = chunk_documents(
        [
            {"id": "empty", "text": "   ", "metadata": {"article": "L0000-0"}},
            {"id": "valid", "text": "Un contenu exploitable.", "metadata": {}},
        ]
    )

    assert [chunk.document_id for chunk in chunks] == ["valid"]


def test_chunking_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        ChunkingConfig(max_chars=0)

    with pytest.raises(ValueError, match="overlap_chars"):
        ChunkingConfig(max_chars=100, overlap_chars=100)


def test_chunk_document_requires_normalized_document_fields() -> None:
    with pytest.raises(ValueError, match="'id'"):
        chunk_document({"text": "contenu"})

    with pytest.raises(ValueError, match="'text'"):
        chunk_document({"id": "article"})
