import json

import pytest

import index_pipeline
from config import load_config
from index_pipeline import (
    IndexPipelineError,
    build_chunks_from_processed_documents,
    load_processed_documents,
    run_indexing_pipeline,
)


def _config(tmp_path):
    return load_config(
        env_file=None,
        environ={
            "RAW_DATA_DIR": str(tmp_path / "raw"),
            "PROCESSED_DATA_DIR": str(tmp_path / "processed"),
            "CHROMA_DB_DIR": str(tmp_path / "chroma"),
            "PROMPTS_DIR": "prompts",
        },
    )


def test_load_processed_documents_reads_documents_list(tmp_path) -> None:
    config = _config(tmp_path)
    config.paths.processed_data_dir.mkdir(parents=True)
    path = config.paths.processed_data_dir / "code_du_travail_documents.json"
    path.write_text(json.dumps({"documents": [{"id": "doc-1", "text": "Texte"}]}), encoding="utf-8")

    documents = load_processed_documents(config)

    assert documents == [{"id": "doc-1", "text": "Texte"}]


def test_load_processed_documents_rejects_missing_file(tmp_path) -> None:
    config = _config(tmp_path)

    with pytest.raises(IndexPipelineError, match="not found"):
        load_processed_documents(config)


def test_load_processed_documents_rejects_invalid_json(tmp_path) -> None:
    config = _config(tmp_path)
    config.paths.processed_data_dir.mkdir(parents=True)
    path = config.paths.processed_data_dir / "code_du_travail_documents.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(IndexPipelineError, match="not valid JSON"):
        load_processed_documents(config)


def test_build_chunks_from_processed_documents_preserves_metadata(tmp_path) -> None:
    config = _config(tmp_path)
    config.paths.processed_data_dir.mkdir(parents=True)
    path = config.paths.processed_data_dir / "code_du_travail_documents.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "article-L3121-27",
                        "text": "La duree legale de travail effectif des salaries est fixee.",
                        "metadata": {"article": "L3121-27", "source": "Legifrance"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    chunks = build_chunks_from_processed_documents(config)

    assert len(chunks) == 1
    assert chunks[0]["id"] == "article-L3121-27::chunk-001"
    assert chunks[0]["metadata"]["article"] == "L3121-27"
    assert chunks[0]["metadata"]["document_id"] == "article-L3121-27"


def test_run_indexing_pipeline_calls_optional_steps(monkeypatch, tmp_path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(index_pipeline, "load_corpus", lambda config: calls.append("load"))
    monkeypatch.setattr(index_pipeline, "parse_corpus", lambda config: calls.append("parse"))

    def fake_index_processed_corpus(config, recreate=False):
        calls.append(f"index:{recreate}")
        return 12

    monkeypatch.setattr(index_pipeline, "index_processed_corpus", fake_index_processed_corpus)

    count = run_indexing_pipeline(
        config=config,
        fetch_raw=True,
        parse_raw=True,
        recreate=True,
    )

    assert count == 12
    assert calls == ["load", "parse", "index:True"]
