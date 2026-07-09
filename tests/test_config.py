from pathlib import Path

import pytest

from config import LEGAL_DISCLAIMER, PROJECT_ROOT, load_config


def test_load_config_uses_safe_defaults() -> None:
    config = load_config(env_file=None, environ={})

    assert config.environment == "development"
    assert config.paths.raw_data_dir == PROJECT_ROOT / "data" / "raw"
    assert config.paths.processed_data_dir == PROJECT_ROOT / "data" / "processed"
    assert config.paths.chroma_db_dir == PROJECT_ROOT / "data" / "chroma"
    assert config.paths.prompts_dir == PROJECT_ROOT / "prompts"
    assert config.embedding.collection_name == "code_du_travail"
    assert config.retrieval.top_k == 5
    assert config.retrieval.enable_hyde is False
    assert config.llm.provider == "groq"
    assert config.llm.temperature == 0.2
    assert config.llm.api_key is None
    assert config.legal_disclaimer == LEGAL_DISCLAIMER


def test_load_config_reads_environment_overrides(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    config = load_config(
        env_file=None,
        environ={
            "APP_ENV": "test",
            "RAW_DATA_DIR": str(raw_dir),
            "PROCESSED_DATA_DIR": str(processed_dir),
            "RETRIEVAL_TOP_K": "8",
            "RETRIEVAL_ENABLE_HYDE": "true",
            "GROQ_API_KEY": "test-secret",
            "GROQ_TEMPERATURE": "0.1",
            "CORPUS_SOURCE": "legi-data",
            "CORPUS_DATE": "2026-07-08",
        },
    )

    assert config.environment == "test"
    assert config.paths.raw_data_dir == raw_dir.resolve()
    assert config.paths.processed_data_dir == processed_dir.resolve()
    assert config.retrieval.top_k == 8
    assert config.retrieval.enable_hyde is True
    assert config.llm.api_key == "test-secret"
    assert config.llm.temperature == 0.1
    assert config.corpus.source == "legi-data"
    assert config.corpus.date == "2026-07-08"


def test_load_config_rejects_invalid_numeric_values() -> None:
    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K"):
        load_config(env_file=None, environ={"RETRIEVAL_TOP_K": "0"})

    with pytest.raises(ValueError, match="GROQ_TEMPERATURE"):
        load_config(env_file=None, environ={"GROQ_TEMPERATURE": "3"})
