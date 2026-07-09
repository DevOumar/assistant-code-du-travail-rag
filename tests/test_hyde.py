from types import SimpleNamespace
import sys

import pytest

from config import load_config
from hyde import GroqHyDEGenerator, HyDEGenerationError, build_hyde_search_document


def test_hyde_requires_api_key() -> None:
    config = load_config(env_file=None, environ={})
    generator = GroqHyDEGenerator(config=config)

    with pytest.raises(HyDEGenerationError, match="GROQ_API_KEY"):
        generator.generate("Quelle est la duree du travail ?")


def test_hyde_calls_groq_with_specific_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            message = SimpleNamespace(content="Un extrait hypothetique utile.")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))

    config = load_config(
        env_file=None,
        environ={
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "llama-test",
        },
    )

    document = build_hyde_search_document("Quelle est la duree legale du travail ?", config=config)

    assert document == "Un extrait hypothetique utile."
    assert calls["api_key"] == "test-key"
    assert calls["model"] == "llama-test"
    assert calls["temperature"] == 0.0
    assert calls["max_tokens"] == 256
