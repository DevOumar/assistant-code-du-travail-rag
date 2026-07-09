from types import SimpleNamespace
import sys

import pytest

from config import load_config
from llm import GroqAnswerGenerator, LlmGenerationError
from prompting import PromptMessages


def test_groq_generator_requires_api_key() -> None:
    config = load_config(env_file=None, environ={"GROQ_API_KEY": ""})
    generator = GroqAnswerGenerator(config=config)

    with pytest.raises(LlmGenerationError, match="GROQ_API_KEY"):
        generator.generate(PromptMessages(system="system", user="user"))


def test_groq_generator_calls_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            message = SimpleNamespace(content="Reponse sourcee.")
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
            "GROQ_TEMPERATURE": "0.1",
            "GROQ_MAX_TOKENS": "256",
        },
    )

    answer = GroqAnswerGenerator(config=config).generate(
        PromptMessages(system="System prompt", user="Question utilisateur")
    )

    assert answer == "Reponse sourcee."
    assert calls["api_key"] == "test-key"
    assert calls["model"] == "llama-test"
    assert calls["temperature"] == 0.1
    assert calls["max_tokens"] == 256
    assert calls["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Question utilisateur"},
    ]
