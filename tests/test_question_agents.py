from question_agents import QuestionFormatter, ReferenceRetriever


def test_question_formatter_marks_small_talk_and_cleans_questions() -> None:
    routing = QuestionFormatter().format("Bonjour, peux-tu me dire la durée légale du travail ?")

    assert routing.cleaned_question == "la durée légale du travail ?"
    assert routing.is_small_talk is False
    assert routing.should_skip_retrieval is False


def test_question_formatter_detects_chatty_messages() -> None:
    routing = QuestionFormatter().format("Tu fais quoi ?")

    assert routing.is_small_talk is True
    assert routing.should_skip_retrieval is True


def test_reference_retriever_exposes_retrieval_contract(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []

    class FakeRetriever:
        def __init__(self, config=None, decompose=True) -> None:
            self.config = config
            self.decompose = decompose

        def retrieve(self, question: str, top_k: int | None = None):
            calls.append((question, top_k))
            return []

    monkeypatch.setattr("retrieval.VectorStoreRetriever", FakeRetriever)

    retriever = ReferenceRetriever(config=None, decompose=True)
    assert retriever.retrieve_references("Quelle est la durée légale du travail ?", top_k=3) == []
    assert calls == [("Quelle est la durée légale du travail ?", 3)]
