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


def test_question_formatter_can_keep_original_question_without_reformulation() -> None:
    routing = QuestionFormatter(enable_reformulation=False).format(
        "Bonjour, peux-tu me dire la durée légale du travail ?"
    )

    assert routing.cleaned_question == "Bonjour, peux-tu me dire la durée légale du travail ?"
    assert routing.removed_fragments == ()
    assert routing.is_small_talk is False


def test_reference_retriever_exposes_retrieval_contract(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []

    class FakeRetriever:
        def __init__(self, config=None, decompose=True, enable_reformulation=True, enable_hyde=None, enable_hybrid_search=None) -> None:
            self.config = config
            self.decompose = decompose

        def retrieve(self, question: str, top_k: int | None = None):
            calls.append((question, top_k))
            return []

    monkeypatch.setattr("retrieval.VectorStoreRetriever", FakeRetriever)

    retriever = ReferenceRetriever(config=None, decompose=True)
    assert retriever.retrieve_references("Quelle est la durée légale du travail ?", top_k=3) == []
    assert calls == [("Quelle est la durée légale du travail ?", 3)]


def test_reference_retriever_propagates_runtime_flags(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeRetriever:
        def __init__(
            self,
            config=None,
            decompose=True,
            enable_reformulation=True,
            enable_hyde=None,
            enable_hybrid_search=None,
        ) -> None:
            seen["config"] = config
            seen["decompose"] = decompose
            seen["enable_reformulation"] = enable_reformulation
            seen["enable_hyde"] = enable_hyde
            seen["enable_hybrid_search"] = enable_hybrid_search

        def retrieve(self, question: str, top_k: int | None = None):
            seen["question"] = question
            seen["top_k"] = top_k
            return []

    monkeypatch.setattr("retrieval.VectorStoreRetriever", FakeRetriever)

    retriever = ReferenceRetriever(
        config=None,
        decompose=False,
        enable_reformulation=False,
        enable_hyde=True,
        enable_hybrid_search=True,
    )
    retriever.retrieve_references("Question test", top_k=2)

    assert seen["decompose"] is False
    assert seen["enable_reformulation"] is False
    assert seen["enable_hyde"] is True
    assert seen["enable_hybrid_search"] is True
    assert seen["question"] == "Question test"
