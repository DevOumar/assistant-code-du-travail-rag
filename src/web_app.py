"""Streamlit chat interface for the legal RAG assistant.

This module provides the user-facing shell: chat history, moderation feedback,
corpus metadata, source rendering, and legal disclaimer display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from config import AppConfig, LEGAL_DISCLAIMER, load_config
from moderator import InputModerator
from pipeline import build_rag_pipeline
from rag import RagResponse, RetrievedChunk


CHAT_HISTORY_KEY = "chat_messages"


class AnsweringPipeline(Protocol):
    def answer(self, question: str) -> RagResponse:
        """Return a RAG answer for the user question."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    sources: list[RetrievedChunk] = field(default_factory=list)


@dataclass(frozen=True)
class UnavailablePipeline:
    """Fallback pipeline used when the vector database or LLM is unavailable."""

    legal_disclaimer: str = LEGAL_DISCLAIMER

    def answer(self, question: str) -> RagResponse:
        return RagResponse(
            question=question,
            answer=(
                "Le pipeline RAG complet n'est pas disponible dans cette session. "
                "Verifiez la configuration, la cle Groq et l'indexation ChromaDB "
                "avant de repondre sur le fond.\n\n"
                f"{self.legal_disclaimer}"
            ),
            sources=[],
            used_context=False,
        )


def format_sources_markdown(sources: list[RetrievedChunk]) -> str:
    """Render retrieved sources as compact Markdown."""

    if not sources:
        return ""

    lines = ["**Sources utilisees**"]
    for index, source in enumerate(sources, 1):
        article = source.metadata.get("article") or "article non renseigne"
        origin = source.metadata.get("source") or "source non renseignee"
        score = f" - score {source.score:.4f}" if source.score is not None else ""
        lines.append(f"- [{index}] `{article}` - {origin}{score}")

    return "\n".join(lines)


def build_corpus_status(config: AppConfig) -> str:
    """Return a concise status line about corpus freshness."""

    source = config.corpus.source or "source non renseignee"
    date = config.corpus.date or "date non renseignee"
    return f"Corpus : {source} | Date : {date}"


def main() -> None:
    """Run the Streamlit application."""

    import streamlit as st

    config = load_config()
    moderator = InputModerator()
    pipeline: AnsweringPipeline = _build_pipeline_or_fallback(config)

    st.set_page_config(
        page_title="Assistant Code du travail",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles(st)
    _initialize_history(st)

    with st.sidebar:
        st.title("Assistant")
        st.caption(build_corpus_status(config))
        st.caption(f"Top-k retrieval : {config.retrieval.top_k}")
        st.divider()
        st.caption(config.legal_disclaimer)
        if st.button("Nouvelle conversation", use_container_width=True):
            st.session_state[CHAT_HISTORY_KEY] = []
            st.rerun()

    st.title("Assistant Code du travail")
    st.caption("Posez une question sur le droit du travail francais.")

    for message in st.session_state[CHAT_HISTORY_KEY]:
        _render_message(st, message)

    question = st.chat_input("Exemple : Quelle est la duree legale du travail ?")
    if not question:
        return

    user_message = ChatMessage(role="user", content=question)
    st.session_state[CHAT_HISTORY_KEY].append(user_message)
    _render_message(st, user_message)

    decision = moderator.moderate(question)
    if not decision.is_allowed:
        answer = "Question refusee par la moderation :\n" + "\n".join(
            f"- {reason}" for reason in decision.reasons
        )
        assistant_message = ChatMessage(role="assistant", content=answer)
    else:
        try:
            response = pipeline.answer(decision.sanitized_question or question)
            assistant_message = ChatMessage(
                role="assistant",
                content=response.answer,
                sources=response.sources,
            )
        except Exception as exc:
            assistant_message = ChatMessage(
                role="assistant",
                content=(
                    "Le pipeline RAG n'est pas pret pour repondre a cette question. "
                    f"Detail technique : {exc}"
                ),
            )

    st.session_state[CHAT_HISTORY_KEY].append(assistant_message)
    _render_message(st, assistant_message)


def _build_pipeline_or_fallback(config: AppConfig) -> AnsweringPipeline:
    try:
        return build_rag_pipeline(config)
    except Exception:
        return UnavailablePipeline(config.legal_disclaimer)


def _initialize_history(st: object) -> None:
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []


def _render_message(st: object, message: ChatMessage) -> None:
    with st.chat_message(message.role):
        st.markdown(message.content.replace("\n", "  \n"))
        sources = format_sources_markdown(message.sources)
        if sources:
            st.markdown(sources)


def _inject_styles(st: object) -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 2rem;
        }
        [data-testid="stChatMessage"] {
            border-radius: 8px;
            padding: 0.25rem 0.1rem;
        }
        [data-testid="stChatInput"] textarea {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
