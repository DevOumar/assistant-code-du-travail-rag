"""Streamlit chat interface for the legal RAG assistant.

This module provides the user-facing shell: chat history, moderation feedback,
corpus metadata, source rendering, and legal disclaimer display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from datetime import datetime, date

from config import AppConfig, LEGAL_DISCLAIMER, load_config
from moderator import InputModerator
from pipeline import build_rag_pipeline
from rag import RagResponse, RetrievedChunk


CHAT_HISTORY_KEY = "chat_messages"
CONVERSATIONS_KEY = "conversations"
CURRENT_CONVERSATION_INDEX_KEY = "current_conversation_index"


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

    lines = ["**Sources utilisées**"]
    for index, source in enumerate(sources, 1):
        article = source.metadata.get("article") or "article non renseigne"
        origin = source.metadata.get("source") or "source non renseignee"
        score = f" - score {source.score:.4f}" if source.score is not None else ""
        lines.append(f"- [{index}] `{article}` - {origin}{score}")

    return "\n".join(lines)


def build_corpus_status(config: AppConfig) -> str:
    """Return a concise status line about corpus freshness."""

    source = config.corpus.source or "source non renseignée"
    corpus_date_str = config.corpus.date

    if not corpus_date_str:
        return f"Corpus : {source} | Date : non renseignée"

    # Try parsing ISO date `YYYY-MM-DD`
    try:
        corpus_date = datetime.fromisoformat(corpus_date_str).date()
    except Exception:
        return f"Corpus : {source} | Date : {corpus_date_str}"

    today = date.today()
    delta = today - corpus_date
    months = max(0, delta.days // 30)

    if months <= 3:
        risk = "Faible"
    elif months <= 12:
        risk = "Moyen"
    else:
        risk = "Élevé"

    age_text = f"{months} mois" if months > 0 else "<1 mois"

    return f"Corpus : {source} | Date : {corpus_date_str} | Âge : {age_text} | Risque d'obsolescence : {risk}"


def main() -> None:
    """Run the Streamlit application."""

    import streamlit as st

    config = load_config()

    st.set_page_config(
        page_title="Assistant Code du travail",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not config.llm.api_key:
        st.sidebar.error(
            "GROQ_API_KEY n'est pas défini. Vérifiez votre fichier .env et le répertoire de démarrage de Streamlit."
        )

    moderator = InputModerator()
    pipeline: AnsweringPipeline = _build_pipeline_or_fallback(config)

    _inject_styles(st)
    _initialize_history(st)

    with st.sidebar:
        st.title("Assistant")
        st.markdown(
            f"<div class='sidebar-panel'>"
            f"<p><strong>Corpus</strong> : {config.corpus.source or 'non renseigné'}</p>"
            f"<p><strong>Date</strong> : {config.corpus.date or 'non renseignée'}</p>"
            f"<p><strong>Top-k</strong> : {config.retrieval.top_k}</p>"
            f"<p><strong>Statut</strong> : {build_corpus_status(config)}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.divider()
        st.markdown(
            "<div class='legal-badge'>Cet assistant ne fournit pas de conseil juridique. "
            "Consultez un avocat ou l'inspection du travail pour votre situation personnelle.</div>",
            unsafe_allow_html=True,
        )
        

    st.title("Assistant Code du travail")
    st.caption("Posez une question sur le droit du travail français.")

    current_messages = st.session_state[CONVERSATIONS_KEY][st.session_state[CURRENT_CONVERSATION_INDEX_KEY]]['messages']
    for message in current_messages:
        _render_message(st, message)

    question = st.chat_input("Exemple : Quelle est la durée légale du travail ?")
    if not question:
        return

    user_message = ChatMessage(role="user", content=question)
    current_messages.append(user_message)
    _render_message(st, user_message)

    decision = moderator.moderate(question)
    if not decision.is_allowed:
        answer = "Question refusée par la modération :\n" + "\n".join(
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
                    "Le pipeline RAG n'est pas prêt pour répondre à cette question. "
                    f"Détail technique : {exc}"
                ),
            )

    current_messages.append(assistant_message)
    _render_message(st, assistant_message)


def _build_pipeline_or_fallback(config: AppConfig) -> AnsweringPipeline:
    try:
        return build_rag_pipeline(config)
    except Exception:
        return UnavailablePipeline(config.legal_disclaimer)


def _initialize_history(st: object) -> None:
    if CONVERSATIONS_KEY not in st.session_state:
        st.session_state[CONVERSATIONS_KEY] = []
    if CURRENT_CONVERSATION_INDEX_KEY not in st.session_state:
        st.session_state[CURRENT_CONVERSATION_INDEX_KEY] = 0

    if not st.session_state[CONVERSATIONS_KEY]:
        st.session_state[CONVERSATIONS_KEY].append(
            {"name": "Conversation 1", "messages": []}
        )

    current_index = st.session_state[CURRENT_CONVERSATION_INDEX_KEY]
    if current_index >= len(st.session_state[CONVERSATIONS_KEY]):
        st.session_state[CURRENT_CONVERSATION_INDEX_KEY] = len(st.session_state[CONVERSATIONS_KEY]) - 1


def _create_new_conversation(st: object) -> None:
    conversation_count = len(st.session_state[CONVERSATIONS_KEY])
    st.session_state[CONVERSATIONS_KEY].append(
        {"name": f"Conversation {conversation_count + 1}", "messages": []}
    )
    st.session_state[CURRENT_CONVERSATION_INDEX_KEY] = conversation_count


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
            border-radius: 12px;
            padding: 0.6rem 0.8rem;
        }
        [data-testid="stChatInput"] textarea {
            border-radius: 12px;
        }
        .sidebar-panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        .sidebar-panel p {
            margin: 0 0 0.5rem;
            color: #334155;
        }
        .sidebar-panel strong {
            color: #0f172a;
        }
        .stRadio > div {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .stRadio label {
            display: block;
            padding: 0.85rem 1rem;
            border-radius: 0.95rem;
            border: 1px solid transparent;
            background: #f8fafc;
            color: #0f172a;
            cursor: pointer;
            transition: all 120ms ease-in-out;
        }
        .stRadio label:hover {
            background: #eef2ff;
        }
        .stRadio input:checked + label {
            border-color: #3b82f6;
            background: #eff6ff;
            color: #1d4ed8;
        }
        .stButton button {
            border-radius: 0.95rem;
        }
        .stSidebar .block-container {
            padding-top: 1rem;
        }
        .legal-badge {
            background: #f1f5ff;
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            color: #0f172a;
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        .stSidebar {
            padding-top: 1rem;
        }
        .stRadio > div {
            gap: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
