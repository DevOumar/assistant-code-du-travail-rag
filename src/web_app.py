"""Streamlit chat interface for the legal RAG assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

from config import AppConfig, LEGAL_DISCLAIMER, load_config
from moderator import InputModerator
from pipeline import build_rag_pipeline
from rag import RagResponse, RetrievedChunk


CONVERSATIONS_KEY = "conversations"
CURRENT_CONVERSATION_INDEX_KEY = "current_conversation_index"
PRESET_QUESTION_KEY = "preset_question"

QUESTION_PRESETS = (
    "Quelle est la durée légale du travail ?",
    "Quels sont les congés payés acquis après un an de travail ?",
    "Quel est le préavis en cas de démission d'un CDI ?",
    "En cas de rupture conventionnelle, quelles indemnités sont dues ?",
    "En cas de fusion-acquisition, que devient le contrat de travail des salariés ?",
)


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
                "Vérifiez la configuration, la clé Groq et l'indexation ChromaDB "
                "avant de répondre sur le fond.\n\n"
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
        article = source.metadata.get("article") or "article non renseigné"
        origin = source.metadata.get("source") or "source non renseignée"
        score = f" - score {source.score:.4f}" if source.score is not None else ""
        lines.append(f"- [{index}] `{article}` - {origin}{score}")

    return "\n".join(lines)


def build_corpus_status(config: AppConfig) -> str:
    """Return a concise status line about corpus freshness."""

    source = config.corpus.source or "source non renseignée"
    corpus_date_str = config.corpus.date

    if not corpus_date_str:
        return f"Corpus : {source} | Date : non renseignée"

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


def build_corpus_freshness(config: AppConfig) -> str:
    """Return only the freshness part of the corpus status."""

    corpus_date_str = config.corpus.date

    if not corpus_date_str:
        return "Fraîcheur : non renseignée"

    try:
        corpus_date = datetime.fromisoformat(corpus_date_str).date()
    except Exception:
        return f"Fraîcheur : date invalide ({corpus_date_str})"

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
    return f"Fraîcheur : {age_text} | Risque : {risk}"


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
        st.markdown("<div class='sidebar-title'>Assistant Code du travail</div>", unsafe_allow_html=True)
        st.caption("RAG documentaire et réponses sourcées")

        if not config.llm.api_key:
            st.warning("GROQ_API_KEY n'est pas définie.")

        st.markdown(
            f"""
            <div class='sidebar-card'>
                <div class='sidebar-card__label'>État du corpus</div>
                <div class='sidebar-card__value'>{config.corpus.source or 'non renseigné'}</div>
                <div class='sidebar-card__grid'>
                    <div class='sidebar-card__field'>
                        <div class='sidebar-card__kicker'>Date du corpus</div>
                        <div class='sidebar-card__meta'>{config.corpus.date or 'non renseignée'}</div>
                    </div>
                    <div class='sidebar-card__field'>
                        <div class='sidebar-card__kicker'>Top-k</div>
                        <div class='sidebar-card__meta'>{config.retrieval.top_k}</div>
                    </div>
                    <div class='sidebar-card__field sidebar-card__field--full'>
                        <div class='sidebar-card__kicker'>Fraîcheur</div>
                        <div class='sidebar-card__meta'>{build_corpus_freshness(config)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**▸ Conversations**")
        conversation_names = [item["name"] for item in st.session_state[CONVERSATIONS_KEY]]
        selected_name = st.radio(
            "Choisir une conversation",
            conversation_names,
            index=st.session_state[CURRENT_CONVERSATION_INDEX_KEY],
            label_visibility="collapsed",
        )
        st.session_state[CURRENT_CONVERSATION_INDEX_KEY] = conversation_names.index(selected_name)

        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("Nouvelle", use_container_width=True):
                _create_new_conversation(st)
                st.rerun()
        with col_clear:
            if st.button("Vider", use_container_width=True):
                st.session_state[CONVERSATIONS_KEY][st.session_state[CURRENT_CONVERSATION_INDEX_KEY]][
                    "messages"
                ] = []
                st.rerun()

        st.divider()
        st.markdown("**▸ Thèmes couverts**")
        st.markdown(
            """
            - Durée légale du travail
            - Congés payés et acquisition
            - Préavis en CDI
            - Rupture conventionnelle
            - Fusion-acquisition et contrat de travail
            """
        )

        st.markdown("**▸ Questions rapides**")
        for preset in QUESTION_PRESETS:
            if st.button(preset, use_container_width=True, key=f"preset::{preset}"):
                st.session_state[PRESET_QUESTION_KEY] = preset
                st.rerun()

        st.markdown(
            """
            <div class='legal-badge'>
                Cet assistant ne fournit pas de conseil juridique.
                Consultez un avocat ou l'inspection du travail pour votre situation personnelle.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.title("Assistant Code du travail")
    st.caption("Posez une question sur le droit du travail français.")

    current_messages = st.session_state[CONVERSATIONS_KEY][st.session_state[CURRENT_CONVERSATION_INDEX_KEY]][
        "messages"
    ]
    for message in current_messages:
        _render_message(st, message)

    question = st.session_state.pop(PRESET_QUESTION_KEY, None)
    if question is None:
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
        st.session_state[CONVERSATIONS_KEY].append({"name": "Conversation 1", "messages": []})

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
        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .sidebar-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            margin: 0.75rem 0 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .sidebar-card__label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .sidebar-card__value {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .sidebar-card__grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem 0.85rem;
            margin-top: 0.65rem;
        }
        .sidebar-card__field--full {
            grid-column: 1 / -1;
        }
        .sidebar-card__kicker {
            font-size: 0.72rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.15rem;
        }
        .sidebar-card__meta {
            font-size: 0.88rem;
            color: #475569;
            line-height: 1.45;
            word-break: break-word;
        }
        .stButton button {
            border-radius: 0.95rem;
        }
        .stSidebar .block-container {
            padding-top: 1rem;
        }
        .legal-badge {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #0f172a;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: #334155;
            font-size: 0.95rem;
            line-height: 1.6;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
