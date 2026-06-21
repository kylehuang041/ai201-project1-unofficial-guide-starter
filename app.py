"""Gradio query interface for The Unofficial Guide.

Run:  python app.py   ->  http://localhost:7860

Features exposed in the UI:
  * Conversational memory (multi-turn chat; the system rewrites follow-ups).
  * Search-mode toggle: hybrid (BM25 + semantic), semantic-only, or keyword-only.
  * Metadata filtering: restrict answers to one or more source documents.
"""
from __future__ import annotations

import warnings

import gradio as gr

from config import DOCUMENTS_DIR
from src.rag import ConversationSession

warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    category=DeprecationWarning,
)

DOC_CHOICES = sorted(
    [p.name for p in DOCUMENTS_DIR.glob("*.pdf")]
    + [p.name for p in DOCUMENTS_DIR.glob("*.md")]
    + [p.name for p in DOCUMENTS_DIR.glob("*.txt")]
)


def _build_where(selected: list[str] | None) -> dict | None:
    if not selected:
        return None
    return {"source": {"$in": selected}}


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = ["| # | Document | Section | Score |", "|---|---|---|---|"]
    for s in sources:
        lines.append(
            f"| [{s['n']}] | {s['source']} | {s['section']} | {s['score']} |"
        )
    return "\n".join(lines)


def respond(message, chat_history, session, mode, doc_filter):
    if session is None:
        session = ConversationSession()
    session.mode = mode
    session.where = _build_where(doc_filter)

    result = session.ask(message)
    answer = result["answer"]
    sources_md = _format_sources(result["sources"])
    note = ""
    if result["standalone_question"].strip().lower() != message.strip().lower():
        note = f"\n\n_Interpreted as: “{result['standalone_question']}”_"

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer + note},
    ]
    return chat_history, session, sources_md, ""


def reset(session):
    if session is not None:
        session.reset()
    return [], session, "_No sources retrieved._"


with gr.Blocks(title="The Unofficial Guide — DSA RAG") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask questions about Data Structures & Algorithms. Answers are grounded "
        "in the study guides and cite their sources."
    )
    session_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=440, label="Conversation")
            msg = gr.Textbox(
                label="Your question",
                placeholder="e.g. When should I use a sliding window instead of two pointers?",
            )
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("New conversation")
        with gr.Column(scale=2):
            mode = gr.Radio(
                ["hybrid", "semantic", "bm25"],
                value="hybrid",
                label="Search mode",
                info="hybrid = BM25 keyword + semantic (recommended)",
            )
            doc_filter = gr.Dropdown(
                DOC_CHOICES,
                value=[],
                multiselect=True,
                label="Filter by document (metadata filtering)",
                info="Leave empty to search all guides.",
            )
            sources_box = gr.Markdown("_No sources retrieved._", label="Retrieved from")

    inputs = [msg, chatbot, session_state, mode, doc_filter]
    outputs = [chatbot, session_state, sources_box, msg]
    ask_btn.click(respond, inputs=inputs, outputs=outputs)
    msg.submit(respond, inputs=inputs, outputs=outputs)
    clear_btn.click(reset, inputs=session_state,
                    outputs=[chatbot, session_state, sources_box])


if __name__ == "__main__":
    demo.launch()
