"""Grounded answer generation (Groq) with conversational memory."""
from __future__ import annotations

from dataclasses import dataclass, field

from config import GROQ_API_KEY, GROQ_MODEL, MAX_HISTORY_TURNS, TOP_K
from .retrieval import Result, get_retriever

SYSTEM_PROMPT = """You are The Unofficial Guide, a digital personal assistant \
that answers questions ONLY using the documents provided as CONTEXT.

Rules you must follow:
1. Answer ONLY using the numbered CONTEXT passages provided. Do not use any \
outside or prior knowledge, even if you are confident.
2. If the user greets you (e.g., "hi", "hello"), return a greeting and say that \
you are a digital assistant for answering questions regarding the documents.
3. If the question is unrelated to the documents or the CONTEXT does not contain \
enough information, reply: "I can only help answer questions regarding these \
documents or help with the documents."
4. Cite the sources you used inline with their bracket numbers, e.g. [1], [2].
5. Be concise and concrete. Prefer the wording and code from the passages.
"""

CONTEXTUALIZE_PROMPT = """Given the conversation so far and a follow-up message, \
rewrite the follow-up as a standalone question that can be understood without \
the history (resolve pronouns like "it"/"that"). If it is already standalone, \
return it unchanged. Output ONLY the rewritten question, nothing else."""


@dataclass
class Turn:
    question: str
    answer: str


def _client():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    from groq import Groq

    return Groq(api_key=GROQ_API_KEY)


def _format_context(results: list[Result]) -> tuple[str, list[dict]]:
    blocks, sources = [], []
    for i, r in enumerate(results, start=1):
        m = r.metadata
        label = f"{m.get('source')} — {m.get('section_path', m.get('section'))}"
        blocks.append(f"[{i}] (source: {label})\n{r.text}")
        sources.append(
            {
                "n": i,
                "source": m.get("source"),
                "section": m.get("section_path", m.get("section")),
                "topic": m.get("topic"),
                "score": round(r.score, 4),
            }
        )
    return "\n\n".join(blocks), sources


def contextualize(question: str, history: list[Turn]) -> str:
    """Rewrite a follow-up into a standalone query using conversation memory."""
    if not history:
        return question
    convo = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in history)
    client = _client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=128,
        messages=[
            {"role": "system", "content": CONTEXTUALIZE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{convo}\n\nFollow-up: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip() or question


def ask(
    question: str,
    where: dict | None = None,
    history: list[Turn] | None = None,
    top_k: int = TOP_K,
    mode: str = "hybrid",
) -> dict:
    """Retrieve, ground, and generate an answer with source attribution."""
    history = history or []
    standalone = contextualize(question, history)
    results = get_retriever().search(standalone, where=where, top_k=top_k, mode=mode)

    if not results:
        return {
            "answer": "No relevant information was found for your request.",
            "sources": [],
            "results": [],
            "standalone_question": standalone,
        }

    context, sources = _format_context(results)
    client = _client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {standalone}\n\nAnswer:",
            },
        ],
    )
    answer = resp.choices[0].message.content.strip()
    return {
        "answer": answer,
        "sources": sources,
        "results": results,
        "standalone_question": standalone,
    }


@dataclass
class ConversationSession:
    """Stateful multi-turn wrapper that remembers prior questions/answers."""

    where: dict | None = None
    mode: str = "hybrid"
    history: list[Turn] = field(default_factory=list)

    def ask(self, question: str) -> dict:
        result = ask(question, where=self.where, history=self.history, mode=self.mode)
        self.history.append(Turn(question=question, answer=result["answer"]))
        # Keep only the most recent turns to bound prompt size.
        self.history = self.history[-MAX_HISTORY_TURNS:]
        return result

    def reset(self):
        self.history = []
