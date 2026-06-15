"""Parent-child chunking.

Strategy (see report.md for the full rationale):

  * A **parent** is a whole document section (one header + its body). Parents are
    NOT embedded - they are stored verbatim and handed to the LLM at answer time
    so it sees a complete, coherent block of context.
  * A **child** is a small, sentence-aligned window inside a section, sized to
    stay under the embedder's 256-token window. Children are what we embed and
    search, so similarity is computed over precise, undiluted text.

Every chunk carries metadata: the document name, the topic, the section header,
the full section breadcrumb (h1 > h2), the page, and its position. spaCy
(``en_core_web_sm``) provides the sentence boundaries we group on, and the
header detection in ``extractors`` provides the section structure we inherit as
metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import (
    CHILD_OVERLAP_TOKENS,
    CHILD_TARGET_TOKENS,
    PARENT_MAX_TOKENS,
    SPACY_MODEL,
)
from .embedder import count_tokens
from .extractors import Block, EXTRACTORS


@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    nlp = spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer"])
    return nlp


@dataclass
class Chunk:
    id: str
    text: str
    embed_text: str
    kind: str                       # "parent" | "child"
    parent_id: str | None
    metadata: dict = field(default_factory=dict)


def _slug(path: Path) -> str:
    return path.stem


def _topic(title: str, doc: str) -> str:
    title = title.replace(" Study Guide", "").strip()
    if title:
        return title
    return doc.replace("_", " ").replace(" Study Guide", "").strip()


def _split_sentences(units: list[tuple[str, str]]) -> list[str]:
    """Turn (sub_type, text) units into a flat list of atomic pieces.

    Prose is sentence-split with spaCy; code blocks stay whole so we never cut a
    snippet in half.
    """
    nlp = get_nlp()
    pieces: list[str] = []
    for sub_type, text in units:
        text = text.strip()
        if not text:
            continue
        if sub_type == "code":
            pieces.append(text)
            continue
        for sent in nlp(text).sents:
            s = sent.text.strip()
            if s:
                pieces.append(s)
    return pieces


def _window(pieces: list[str]) -> list[str]:
    """Greedily pack sentences into ~CHILD_TARGET_TOKENS windows with overlap."""
    windows: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        ptoks = count_tokens(piece)
        if cur and cur_tokens + ptoks > CHILD_TARGET_TOKENS:
            windows.append(" ".join(cur))
            # Build overlap tail from the end of the current window.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(cur):
                t = count_tokens(prev)
                if tail_tokens + t > CHILD_OVERLAP_TOKENS:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            cur = tail
            cur_tokens = tail_tokens
            continue
        cur.append(piece)
        cur_tokens += ptoks
        i += 1
    if cur:
        windows.append(" ".join(cur))
    return windows


def _select_extractor(path: Path, extractor: str) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        return "markdown"
    return extractor


def chunk_document(path: str | Path, extractor: str = "pdfplumber") -> list[Chunk]:
    """Extract a document and split it into parent + child chunks."""
    path = Path(path)
    extractor = _select_extractor(path, extractor)
    blocks: list[Block] = EXTRACTORS[extractor](path)
    doc = _slug(path)

    title = next((b.text for b in blocks if b.sub_type == "title"), doc)
    topic = _topic(title, doc)

    chunks: list[Chunk] = []
    header_stack: list[tuple[int, str]] = []   # (level, text)
    section_header = title
    pending_units: list[tuple[str, str]] = []
    parent_idx = 0

    def breadcrumb() -> str:
        parts = [t for _, t in header_stack]
        return " > ".join(parts) if parts else title

    def flush_section():
        nonlocal pending_units, parent_idx
        if not pending_units:
            return
        body = "\n".join(t for _, t in pending_units).strip()
        if not body:
            pending_units = []
            return
        crumb = breadcrumb()
        h1 = header_stack[0][1] if header_stack else title
        parent_id = f"{doc}::p{parent_idx}"
        base_meta = {
            "doc": doc,
            "topic": topic,
            "source": path.name,
            "section": section_header,
            "section_path": crumb,
            "h1": h1,
            "page": pending_units_page[0],
        }
        # Parent: full section, capped so it can't overflow LLM context.
        parent_text = f"{crumb}\n{body}".strip()
        if count_tokens(parent_text) > PARENT_MAX_TOKENS:
            parent_text = parent_text[: PARENT_MAX_TOKENS * 6]
        chunks.append(
            Chunk(
                id=parent_id,
                text=parent_text,
                embed_text=parent_text,
                kind="parent",
                parent_id=None,
                metadata={**base_meta, "kind": "parent", "position": parent_idx},
            )
        )
        # Children: sentence-aligned windows under the embedder limit.
        pieces = _split_sentences(pending_units)
        for j, win in enumerate(_window(pieces)):
            child_id = f"{parent_id}::c{j}"
            # Prepend the breadcrumb so the embedding carries section context.
            embed_text = f"{topic} — {crumb}\n{win}"
            chunks.append(
                Chunk(
                    id=child_id,
                    text=win,
                    embed_text=embed_text,
                    kind="child",
                    parent_id=parent_id,
                    metadata={
                        **base_meta,
                        "kind": "child",
                        "position": j,
                        "n_tokens": count_tokens(win),
                    },
                )
            )
        parent_idx += 1
        pending_units = []

    pending_units_page = [1]
    for b in blocks:
        if b.kind == "header":
            if b.sub_type == "title":
                continue
            flush_section()
            # Maintain breadcrumb stack by header level.
            while header_stack and header_stack[-1][0] >= b.level:
                header_stack.pop()
            header_stack.append((b.level, b.text))
            section_header = b.text
            pending_units_page = [b.page]
        else:
            if not pending_units:
                pending_units_page = [b.page]
            pending_units.append((b.sub_type, b.text))
    flush_section()
    return chunks


def chunk_corpus(documents_dir: str | Path, extractor: str = "pdfplumber") -> list[Chunk]:
    documents_dir = Path(documents_dir)
    all_chunks: list[Chunk] = []
    for path in sorted(documents_dir.glob("*.pdf")):
        all_chunks.extend(chunk_document(path, extractor=extractor))
    for path in sorted(documents_dir.glob("*.md")):
        all_chunks.extend(chunk_document(path, extractor=extractor))
    for path in sorted(documents_dir.glob("*.txt")):
        all_chunks.extend(chunk_document(path, extractor=extractor))
    return all_chunks
