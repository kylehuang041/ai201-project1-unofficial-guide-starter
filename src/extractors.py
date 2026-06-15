"""Document text extraction.

Two interchangeable PDF extractors are provided so they can be benchmarked
head-to-head (see report.md):

  * ``extract_pdfplumber`` - fast, lightweight, no structural labels. We add a
    regex-based header detector tuned to these study guides' numbered headings.
  * ``extract_spacy_layout`` - layout-aware (docling under the hood). Returns
    ``section_header`` / ``list_item`` / ``text`` labels for free, at the cost
    of heavyweight models and ~40x slower extraction.

Both PDF extractors return a list of :class:`Block` so the rest of the pipeline
is agnostic to which extractor produced the text. Markdown/text ingestion uses a
lightweight parser that recognizes ``#`` headings and paragraph breaks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# Footer that Copilot stamps on every page, optionally followed by a page no.
_FOOTER_RE = re.compile(r"^\s*Copilot may make mistakes\s*\d*\s*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[\u25aa\u2022\u25cf\u2023\-]\s*$")  # lone bullet glyph
_PAGENUM_RE = re.compile(r"^\s*\d+\s*$")
# Numbered headings: "1. Overview", "2.1 Two-Pointer Traversal", "3.1 1D DP".
# The title may start with a digit ("1D DP") so we allow alphanumerics and
# enforce "looks like a heading" constraints in _classify_header instead.
_HEADER_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Za-z0-9].{0,80})$")
# Lines that look like source code rather than prose.
_CODE_RE = re.compile(
    r"(=|==|!=|<=|>=|\+=|-=|\[|\]|\bwhile\b|\bfor\b|\bif\b|\bdef\b|\breturn\b|->|::|\.\.\.)"
)
_MD_HEADER_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+)$")


@dataclass
class Block:
    """A labelled span of document text."""

    text: str
    kind: str            # "header" | "content"
    sub_type: str        # "title" | "paragraph" | "list_item" | "code"
    level: int           # header depth (0 = doc title, 1 = H1, ...); 99 for content
    page: int


def _is_codey(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # Short symbol-heavy lines, or lines starting with a control keyword.
    return bool(_CODE_RE.search(stripped)) and (
        len(stripped.split()) <= 8 or stripped.endswith(":") or stripped == "..."
    )


def _classify_header(line: str) -> tuple[int, str] | None:
    m = _HEADER_RE.match(line.strip())
    if not m:
        return None
    title = m.group("title").strip()
    # Reject prose that merely starts with a number (e.g. "70% of problems...").
    if title.endswith((".", ",", ";", ":")) or len(title.split()) > 10:
        return None
    # Real headings are title-cased: require at least one uppercase letter.
    if not any(ch.isupper() for ch in title):
        return None
    level = m.group("num").count(".") + 1
    return level, line.strip()


def _clean_lines(raw: str) -> list[str]:
    out: list[str] = []
    prev = None
    for line in raw.splitlines():
        if _FOOTER_RE.match(line) or _BULLET_RE.match(line) or _PAGENUM_RE.match(line):
            continue
        stripped = line.rstrip()
        if not stripped.strip():
            out.append("")  # preserve paragraph breaks
            prev = ""
            continue
        if stripped.strip() == prev:  # drop duplicated title line
            continue
        out.append(stripped)
        prev = stripped.strip()
    return out


def extract_pdfplumber(path: str | Path) -> list[Block]:
    """Extract blocks with pdfplumber.

    ``x_tolerance=1`` is essential here: the default (3) glues words together on
    these PDFs ("Arraysarethefoundation..."). Dropping it to 1 restores spaces.
    """
    path = Path(path)
    blocks: list[Block] = []
    title_seen = False
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text(x_tolerance=1) or ""
            lines = _clean_lines(raw)
            buf: list[str] = []
            buf_type = "paragraph"

            def flush():
                nonlocal buf, buf_type
                if buf:
                    text = "\n".join(buf).strip() if buf_type == "code" else " ".join(buf).strip()
                    if text:
                        blocks.append(Block(text, "content", buf_type, 99, page_no))
                buf = []
                buf_type = "paragraph"

            for line in lines:
                if not line.strip():
                    flush()
                    continue
                header = _classify_header(line)
                if header:
                    flush()
                    level, htext = header
                    blocks.append(Block(htext, "header", "header", level, page_no))
                    continue
                # The first non-header line of the doc is the title.
                if not title_seen and not blocks:
                    blocks.append(Block(line.strip(), "header", "title", 0, page_no))
                    title_seen = True
                    continue
                line_type = "code" if _is_codey(line) else "paragraph"
                if buf and line_type != buf_type:
                    flush()
                buf_type = line_type
                buf.append(line.strip())
            flush()
    return blocks


def extract_spacy_layout(path: str | Path) -> list[Block]:
    """Extract blocks with spacy-layout (docling). Layout labels come for free."""
    import spacy
    from spacy_layout import spaCyLayout

    nlp = spacy.blank("en")
    layout = spaCyLayout(nlp)
    doc = layout(str(path))

    label_map = {
        "section_header": ("header", "header"),
        "title": ("header", "title"),
        "list_item": ("content", "list_item"),
        "code": ("content", "code"),
        "formula": ("content", "code"),
    }
    blocks: list[Block] = []
    for span in doc.spans["layout"]:
        text = span.text.strip()
        if not text or _FOOTER_RE.match(text):
            continue
        kind, sub_type = label_map.get(span.label_, ("content", "paragraph"))
        if kind == "header":
            header = _classify_header(text)
            level = header[0] if header else (0 if sub_type == "title" else 1)
        else:
            level = 99
        page = getattr(getattr(span, "_", None), "layout", None)
        page_no = getattr(page, "page_no", 1) if page else 1
        blocks.append(Block(text, kind, sub_type, level, page_no))
    return blocks


def extract_markdown(path: str | Path) -> list[Block]:
    """Extract blocks from a Markdown or plain-text file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks: list[Block] = []
    title_seen = False
    page_no = 1
    buf: list[str] = []
    buf_type = "paragraph"

    def flush():
        nonlocal buf, buf_type
        if buf:
            text_block = "\n".join(buf).strip() if buf_type == "code" else " ".join(buf).strip()
            if text_block:
                blocks.append(Block(text_block, "content", buf_type, 99, page_no))
        buf = []
        buf_type = "paragraph"

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        md_header = _MD_HEADER_RE.match(line)
        if md_header:
            flush()
            level = len(md_header.group("hashes"))
            title = md_header.group("title").strip()
            if not title_seen:
                blocks.append(Block(title, "header", "title", 0, page_no))
                title_seen = True
            else:
                blocks.append(Block(title, "header", "header", level, page_no))
            continue
        line_type = "code" if _is_codey(line) else "paragraph"
        if buf and line_type != buf_type:
            flush()
        buf_type = line_type
        buf.append(line.strip())
    flush()
    return blocks


EXTRACTORS = {
    "pdfplumber": extract_pdfplumber,
    "spacy-layout": extract_spacy_layout,
    "markdown": extract_markdown,
}
