import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import Chunk, chunk_corpus  # noqa: E402
from config import DOCUMENTS_DIR  # noqa: E402
import src.vectorstore as vectorstore  # noqa: E402


def test_build_index_persists_children_and_parents(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    parents_path = chroma_dir / "parents.json"
    children_path = chroma_dir / "children.json"

    monkeypatch.setattr(vectorstore, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(vectorstore, "PARENTS_PATH", parents_path)
    monkeypatch.setattr(vectorstore, "_CHILDREN_PATH", children_path)
    monkeypatch.setattr(vectorstore, "embed", lambda texts: [[0.0, 1.0, 0.0] for _ in texts])

    parent = Chunk(
        id="doc::p0",
        text="Section A\nThis is the parent section.",
        embed_text="Section A\nThis is the parent section.",
        kind="parent",
        parent_id=None,
        metadata={
            "doc": "doc",
            "topic": "Doc",
            "source": "doc.txt",
            "section": "Section A",
            "section_path": "Section A",
            "h1": "Section A",
            "page": 1,
            "kind": "parent",
            "position": 0,
        },
    )
    child = Chunk(
        id="doc::p0::c0",
        text="This is the child chunk.",
        embed_text="Doc — Section A\nThis is the child chunk.",
        kind="child",
        parent_id="doc::p0",
        metadata={
            "doc": "doc",
            "topic": "Doc",
            "source": "doc.txt",
            "section": "Section A",
            "section_path": "Section A",
            "h1": "Section A",
            "page": 1,
            "kind": "child",
            "position": 0,
            "n_tokens": 7,
        },
    )

    stats = vectorstore.build_index([parent, child])
    assert stats == {"children": 1, "parents": 1}

    assert parents_path.exists()
    assert children_path.exists()

    with open(parents_path) as f:
        parents = json.load(f)
    assert "doc::p0" in parents
    assert parents["doc::p0"]["text"].startswith("Section A")

    with open(children_path) as f:
        children = json.load(f)
    assert len(children) == 1
    assert children[0]["parent_id"] == "doc::p0"


def test_print_chunks_for_keywords():
    keywords = ["chunk", "child", "dog"]
    chunks = [c for c in chunk_corpus(DOCUMENTS_DIR) if c.kind == "child"]

    for keyword in keywords:
        print(f"\n=== keyword: {keyword} ===")
        matches = [
            c
            for c in chunks
            if keyword.lower() in c.text.lower()
            or keyword.lower() in c.metadata.get("section_path", "").lower()
        ]
        if not matches:
            print("No chunks found.")
            continue
        for c in matches:
            meta = c.metadata
            print(
                f"[{meta.get('source')}] {meta.get('section_path')} "
                f"(page {meta.get('page')}, tokens {meta.get('n_tokens')})"
            )
            print(c.text)
            print("-" * 60)

    assert True
