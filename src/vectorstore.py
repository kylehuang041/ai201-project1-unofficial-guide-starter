"""ChromaDB vector store + parent/child persistence."""
from __future__ import annotations

import json
from pathlib import Path

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, PARENTS_PATH
from .chunking import Chunk
from .embedder import embed

_CHILDREN_PATH = Path(CHROMA_DIR) / "children.json"


def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    client = _client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def build_index(chunks: list[Chunk]) -> dict:
    """Embed children into Chroma; persist parents + children sidecar files."""
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = _client()
    # Fresh build each time so re-running ingestion is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    children = [c for c in chunks if c.kind == "child"]
    parents = [c for c in chunks if c.kind == "parent"]

    embeddings = embed([c.embed_text for c in children])
    collection.add(
        ids=[c.id for c in children],
        documents=[c.text for c in children],
        embeddings=embeddings,
        metadatas=[c.metadata for c in children],
    )

    PARENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARENTS_PATH, "w") as f:
        json.dump(
            {p.id: {"text": p.text, "metadata": p.metadata} for p in parents}, f
        )
    with open(_CHILDREN_PATH, "w") as f:
        json.dump(
            [
                {"id": c.id, "text": c.text, "parent_id": c.parent_id, "metadata": c.metadata}
                for c in children
            ],
            f,
        )
    return {"children": len(children), "parents": len(parents)}


def load_parents() -> dict:
    with open(PARENTS_PATH) as f:
        return json.load(f)


def load_children() -> list[dict]:
    with open(_CHILDREN_PATH) as f:
        return json.load(f)
