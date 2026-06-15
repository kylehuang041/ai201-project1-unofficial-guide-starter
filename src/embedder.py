"""Lazily-loaded sentence-transformers embedder shared across the pipeline."""
from __future__ import annotations

from functools import lru_cache

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    """Word-piece token count using the embedder's own tokenizer."""
    tok = get_model().tokenizer
    return len(tok.encode(text, add_special_tokens=False))


def embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vecs.tolist()
