"""Hybrid retrieval: BM25 keyword + dense semantic, fused with RRF.

Children are searched (precise), then expanded to their parent sections (rich
context) before being returned. Metadata filtering is applied to *both*
retrievers so a filter can never leak through one path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from config import CANDIDATE_K, MIN_SEMANTIC_SIMILARITY, RRF_K, TOP_K
from .embedder import embed
from .vectorstore import get_collection, load_children, load_parents

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _matches(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and val not in cond["$in"]:
                return False
            if "$eq" in cond and val != cond["$eq"]:
                return False
        elif val != cond:
            return False
    return True


@dataclass
class Result:
    parent_id: str
    text: str
    metadata: dict
    score: float
    matched_children: list[str]


class HybridRetriever:
    def __init__(self):
        self.collection = get_collection()
        self.children = load_children()
        self.parents = load_parents()
        self._child_by_id = {c["id"]: c for c in self.children}
        # BM25 index is keyword-matched against text + section breadcrumb.
        corpus = [
            _tokenize(c["text"] + " " + c["metadata"].get("section_path", ""))
            for c in self.children
        ]
        self.bm25 = BM25Okapi(corpus)

    # --- individual retrievers -------------------------------------------
    def _semantic(self, query: str, where: dict | None, k: int) -> list[tuple[str, float]]:
        qvec = embed([query])[0]
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=k,
            where=where or None,
        )
        ids = res["ids"][0] if res["ids"] else []
        dists = res.get("distances", [[]])[0] if res.get("distances") else []
        pairs: list[tuple[str, float]] = []
        for cid, dist in zip(ids, dists):
            # Chroma cosine distance = 1 - cosine similarity.
            similarity = 1.0 - float(dist)
            pairs.append((cid, similarity))
        return pairs

    def _bm25(self, query: str, where: dict | None, k: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[str] = []
        for i in ranked:
            child = self.children[i]
            if scores[i] <= 0:
                break
            if _matches(child["metadata"], where):
                out.append(child["id"])
            if len(out) >= k:
                break
        return out

    # --- fusion ----------------------------------------------------------
    @staticmethod
    def _rrf(rankings: list[list[str]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        return fused

    def search(
        self,
        query: str,
        where: dict | None = None,
        top_k: int = TOP_K,
        mode: str = "hybrid",
    ) -> list[Result]:
        """Return up to ``top_k`` parent sections ranked by fused child scores.

        ``mode`` is one of ``"hybrid"``, ``"semantic"``, or ``"bm25"`` so the
        three can be compared directly (used by the evaluation harness).
        """
        sem_pairs = self._semantic(query, where, CANDIDATE_K) if mode != "bm25" else []
        sem_scores = {cid: sim for cid, sim in sem_pairs}
        if MIN_SEMANTIC_SIMILARITY is not None:
            sem_ids = [cid for cid, sim in sem_pairs if sim >= MIN_SEMANTIC_SIMILARITY]
        else:
            sem_ids = [cid for cid, _ in sem_pairs]
        kw = self._bm25(query, where, CANDIDATE_K) if mode != "semantic" else []

        if mode == "semantic" and MIN_SEMANTIC_SIMILARITY is not None and not sem_ids:
            return []
        if mode == "semantic":
            fused = {cid: 1.0 / (RRF_K + r) for r, cid in enumerate(sem_ids, 1)}
        elif mode == "bm25":
            fused = {cid: 1.0 / (RRF_K + r) for r, cid in enumerate(kw, 1)}
        else:
            fused = self._rrf([sem_ids, kw])

        # Expand children -> parents, accumulating child scores per parent.
        parent_scores: dict[str, float] = {}
        parent_children: dict[str, list[str]] = {}
        parent_sem_max: dict[str, float] = {}
        for cid, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            child = self._child_by_id.get(cid)
            if not child:
                continue
            pid = child["parent_id"]
            parent_scores[pid] = parent_scores.get(pid, 0.0) + score
            parent_children.setdefault(pid, []).append(cid)
            if cid in sem_scores:
                parent_sem_max[pid] = max(parent_sem_max.get(pid, 0.0), sem_scores[cid])

        ranked_parents = sorted(
            parent_scores.items(), key=lambda x: x[1], reverse=True
        )
        if MIN_SEMANTIC_SIMILARITY is not None and mode == "semantic":
            ranked_parents = [
                (pid, score)
                for pid, score in ranked_parents
                if parent_sem_max.get(pid, 0.0) >= MIN_SEMANTIC_SIMILARITY
            ]
        ranked_parents = ranked_parents[:top_k]
        results: list[Result] = []
        for pid, score in ranked_parents:
            parent = self.parents.get(pid)
            if not parent:
                continue
            results.append(
                Result(
                    parent_id=pid,
                    text=parent["text"],
                    metadata=parent["metadata"],
                    score=score,
                    matched_children=parent_children[pid],
                )
            )
        return results


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
