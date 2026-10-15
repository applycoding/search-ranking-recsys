"""Hybrid candidate generation: BM25 + dense, fused with reciprocal rank fusion."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .data import Item
from .embed import Embedder, LsaEmbedder


def item_text(it: Item) -> str:
    return f"{it.title}. {it.description}"


@dataclass
class Candidate:
    item_id: int
    bm25: float
    dense: float
    rrf: float
    bm25_rank: int
    dense_rank: int


class HybridRetriever:
    def __init__(self, items: list[Item], embedder: Embedder | None = None, rrf_k: int = 60):
        self.items = items
        docs = [item_text(i) for i in items]
        self.bm25 = BM25().fit(docs)
        self.embedder = (embedder or LsaEmbedder()).fit(docs)
        self.doc_vecs = self.embedder.encode(docs)
        self.rrf_k = rrf_k

    def bm25_scores(self, q: str) -> np.ndarray:
        return self.bm25.scores(q)

    def dense_scores(self, q: str) -> np.ndarray:
        return self.doc_vecs @ self.embedder.encode([q])[0]

    def retrieve(self, q: str, k: int = 100, mode: str = "hybrid") -> list[Candidate]:
        b = self.bm25_scores(q)
        d = self.dense_scores(q)
        n = len(self.items)
        b_rank = np.empty(n, dtype=int); b_rank[np.argsort(-b, kind="stable")] = np.arange(1, n + 1)
        d_rank = np.empty(n, dtype=int); d_rank[np.argsort(-d, kind="stable")] = np.arange(1, n + 1)
        rrf = 1.0 / (self.rrf_k + b_rank) + 1.0 / (self.rrf_k + d_rank)
        key = {"hybrid": rrf, "bm25": b, "dense": d}[mode]
        top = np.argsort(-key, kind="stable")[:k]
        return [Candidate(int(i), float(b[i]), float(d[i]), float(rrf[i]), int(b_rank[i]), int(d_rank[i])) for i in top]
