"""Query/document features for the learning-to-rank stage."""
from __future__ import annotations

import numpy as np

from .data import Item
from .retrieval import Candidate
from .text import tokenize

FEATURE_NAMES = [
    "bm25", "dense", "rrf", "bm25_rank_inv", "dense_rank_inv",
    "title_overlap", "title_coverage", "log_popularity", "rating", "log_price", "query_len",
]


def featurize(query: str, cands: list[Candidate], items: list[Item]) -> np.ndarray:
    qt = set(tokenize(query))
    rows = []
    for c in cands:
        it = items[c.item_id]
        tt = set(tokenize(it.title))
        overlap = len(qt & tt)
        rows.append([
            c.bm25, c.dense, c.rrf, 1.0 / c.bm25_rank, 1.0 / c.dense_rank,
            overlap, overlap / max(len(qt), 1), np.log1p(it.popularity), it.rating, np.log1p(it.price), len(qt),
        ])
    return np.asarray(rows, dtype=np.float64)
