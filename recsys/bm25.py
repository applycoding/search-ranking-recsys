"""Okapi BM25 over a sparse term matrix (no external search engine needed)."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]) -> BM25:
        toks = [tokenize(d) for d in docs]
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, t in enumerate(toks):
            counts: dict[int, int] = {}
            for w in t:
                j = self.vocab.setdefault(w, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i); cols.append(j); vals.append(c)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float64)
        dl = np.array([len(t) for t in toks], dtype=np.float64)
        avgdl = dl.mean() if len(dl) else 0.0
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        n = len(docs)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        # Precompute the saturated term weight per (doc, term).
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = tf.data * (self.k1 + 1) / denom
        self.weights = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        ids = [self.vocab[w] for w in tokenize(query) if w in self.vocab]
        if not ids:
            return np.zeros(self.weights.shape[0])
        sub = self.weights[:, ids]
        return np.asarray(sub @ self.idf[ids]).ravel()
