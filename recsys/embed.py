"""Dense embeddings without downloading a model.

Default: LSA (TF-IDF on word + char n-grams, reduced with TruncatedSVD), L2 normalised.
It captures some subword similarity (e.g. "runner" vs "runners") and runs on CPU in
milliseconds. A sentence-transformers encoder can be plugged in through the same interface.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder(Protocol):
    def fit(self, docs: list[str]) -> Embedder: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LsaEmbedder:
    def __init__(self, dim: int = 96, seed: int = 0):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dim, random_state=seed)

    def fit(self, docs: list[str]) -> LsaEmbedder:
        x = sparse.hstack([self.word.fit_transform(docs), self.char.fit_transform(docs)]).tocsr()
        self.svd.fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        x = sparse.hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return normalize(self.svd.transform(x))


class SentenceTransformerEmbedder:  # pragma: no cover - optional, needs a model download
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def fit(self, docs: list[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))
