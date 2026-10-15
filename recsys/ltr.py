"""Learning-to-rank: LightGBM LambdaMART with a scikit-learn pointwise baseline."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Query
from .features import FEATURE_NAMES, featurize
from .retrieval import HybridRetriever


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: list[int]
    item_ids: list[list[int]]


def build_dataset(retriever: HybridRetriever, queries: list[Query], k: int = 100) -> Dataset:
    xs, ys, groups, ids = [], [], [], []
    for q in queries:
        cands = retriever.retrieve(q.text, k=k)
        xs.append(featurize(q.text, cands, retriever.items))
        ys.append([q.labels.get(c.item_id, 0) for c in cands])
        groups.append(len(cands))
        ids.append([c.item_id for c in cands])
    return Dataset(np.vstack(xs), np.concatenate(ys).astype(int), groups, ids)


class LambdaMartRanker:
    def __init__(self, seed: int = 0, n_estimators: int = 200):
        self.model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
            random_state=seed, deterministic=True, force_row_wise=True, verbose=-1,
        )

    def fit(self, ds: Dataset) -> LambdaMartRanker:
        self.model.fit(ds.X, ds.y, group=ds.groups)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> dict[str, int]:
        return dict(sorted(zip(FEATURE_NAMES, map(int, self.model.feature_importances_)), key=lambda kv: -kv[1]))


class PointwiseLogReg:
    """Baseline: predict P(relevant >= 2) per document, rank by probability."""

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def fit(self, ds: Dataset) -> PointwiseLogReg:
        self.model.fit(ds.X, (ds.y >= 2).astype(int))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def rerank(retriever: HybridRetriever, ranker, query: str, k: int = 100) -> list[int]:
    cands = retriever.retrieve(query, k=k)
    scores = ranker.score(featurize(query, cands, retriever.items))
    order = np.argsort(-scores, kind="stable")
    return [cands[i].item_id for i in order]
