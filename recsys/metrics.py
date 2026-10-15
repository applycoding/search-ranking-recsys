"""Offline ranking metrics."""
from __future__ import annotations

import numpy as np


def dcg(gains: list[int] | np.ndarray, k: int) -> float:
    g = np.asarray(gains, dtype=float)[:k]
    if g.size == 0:
        return 0.0
    return float(np.sum((2**g - 1) / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(ranked_gains: list[int], all_gains: list[int], k: int = 10) -> float:
    ideal = dcg(sorted(all_gains, reverse=True), k)
    return 0.0 if ideal == 0 else dcg(ranked_gains, k) / ideal


def mrr(ranked_gains: list[int], threshold: int = 2) -> float:
    for i, g in enumerate(ranked_gains):
        if g >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_gains: list[int], all_gains: list[int], k: int, threshold: int = 2) -> float:
    total = sum(1 for g in all_gains if g >= threshold)
    if total == 0:
        return 0.0
    return sum(1 for g in ranked_gains[:k] if g >= threshold) / total
