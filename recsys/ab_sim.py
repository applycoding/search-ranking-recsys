"""Online A/B simulation with a position-biased cascade click model.

Each simulated user scans results top down, examines position i with probability
p_exam(i) = 1 / (i + 1) ** eta, and clicks an examined result with probability tied to its
hidden relevance grade. We compare a control ranker against a treatment ranker on CTR@10
and report a two-proportion z-test plus a bootstrap confidence interval for the lift.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .data import make_catalog, make_queries, split_queries
from .ltr import LambdaMartRanker, build_dataset, rerank
from .retrieval import HybridRetriever

CLICK_PROB = {0: 0.02, 1: 0.10, 2: 0.35, 3: 0.65}


def simulate_session(gains: list[int], rng: np.random.Generator, eta: float = 1.0, k: int = 10) -> int:
    for i, g in enumerate(gains[:k]):
        if rng.random() < 1.0 / (i + 1) ** eta and rng.random() < CLICK_PROB[g]:
            return 1  # cascade: user stops after first click
    return 0


def two_proportion_z(c1: int, n1: int, c2: int, n2: int) -> tuple[float, float]:
    p1, p2 = c1 / n1, c2 / n2
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-12
    z = (p2 - p1) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def run(sessions: int = 20000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    items = make_catalog()
    queries = make_queries(items)
    train, test = split_queries(queries)
    retriever = HybridRetriever(items)
    ranker = LambdaMartRanker().fit(build_dataset(retriever, train))
    control = {q.id: [q.labels.get(c.item_id, 0) for c in retriever.retrieve(q.text, 10, "bm25")] for q in test}
    treat = {q.id: [q.labels.get(i, 0) for i in rerank(retriever, ranker, q.text)[:10]] for q in test}
    arms = rng.integers(0, 2, size=sessions)  # 50/50 random assignment
    qs = rng.integers(0, len(test), size=sessions)
    clicks = np.array([simulate_session((treat if a else control)[test[qi].id], rng) for a, qi in zip(arms, qs)])
    c_clicks, c_n = int(clicks[arms == 0].sum()), int((arms == 0).sum())
    t_clicks, t_n = int(clicks[arms == 1].sum()), int((arms == 1).sum())
    z, p = two_proportion_z(c_clicks, c_n, t_clicks, t_n)
    boots = []
    c_arr, t_arr = clicks[arms == 0], clicks[arms == 1]
    for _ in range(500):
        boots.append(rng.choice(t_arr, t_arr.size).mean() - rng.choice(c_arr, c_arr.size).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "sessions": sessions,
        "control": {"ranker": "BM25", "n": c_n, "ctr": c_clicks / c_n},
        "treatment": {"ranker": "hybrid + LambdaMART", "n": t_n, "ctr": t_clicks / t_n},
        "abs_lift": t_clicks / t_n - c_clicks / c_n,
        "ci95": [float(lo), float(hi)],
        "z": z,
        "p_value": p,
    }


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--sessions", type=int, default=20000)
    a.add_argument("--out", type=Path, default=Path("reports"))
    args = a.parse_args()
    r = run(args.sessions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ab_simulation.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
