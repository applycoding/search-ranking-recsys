"""Offline evaluation: compare retrieval modes and rankers on held-out queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import make_catalog, make_queries, split_queries
from .ltr import LambdaMartRanker, PointwiseLogReg, build_dataset, rerank
from .metrics import mrr, ndcg_at_k, recall_at_k
from .retrieval import HybridRetriever


def score_rankings(rankings: dict[int, list[int]], queries) -> dict[str, float]:
    n10, n5, rr, rec = [], [], [], []
    for q in queries:
        gains = [q.labels.get(i, 0) for i in rankings[q.id]]
        allg = list(q.labels.values())
        n10.append(ndcg_at_k(gains, allg, 10)); n5.append(ndcg_at_k(gains, allg, 5))
        rr.append(mrr(gains)); rec.append(recall_at_k(gains, allg, 50))
    return {"ndcg@5": float(np.mean(n5)), "ndcg@10": float(np.mean(n10)), "mrr": float(np.mean(rr)), "recall@50": float(np.mean(rec))}


def run(n_items: int = 1200, n_queries: int = 300, k: int = 100, seed: int = 0) -> dict:
    items = make_catalog(n_items)
    queries = make_queries(items, n_queries)
    train, test = split_queries(queries)
    retriever = HybridRetriever(items)
    results: dict[str, dict[str, float]] = {}
    for mode in ("bm25", "dense", "hybrid"):
        results[f"{mode} retrieval"] = score_rankings({q.id: [c.item_id for c in retriever.retrieve(q.text, k, mode)] for q in test}, test)
    ds = build_dataset(retriever, train, k)
    logreg = PointwiseLogReg().fit(ds)
    lmart = LambdaMartRanker(seed=seed).fit(ds)
    results["hybrid + logistic regression"] = score_rankings({q.id: rerank(retriever, logreg, q.text, k) for q in test}, test)
    results["hybrid + LightGBM LambdaMART"] = score_rankings({q.id: rerank(retriever, lmart, q.text, k) for q in test}, test)
    return {"config": {"items": n_items, "queries": n_queries, "train": len(train), "test": len(test), "k": k},
            "results": results, "feature_importance": lmart.feature_importance()}


def to_markdown(report: dict) -> str:
    lines = ["| System | NDCG@5 | NDCG@10 | MRR | Recall@50 |", "|---|---|---|---|---|"]
    for name, m in report["results"].items():
        lines.append(f"| {name} | {m['ndcg@5']:.3f} | {m['ndcg@10']:.3f} | {m['mrr']:.3f} | {m['recall@50']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=int, default=1200)
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args()
    report = run(a.items, a.queries)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "offline_eval.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (a.out / "offline_eval.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
