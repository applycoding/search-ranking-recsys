"""Minimal search API: hybrid retrieval then LambdaMART rerank, trained at startup."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Query

from .data import make_catalog, make_queries, split_queries
from .ltr import LambdaMartRanker, build_dataset, rerank
from .retrieval import HybridRetriever

app = FastAPI(title="search-ranking-recsys")


@lru_cache(maxsize=1)
def engine():
    items = make_catalog()
    train, _ = split_queries(make_queries(items))
    retriever = HybridRetriever(items)
    return retriever, LambdaMartRanker().fit(build_dataset(retriever, train))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(q: str = Query(..., min_length=1, max_length=200), k: int = Query(10, ge=1, le=50), mode: str = Query("ltr", pattern="^(ltr|hybrid|bm25|dense)$")):
    retriever, ranker = engine()
    if mode == "ltr":
        ids = rerank(retriever, ranker, q)[:k]
    else:
        ids = [c.item_id for c in retriever.retrieve(q, k, mode)]
    return {"query": q, "mode": mode, "results": [{"id": i, "title": retriever.items[i].title, "category": retriever.items[i].category} for i in ids]}
