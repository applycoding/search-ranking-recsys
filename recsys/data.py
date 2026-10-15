"""Deterministic synthetic product catalogue, queries and graded relevance labels.

Synthetic data keeps the demo offline and reproducible. Labels come from the generator's
hidden intent (category + attribute), so they are noisy proxies, not human judgements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CATEGORIES: dict[str, list[str]] = {
    "headphones": ["wireless", "noise cancelling", "over ear", "earbuds", "bluetooth", "studio"],
    "running shoes": ["trail", "cushioned", "lightweight", "waterproof", "racing", "wide fit"],
    "coffee": ["espresso", "decaf", "single origin", "dark roast", "organic", "beans"],
    "laptop": ["gaming", "ultralight", "16 inch", "student", "oled", "long battery"],
    "backpack": ["hiking", "travel", "laptop sleeve", "waterproof", "35l", "anti theft"],
    "smartwatch": ["gps", "heart rate", "fitness", "sleep tracking", "lte", "long battery"],
    "camera": ["mirrorless", "full frame", "vlogging", "4k", "compact", "weather sealed"],
    "desk chair": ["ergonomic", "mesh", "lumbar support", "gaming", "adjustable", "office"],
}
SYNONYMS = {"headphones": "headset", "running shoes": "runners", "laptop": "notebook", "smartwatch": "fitness watch", "backpack": "rucksack"}
BRANDS = ["acme", "nordic", "zenith", "orbit", "kestrel", "lumen", "vertex", "halo"]


@dataclass
class Item:
    id: int
    title: str
    description: str
    category: str
    attributes: list[str]
    popularity: float
    rating: float
    price: float


@dataclass
class Query:
    id: int
    text: str
    category: str
    attributes: list[str]
    labels: dict[int, int] = field(default_factory=dict)  # item id -> grade 0..3


def make_catalog(n_items: int = 1200, seed: int = 7) -> list[Item]:
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    items: list[Item] = []
    for i in range(n_items):
        cat = cats[i % len(cats)]
        attrs = list(rng.choice(CATEGORIES[cat], size=rng.integers(1, 4), replace=False))
        brand = BRANDS[rng.integers(len(BRANDS))]
        noun = SYNONYMS[cat] if cat in SYNONYMS and rng.random() < 0.3 else cat
        title = f"{brand} {' '.join(attrs[:2])} {noun}"
        # Distractor: descriptions mention a random attribute from another category,
        # so lexical matching alone picks up some wrong items.
        other = cats[(i + 1 + int(rng.integers(len(cats) - 1))) % len(cats)]
        distractor = CATEGORIES[other][int(rng.integers(len(CATEGORIES[other])))]
        desc = f"{brand} {noun} with {', '.join(attrs)} features. Pairs well with {distractor} {other} gear."
        items.append(
            Item(
                id=i,
                title=title,
                description=desc,
                category=cat,
                attributes=attrs,
                popularity=float(rng.pareto(2.0)),
                rating=float(np.clip(rng.normal(4.0, 0.6), 1, 5)),
                price=float(np.round(rng.lognormal(4, 0.6), 2)),
            )
        )
    return items


def grade(item: Item, q: Query) -> int:
    """Hidden relevance: topical match, then demoted for poorly rated items.

    The rating demotion is what a text-only retriever cannot see, so a learned ranker
    with rating and popularity features has something real to learn.
    """
    if item.category != q.category:
        return 0
    overlap = len(set(item.attributes) & set(q.attributes))
    g = 3 if overlap == len(q.attributes) else 2 if overlap > 0 else 1
    if item.rating < 3.6:
        g = max(1, g - 1)
    return g


def make_queries(items: list[Item], n_queries: int = 300, seed: int = 11) -> list[Query]:
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    queries: list[Query] = []
    for qid in range(n_queries):
        cat = cats[rng.integers(len(cats))]
        attrs = list(rng.choice(CATEGORIES[cat], size=rng.integers(1, 3), replace=False))
        noun = SYNONYMS[cat] if cat in SYNONYMS and rng.random() < 0.35 else cat
        text = f"{' '.join(attrs)} {noun}"
        q = Query(id=qid, text=text, category=cat, attributes=attrs)
        q.labels = {it.id: grade(it, q) for it in items if it.category == cat}
        queries.append(q)
    return queries


def split_queries(queries: list[Query], test_frac: float = 0.3, seed: int = 3) -> tuple[list[Query], list[Query]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(queries))
    cut = int(len(queries) * (1 - test_frac))
    return [queries[i] for i in idx[:cut]], [queries[i] for i in idx[cut:]]
