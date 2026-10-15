import numpy as np
import pytest

from recsys.bm25 import BM25
from recsys.data import make_catalog, make_queries
from recsys.retrieval import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(make_catalog(400))


def test_bm25_ranks_matching_doc_first():
    bm = BM25().fit(["red apple pie", "green apple", "blue car engine", "apple apple apple"])
    s = bm.scores("car engine")
    assert int(np.argmax(s)) == 2
    assert bm.scores("unknownterm").sum() == 0


def test_bm25_idf_downweights_common_terms():
    bm = BM25().fit(["the cat", "the dog", "the bird", "a fish"])
    assert bm.idf[bm.vocab["the"]] < bm.idf[bm.vocab["fish"]]


def test_dense_vectors_are_normalised(retriever):
    norms = np.linalg.norm(retriever.doc_vecs, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_retrieve_returns_k_unique(retriever, mode):
    cands = retriever.retrieve("wireless noise cancelling headphones", k=25, mode=mode)
    ids = [c.item_id for c in cands]
    assert len(ids) == 25 and len(set(ids)) == 25


def test_hybrid_top_results_are_on_topic(retriever):
    cands = retriever.retrieve("trail running shoes", k=10)
    cats = [retriever.items[c.item_id].category for c in cands]
    assert cats.count("running shoes") >= 8


def test_dense_handles_synonyms(retriever):
    cands = retriever.retrieve("rucksack", k=10, mode="dense")
    cats = [retriever.items[c.item_id].category for c in cands]
    assert cats.count("backpack") >= 7


def test_data_is_deterministic():
    a = make_catalog(50)
    b = make_catalog(50)
    assert [i.title for i in a] == [i.title for i in b]
    qa = make_queries(a, 10)
    assert all(set(q.labels.values()) <= {0, 1, 2, 3} for q in qa)
