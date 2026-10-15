import math

from recsys.metrics import dcg, mrr, ndcg_at_k, recall_at_k


def test_dcg_matches_formula():
    assert math.isclose(dcg([3, 2, 0], 3), (2**3 - 1) / 1 + (2**2 - 1) / math.log2(3))


def test_ndcg_perfect_and_worst():
    gains = [3, 2, 1, 0]
    assert ndcg_at_k(sorted(gains, reverse=True), gains, 4) == 1.0
    assert ndcg_at_k([0, 0], gains, 2) == 0.0
    assert ndcg_at_k([0, 0], [0, 0], 2) == 0.0


def test_ndcg_is_order_sensitive():
    gains = [3, 0, 1]
    assert ndcg_at_k([3, 1, 0], gains, 3) > ndcg_at_k([0, 1, 3], gains, 3)


def test_mrr_and_recall():
    assert mrr([0, 1, 2, 3]) == 1 / 3
    assert mrr([0, 0]) == 0.0
    assert recall_at_k([2, 0, 3], [2, 3, 3, 0], 2) == 1 / 3
    assert recall_at_k([0], [0, 1], 1) == 0.0
