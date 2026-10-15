import math

import numpy as np

from recsys import ab_sim, evaluate
from recsys.ab_sim import simulate_session, two_proportion_z


def test_ltr_beats_bm25_on_heldout_queries():
    report = evaluate.run(n_items=600, n_queries=150)
    r = report["results"]
    assert r["hybrid + LightGBM LambdaMART"]["ndcg@10"] > r["bm25 retrieval"]["ndcg@10"]
    assert r["hybrid + LightGBM LambdaMART"]["ndcg@10"] >= r["hybrid + logistic regression"]["ndcg@10"] - 0.01
    assert "rating" in report["feature_importance"]
    md = evaluate.to_markdown(report)
    assert md.startswith("| System |")


def test_click_model_prefers_relevant_top_results():
    rng = np.random.default_rng(0)
    good = np.mean([simulate_session([3] * 10, rng) for _ in range(3000)])
    bad = np.mean([simulate_session([0] * 10, rng) for _ in range(3000)])
    assert good > bad + 0.3


def test_z_test_sanity():
    z, p = two_proportion_z(500, 1000, 500, 1000)
    assert z == 0 and math.isclose(p, 1.0)
    z, p = two_proportion_z(400, 1000, 500, 1000)
    assert z > 4 and p < 1e-4


def test_ab_simulation_detects_lift():
    r = ab_sim.run(sessions=4000, seed=1)
    assert r["treatment"]["ctr"] > r["control"]["ctr"]
    assert r["ci95"][0] < r["abs_lift"] < r["ci95"][1]
