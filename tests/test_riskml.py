"""Tests on synthetic data: `python -m pytest -q`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riskml.fraud import best_threshold, correct_undersampling, cost, time_split  # noqa: E402
from riskml.scorecard import WOEScorecard, ks_statistic  # noqa: E402


def test_undersampling_correction_inverts_prior_shift():
    # true p = 0.01; keeping 10% of negatives inflates it to p_s = 0.01/(0.01+0.99*0.1)
    p, beta = 0.01, 0.1
    p_s = p / (p + (1 - p) * beta)
    assert abs(correct_undersampling(np.array([p_s]), beta)[0] - p) < 1e-12


def test_cost_and_threshold():
    y = np.array([0, 0, 1, 1]); amt = np.array([10.0, 10.0, 100.0, 1.0])
    assert cost(y, amt, np.array([False, False, True, False]), review_cost=5) == 1 + 5
    scores = np.array([0.1, 0.2, 0.9, 0.3])
    thr, c = best_threshold(y, amt, scores, review_cost=5)
    assert c <= cost(y, amt, scores >= 0.5, review_cost=5)


def test_time_split_is_ordered():
    df = pd.DataFrame({"Time": np.random.default_rng(0).permutation(100), "Class": 0})
    a, b, c = time_split(df)
    assert a.Time.max() < b.Time.min() and b.Time.max() < c.Time.min()


def test_scorecard_learns_signal_and_points_add_up():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"status": rng.integers(1, 5, n), "duration": rng.integers(6, 60, n), "age": rng.integers(20, 70, n)})
    logit = -1.5 + 0.8 * (X.status <= 2) + 0.03 * (X.duration - 30)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    sc = WOEScorecard(n_bins=5).fit(X, y)
    p = sc.predict_proba(X)[:, 1]
    assert ks_statistic(y, -p) > 0.15
    assert sc.iv_["status"] > sc.iv_["age"]
    # summing per-attribute points reproduces the score
    B = sc._bin(X)
    tab = sc.points_table().set_index(["attribute", "bin"])["points"]
    total = sum(B[c].map(lambda b, c=c: tab.loc[(c, b)]) for c in sc.features_)
    assert np.allclose(total, sc.score_points(X), atol=0.5 * len(sc.features_))
