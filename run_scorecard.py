"""Credit scorecard study on South German Credit.

    python run_scorecard.py

Compares a WOE logistic-regression scorecard with gradient boosting under
5 x 5 repeated stratified cross-validation, then picks an approval cut-off with
the dataset's published cost matrix (accepting a bad loan costs 5, rejecting a
good one costs 1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from riskml.scorecard import EXCLUDE, NUMERIC, WOEScorecard, ks_statistic, load  # noqa: E402

COST_BAD_ACCEPTED, COST_GOOD_REJECTED = 5.0, 1.0


def expected_cost(y, p_bad, thr):
    reject = p_bad >= thr
    return float((COST_BAD_ACCEPTED * ((y == 1) & ~reject).sum() + COST_GOOD_REJECTED * ((y == 0) & reject).sum()) / len(y))


def main():
    out = ROOT / "results" / "scorecard"; out.mkdir(parents=True, exist_ok=True)
    df = load(ROOT / "data" / "SouthGermanCredit.asc")
    X, y = df.drop(columns=["bad"] + EXCLUDE), df["bad"].to_numpy()
    cat_cols = [c for c in X.columns if c not in NUMERIC]
    print(f"{len(df)} loans, {y.sum()} bad ({100*y.mean():.0f}%); {X.shape[1]} attributes (excluded: {EXCLUDE})")
    models = {
        "WOE scorecard (logistic)": WOEScorecard(n_bins=5, C=1.0),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3, categorical_features=[X.columns.get_loc(c) for c in cat_cols], random_state=0),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    res = {"n": int(len(df)), "bad_rate": float(y.mean()), "excluded_attributes": EXCLUDE, "models": {}}
    oof_p = {}
    for name, model in models.items():
        aucs, kss, costs, p_acc = [], [], [], np.zeros(len(y))
        for k, (tr, te) in enumerate(cv.split(X, y)):
            m = clone(model).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            if k < 5:
                p_acc[te] = p  # first repeat -> one out-of-fold prediction per loan
            aucs.append(roc_auc_score(y[te], p)); kss.append(ks_statistic(y[te], -p))
            # Bayes-optimal cut-off for the cost matrix: reject if p_bad > 1/(1+5)
            costs.append(expected_cost(y[te], p, COST_GOOD_REJECTED / (COST_GOOD_REJECTED + COST_BAD_ACCEPTED)))
        oof_p[name] = p_acc
        res["models"][name] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                               "gini_mean": float(2 * np.mean(aucs) - 1), "ks_mean": float(np.mean(kss)),
                               "expected_cost_per_loan": float(np.mean(costs))}
        r = res["models"][name]
        print(f"{name:26s} AUC {r['auc_mean']:.3f} ± {r['auc_std']:.3f} | Gini {r['gini_mean']:.3f} | KS {r['ks_mean']:.3f} | cost/loan {r['expected_cost_per_loan']:.3f}")
    res["cost_reject_everyone"] = float(COST_GOOD_REJECTED * (y == 0).mean())
    res["cost_accept_everyone"] = float(COST_BAD_ACCEPTED * y.mean())
    print(f"reference costs per loan: accept everyone {res['cost_accept_everyone']:.3f}, reject everyone {res['cost_reject_everyone']:.3f}")

    sc = WOEScorecard(n_bins=5).fit(X, y)
    iv = pd.Series(sc.iv_).sort_values(ascending=False)
    res["information_value"] = iv.round(3).to_dict()
    res["features_kept_iv_ge_0.02"] = sc.features_
    sc.points_table().to_csv(out / "scorecard_points.csv", index=False)
    scores = sc.score_points(X)
    res["score_range"] = [float(scores.min()), float(scores.max())]
    bands = pd.qcut(scores, 5, labels=["lowest", "low", "mid", "high", "highest"])
    res["bad_rate_by_score_quintile"] = pd.Series(y).groupby(bands, observed=True).mean().round(3).to_dict()
    print("IV top 5:", iv.head(5).round(3).to_dict())
    print("bad rate by score quintile:", res["bad_rate_by_score_quintile"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    iv.plot.barh(ax=axes[0]); axes[0].invert_yaxis(); axes[0].axvline(0.02, ls="--", c="gray")
    axes[0].set_title("Information Value (dashed: 0.02 cut)")
    pd.Series(res["bad_rate_by_score_quintile"]).plot.bar(ax=axes[1], rot=0)
    axes[1].set_ylabel("bad rate"); axes[1].set_title("Bad rate by scorecard quintile (in-sample)")
    fig.tight_layout(); fig.savefig(out / "scorecard_iv_and_bands.png", dpi=140); plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
