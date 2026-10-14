"""Card-fraud study: out-of-time evaluation, cost-based thresholds, calibration.

    python run_fraud.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from riskml.fraud import (FEATURES, REVIEW_COST, add_features, best_threshold, brier,  # noqa: E402
                          correct_undersampling, cost, models, ranking_metrics, time_split, undersample)


def load(data_dir: Path) -> pd.DataFrame:
    z = data_dir / "creditcard.csv.zip"
    if z.exists():
        with zipfile.ZipFile(z) as f:
            return pd.read_csv(f.open("creditcard.csv"))
    return pd.read_csv(data_dir / "creditcard.csv")


def main():
    out = ROOT / "results" / "fraud"; figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    df = add_features(load(ROOT / "data"))
    tr, va, te = time_split(df)
    print(f"{len(df):,} transactions, {int(df.Class.sum())} frauds | out-of-time split: "
          f"train {len(tr):,} ({int(tr.Class.sum())} fraud), val {len(va):,} ({int(va.Class.sum())}), test {len(te):,} ({int(te.Class.sum())})")
    res = {"n": int(len(df)), "n_fraud": int(df.Class.sum()), "review_cost": REVIEW_COST,
           "split": {k: {"n": int(len(d)), "fraud": int(d.Class.sum())} for k, d in [("train", tr), ("val", va), ("test", te)]},
           "models": {}}
    yte, ate = te.Class.to_numpy(), te.Amount.to_numpy()
    no_model_cost = cost(yte, ate, np.zeros(len(te), bool))
    res["test_cost_no_model"] = no_model_cost
    curves = {}
    for name, model in models().items():
        m = clone(model).fit(tr[FEATURES], tr.Class)
        s_va, s_te = m.predict_proba(va[FEATURES])[:, 1], m.predict_proba(te[FEATURES])[:, 1]
        thr, _ = best_threshold(va.Class.to_numpy(), va.Amount.to_numpy(), s_va)
        c_opt, c_half = cost(yte, ate, s_te >= thr), cost(yte, ate, s_te >= 0.5)
        flagged = s_te >= thr
        r = {"out_of_time": ranking_metrics(yte, s_te), "threshold_cost_optimal": thr,
             "test_cost_at_cost_optimal_threshold": c_opt, "test_cost_at_0.5": c_half,
             "alerts_at_cost_optimal": int(flagged.sum()),
             "frauds_caught_at_cost_optimal": int((flagged & (yte == 1)).sum()),
             "fraud_amount_caught_pct": float(100 * ate[flagged & (yte == 1)].sum() / ate[yte == 1].sum()),
             "brier_out_of_time": brier(yte, s_te)}
        # random split, same model, for contrast
        Xa, Xb, ya, yb = train_test_split(df[FEATURES], df.Class, test_size=0.2, stratify=df.Class, random_state=0)
        r["random_split"] = ranking_metrics(yb.to_numpy(), clone(model).fit(Xa, ya).predict_proba(Xb)[:, 1])
        res["models"][name] = r
        curves[name] = precision_recall_curve(yte, s_te)
        print(f"{name:21s} out-of-time PR-AUC {r['out_of_time']['pr_auc']:.3f} (random split {r['random_split']['pr_auc']:.3f}) | "
              f"cost {c_opt:,.0f} vs {c_half:,.0f} at 0.5 vs {no_model_cost:,.0f} with no model | caught {r['frauds_caught_at_cost_optimal']}/{int(yte.sum())} with {r['alerts_at_cost_optimal']} alerts")

    # Undersampling distorts probabilities; correct them and compare calibration.
    hgb = models()["HistGradientBoosting"]
    sub, beta = undersample(tr, ratio=0.02, seed=0)
    p_s = clone(hgb).fit(sub[FEATURES], sub.Class).predict_proba(te[FEATURES])[:, 1]
    p_c = correct_undersampling(p_s, beta)
    res["undersampling"] = {"beta": beta, "train_rows": int(len(sub)),
                            "brier_raw": brier(yte, p_s), "brier_corrected": brier(yte, p_c),
                            "mean_pred_raw": float(p_s.mean()), "mean_pred_corrected": float(p_c.mean()),
                            "test_fraud_rate": float(yte.mean()),
                            "pr_auc": ranking_metrics(yte, p_s)["pr_auc"]}
    u = res["undersampling"]
    print(f"undersampled (beta={beta}) HGB: Brier {u['brier_raw']:.5f} -> {u['brier_corrected']:.5f} after correction; "
          f"mean p {u['mean_pred_raw']:.4f} -> {u['mean_pred_corrected']:.5f} (true rate {u['test_fraud_rate']:.5f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, (p, r, _) in curves.items():
        axes[0].plot(r, p, label=f"{name} (AP {res['models'][name]['out_of_time']['pr_auc']:.3f})")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].set_title("Out-of-time test: precision-recall"); axes[0].legend()
    for lab, p in [("undersampled, raw", p_s), ("undersampled, corrected", p_c)]:
        fy, fx = calibration_curve(yte, np.clip(p, 0, 1), n_bins=10, strategy="quantile")
        axes[1].plot(fx, fy, "o-", label=lab)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8); axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("mean predicted probability"); axes[1].set_ylabel("observed fraud rate"); axes[1].set_title("Calibration after undersampling"); axes[1].legend()
    fig.tight_layout(); fig.savefig(figs / "fraud_pr_and_calibration.png", dpi=140); plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
