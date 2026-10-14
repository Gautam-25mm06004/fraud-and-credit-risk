"""Card-fraud detection on the ULB / Worldline dataset (284,807 transactions over
two days in September 2013, 492 frauds; V1-V28 are PCA components released by
the bank, plus Time and Amount).

What this module adds beyond "train a classifier":
* an out-of-time split (train on the first 60% of the timeline, choose the
  threshold on the next 20%, test on the last 20%), compared with a random split;
* cost-based thresholding: a missed fraud costs its Amount, an alert costs a
  fixed review fee, so the threshold is chosen to minimise money lost;
* probability calibration after undersampling, using the correction of
  Dal Pozzolo et al. (2015): p = beta*p_s / (beta*p_s - p_s + 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REVIEW_COST = 5.0  # cost of one analyst review, in the dataset's currency (assumption)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    out["hour"] = (out["Time"] // 3600) % 24
    return out


FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]


def time_split(df: pd.DataFrame, train=0.6, val=0.2):
    df = df.sort_values("Time").reset_index(drop=True)
    a, b = int(train * len(df)), int((train + val) * len(df))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def models(seed: int = 0):
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed),
    }


def undersample(df: pd.DataFrame, ratio: float, seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Keep all frauds and a random `ratio` fraction of legitimate transactions.
    Returns the sample and beta = ratio (the probability a negative is kept)."""
    rng = np.random.default_rng(seed)
    keep = (df.Class == 1) | (rng.random(len(df)) < ratio)
    return df[keep], ratio


def correct_undersampling(p_s: np.ndarray, beta: float) -> np.ndarray:
    return beta * p_s / (beta * p_s - p_s + 1.0)


def cost(y: np.ndarray, amount: np.ndarray, flag: np.ndarray, review_cost: float = REVIEW_COST) -> float:
    missed = (y == 1) & (~flag)
    return float(amount[missed].sum() + review_cost * flag.sum())


def best_threshold(y, amount, scores, review_cost: float = REVIEW_COST):
    grid = np.unique(np.quantile(scores, np.linspace(0.90, 0.99999, 400)))
    costs = [cost(y, amount, scores >= t, review_cost) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def ranking_metrics(y, scores) -> dict[str, float]:
    order = np.argsort(-scores)
    k = max(1, int(0.001 * len(y)))  # top 0.1% of transactions reviewed
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision_at_top_0.1pct": float(y[order[:k]].mean()),
        "recall_at_top_0.1pct": float(y[order[:k]].sum() / max(1, y.sum())),
    }


def brier(y, p) -> float:
    return float(brier_score_loss(y, np.clip(p, 0, 1)))
