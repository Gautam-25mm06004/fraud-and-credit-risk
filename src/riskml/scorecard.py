"""A points-based credit scorecard on the South German Credit data (1,000 loans,
300 bad; corrected 2019 version by Groemping).

Pipeline, as in Siddiqi's "Credit Risk Scorecards":
1. bin every attribute (categories as-is, numeric attributes into quantile bins);
2. replace each bin by its Weight of Evidence  WOE = ln(%good / %bad);
3. rank attributes by Information Value  IV = sum (%good - %bad) * WOE;
4. fit logistic regression on WOE values;
5. scale to points: score = offset + factor * ln(odds), with PDO = 20 points to
   double the odds and 600 points at odds 50:1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

# German -> English names from the dataset's code table
COLUMNS = {
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history", "verw": "purpose",
    "hoehe": "amount", "sparkont": "savings", "beszeit": "employment_duration", "rate": "installment_rate",
    "famges": "personal_status_sex", "buerge": "other_debtors", "wohnzeit": "present_residence",
    "verm": "property", "alter": "age", "weitkred": "other_installment_plans", "wohn": "housing",
    "bishkred": "number_credits", "beruf": "job", "pers": "people_liable", "telef": "telephone",
    "gastarb": "foreign_worker", "kredit": "credit_risk",
}
NUMERIC = ["duration", "amount", "age"]
# Protected or ethically sensitive attributes are excluded from the scorecard.
EXCLUDE = ["personal_status_sex", "foreign_worker"]


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+").rename(columns=COLUMNS)
    df["bad"] = (df.pop("credit_risk") == 0).astype(int)  # 0 = bad in the corrected data
    return df


class WOEScorecard(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins: int = 5, C: float = 1.0, pdo: float = 20.0, base_score: float = 600.0,
                 base_odds: float = 50.0, min_iv: float = 0.02):
        self.n_bins, self.C, self.pdo, self.base_score, self.base_odds, self.min_iv = n_bins, C, pdo, base_score, base_odds, min_iv

    def _bin(self, X: pd.DataFrame) -> pd.DataFrame:
        B = X.copy()
        for c in self.edges_:
            B[c] = np.digitize(X[c], self.edges_[c])
        return B

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.edges_ = {c: np.unique(np.quantile(X[c], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                       for c in X.columns if c in NUMERIC}
        B = self._bin(X)
        good_tot, bad_tot = (y == 0).sum(), (y == 1).sum()
        self.woe_, self.iv_ = {}, {}
        for c in B.columns:
            tab = pd.crosstab(B[c], y).reindex(columns=[0, 1], fill_value=0) + 0.5  # Laplace smoothing
            pg, pb = tab[0] / (good_tot + 0.5 * len(tab)), tab[1] / (bad_tot + 0.5 * len(tab))
            w = np.log(pg / pb)
            self.woe_[c] = w.to_dict()
            self.iv_[c] = float(((pg - pb) * w).sum())
        self.features_ = [c for c in B.columns if self.iv_[c] >= self.min_iv]
        W = self._woe(B)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(W, y)
        self.classes_ = np.array([0, 1])
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        return self

    def _woe(self, B: pd.DataFrame) -> np.ndarray:
        return np.column_stack([B[c].map(self.woe_[c]).fillna(0.0).to_numpy() for c in self.features_])

    def predict_proba(self, X: pd.DataFrame):
        p = self.lr_.predict_proba(self._woe(self._bin(X)))
        return p

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score_points(self, X: pd.DataFrame) -> np.ndarray:
        """Credit score: higher = safer. ln(odds of good) = -logit(bad)."""
        p_bad = self.predict_proba(X)[:, 1]
        log_odds_good = np.log((1 - p_bad) / p_bad)
        return self.offset_ + self.factor_ * log_odds_good

    def points_table(self) -> pd.DataFrame:
        """Points per attribute bin (the printable scorecard)."""
        n = len(self.features_)
        intercept = self.lr_.intercept_[0]
        rows = []
        for coef, c in zip(self.lr_.coef_[0], self.features_):
            for b, w in self.woe_[c].items():
                # each attribute's share of the score; minus because coef is for 'bad'
                pts = -(coef * w + intercept / n) * self.factor_ + self.offset_ / n
                rows.append({"attribute": c, "bin": b, "woe": round(w, 3), "points": round(pts, 1)})
        return pd.DataFrame(rows)


def ks_statistic(y, score) -> float:
    order = np.argsort(score)
    y = np.asarray(y)[order]
    cb, cg = np.cumsum(y) / y.sum(), np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cb - cg)))
