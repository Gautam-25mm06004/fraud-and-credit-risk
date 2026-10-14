# Fraud Detection and Credit Scorecards: Cost-Aware Risk Modelling

Two classic fintech risk problems, built the way a risk team would judge them: decisions are measured in money, not only in AUC.

1. **Card-fraud detection** on 284,807 real card transactions (ULB / Worldline, 492 frauds). It uses an out-of-time split, cost-optimal alert thresholds and probability calibration after undersampling.
2. **Credit scorecard** on the South German Credit data (1,000 loans, 300 bad). It is a Weight-of-Evidence logistic scorecard scaled to points, compared with gradient boosting, with a cut-off chosen from the published cost matrix.

All numbers below come from `results/*/metrics.json`, produced by the two scripts.

## 1. Fraud detection (`run_fraud.py`)

**Split by time**: train on the first 60% of the two-day timeline (170,884 transactions, 360 frauds), choose the alert threshold on the next 20%, and test once on the last 20% (56,962 transactions, 75 frauds).

**Cost model**: a missed fraud costs its transaction amount, and each alert costs a fixed review fee of 5. Both are stated assumptions, set in `riskml/fraud.py`.

| Model | PR-AUC (out-of-time) | ROC-AUC | Test loss at cost-optimal threshold | Loss at threshold 0.5 | Frauds caught | Alerts |
|---|---|---|---|---|---|---|
| No model (approve all) | – | – | 7,729 | – | 0 / 75 | 0 |
| Logistic regression | 0.688 | – | 3,468 | 5,388 | 55 / 75 | 94 |
| **Hist. Gradient Boosting** | **0.802** | **0.985** | **2,771** | 3,038 | **57 / 75** | **73** |

* Gradient boosting with a cost-chosen threshold **cuts fraud loss on the unseen final period by 64%** and stops 68.9% of the fraud value. Reviewing the top 0.1% of transactions gives 91% precision and 68% recall.
* Choosing the threshold by cost instead of using 0.5 saves a further 9% for boosting and 36% for logistic regression.
* The random split is optimistic for the linear model: its PR-AUC is 0.724 on a random split against 0.688 out-of-time.
* **Calibration after undersampling**: training on all frauds plus 2% of legitimate transactions (3,705 rows) leaves ranking almost intact (PR-AUC 0.798). However, it inflates the probabilities: the mean predicted fraud rate is 0.41% against a true 0.13%. Applying the Dal Pozzolo et al. correction p = βpₛ / (βpₛ − pₛ + 1) brings the mean back to 0.114% and **cuts the Brier score 3.2× (0.00130 → 0.00041)**. Figure: `results/fraud/figures/fraud_pr_and_calibration.png`.

## 2. Credit scorecard (`run_scorecard.py`)

1. Bin each attribute: categories as-is, and duration, amount and age into quantile bins.
2. Compute the Weight of Evidence, WOE = ln(%good / %bad), and the Information Value of each attribute.
3. Keep the 13 attributes with IV ≥ 0.02 and fit a logistic regression on the WOE values.
4. Scale to points: PDO = 20, and 600 points at odds of 50:1. The printable points table is in `results/scorecard/scorecard_points.csv`.

Gender/marital status and foreign-worker status are **excluded on purpose** (fair-lending practice).

Validation: 5 × 5 repeated stratified cross-validation. The cost matrix is the one published with the data: accepting a bad loan costs 5, rejecting a good loan costs 1. The Bayes-optimal rule rejects when p(bad) > 1/6.

| Model | AUC | Gini | KS | Expected cost per loan |
|---|---|---|---|---|
| Accept everyone | – | – | – | 1.500 |
| Reject everyone | – | – | – | 0.700 |
| **WOE scorecard (logistic)** | 0.778 ± 0.026 | 0.557 | 0.477 | **0.530** |
| Hist. Gradient Boosting | 0.786 ± 0.029 | 0.573 | 0.480 | 0.547 |

* The interpretable scorecard is within 0.008 AUC of gradient boosting and has the **lower expected cost**.
* It cuts cost per loan by 24% against the best trivial policy.
* Strongest attributes by IV: checking-account status (0.66), credit history (0.29), duration (0.22), savings (0.19) and purpose (0.16).
* Bad rate falls monotonically across score quintiles, from 65.5% in the lowest to 4.0% in the highest (in-sample).

## Run it in VS Code (Windows)

```powershell
cd fraud-and-credit-risk
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py   # ~70 MB
python run_fraud.py               # ~1-2 minutes
python run_scorecard.py           # ~30 seconds
python -m pytest -q
```

## Layout

```
src/riskml/fraud.py       features, time split, cost, thresholding, undersampling correction
src/riskml/scorecard.py   WOE binning, Information Value, logistic scorecard, points table, KS
run_fraud.py              fraud study -> results/fraud/
run_scorecard.py          scorecard study -> results/scorecard/
tests/test_riskml.py      4 tests (calibration correction, cost, split order, scorecard points)
```

## Explaining it in an interview

* **Why PR-AUC, not accuracy or ROC-AUC?** With 0.17% fraud, a model that flags nothing is 99.8% accurate, and ROC-AUC is dominated by the huge negative class. Precision-recall reflects the alert queue an analyst actually works through.
* **Why choose the threshold by cost?** The business loses money, not F1. A missed R$500 fraud and a missed R$2 fraud are different, so the threshold is chosen on a validation period to minimise loss, then frozen.
* **Why correct probabilities after undersampling?** Undersampling changes the class prior, so raw scores overstate risk (here 3×). Downstream uses such as expected-loss pricing and limits need calibrated probabilities.
* **Why a scorecard if boosting has higher AUC?** It is within 0.008 AUC, has lower expected cost here, and every decision can be explained as points per attribute, which regulators and adverse-action notices require. Protected attributes are excluded.

## References

* A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi, "Calibrating probability with undersampling for unbalanced classification," *IEEE Symposium on Computational Intelligence and Data Mining*, 2015.
* A. Dal Pozzolo, O. Caelen, Y.-A. Le Borgne, S. Waterschoot and G. Bontempi, "Learned lessons in credit card fraud detection from a practitioner perspective," *Expert Systems with Applications*, 41(10), 2014.
* C. Elkan, "The foundations of cost-sensitive learning," *IJCAI*, 2001.
* N. Siddiqi, *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*, Wiley, 2006.
* U. Grömping, "South German Credit Data: Correcting a Widely Used Data Set," Beuth University of Applied Sciences Berlin, Report 4/2019 (UCI dataset 573).
* Data mirrors: [stat432/credit-analysis](https://github.com/stat432/credit-analysis) (card fraud), [mayurborkar/Internship-South-German-Bank-Credit-Risk](https://github.com/mayurborkar/Internship-South-German-Bank-Credit-Risk) (South German Credit).
