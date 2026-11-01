# BorewellGuard - ML module

Predicts borewell hazard priority (Low / Medium / High / Critical) and explains the top contributing factors.

## IMPORTANT: the data is SYNTHETIC
No public labelled dataset of borewell hazard reports exists, so `generate_data.py` creates
**artificial** reports. Labels come from a hand-designed hazard model (with noise and feature
interactions), not from real incidents or expert inspections. Reported metrics therefore show
that the model learns the encoded hazard logic; they are **not** real-world field accuracy.
To get real metrics, replace `data/synthetic_borewell_reports.csv` with real inspection data
(same columns) and re-run `train_model.py`.

## Run
```
pip install -r requirements.txt
python generate_data.py     # writes data/synthetic_borewell_reports.csv
python train_model.py       # trains, evaluates, saves models/ and outputs/
python predict.py           # demo prediction with explanation
```

## What it compares (held-out test set, synthetic data)
Rule-based baseline (the app's original `calculateRisk`) vs Logistic Regression, Random Forest,
Gradient Boosting. The best ML model is chosen by 5-fold cross-validated macro-F1.
Outputs: `outputs/metrics.json`, `confusion_matrix.png`, `feature_importance.png`, `model_comparison.png`.

## Critical escalation (safety rule)
Missing a truly Critical borewell is the worst error, so if P(Critical) >= a threshold the report is
raised to Critical even when another class is more likely. The threshold is chosen on out-of-fold
TRAINING predictions (target Critical recall 92%), then checked once on the held-out test set:

| Held-out test, synthetic data | Critical recall | Critical precision | Accuracy |
|---|---|---|---|
| Rule-based (original app logic) | 90% | 42% | 51% |
| Logistic Regression | 74% | 83% | 79% |
| Logistic Regression + escalation (P >= 0.25) | 91% | 73% | 79% |

Trade-off: more false alarms than plain ML (precision 83% -> 73%), far fewer than the rule-based logic.
The response flags `escalated: true` with a plain-language note.

## Explanations
`predict.py` scores each factor by switching it to a safe value (and, from a fully safe report,
switching it on), on a log-odds scale. This ignores feature interactions, so it is an
approximation, not exact attribution.

Decision support only. A human inspection is always required.
