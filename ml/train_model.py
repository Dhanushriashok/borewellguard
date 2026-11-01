"""
train_model.py - train and evaluate borewell risk classifiers.

Compares:
  1. Rule-based baseline  (the calculateRisk() logic from the React app, ported to Python)
  2. Logistic Regression
  3. Random Forest
  4. Gradient Boosting (HistGradientBoosting)

Outputs (in ml/outputs and ml/models):
  metrics.json, confusion_matrix.png, feature_importance.png, model_comparison.png,
  risk_model.joblib (best model by cross-validated macro-F1)

NOTE: the dataset is SYNTHETIC (see generate_data.py). Metrics measure how well the
model learns the encoded hazard logic, not real-world field accuracy.

Run:  python train_model.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
CATEGORICAL = ["cover_condition", "nearby_access"]
NUMERIC = [
    "nearby_population",
    "children_nearby",
    "borewell_age_years",
    "depth_ft",
    "abandoned",
    "fenced_or_marked",
    "rainfall_30d_mm",
    "days_since_inspection",
    "prior_incidents_2km",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "risk_level"
SEED = 42

# Safety policy: the escalation threshold is the HIGHEST P(Critical) cut-off that still
# catches at least this share of truly Critical borewells in out-of-fold TRAINING data.
CRITICAL_RECALL_TARGET = 0.92

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "synthetic_borewell_reports.csv"
OUT_DIR = HERE / "outputs"
MODEL_DIR = HERE / "models"


def rule_based_risk(condition: str, access: str) -> str:
    """Exact port of calculateRisk() from frontend/src/App.jsx."""
    score = 0
    if condition == "Uncovered":
        score += 6
    elif condition == "Damaged cover":
        score += 4
    elif condition == "Unknown":
        score += 3

    if access == "Near public pathway":
        score += 3
    elif access == "Near homes or school":
        score += 4
    elif access == "Near farmland":
        score += 2
    elif access == "Unknown":
        score += 2

    if score >= 8:
        return "Critical"
    if score >= 5:
        return "High"
    if score >= 3:
        return "Medium"
    return "Low"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", build_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", build_preprocessor()),
                ("clf", HistGradientBoostingClassifier(max_iter=200, random_state=SEED)),
            ]
        ),
    }


def safety_metrics(y_true, y_pred) -> dict:
    """Metrics that matter for a safety triage tool."""
    t = np.array([RISK_LEVELS.index(v) for v in y_true])
    p = np.array([RISK_LEVELS.index(v) for v in y_pred])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RISK_LEVELS, average="macro")),
        "within_one_level": float(np.mean(np.abs(t - p) <= 1)),
        # Dangerous mistake: predicting a LOWER risk than the truth.
        "under_triage_rate": float(np.mean(p < t)),
        # Missing a truly Critical borewell is the worst case.
        "critical_recall": float(
            recall_score(y_true, y_pred, labels=["Critical"], average=None, zero_division=0)[0]
        ),
        # Of everything flagged Critical, how much really is (false-alarm control).
        "critical_precision": float(
            precision_score(y_true, y_pred, labels=["Critical"], average=None, zero_division=0)[0]
        ),
    }


def apply_escalation(proba, classes, threshold):
    """Most probable class, EXCEPT P(Critical) >= threshold always becomes Critical."""
    classes = list(classes)
    labels = np.array(classes, dtype=object)[proba.argmax(axis=1)]
    if threshold is not None:
        labels[proba[:, classes.index("Critical")] >= threshold] = "Critical"
    return labels


def choose_threshold(oof_proba, classes, y_true, target):
    """Highest threshold whose out-of-fold Critical recall >= target. Uses TRAINING
    data only, so the held-out test set stays an honest check."""
    is_crit = np.asarray(y_true) == "Critical"
    sweep, chosen, thr = [], None, None
    for thr in np.round(np.arange(0.50, 0.019, -0.01), 2):
        labels = apply_escalation(oof_proba, classes, thr)
        flagged = labels == "Critical"
        recall = float(np.mean(labels[is_crit] == "Critical"))
        precision = float(np.mean(is_crit[flagged])) if flagged.any() else 0.0
        sweep.append({"threshold": float(thr), "recall": round(recall, 4), "precision": round(precision, 4)})
        if chosen is None and recall >= target:
            chosen = float(thr)
    return (chosen if chosen is not None else float(thr)), sweep


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}. Run generate_data.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}  (SYNTHETIC data)")

    results = {}

    # ---- 1. rule-based baseline (what the app uses today) ----
    rule_pred = [rule_based_risk(c, a) for c, a in zip(X_test["cover_condition"], X_test["nearby_access"])]
    results["Rule-based (current app)"] = safety_metrics(y_test, rule_pred)

    # ---- 2-4. ML models ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fitted = {}
    cv_scores = {}
    for name, model in build_models().items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = safety_metrics(y_test, pred)
        results[name]["cv_macro_f1_mean"] = float(cv_f1.mean())
        results[name]["cv_macro_f1_std"] = float(cv_f1.std())
        fitted[name] = model
        cv_scores[name] = cv_f1.mean()
        print(f"{name:22s} acc={results[name]['accuracy']:.3f} "
              f"macroF1={results[name]['macro_f1']:.3f} "
              f"CV F1={cv_f1.mean():.3f}+/-{cv_f1.std():.3f}")
    print(f"{'Rule-based (current app)':22s} acc={results['Rule-based (current app)']['accuracy']:.3f} "
          f"macroF1={results['Rule-based (current app)']['macro_f1']:.3f}")

    # ---- pick best ML model by CV macro-F1 (not by test score) ----
    best_name = max(cv_scores, key=cv_scores.get)
    best = fitted[best_name]
    best_pred = best.predict(X_test)
    print(f"\nBest model (by CV macro-F1): {best_name}\n")
    print(classification_report(y_test, best_pred, labels=RISK_LEVELS, digits=3))

    # ---- Critical escalation: threshold chosen on TRAINING data only ----
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)
    threshold, sweep = choose_threshold(oof, best.classes_, y_train, CRITICAL_RECALL_TARGET)
    esc_name = f"{best_name} + Critical escalation"
    esc_pred = apply_escalation(best.predict_proba(X_test), best.classes_, threshold)
    results[esc_name] = safety_metrics(y_test, esc_pred)
    print(f"Escalation: if P(Critical) >= {threshold:.2f} -> Critical "
          f"(threshold chosen on training data, target recall {CRITICAL_RECALL_TARGET:.0%})")
    print(f"{esc_name}: acc={results[esc_name]['accuracy']:.3f} "
          f"critical_recall={results[esc_name]['critical_recall']:.3f} "
          f"critical_precision={results[esc_name]['critical_precision']:.3f}")
    print(f"Rule-based baseline:   critical_recall={results['Rule-based (current app)']['critical_recall']:.3f} "
          f"critical_precision={results['Rule-based (current app)']['critical_precision']:.3f}\n")

    # ---- confusion matrices: rule-based vs best ML ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, title, pred in [
        (axes[0], "Rule-based (current app)", rule_pred),
        (axes[1], f"ML + Critical escalation (P>={threshold:.2f})", esc_pred),
    ]:
        cm = confusion_matrix(y_test, pred, labels=RISK_LEVELS)
        ConfusionMatrixDisplay(cm, display_labels=RISK_LEVELS).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(title)
    fig.suptitle("Confusion matrix on held-out test set (SYNTHETIC data)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- model comparison bar chart ----
    names = list(results.keys())
    metrics_to_plot = ["accuracy", "macro_f1", "critical_recall", "critical_precision"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + (i - 1.5) * width, [results[n][m] for n in names], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Rule-based vs ML (test set, SYNTHETIC data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # ---- permutation importance (on original columns) ----
    perm = permutation_importance(
        best, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=SEED, n_jobs=-1
    )
    imp = (
        pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean, "std": perm.importances_std})
        .sort_values("importance", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#4c9f70")
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    ax.set_title(f"Permutation importance - {best_name} (SYNTHETIC data)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # ---- save model + metrics ----
    joblib.dump(
        {"model": best, "features": FEATURES, "classes": list(best.classes_),
         "model_name": best_name, "data_source": "synthetic",
         "critical_threshold": threshold, "critical_recall_target": CRITICAL_RECALL_TARGET},
        MODEL_DIR / "risk_model.joblib",
    )
    report = {
        "data_source": "synthetic",
        "best_model": best_name,
        "critical_escalation": {
            "rule": "P(Critical) >= threshold -> Critical",
            "threshold": threshold,
            "target_recall_on_training_oof": CRITICAL_RECALL_TARGET,
            "oof_sweep": sweep,
        },
        "test_rows": int(len(X_test)),
        "results": results,
        "permutation_importance": imp.sort_values("importance", ascending=False)
        .round(4).to_dict(orient="records"),
        "caveat": "Metrics are on SYNTHETIC data; they do not measure real-world accuracy.",
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model to {MODEL_DIR / 'risk_model.joblib'}")
    print(f"Saved metrics and plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
