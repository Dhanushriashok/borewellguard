"""
predict.py - risk prediction with confidence and a simple explanation.

Used by the future backend (/predict endpoint) and runnable as a demo:
    python predict.py

Explanation method: one-at-a-time "occlusion". For each input feature we replace
its value with a SAFE reference value and measure how much the model's risk score drops. Large drops = factors pushing the risk up. This is an
approximation (it ignores feature interactions) but is simple and model-agnostic.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "risk_model.joblib"
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

# Minimum effect (log-odds scale) for a factor to be shown as a reason.
MIN_FACTOR_EFFECT = 0.25

# What a "safe" value looks like for each feature (used for occlusion).
SAFE_REFERENCE = {
    "cover_condition": "Covered",
    "nearby_access": "Restricted access",
    "nearby_population": 0,
    "children_nearby": 0,
    "borewell_age_years": 5.0,
    "depth_ft": 300.0,
    "abandoned": 0,
    "fenced_or_marked": 1,
    "rainfall_30d_mm": 50.0,
    "days_since_inspection": 30.0,
    "prior_incidents_2km": 0,
}

FRIENDLY = {
    "cover_condition": "Cover condition",
    "nearby_access": "Nearby access",
    "nearby_population": "People living or passing nearby",
    "children_nearby": "Children nearby",
    "borewell_age_years": "Borewell age",
    "depth_ft": "Borewell depth",
    "abandoned": "Abandoned borewell",
    "fenced_or_marked": "No fence or warning marker",
    "rainfall_30d_mm": "Recent heavy rainfall",
    "days_since_inspection": "Long time since last inspection",
    "prior_incidents_2km": "Earlier incidents in the area",
}

# Inputs the caller must provide (validated below).
CATEGORICAL_CHOICES = {
    "cover_condition": ["Uncovered", "Damaged cover", "Covered", "Unknown"],
    "nearby_access": [
        "Near public pathway",
        "Near homes or school",
        "Near farmland",
        "Restricted access",
        "Unknown",
    ],
}
NUMERIC_RANGES = {
    "nearby_population": (0, 5000),
    "children_nearby": (0, 1),
    "borewell_age_years": (0, 100),
    "depth_ft": (0, 3000),
    "abandoned": (0, 1),
    "fenced_or_marked": (0, 1),
    "rainfall_30d_mm": (0, 2000),
    "days_since_inspection": (0, 20000),
    "prior_incidents_2km": (0, 100),
}


class InvalidReportError(ValueError):
    """Raised when the input report has missing or out-of-range fields."""


@lru_cache(maxsize=1)
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run generate_data.py then train_model.py."
        )
    return joblib.load(MODEL_PATH)


def validate_report(report: dict) -> dict:
    """Check types/ranges and return a clean copy. Raises InvalidReportError."""
    clean = {}
    for name, choices in CATEGORICAL_CHOICES.items():
        value = report.get(name)
        if value not in choices:
            raise InvalidReportError(f"'{name}' must be one of {choices}, got {value!r}")
        clean[name] = value
    for name, (low, high) in NUMERIC_RANGES.items():
        if name not in report:
            raise InvalidReportError(f"Missing field '{name}'")
        try:
            value = float(report[name])
        except (TypeError, ValueError):
            raise InvalidReportError(f"'{name}' must be a number, got {report[name]!r}")
        if not (low <= value <= high):
            raise InvalidReportError(f"'{name}' must be between {low} and {high}, got {value}")
        clean[name] = value
    return clean


def _risk_score(model, frame: pd.DataFrame) -> np.ndarray:
    """Continuous ordinal risk score = mean of the cumulative log-odds
    P(>=Medium), P(>=High), P(Critical). Unlike the expected level, it does not
    saturate when the model is very sure, so small contributions stay visible."""
    proba = model.predict_proba(frame)
    order = [list(model.classes_).index(c) for c in RISK_LEVELS]
    proba = proba[:, order]
    eps = 1e-12
    logits = []
    for k in (1, 2, 3):
        upper = np.clip(proba[:, k:].sum(axis=1), eps, 1 - eps)
        logits.append(np.log(upper) - np.log1p(-upper))
    return np.mean(logits, axis=0)


def predict_risk(report: dict, top_k: int = 3) -> dict:
    """Return predicted risk, confidence, probabilities and top risk factors."""
    clean = validate_report(report)
    bundle = load_bundle()
    model, features = bundle["model"], bundle["features"]

    row = pd.DataFrame([clean])[features]
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    probs = {c: float(proba[classes.index(c)]) for c in RISK_LEVELS}
    predicted = max(probs, key=probs.get)

    # Safety escalation: a meaningful chance of Critical is treated as Critical, because
    # missing a truly Critical borewell is the worst mistake. Older model files without a
    # threshold simply skip this step.
    threshold = bundle.get("critical_threshold")
    escalated, escalation_note = False, None
    if threshold is not None and predicted != "Critical" and probs["Critical"] >= threshold:
        escalated = True
        escalation_note = (
            f"Raised from {predicted} to Critical as a precaution: the model sees a "
            f"{probs['Critical']:.0%} chance of Critical (escalation threshold {threshold:.0%})."
        )
        predicted = "Critical"

    # Two-sided occlusion. Remove-one alone fails when the prediction is saturated
    # (several hazards together keep it at Critical even if one is removed), so we
    # average two views of each feature:
    #   remove-one: expected risk drop if ONLY this feature were made safe
    #   add-one:    expected risk rise if ONLY this feature were hazardous,
    #               starting from a fully safe report
    base_level = _risk_score(model, row)[0]
    safe_row = pd.DataFrame([SAFE_REFERENCE])[features]
    safe_level = _risk_score(model, safe_row)[0]
    factors = []
    for name in features:
        if clean[name] == SAFE_REFERENCE[name]:
            continue
        removed = row.copy()
        removed.at[0, name] = SAFE_REFERENCE[name]
        remove_drop = base_level - _risk_score(model, removed)[0]

        added = safe_row.copy()
        added.at[0, name] = clean[name]
        add_rise = _risk_score(model, added)[0] - safe_level

        factors.append((name, float(0.5 * (remove_drop + add_rise))))
    factors.sort(key=lambda t: t[1], reverse=True)

    # Ignore negligible effects (a safe borewell with 2 people nearby must not show
    # "people nearby = 100% of the risk"). Shares are relative to the meaningful factors.
    meaningful = [(n, d) for n, d in factors if d >= MIN_FACTOR_EFFECT]
    positive_total = sum(d for _, d in meaningful) or 1.0
    top = [
        {
            "feature": name,
            "label": FRIENDLY[name],
            "value": clean[name],
            "share": round(drop / positive_total, 3),
        }
        for name, drop in meaningful[:top_k]
        if drop / positive_total >= 0.05
    ]

    return {
        "risk_level": predicted,
        "confidence": round(probs[predicted], 3),
        "probabilities": {k: round(v, 3) for k, v in probs.items()},
        "top_factors": top,
        "escalated": escalated,
        "escalation_note": escalation_note,
        "model": bundle["model_name"],
        "data_source": bundle.get("data_source", "unknown"),
        "disclaimer": "Decision support only. A human inspection is always required.",
    }


if __name__ == "__main__":
    demo = {
        "cover_condition": "Uncovered",
        "nearby_access": "Near homes or school",
        "nearby_population": 45,
        "children_nearby": 1,
        "borewell_age_years": 12,
        "depth_ft": 520,
        "abandoned": 1,
        "fenced_or_marked": 0,
        "rainfall_30d_mm": 120,
        "days_since_inspection": 400,
        "prior_incidents_2km": 1,
    }
    result = predict_risk(demo)
    print(f"Predicted risk: {result['risk_level']} (confidence {result['confidence']:.0%})")
    print("Probabilities:", result["probabilities"])
    print("Top factors pushing risk up:")
    for f in result["top_factors"]:
        print(f"  - {f['label']} = {f['value']}  ({f['share']:.0%} of risk push)")
