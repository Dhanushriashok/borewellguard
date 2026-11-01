"""
generate_data.py - SYNTHETIC borewell hazard dataset generator for BorewellGuard.

IMPORTANT / HONESTY NOTE
------------------------
No public, labelled dataset of borewell hazard reports with risk levels exists
that we could use. This script therefore generates SYNTHETIC data. Every row is
artificial. The risk labels come from a hand-designed hazard model (with noise
and feature interactions), NOT from real incidents or expert inspections.

What this means for reported accuracy:
  * ML accuracy on this data shows the model can learn the hazard logic that
    we encoded. It is NOT evidence of real-world field accuracy.
  * Replace data/synthetic_borewell_reports.csv with real inspection data
    (same columns) and re-run train_model.py to get honest real-world metrics.

Run:  python generate_data.py            (writes data/synthetic_borewell_reports.csv)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

COVER_OPTIONS = ["Uncovered", "Damaged cover", "Covered", "Unknown"]
ACCESS_OPTIONS = [
    "Near public pathway",
    "Near homes or school",
    "Near farmland",
    "Restricted access",
    "Unknown",
]

# Hazard weights of our hand-designed (synthetic) hazard model.
COVER_WEIGHT = {"Uncovered": 3.0, "Damaged cover": 2.0, "Covered": 0.0, "Unknown": 1.2}
ACCESS_WEIGHT = {
    "Near public pathway": 1.2,
    "Near homes or school": 1.8,
    "Near farmland": 0.8,
    "Restricted access": -0.8,
    "Unknown": 0.7,
}
# Target share of each risk level in the dataset (Low, Medium, High, Critical).
LEVEL_SHARES = [0.35, 0.30, 0.22, 0.13]


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cover = rng.choice(COVER_OPTIONS, size=n_rows, p=[0.28, 0.27, 0.35, 0.10])
    access = rng.choice(ACCESS_OPTIONS, size=n_rows, p=[0.22, 0.18, 0.28, 0.20, 0.12])

    # Nearby population (people living/passing within ~200 m) depends on surroundings.
    pop_lambda = {
        "Near public pathway": 20,
        "Near homes or school": 40,
        "Near farmland": 5,
        "Restricted access": 8,
        "Unknown": 15,
    }
    population = np.array([rng.poisson(pop_lambda[a]) for a in access])

    # Children living/playing close by (0 / 1).
    child_prob = {
        "Near public pathway": 0.25,
        "Near homes or school": 0.80,
        "Near farmland": 0.10,
        "Restricted access": 0.05,
        "Unknown": 0.20,
    }
    children = np.array([rng.binomial(1, child_prob[a]) for a in access])

    age_years = np.clip(rng.gamma(3.0, 4.0, n_rows), 1, 40)
    depth_ft = np.clip(rng.normal(400, 150, n_rows), 100, 1000)

    abandoned_prob = np.where(
        cover == "Uncovered", 0.45, np.where(cover == "Damaged cover", 0.30, 0.05)
    )
    abandoned = rng.binomial(1, abandoned_prob)

    fenced_prob = np.where(abandoned == 1, 0.10, 0.30)
    fenced = rng.binomial(1, fenced_prob)

    rainfall_30d = np.clip(rng.gamma(2.0, 40.0, n_rows), 0, 400)
    days_since_inspection = np.clip(rng.exponential(200, n_rows), 0, 900)
    prior_incidents = rng.poisson(0.3, n_rows)

    # ---- hidden hazard model that generates the labels (synthetic) ----
    cover_w = np.array([COVER_WEIGHT[c] for c in cover])
    access_w = np.array([ACCESS_WEIGHT[a] for a in access])

    exposure = (
        0.035 * np.minimum(population, 80)
        + 1.2 * children
        + 0.04 * age_years
        + 0.0015 * depth_ft
        + 0.9 * abandoned
        - 1.0 * fenced
        + 0.003 * days_since_inspection
        + 0.5 * prior_incidents
    )
    is_open = np.isin(cover, ["Uncovered", "Damaged cover"])
    rain_effect = 0.004 * rainfall_30d * is_open  # soft soil around open/damaged mouth
    open_child_interaction = 1.5 * (cover == "Uncovered") * children * (1 - fenced)

    # A proper cover strongly dampens everything except the cover term itself.
    dampen = np.where(cover == "Covered", 0.5, 1.0)
    score = cover_w + access_w + dampen * (exposure + rain_effect) + open_child_interaction
    score = score + rng.normal(0, 0.7, n_rows)  # labelling noise

    # Convert the latent score into 4 risk classes using fixed population quantiles.
    cuts = np.quantile(score, np.cumsum(LEVEL_SHARES)[:-1])
    risk_idx = np.digitize(score, cuts)
    risk_level = np.array(RISK_LEVELS)[risk_idx]

    df = pd.DataFrame(
        {
            "cover_condition": cover,
            "nearby_access": access,
            "nearby_population": population,
            "children_nearby": children,
            "borewell_age_years": age_years.round(1),
            "depth_ft": depth_ft.round(0),
            "abandoned": abandoned,
            "fenced_or_marked": fenced,
            "rainfall_30d_mm": rainfall_30d.round(1),
            "days_since_inspection": days_since_inspection.round(0),
            "prior_incidents_2km": prior_incidents,
            "risk_level": risk_level,
            "data_source": "synthetic",
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC borewell data")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else here / "data" / "synthetic_borewell_reports.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(args.rows, args.seed)
    df.to_csv(out_path, index=False)

    meta = {
        "data_source": "synthetic",
        "rows": int(len(df)),
        "seed": args.seed,
        "note": "Artificial data from a hand-designed hazard model. Not real incidents.",
        "class_counts": df["risk_level"].value_counts().to_dict(),
    }
    (out_path.parent / "DATA_NOTICE.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(df)} SYNTHETIC rows to {out_path}")
    print(df["risk_level"].value_counts().reindex(RISK_LEVELS).to_string())


if __name__ == "__main__":
    main()
