"""ML module tests. Run from the BorewellRiskAI folder:  py -m pytest -q"""

import pytest

from ml.predict import InvalidReportError, load_bundle, predict_risk, validate_report

BASE = {
    "cover_condition": "Covered",
    "nearby_access": "Restricted access",
    "nearby_population": 2,
    "children_nearby": 0,
    "borewell_age_years": 4,
    "depth_ft": 300,
    "abandoned": 0,
    "fenced_or_marked": 1,
    "rainfall_30d_mm": 40,
    "days_since_inspection": 20,
    "prior_incidents_2km": 0,
}

# Real row from the synthetic data: the model's top class is High (59.5%) but
# P(Critical) is 38.9%, above the escalation threshold, so it must be raised.
BORDERLINE = {
    "cover_condition": "Uncovered",
    "nearby_access": "Unknown",
    "nearby_population": 19,
    "children_nearby": 0,
    "borewell_age_years": 7.7,
    "depth_ft": 657,
    "abandoned": 1,
    "fenced_or_marked": 0,
    "rainfall_30d_mm": 14.1,
    "days_since_inspection": 158,
    "prior_incidents_2km": 1,
}


def test_bundle_has_sane_escalation_threshold():
    bundle = load_bundle()
    assert bundle["data_source"] == "synthetic"
    assert 0 < bundle["critical_threshold"] <= 0.5


def test_safe_borewell_is_low_and_not_escalated():
    result = predict_risk(BASE)
    assert result["risk_level"] == "Low"
    assert result["escalated"] is False
    assert result["top_factors"] == []


def test_borderline_case_is_escalated_to_critical():
    result = predict_risk(BORDERLINE)
    top_class = max(result["probabilities"], key=result["probabilities"].get)
    assert top_class != "Critical"
    assert result["risk_level"] == "Critical"
    assert result["escalated"] is True
    assert result["escalation_note"]
    assert result["confidence"] == result["probabilities"]["Critical"]


def test_probabilities_sum_to_one():
    result = predict_risk(BORDERLINE)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_factor_shares_are_fractions():
    shares = [f["share"] for f in predict_risk(BORDERLINE)["top_factors"]]
    assert shares and all(0 < s <= 1 for s in shares)


@pytest.mark.parametrize(
    "patch",
    [
        {"cover_condition": "Open"},
        {"nearby_access": "Moon"},
        {"depth_ft": -1},
        {"nearby_population": 10**7},
        {"children_nearby": "yes please"},
    ],
)
def test_invalid_inputs_raise(patch):
    with pytest.raises(InvalidReportError):
        validate_report({**BASE, **patch})


def test_missing_field_raises():
    bad = {k: v for k, v in BASE.items() if k != "depth_ft"}
    with pytest.raises(InvalidReportError):
        validate_report(bad)
