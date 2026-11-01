"""API tests. Run from the BorewellRiskAI folder:  py -m pytest -q"""

import pytest
from fastapi.testclient import TestClient

from backend.app import app

RISKY = {
    "location": "Demo Village - Zone A",
    "cover_condition": "Uncovered",
    "nearby_access": "Near homes or school",
    "nearby_population": 45,
    "children_nearby": True,
    "abandoned": True,
    "fenced_or_marked": False,
}
SAFE = {
    "location": "Demo Village - Zone C",
    "cover_condition": "Covered",
    "nearby_access": "Restricted access",
    "nearby_population": 2,
    "children_nearby": False,
    "abandoned": False,
    "fenced_or_marked": True,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    return TestClient(app)


def test_health_reports_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["data_source"] == "synthetic"


def test_risky_report_is_high_or_critical_with_factors(client):
    res = client.post("/api/reports", json=RISKY)
    assert res.status_code == 201
    body = res.json()
    assert body["id"].startswith("BW-")
    assert body["risk_level"] in ("High", "Critical")
    assert body["top_factors"], "explanation should not be empty for a hazardous report"
    assert body["status"] == "Pending inspection"
    assert "human inspection" in body["disclaimer"].lower()


def test_safe_report_is_low(client):
    assert client.post("/api/reports", json=SAFE).json()["risk_level"] == "Low"


def test_missing_optional_fields_are_reported_as_assumed(client):
    minimal = {
        "location": "Near the old temple",
        "cover_condition": "Damaged cover",
        "nearby_access": "Near farmland",
    }
    body = client.post("/api/reports", json=minimal).json()
    assert "depth_ft" in body["assumed_defaults"]
    assert "nearby_population" in body["assumed_defaults"]


def test_supplied_fields_are_not_marked_assumed(client):
    body = client.post("/api/reports", json=RISKY).json()
    assert "nearby_population" not in body["assumed_defaults"]
    assert "children_nearby" not in body["assumed_defaults"]


def test_reports_persist_and_list_newest_first(client):
    first = client.post("/api/reports", json=RISKY).json()
    second = client.post("/api/reports", json=SAFE).json()
    listing = client.get("/api/reports").json()
    assert [r["id"] for r in listing] == [second["id"], first["id"]]


def test_filter_by_risk(client):
    client.post("/api/reports", json=RISKY)
    client.post("/api/reports", json=SAFE)
    lows = client.get("/api/reports", params={"risk": "Low"}).json()
    assert len(lows) == 1 and lows[0]["risk_level"] == "Low"


def test_mark_inspected_updates_status(client):
    rid = client.post("/api/reports", json=RISKY).json()["id"]
    res = client.patch(f"/api/reports/{rid}/status", json={"status": "Inspection completed"})
    assert res.status_code == 200
    assert client.get(f"/api/reports/{rid}").json()["status"] == "Inspection completed"


def test_invalid_status_rejected(client):
    rid = client.post("/api/reports", json=RISKY).json()["id"]
    res = client.patch(f"/api/reports/{rid}/status", json={"status": "Whatever"})
    assert res.status_code == 422


@pytest.mark.parametrize(
    "patch",
    [
        {"cover_condition": "Open"},
        {"nearby_access": "Somewhere"},
        {"location": "ab"},
        {"depth_ft": -5},
        {"nearby_population": 999999},
        {"latitude": 123},
        {"longitude": -500},
    ],
)
def test_invalid_input_rejected(client, patch):
    res = client.post("/api/reports", json={**RISKY, **patch})
    assert res.status_code == 422


def test_missing_required_field_rejected(client):
    bad = {k: v for k, v in RISKY.items() if k != "cover_condition"}
    assert client.post("/api/reports", json=bad).status_code == 422


def test_unknown_report_returns_404(client):
    assert client.get("/api/reports/BW-9999").status_code == 404
    assert client.get("/api/reports/not-an-id").status_code == 404
    res = client.patch("/api/reports/BW-9999/status", json={"status": "Verified safe"})
    assert res.status_code == 404


def test_predict_does_not_save(client):
    payload = {k: v for k, v in RISKY.items() if k != "location"}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    assert res.json()["risk_level"] in ("High", "Critical")
    assert client.get("/api/reports").json() == []


def test_stats_counts(client):
    client.post("/api/reports", json=RISKY)
    client.post("/api/reports", json=SAFE)
    stats = client.get("/api/stats").json()
    assert stats["total"] == 2
    assert sum(stats["by_risk"].values()) == 2
    assert stats["by_status"]["Pending inspection"] == 2
