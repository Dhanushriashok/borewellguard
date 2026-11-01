"""
BorewellGuard backend - FastAPI + SQLite.

Run (from the BorewellRiskAI folder):
    py -m uvicorn backend.app:app --reload --port 8000
Interactive API docs:  http://localhost:8000/docs

Endpoints
    GET   /api/health                  service + model status
    POST  /api/predict                 ML risk prediction only (nothing saved)
    POST  /api/reports                 predict + save a new borewell report
    GET   /api/reports                 list reports (filters: risk, status)
    GET   /api/reports/{id}            one report
    PATCH /api/reports/{id}/status     update inspection status
    GET   /api/stats                   counts by risk level and status

The ML model is trained on SYNTHETIC data (see ml/README.md). Every response that
contains a prediction says so, and a human inspection is always required.
"""

import json
import os
import sqlite3
import sys
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.predict import InvalidReportError, load_bundle, predict_risk  # noqa: E402

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Pending inspection", "Inspection completed", "Verified safe"]

# Values used when a citizen cannot supply a field. Reported back to the caller
# in `assumed_defaults` so nobody mistakes them for real observations.
DEFAULTS = {
    "nearby_population": 10,
    "children_nearby": 0,
    "abandoned": 0,
    "fenced_or_marked": 0,
    "borewell_age_years": 10.0,
    "depth_ft": 400.0,
    "rainfall_30d_mm": 60.0,
    "days_since_inspection": 150.0,
    "prior_incidents_2km": 0,
}

CoverCondition = Literal["Uncovered", "Damaged cover", "Covered", "Unknown"]
NearbyAccess = Literal[
    "Near public pathway",
    "Near homes or school",
    "Near farmland",
    "Restricted access",
    "Unknown",
]


class ReportIn(BaseModel):
    """What the report form sends. Only the first three fields are required."""

    location: str = Field(min_length=3, max_length=200)
    cover_condition: CoverCondition
    nearby_access: NearbyAccess
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    nearby_population: Optional[int] = Field(default=None, ge=0, le=5000)
    children_nearby: Optional[bool] = None
    abandoned: Optional[bool] = None
    fenced_or_marked: Optional[bool] = None
    borewell_age_years: Optional[float] = Field(default=None, ge=0, le=100)
    depth_ft: Optional[float] = Field(default=None, ge=0, le=3000)
    rainfall_30d_mm: Optional[float] = Field(default=None, ge=0, le=2000)
    days_since_inspection: Optional[float] = Field(default=None, ge=0, le=20000)
    prior_incidents_2km: Optional[int] = Field(default=None, ge=0, le=100)


class PredictIn(BaseModel):
    """Same as ReportIn but without a location (prediction only)."""

    cover_condition: CoverCondition
    nearby_access: NearbyAccess
    nearby_population: Optional[int] = Field(default=None, ge=0, le=5000)
    children_nearby: Optional[bool] = None
    abandoned: Optional[bool] = None
    fenced_or_marked: Optional[bool] = None
    borewell_age_years: Optional[float] = Field(default=None, ge=0, le=100)
    depth_ft: Optional[float] = Field(default=None, ge=0, le=3000)
    rainfall_30d_mm: Optional[float] = Field(default=None, ge=0, le=2000)
    days_since_inspection: Optional[float] = Field(default=None, ge=0, le=20000)
    prior_incidents_2km: Optional[int] = Field(default=None, ge=0, le=100)


class StatusIn(BaseModel):
    status: Literal["Pending inspection", "Inspection completed", "Verified safe"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Free hosting wipes local files whenever the service restarts, so an empty database
    # would make a live demo look broken. SEED_DEMO=1 adds the labelled "(demo)" reports,
    # and only when the database has no reports yet.
    if os.environ.get("SEED_DEMO") == "1":
        from backend.seed_demo import main as seed_demo

        seed_demo()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="BorewellGuard API",
    version="0.3.0",
    description="Borewell hazard reporting with ML risk prediction. "
    "Model trained on SYNTHETIC data. Decision support only.",
)

# Any localhost / 127.0.0.1 port is allowed (Vite moves to 5174, 5175... when 5173 is
# busy). For a deployed site set ALLOWED_ORIGINS="https://your-site.example".
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
def db_path() -> str:
    return os.environ.get("DB_PATH", str(Path(__file__).resolve().parent / "borewellguard.db"))


SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    location         TEXT NOT NULL,
    latitude         REAL,
    longitude        REAL,
    features         TEXT NOT NULL,
    assumed_defaults TEXT NOT NULL,
    risk_level       TEXT NOT NULL,
    confidence       REAL NOT NULL,
    probabilities    TEXT NOT NULL,
    top_factors      TEXT NOT NULL,
    model            TEXT NOT NULL,
    data_source      TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'Pending inspection',
    escalated        INTEGER NOT NULL DEFAULT 0,
    escalation_note  TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_risk ON reports(risk_level);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Upgrade databases created before escalation existed (keeps old reports).
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(reports)")}
    if "escalated" not in existing:
        conn.execute("ALTER TABLE reports ADD COLUMN escalated INTEGER NOT NULL DEFAULT 0")
    if "escalation_note" not in existing:
        conn.execute("ALTER TABLE reports ADD COLUMN escalation_note TEXT")
    conn.commit()
    return conn


def public_id(row_id: int) -> str:
    return f"BW-{row_id:04d}"


def parse_id(report_id: str) -> int:
    text = report_id.strip().upper()
    if text.startswith("BW-"):
        text = text[3:]
    if not text.isdigit():
        raise HTTPException(status_code=404, detail="Report not found")
    return int(text)


def row_to_report(row: sqlite3.Row) -> dict:
    return {
        "id": public_id(row["id"]),
        "created_at": row["created_at"],
        "location": row["location"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "inputs": json.loads(row["features"]),
        "assumed_defaults": json.loads(row["assumed_defaults"]),
        "risk_level": row["risk_level"],
        "confidence": row["confidence"],
        "probabilities": json.loads(row["probabilities"]),
        "top_factors": json.loads(row["top_factors"]),
        "escalated": bool(row["escalated"]),
        "escalation_note": row["escalation_note"],
        "model": row["model"],
        "data_source": row["data_source"],
        "status": row["status"],
        "disclaimer": "Decision support only. A human inspection is always required.",
    }


# --------------------------------------------------------------------------- #
# prediction helper
# --------------------------------------------------------------------------- #
def build_features(payload: BaseModel) -> tuple[dict, list[str]]:
    """Fill gaps with DEFAULTS; return (model inputs, names of assumed fields)."""
    data = payload.model_dump()
    features = {
        "cover_condition": data["cover_condition"],
        "nearby_access": data["nearby_access"],
    }
    assumed = []
    for name, default in DEFAULTS.items():
        value = data.get(name)
        if value is None:
            features[name] = default
            assumed.append(name)
        else:
            features[name] = int(value) if isinstance(value, bool) else value
    return features, assumed


def run_model(features: dict) -> dict:
    try:
        return predict_risk(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"ML model not available: {exc}")
    except InvalidReportError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"ML model not available: {exc}")
    return {
        "status": "ok",
        "model": bundle["model_name"],
        "data_source": bundle.get("data_source", "unknown"),
    }


@app.post("/api/predict")
def predict(payload: PredictIn):
    features, assumed = build_features(payload)
    result = run_model(features)
    return {**result, "inputs": features, "assumed_defaults": assumed}


@app.post("/api/reports", status_code=201)
def create_report(payload: ReportIn):
    features, assumed = build_features(payload)
    result = run_model(features)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with closing(connect()) as conn, conn:
        cur = conn.execute(
            """INSERT INTO reports
               (created_at, location, latitude, longitude, features, assumed_defaults,
                risk_level, confidence, probabilities, top_factors, model, data_source,
                escalated, escalation_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                created_at,
                payload.location.strip(),
                payload.latitude,
                payload.longitude,
                json.dumps(features),
                json.dumps(assumed),
                result["risk_level"],
                result["confidence"],
                json.dumps(result["probabilities"]),
                json.dumps(result["top_factors"]),
                result["model"],
                result["data_source"],
                int(result["escalated"]),
                result["escalation_note"],
            ),
        )
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (cur.lastrowid,)).fetchone()
    return row_to_report(row)


@app.get("/api/reports")
def list_reports(
    risk: Optional[Literal["Low", "Medium", "High", "Critical"]] = None,
    status: Optional[Literal["Pending inspection", "Inspection completed", "Verified safe"]] = None,
    limit: int = Query(200, ge=1, le=1000),
):
    sql, args = "SELECT * FROM reports WHERE 1=1", []
    if risk:
        sql += " AND risk_level = ?"
        args.append(risk)
    if status:
        sql += " AND status = ?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with closing(connect()) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [row_to_report(r) for r in rows]


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    row_id = parse_id(report_id)
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (row_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return row_to_report(row)


@app.patch("/api/reports/{report_id}/status")
def update_status(report_id: str, payload: StatusIn):
    row_id = parse_id(report_id)
    with closing(connect()) as conn, conn:
        cur = conn.execute(
            "UPDATE reports SET status = ? WHERE id = ?", (payload.status, row_id)
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Report not found")
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (row_id,)).fetchone()
    return row_to_report(row)


@app.get("/api/stats")
def stats():
    with closing(connect()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        by_risk = {r: 0 for r in RISK_LEVELS}
        for r in conn.execute("SELECT risk_level, COUNT(*) c FROM reports GROUP BY risk_level"):
            by_risk[r["risk_level"]] = r["c"]
        by_status = {s: 0 for s in STATUSES}
        for r in conn.execute("SELECT status, COUNT(*) c FROM reports GROUP BY status"):
            by_status[r["status"]] = r["c"]
    return {"total": total, "by_risk": by_risk, "by_status": by_status}
