import { useEffect, useState } from "react";
import "./App.css";

// Backend address. For deployment set VITE_API_URL in the frontend environment.
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const IS_LOCAL = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/|$)/.test(API);

const emptyForm = {
  location: "",
  cover_condition: "Uncovered",
  nearby_access: "Near public pathway",
  latitude: "",
  longitude: "",
  nearby_population: "",
  children_nearby: "",
  abandoned: "",
  fenced_or_marked: "",
  borewell_age_years: "",
  depth_ft: "",
};

const ASSUMED_LABELS = {
  nearby_population: "people nearby",
  children_nearby: "children nearby",
  abandoned: "abandoned status",
  fenced_or_marked: "fence / warning marker",
  borewell_age_years: "borewell age",
  depth_ft: "depth",
  rainfall_30d_mm: "recent rainfall",
  days_since_inspection: "time since last inspection",
  prior_incidents_2km: "earlier incidents nearby",
};

function describeError(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => `${(d.loc || []).slice(1).join(".") || "input"}: ${d.msg}`)
      .join("; ");
  }
  return "Request failed.";
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    const err = new Error(
      IS_LOCAL
        ? "Cannot reach the backend. Start it with: py -m uvicorn backend.app:app --port 8000"
        : "Cannot reach the server. It may be waking up from idle; please try again in a minute."
    );
    err.network = true;
    throw err;
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(describeError(body && body.detail) || `Request failed (${res.status})`);
    err.network = [502, 503, 504].includes(res.status); // gateway errors while a free server wakes up
    throw err;
  }
  return body;
}

function buildPayload(form) {
  const num = (v) => (v === "" ? undefined : Number(v));
  const yesNo = (v) => (v === "" ? undefined : v === "yes");
  return {
    location: form.location.trim(),
    cover_condition: form.cover_condition,
    nearby_access: form.nearby_access,
    latitude: num(form.latitude),
    longitude: num(form.longitude),
    nearby_population: num(form.nearby_population),
    children_nearby: yesNo(form.children_nearby),
    abandoned: yesNo(form.abandoned),
    fenced_or_marked: yesNo(form.fenced_or_marked),
    borewell_age_years: num(form.borewell_age_years),
    depth_ft: num(form.depth_ft),
  };
}

function percent(value) {
  return value >= 0.995 ? ">99%" : `${Math.round(value * 100)}%`;
}

// A report is "escalated" when the safety rule raised it to Critical although another
// class was more probable (see ml/README.md, "Critical escalation").
function isEscalated(report) {
  const probs = report.probabilities || {};
  const top = Object.keys(probs).reduce((a, b) => (probs[a] >= probs[b] ? a : b), "Low");
  return report.risk_level === "Critical" && top !== "Critical";
}

const precautionStyle = {
  display: "inline-block",
  marginLeft: 8,
  padding: "2px 8px",
  border: "1px solid #6b5a2a",
  background: "#2e2814",
  color: "#ffe49a",
  borderRadius: 999,
  fontSize: 11,
};

function App() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [waking, setWaking] = useState(false);
  const [filter, setFilter] = useState("All");
  const [showForm, setShowForm] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");
  const [result, setResult] = useState(null);
  const [form, setForm] = useState(emptyForm);

  // Free hosting puts the server to sleep when idle; the first request can take about a
  // minute. On a deployed site, retry quietly (up to ~90 s) instead of showing an error.
  function loadReports(attempt = 0) {
    setLoading(true);
    api("/api/reports")
      .then((data) => {
        setReports(data);
        setLoadError("");
        setWaking(false);
        setLoading(false);
      })
      .catch((err) => {
        if (err.network && !IS_LOCAL && attempt < 18) {
          setWaking(true);
          setTimeout(() => loadReports(attempt + 1), 5000);
          return;
        }
        setWaking(false);
        setLoadError(err.message);
        setLoading(false);
      });
  }

  useEffect(() => {
    loadReports();
  }, []);

  const criticalCount = reports.filter((r) => r.risk_level === "Critical").length;
  const openCount = reports.filter((r) => r.status !== "Verified safe").length;
  const verifiedCount = reports.filter((r) => r.status === "Verified safe").length;

  const visibleReports =
    filter === "All" ? reports : reports.filter((r) => r.risk_level === filter);

  function updateField(event) {
    setForm({ ...form, [event.target.name]: event.target.value });
  }

  async function submitReport(event) {
    event.preventDefault();
    setSubmitting(true);
    setFormError("");
    try {
      const created = await api("/api/reports", {
        method: "POST",
        body: JSON.stringify(buildPayload(form)),
      });
      setReports((current) => [created, ...current]);
      setResult(created);
      setFilter("All");
      setForm(emptyForm);
      setShowForm(false);
    } catch (err) {
      setFormError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function markInspected(id) {
    try {
      const updated = await api(`/api/reports/${id}/status`, {
        method: "PATCH",
        body: JSON.stringify({ status: "Inspection completed" }),
      });
      setReports((current) => current.map((r) => (r.id === id ? updated : r)));
    } catch (err) {
      setLoadError(err.message);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">⌖</div>
          <div>
            <h2>Borewell<span>Guard</span></h2>
            <p>COMMUNITY SAFETY NETWORK</p>
          </div>
        </div>

        <p className="nav-heading">WORKSPACE</p>
        <div className="nav-item active">▦ &nbsp; Overview</div>
        <div className="nav-item">⌖ &nbsp; Borewell reports</div>
        <div className="nav-item">⚑ &nbsp; Risk monitoring</div>
        <div className="nav-item">✓ &nbsp; Inspection tracking</div>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <span>Prototype environment</span>
          <p>ML model trained on synthetic data · Not a live safety service</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">SAFETY INTELLIGENCE / OVERVIEW</p>
            <h1>Borewell Risk Dashboard</h1>
            <p className="subtitle">
              Identify hazards. Prioritise inspections. Track safer outcomes.
            </p>
          </div>
          <div className="prototype-badge">● PROTOTYPE</div>
        </header>

        <section className="welcome-panel">
          <div>
            <p className="welcome-label">PREVENTION FIRST</p>
            <h2>Every reported hazard deserves attention.</h2>
            <p>
              Report unsafe borewells. A machine-learning model suggests an
              inspection priority and explains why. A human inspection is
              always required.
            </p>
          </div>
          <div className="welcome-symbol" aria-hidden="true">⌖</div>
        </section>

        {waking && (
          <div className="success-message">
            The server is waking up from idle (free hosting). This can take up to a
            minute; loading automatically…
          </div>
        )}
        {loadError && <div className="error-banner">{loadError}</div>}

        <section className="metrics-grid">
          <Metric label="Registered reports" value={reports.length} note="Stored in database" />
          <Metric label="Critical priority" value={criticalCount} note="Urgent human review" />
          <Metric label="Awaiting verification" value={openCount} note="Not confirmed safe" />
          <Metric label="Verified safe" value={verifiedCount} note="Confirmed records only" />
        </section>

        <section className="section-header">
          <div>
            <h2>Borewell reports</h2>
            <p>Review existing records or submit a new report.</p>
          </div>
          <button
            className="primary-button"
            onClick={() => {
              setShowForm(!showForm);
              setFormError("");
              setResult(null);
            }}
          >
            {showForm ? "Cancel report" : "+ New report"}
          </button>
        </section>

        {result && <ResultPanel report={result} onClose={() => setResult(null)} />}

        {showForm && (
          <form className="report-form" onSubmit={submitReport}>
            <h3>Report an unsafe borewell</h3>
            <p className="form-note">
              Enter what you can see from a safe distance. Do not approach an
              open borewell to collect information.
            </p>

            {formError && <div className="error-banner">{formError}</div>}

            <label>
              Location / address *
              <input
                name="location"
                value={form.location}
                onChange={updateField}
                placeholder="Village, street or nearby landmark"
                minLength={3}
                maxLength={200}
                required
              />
            </label>

            <div className="form-grid">
              <label>
                Observed cover condition *
                <select name="cover_condition" value={form.cover_condition} onChange={updateField}>
                  <option>Uncovered</option>
                  <option>Damaged cover</option>
                  <option>Covered</option>
                  <option>Unknown</option>
                </select>
              </label>

              <label>
                Nearby access *
                <select name="nearby_access" value={form.nearby_access} onChange={updateField}>
                  <option>Near public pathway</option>
                  <option>Near homes or school</option>
                  <option>Near farmland</option>
                  <option>Restricted access</option>
                  <option>Unknown</option>
                </select>
              </label>

              <label>
                Latitude (optional)
                <input
                  name="latitude"
                  type="number"
                  step="any"
                  min="-90"
                  max="90"
                  value={form.latitude}
                  onChange={updateField}
                  placeholder="e.g. 9.9252"
                />
              </label>

              <label>
                Longitude (optional)
                <input
                  name="longitude"
                  type="number"
                  step="any"
                  min="-180"
                  max="180"
                  value={form.longitude}
                  onChange={updateField}
                  placeholder="e.g. 78.1198"
                />
              </label>
            </div>

            <details className="form-details">
              <summary>More details (optional, improves the prediction)</summary>
              <p className="form-note">
                Leave anything you don't know. Typical values are assumed and
                the result tells you which ones.
              </p>
              <div className="form-grid">
                <label>
                  People living or passing nearby
                  <input
                    name="nearby_population"
                    type="number"
                    step="1"
                    min="0"
                    max="5000"
                    value={form.nearby_population}
                    onChange={updateField}
                    placeholder="approx. number within 200 m"
                  />
                </label>

                <label>
                  Children living or playing nearby
                  <select name="children_nearby" value={form.children_nearby} onChange={updateField}>
                    <option value="">Not sure</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </label>

                <label>
                  Looks abandoned
                  <select name="abandoned" value={form.abandoned} onChange={updateField}>
                    <option value="">Not sure</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </label>

                <label>
                  Fence or warning marker present
                  <select name="fenced_or_marked" value={form.fenced_or_marked} onChange={updateField}>
                    <option value="">Not sure</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </label>

                <label>
                  Borewell age (years)
                  <input
                    name="borewell_age_years"
                    type="number"
                    step="any"
                    min="0"
                    max="100"
                    value={form.borewell_age_years}
                    onChange={updateField}
                    placeholder="if known"
                  />
                </label>

                <label>
                  Depth (feet)
                  <input
                    name="depth_ft"
                    type="number"
                    step="any"
                    min="0"
                    max="3000"
                    value={form.depth_ft}
                    onChange={updateField}
                    placeholder="if known"
                  />
                </label>
              </div>
            </details>

            <button className="primary-button" type="submit" disabled={submitting}>
              {submitting ? "Predicting and saving…" : "Submit report"}
            </button>
          </form>
        )}

        <section className="section-header list-heading">
          <div>
            <h2>Report register</h2>
            <p>ML-predicted priority. Model trained on synthetic data; verify on site.</p>
          </div>
          <label className="filter-control">
            <span>Risk level</span>
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option>All</option>
              <option>Critical</option>
              <option>High</option>
              <option>Medium</option>
              <option>Low</option>
            </select>
          </label>
        </section>

        <section className="report-list">
          {loading && <p className="empty-state">Loading reports…</p>}

          {!loading &&
            visibleReports.map((report) => (
              <article className="report-card" key={report.id}>
                <div className="report-marker">⌖</div>
                <div className="report-details">
                  <div className="report-title-row">
                    <h3>{report.location}</h3>
                    <span className={`risk-badge ${report.risk_level.toLowerCase()}`}>
                      {report.risk_level} priority
                    </span>
                  </div>
                  <p className="report-id">
                    {report.id} ·{" "}
                    {isEscalated(report)
                      ? `${percent(report.probabilities.Critical)} chance of Critical`
                      : `confidence ${percent(report.confidence)}`}
                    {isEscalated(report) && (
                      <span
                        style={precautionStyle}
                        title="The model's most likely class was lower, but the chance of Critical was high enough that the safety rule raised it."
                      >
                        Raised as a precaution
                      </span>
                    )}{" "}
                    ·{" "}
                    {report.latitude != null && report.longitude != null
                      ? `${report.latitude}, ${report.longitude}`
                      : "Coordinates not supplied"}
                  </p>
                  <div className="report-facts">
                    <span><b>Condition:</b> {report.inputs.cover_condition}</span>
                    <span><b>Access:</b> {report.inputs.nearby_access}</span>
                  </div>
                  {report.top_factors.length > 0 && (
                    <div className="chips">
                      <span className="chips-label">Why:</span>
                      {report.top_factors.map((f) => (
                        <span className="chip" key={f.feature}>
                          {f.label} ({Math.round(f.share * 100)}%)
                        </span>
                      ))}
                    </div>
                  )}
                  <p className="report-status">Status: {report.status}</p>
                </div>
                <button
                  className="inspect-button"
                  onClick={() => markInspected(report.id)}
                  disabled={report.status === "Inspection completed" || report.status === "Verified safe"}
                >
                  {report.status === "Inspection completed" ? "Inspected ✓" : "Mark inspected"}
                </button>
              </article>
            ))}

          {!loading && reports.length === 0 && !loadError && (
            <p className="empty-state">No reports yet. Use “+ New report” to submit the first one.</p>
          )}
          {!loading && reports.length > 0 && visibleReports.length === 0 && (
            <p className="empty-state">No reports match this risk level.</p>
          )}
        </section>

        <footer>
          <span>BorewellGuard · AI-assisted hazard prevention</span>
          <span>Prototype v0.3 · ML trained on synthetic data, not a live safety service</span>
        </footer>
      </main>
    </div>
  );
}

function ResultPanel({ report, onClose }) {
  const assumed = (report.assumed_defaults || []).map((k) => ASSUMED_LABELS[k] || k);
  return (
    <section className="result-panel">
      <div className="result-head">
        <div>
          <p className="welcome-label">ML PREDICTION · {report.id}</p>
          <h3>
            <span className={`risk-badge ${report.risk_level.toLowerCase()}`}>
              {report.risk_level} priority
            </span>
            <span className="confidence-text">
              {isEscalated(report)
                ? `${percent(report.probabilities.Critical)} chance of Critical`
                : `model confidence ${percent(report.confidence)}`}
            </span>
          </h3>
          {isEscalated(report) && (
            <p className="assumed-note">
              Raised to Critical as a precaution: the model's most likely class was
              lower, but the chance of Critical is high enough that missing it would
              be the worse mistake.
            </p>
          )}
        </div>
        <button className="inspect-button" onClick={onClose}>Dismiss</button>
      </div>

      {report.top_factors.length > 0 ? (
        <>
          <p className="form-note">What pushed the risk up (approximate):</p>
          {report.top_factors.map((f) => (
            <div className="factor-row" key={f.feature}>
              <span className="factor-label">{f.label}</span>
              <div className="factor-bar">
                <span style={{ width: `${Math.round(f.share * 100)}%` }} />
              </div>
              <span className="factor-value">{Math.round(f.share * 100)}%</span>
            </div>
          ))}
        </>
      ) : (
        <p className="form-note">No single hazard factor stands out for this report.</p>
      )}

      {assumed.length > 0 && (
        <p className="assumed-note">
          Typical values were assumed for: {assumed.join(", ")}. Add them in
          “More details” for a better prediction.
        </p>
      )}
      <p className="assumed-note">
        Prediction from a model trained on synthetic data. Human inspection is
        still required.
      </p>
    </section>
  );
}

function Metric({ label, value, note }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{note}</span>
    </article>
  );
}

export default App;
