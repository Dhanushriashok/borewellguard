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

const NAV = [
  { id: "overview", icon: "▦", label: "Overview", eyebrow: "OVERVIEW", title: "Borewell Risk Dashboard",
    subtitle: "Identify hazards. Prioritise inspections. Track safer outcomes." },
  { id: "reports", icon: "⌖", label: "Borewell reports", eyebrow: "REPORTS", title: "Borewell Reports",
    subtitle: "Submit a new report and review every record." },
  { id: "risk", icon: "⚑", label: "Risk monitoring", eyebrow: "RISK MONITORING", title: "Risk Monitoring",
    subtitle: "See how reported hazards are spread across risk levels and what drives them." },
  { id: "inspection", icon: "✓", label: "Inspection tracking", eyebrow: "INSPECTION TRACKING", title: "Inspection Tracking",
    subtitle: "Work through the queue, most urgent first." },
];

const LEVELS = ["Critical", "High", "Medium", "Low"];
const LEVEL_COLORS = { Critical: "#ff6b6b", High: "#ffa84d", Medium: "#ffd24d", Low: "#b8f36a" };
const STATUSES = ["Pending inspection", "Inspection completed", "Verified safe"];
const severity = (r) => LEVELS.indexOf(r.risk_level);

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
  const [view, setView] = useState("overview");
  const [statusFilter, setStatusFilter] = useState("Pending inspection");

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

  function go(id) {
    setView(id);
    window.scrollTo({ top: 0 });
  }

  function startNewReport() {
    setShowForm(true);
    setFormError("");
    setResult(null);
    go("reports");
  }

  async function setStatus(id, status) {
    try {
      const updated = await api(`/api/reports/${id}/status`, {
        method: "PATCH",
        body: JSON.stringify({ status }),
      });
      setReports((current) => current.map((r) => (r.id === id ? updated : r)));
    } catch (err) {
      setLoadError(err.message);
    }
  }

  const current = NAV.find((n) => n.id === view);
  const byUrgency = (list) => [...list].sort((a, b) => severity(a) - severity(b));
  const attention = byUrgency(
    reports.filter((r) => ["Critical", "High"].includes(r.risk_level) && r.status === "Pending inspection")
  );

  const levelRows = LEVELS.map((l) => ({
    label: `${l} priority`,
    count: reports.filter((r) => r.risk_level === l).length,
    color: LEVEL_COLORS[l],
  }));
  const factorTally = {};
  reports.forEach((r) => (r.top_factors || []).forEach((f) => {
    factorTally[f.label] = (factorTally[f.label] || 0) + 1;
  }));
  const factorRows = Object.entries(factorTally)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([label, count]) => ({ label, count, color: "#b8f36a" }));
  const escalatedCount = reports.filter(isEscalated).length;
  const urgentPending = attention.length;
  const highShare = reports.length
    ? `${Math.round((100 * reports.filter((r) => ["Critical", "High"].includes(r.risk_level)).length) / reports.length)}%`
    : "0%";

  const statusRows = STATUSES.map((st) => ({
    label: st, count: reports.filter((r) => r.status === st).length,
  }));
  const inspectionList = byUrgency(reports.filter((r) => r.status === statusFilter));

  function renderCards(list, emptyText) {
    if (loading) return <p className="empty-state">Loading reports…</p>;
    if (list.length === 0) return loadError ? null : <p className="empty-state">{emptyText}</p>;
    return list.map((report) => <ReportCard key={report.id} report={report} onStatus={setStatus} />);
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
        {NAV.map((n) => (
          <div
            key={n.id}
            className={`nav-item${view === n.id ? " active" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => go(n.id)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && go(n.id)}
          >
            {n.icon} &nbsp; {n.label}
          </div>
        ))}

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <span>Prototype environment</span>
          <p>ML model trained on synthetic data · Not a live safety service</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">SAFETY INTELLIGENCE / {current.eyebrow}</p>
            <h1>{current.title}</h1>
            <p className="subtitle">{current.subtitle}</p>
          </div>
          <div className="prototype-badge">● PROTOTYPE</div>
        </header>

        {view === "overview" && (
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
        )}

        {waking && (
          <div className="success-message">
            The server is waking up from idle (free hosting). This can take up to a
            minute; loading automatically…
          </div>
        )}
        {loadError && <div className="error-banner">{loadError}</div>}

        {view === "overview" && (
          <>
            <section className="metrics-grid">
              <Metric label="Registered reports" value={reports.length} note="Stored in database" />
              <Metric label="Critical priority" value={criticalCount} note="Urgent human review" />
              <Metric label="Awaiting verification" value={openCount} note="Not confirmed safe" />
              <Metric label="Verified safe" value={verifiedCount} note="Confirmed records only" />
            </section>

            <section className="section-header">
              <div>
                <h2>Needs attention first</h2>
                <p>Critical and High reports still waiting for an inspection, most urgent first.</p>
              </div>
              <button className="primary-button" onClick={startNewReport}>+ New report</button>
            </section>

            <section className="report-list">
              {renderCards(attention.slice(0, 3), "Nothing urgent is waiting. All Critical and High reports have been inspected.")}
            </section>
            {attention.length > 3 && (
              <p className="assumed-note">
                {attention.length - 3} more waiting in{" "}
                <button className="link-button" onClick={() => go("inspection")}>Inspection tracking</button>.
              </p>
            )}
          </>
        )}

        {view === "reports" && (<>
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
          {renderCards(
            visibleReports,
            reports.length === 0
              ? "No reports yet. Use “+ New report” to submit the first one."
              : "No reports match this risk level."
          )}
        </section>
        </>)}

        {view === "risk" && (
          <>
            <section className="metrics-grid">
              <Metric label="Reports analysed" value={reports.length} note="All stored reports" />
              <Metric label="Critical or High" value={highShare} note="Share of all reports" />
              <Metric label="Urgent and pending" value={urgentPending} note="Critical/High, not inspected" />
              <Metric label="Raised as precaution" value={escalatedCount} note="Escalated to Critical" />
            </section>

            <section className="panel">
              <h3>Reports by risk level</h3>
              <Bars rows={levelRows} />
            </section>

            <section className="panel">
              <h3>Most common risk factors</h3>
              <p className="form-note">How often each factor appears among a report's top reasons.</p>
              {factorRows.length > 0 ? <Bars rows={factorRows} /> : <p className="empty-state">No factor data yet.</p>}
            </section>

            <p className="assumed-note">
              Risk levels come from a model trained on synthetic data and are for prioritising
              inspections only. Factor shares are approximate.
            </p>
          </>
        )}

        {view === "inspection" && (
          <>
            <section className="metrics-grid">
              {statusRows.map((r) => (
                <Metric key={r.label} label={r.label} value={r.count} note={
                  r.label === "Pending inspection" ? "Waiting for a visit"
                  : r.label === "Inspection completed" ? "Visited, awaiting sign-off"
                  : "Confirmed safe on site"} />
              ))}
              <Metric label="Total reports" value={reports.length} note="All stored reports" />
            </section>

            <section className="section-header list-heading">
              <div>
                <h2>Inspection queue</h2>
                <p>Sorted by risk level. Pending → Inspection completed → Verified safe.</p>
              </div>
              <label className="filter-control">
                <span>Status</span>
                <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                  {STATUSES.map((st) => <option key={st}>{st}</option>)}
                </select>
              </label>
            </section>

            <section className="report-list">
              {renderCards(inspectionList, `No reports with status “${statusFilter}”.`)}
            </section>
          </>
        )}

        <footer>
          <span>BorewellGuard · AI-assisted hazard prevention</span>
          <span>Prototype v0.3 · ML trained on synthetic data, not a live safety service</span>
        </footer>
      </main>
    </div>
  );
}

function ReportCard({ report, onStatus }) {
  const next =
    report.status === "Pending inspection"
      ? { label: "Mark inspected", status: "Inspection completed" }
      : report.status === "Inspection completed"
      ? { label: "Mark verified safe", status: "Verified safe" }
      : null;
  return (
    <article className="report-card">
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
      {next ? (
        <button className="inspect-button" onClick={() => onStatus(report.id, next.status)}>
          {next.label}
        </button>
      ) : (
        <button className="inspect-button" disabled>Verified safe ✓</button>
      )}
    </article>
  );
}

function Bars({ rows }) {
  const max = Math.max(1, ...rows.map((r) => r.count));
  return rows.map((r) => (
    <div className="factor-row" key={r.label}>
      <span className="factor-label">{r.label}</span>
      <div className="factor-bar">
        <span style={{ width: `${(100 * r.count) / max}%`, background: r.color }} />
      </div>
      <span className="factor-value">{r.count}</span>
    </div>
  ));
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
