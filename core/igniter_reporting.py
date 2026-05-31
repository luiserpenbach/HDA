"""Advanced HTML reporting for igniter hot-fire post-test analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.igniter_analysis import IgniterAnalysisResult

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a core dep, fallback for safety
    pd = None  # type: ignore

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    go = None  # type: ignore
    _PLOTLY_AVAILABLE = False


def _fmt(value: Optional[float], unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        out = f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"
    return f"{out} {unit}".strip()


def _interpret_eta(eta: Optional[float]) -> Tuple[str, str]:
    if eta is None:
        return ("No CEA baseline", "Install rocketcea to compute c* efficiency.")
    if eta >= 0.9:
        return ("Excellent", "c* efficiency is in a very strong range for this igniter class.")
    if eta >= 0.8:
        return ("Nominal", "c* efficiency is within expected development-test performance.")
    if eta >= 0.7:
        return ("Needs tuning", "c* efficiency is low; inspect atomization and combustion stability.")
    return ("Poor", "c* efficiency is significantly low; review injector behavior and instrumentation.")


def _interpret_cd_delta(cd_delta: Optional[float]) -> Tuple[str, str]:
    if cd_delta is None:
        return ("Unavailable", "No N2O upstream pressure provided for Cd back-calculation.")
    abs_delta = abs(cd_delta)
    if abs_delta <= 0.03:
        return ("Aligned", "Back-calculated Cd is tightly aligned with configured Cd.")
    if abs_delta <= 0.08:
        return ("Watch", "Cd deviation is moderate; monitor trend over the next runs.")
    return ("Investigate", "Cd deviation is large; check orifice condition and metering assumptions.")


def _status_class(label: str) -> str:
    if label in ("Excellent", "Nominal", "Aligned"):
        return "ok"
    if label in ("Needs tuning", "Watch", "No CEA baseline", "Unavailable"):
        return "warn"
    return "bad"


def _detect_time_col(df: "pd.DataFrame") -> Optional[str]:
    for name in ("time_s", "time_ms", "timestamp", "Time", "t"):
        if name in df.columns:
            return name
    return None


def _normalize_time_seconds(df: "pd.DataFrame", time_col: str) -> "pd.Series":
    t = df[time_col].astype(float)
    if time_col in ("time_ms", "timestamp") and float(t.max()) > 100.0:
        return t / 1000.0
    return t


def _find_column_by_keywords(df: "pd.DataFrame", keywords: Tuple[str, ...]) -> Optional[str]:
    columns = [str(c) for c in df.columns]
    lowered = {c: c.lower() for c in columns}
    for col in columns:
        name = lowered[col]
        if all(k in name for k in keywords):
            return col
    for col in columns:
        name = lowered[col]
        if any(k in name for k in keywords):
            return col
    return None


def _resolve_standard_columns(df: "pd.DataFrame", sensor_roles: Dict[str, str]) -> Dict[str, Optional[str]]:
    chamber = sensor_roles.get("chamber_pressure")
    mdot_ox = sensor_roles.get("mass_flow_ox")
    mdot_fuel = sensor_roles.get("mass_flow_fuel")

    if not chamber or chamber not in df.columns:
        chamber = _find_column_by_keywords(df, ("pc",))
        if not chamber:
            chamber = _find_column_by_keywords(df, ("chamber", "press"))
    if not mdot_ox or mdot_ox not in df.columns:
        mdot_ox = _find_column_by_keywords(df, ("ox", "flow"))
    if not mdot_fuel or mdot_fuel not in df.columns:
        mdot_fuel = _find_column_by_keywords(df, ("fuel", "flow"))
        if not mdot_fuel:
            mdot_fuel = _find_column_by_keywords(df, ("eth", "flow"))
    return {
        "chamber_pressure": chamber if chamber in df.columns else None,
        "mass_flow_ox": mdot_ox if mdot_ox in df.columns else None,
        "mass_flow_fuel": mdot_fuel if mdot_fuel in df.columns else None,
    }


def _standard_charts_html(
    df: "pd.DataFrame",
    sensor_roles: Dict[str, str],
    steady_window_s: Optional[Tuple[float, float]],
) -> str:
    if not _PLOTLY_AVAILABLE or df.empty:
        return '<div class="empty">Plotly unavailable or no data provided.</div>'

    time_col = _detect_time_col(df)
    if not time_col:
        return '<div class="empty">No time column found for standard plots.</div>'
    t = _normalize_time_seconds(df, time_col)
    cols = _resolve_standard_columns(df, sensor_roles)
    include_js = True
    blocks = []

    # Standard Plot 1: Chamber pressure
    chamber_col = cols["chamber_pressure"]
    if chamber_col:
        fig_pc = go.Figure()
        fig_pc.add_trace(
            go.Scatter(
                x=t,
                y=df[chamber_col],
                mode="lines",
                line=dict(width=1.7, color="#1d4ed8"),
                name="Chamber Pressure",
            )
        )
        if steady_window_s is not None:
            fig_pc.add_vrect(
                x0=steady_window_s[0],
                x1=steady_window_s[1],
                fillcolor="rgba(22, 163, 74, 0.12)",
                line_width=0,
                annotation_text="steady",
                annotation_position="top left",
            )
        fig_pc.update_layout(
            title="Standard Plot: Chamber Pressure",
            template="plotly_white",
            height=320,
            margin=dict(l=48, r=16, t=44, b=40),
            xaxis_title="Time (s)",
            yaxis_title=chamber_col,
            legend=dict(orientation="h"),
        )
        blocks.append(fig_pc.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False))
        include_js = False
    else:
        blocks.append('<div class="empty">Standard Plot: Chamber Pressure unavailable (no mapped/recognized column).</div>')

    # Standard Plot 2: Mass flow
    fig_mf = go.Figure()
    have_flow = False
    if cols["mass_flow_ox"]:
        fig_mf.add_trace(
            go.Scatter(
                x=t,
                y=df[cols["mass_flow_ox"]],
                mode="lines",
                line=dict(width=1.5, color="#f97316"),
                name="Oxidizer mass flow",
            )
        )
        have_flow = True
    if cols["mass_flow_fuel"]:
        fig_mf.add_trace(
            go.Scatter(
                x=t,
                y=df[cols["mass_flow_fuel"]],
                mode="lines",
                line=dict(width=1.5, color="#16a34a"),
                name="Fuel mass flow",
            )
        )
        have_flow = True

    if have_flow:
        if steady_window_s is not None:
            fig_mf.add_vrect(
                x0=steady_window_s[0],
                x1=steady_window_s[1],
                fillcolor="rgba(22, 163, 74, 0.12)",
                line_width=0,
                annotation_text="steady",
                annotation_position="top left",
            )
        fig_mf.update_layout(
            title="Standard Plot: Mass Flow",
            template="plotly_white",
            height=320,
            margin=dict(l=48, r=16, t=44, b=40),
            xaxis_title="Time (s)",
            yaxis_title="Mass flow channels (g/s)",
            legend=dict(orientation="h"),
        )
        blocks.append(fig_mf.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False))
    else:
        blocks.append('<div class="empty">Standard Plot: Mass Flow unavailable (no mapped/recognized flow columns).</div>')

    return "".join(blocks)


def _traceability_rows(traceability: Dict[str, Any]) -> str:
    keys = (
        "raw_data_path",
        "raw_data_hash",
        "config_hash",
        "analysis_timestamp_utc",
        "analyst_username",
        "analyst_hostname",
        "processing_version",
    )
    rows = []
    for key in keys:
        val = traceability.get(key, "N/A")
        rows.append(f"<tr><td>{escape(key)}</td><td><code>{escape(str(val))}</code></td></tr>")
    return "".join(rows)


def generate_igniter_hotfire_report(
    *,
    test_id: str,
    result: IgniterAnalysisResult,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    traceability: Optional[Dict[str, Any]] = None,
    steady_window_s: Optional[Tuple[float, float]] = None,
    df: Optional["pd.DataFrame"] = None,
    sensor_roles: Optional[Dict[str, str]] = None,
) -> str:
    """Generate a professional engineering HTML report for igniter hot-fire tests."""
    metadata = metadata or {}
    config = config or {}
    traceability = traceability or {}
    sensor_roles = sensor_roles or {}

    eta_label, eta_note = _interpret_eta(result.eta_cstar)
    cd_label, cd_note = _interpret_cd_delta(result.cd_n2o_delta)
    now = datetime.now(timezone.utc).isoformat()
    diag = result.n2o_flow_diagnostics or {}
    config_snapshot = escape(json.dumps(config, indent=2, sort_keys=True, default=str))

    metadata_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in metadata.items()
        if v is not None and str(v).strip()
    )
    warnings_html = "".join(f"<li>{escape(w)}</li>" for w in result.warnings)
    standard_plots = ""
    if df is not None and pd is not None:
        standard_plots = _standard_charts_html(df, sensor_roles, steady_window_s)
    else:
        standard_plots = '<div class="empty">No processed timeseries attached to this report export.</div>'

    assumptions = [
        "N2O flow model: NHNE (SPI/HEM blend) when upstream pressure is provided.",
        "Fuel flow source: measured meter if provided, otherwise SPI estimate from upstream pressure.",
        "c* efficiency baseline uses rocketcea equilibrium model when available.",
        "Steady window bounds and Analyze-tab hardware settings define the reported operating point.",
    ]
    assumptions_html = "".join(f"<li>{escape(item)}</li>" for item in assumptions)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Igniter Hot-Fire Engineering Report - {escape(test_id)}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --surface: #f8fafc;
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe3ef;
      --accent: #1d4ed8;
      --ok-bg: #dcfce7; --ok-ink: #166534;
      --warn-bg: #fef3c7; --warn-ink: #92400e;
      --bad-bg: #fee2e2; --bad-ink: #991b1b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 22px; font-family: Segoe UI, Arial, sans-serif; color: var(--ink); background: var(--bg); }}
    .page {{ max-width: 1240px; margin: 0 auto; }}
    .hero {{ border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); }}
    .hero h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
    .hero .meta {{ color: var(--muted); font-size: 13px; }}
    .section {{ margin-top: 22px; border: 1px solid var(--line); border-radius: 10px; background: #fff; padding: 16px; }}
    .section h2 {{ margin: 0 0 12px 0; font-size: 20px; }}
    .subhead {{ color: var(--muted); font-size: 13px; margin-bottom: 12px; }}
    .kpi-grid {{ display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); }}
    .kpi {{ border: 1px solid var(--line); border-radius: 8px; background: var(--surface); padding: 10px; }}
    .kpi .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi .value {{ margin-top: 4px; font-size: 21px; font-weight: 700; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 700; }}
    .ok {{ background: var(--ok-bg); color: var(--ok-ink); }}
    .warn {{ background: var(--warn-bg); color: var(--warn-ink); }}
    .bad {{ background: var(--bad-bg); color: var(--bad-ink); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: var(--surface); }}
    pre {{ margin: 0; white-space: pre-wrap; background: #0b1220; color: #dbeafe; border-radius: 8px; padding: 12px; font-size: 12px; overflow-x: auto; }}
    code {{ font-family: Consolas, monospace; font-size: 12px; }}
    ul {{ margin: 8px 0 0 18px; }}
    .empty {{ border: 1px dashed var(--line); border-radius: 8px; padding: 12px; color: var(--muted); background: var(--surface); }}
    .footer {{ margin: 22px 0 8px 0; color: var(--muted); font-size: 12px; }}
    @media print {{
      body {{ padding: 0; }}
      .section, .hero {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Igniter Hot-Fire Engineering Report</h1>
      <div class="meta">
        Test ID: <strong>{escape(test_id)}</strong> |
        Generated UTC: {escape(now)} |
        Input source: {escape(result.input_source)} |
        Propellant pair: {escape(result.oxidizer_name)} / {escape(result.fuel_name)}
      </div>
    </div>

    <div class="section">
      <h2>Executive Summary</h2>
      <div class="subhead">Primary performance outcomes at analyzed operating point.</div>
      <div class="kpi-grid">
        <div class="kpi"><div class="label">Chamber pressure</div><div class="value">{_fmt(result.pc_bar, "bar", 2)}</div></div>
        <div class="kpi"><div class="label">Total mass flow</div><div class="value">{_fmt(result.mdot_total_g_s, "g/s", 2)}</div></div>
        <div class="kpi"><div class="label">O/F ratio</div><div class="value">{_fmt(result.of_ratio, "", 3)}</div></div>
        <div class="kpi"><div class="label">c* actual</div><div class="value">{_fmt(result.cstar_actual_m_s, "m/s", 1)}</div></div>
        <div class="kpi"><div class="label">c* theoretical</div><div class="value">{_fmt(result.cstar_theoretical_m_s, "m/s", 1)}</div></div>
        <div class="kpi"><div class="label">eta c*</div><div class="value">{_fmt(result.eta_cstar * 100.0 if result.eta_cstar is not None else None, "%", 1)}</div></div>
      </div>
    </div>

    <div class="section">
      <h2>Standard Plots</h2>
      <div class="subhead">Required baseline evidence for first-pass review: chamber pressure and mass flow timeseries.</div>
      {standard_plots}
    </div>

    <div class="section">
      <h2>Combustion Performance Assessment</h2>
      <table>
        <tbody>
          <tr><th>Check</th><th>Status</th><th>Engineering note</th></tr>
          <tr><td>c* efficiency</td><td><span class="badge {_status_class(eta_label)}">{escape(eta_label)}</span></td><td>{escape(eta_note)}</td></tr>
          <tr><td>N2O Cd consistency</td><td><span class="badge {_status_class(cd_label)}">{escape(cd_label)}</span></td><td>{escape(cd_note)}</td></tr>
          <tr><td>Predicted Pc (target eta c*)</td><td>{_fmt(result.pc_predicted_bar, "bar", 2)}</td><td>Model closure against configured target efficiency.</td></tr>
          <tr><td>Theoretical combustion temperature</td><td>{_fmt(result.tc_theo_k, "K", 0)}</td><td>Reference from CEA thermochemistry.</td></tr>
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>Injector and Feed Diagnostics</h2>
      <table>
        <tbody>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>N2O mass flow</td><td>{_fmt(result.mdot_n2o_g_s, "g/s", 3)}</td></tr>
          <tr><td>Fuel mass flow</td><td>{_fmt(result.mdot_eth_g_s, "g/s", 3)} ({escape(result.mdot_eth_source)})</td></tr>
          <tr><td>N2O Cd back-calculated</td><td>{_fmt(result.cd_n2o_back, "", 4)}</td></tr>
          <tr><td>N2O Cd delta vs configured</td><td>{_fmt(result.cd_n2o_delta, "", 4)}</td></tr>
          <tr><td>NHNE regime</td><td>{escape(str(diag.get("regime", "N/A")))}</td></tr>
          <tr><td>NHNE choked flag</td><td>{escape(str(diag.get("choked", "N/A")))}</td></tr>
          <tr><td>NHNE critical pressure</td><td>{_fmt(diag.get("Pcrit_bar"), "bar", 2)}</td></tr>
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>Traceability and Metadata</h2>
      <table><tbody>{_traceability_rows(traceability)}</tbody></table>
      <div style="height:12px"></div>
      <table><tbody>{metadata_rows or "<tr><td colspan='2'>No metadata available.</td></tr>"}</tbody></table>
    </div>

    <div class="section">
      <h2>Model Assumptions and Notes</h2>
      <ul>{assumptions_html}</ul>
      {"<h3 style='margin:14px 0 8px 0'>Warnings</h3><ul>" + warnings_html + "</ul>" if warnings_html else ""}
    </div>

    <div class="section">
      <h2>Configuration Snapshot</h2>
      <pre>{config_snapshot}</pre>
    </div>

    <div class="footer">
      Generated by Hopper Data Studio igniter reporting pipeline.
    </div>
  </div>
</body>
</html>
"""


def save_igniter_hotfire_report(html: str, filepath: str | Path) -> Path:
    """Persist igniter HTML report to disk."""
    path = Path(filepath)
    path.write_text(html, encoding="utf-8")
    return path
