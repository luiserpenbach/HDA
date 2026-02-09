"""
Enhanced Reporting Module
=========================
Generate professional HTML reports with full traceability,
uncertainty tables, QC summaries, and control charts.

Features:
- Single-test analysis reports
- Campaign summary reports
- SPC/control chart reports
- Exportable to HTML, PDF-ready
- Full audit trail included
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import base64
from io import BytesIO

# Try to import plotly for charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# CHART CONSTANTS
# =============================================================================

_REPORT_COLORS = {
    'primary': '#18181b',
    'accent': '#2563eb',
    'success': '#16a34a',
    'muted': '#71717a',
    'steady_fill': 'rgba(22, 163, 106, 0.12)',
}

_REPORT_LAYOUT = dict(
    template='plotly_white',
    margin=dict(t=40, b=40, l=50, r=20),
    height=280,
    showlegend=False,
)

# Key chart definitions per test type
_KEY_CHARTS = {
    'cold_flow': [
        {'role': 'upstream_pressure', 'title': 'Upstream Pressure', 'y_label': 'Pressure (bar)', 'color': '#2563eb'},
        {'role': 'mass_flow', 'title': 'Mass Flow Rate', 'y_label': 'Flow (g/s)', 'color': '#18181b'},
        {'role': 'downstream_pressure', 'title': 'Downstream Pressure', 'y_label': 'Pressure (bar)', 'color': '#71717a'},
        {'type': 'bar', 'metrics': ['Cd', 'delta_p'], 'title': 'Key Metrics'},
    ],
    'hot_fire': [
        {'role': 'chamber_pressure', 'title': 'Chamber Pressure', 'y_label': 'Pressure (bar)', 'color': '#2563eb'},
        {'role': 'thrust', 'title': 'Thrust', 'y_label': 'Thrust (N)', 'color': '#18181b'},
        {'role': 'mass_flow_ox', 'title': 'Mass Flow', 'y_label': 'Flow (g/s)', 'color': '#71717a',
         'fallbacks': ['mass_flow_total', 'mass_flow_fuel', 'mass_flow']},
        {'type': 'bar', 'metrics': ['Isp', 'c_star', 'of_ratio'], 'title': 'Performance Metrics'},
    ],
}


# =============================================================================
# HTML TEMPLATES
# =============================================================================

HTML_HEAD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --light-bg: #f8f9fa;
            --border-color: #dee2e6;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
        }}
        
        .report-header {{
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .report-header h1 {{
            color: var(--primary-color);
            margin-bottom: 10px;
        }}
        
        .report-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .report-meta span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .section h3 {{
            color: #555;
            margin: 20px 0 10px 0;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        
        .status-pass {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .status-warn {{
            background: #fff3cd;
            color: #856404;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.95em;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--light-bg);
            font-weight: 600;
            color: var(--primary-color);
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .measurement-table td:nth-child(2),
        .measurement-table td:nth-child(3),
        .measurement-table td:nth-child(4) {{
            text-align: right;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        
        .traceability-box {{
            background: var(--light-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85em;
            overflow-x: auto;
        }}
        
        .traceability-box .field {{
            display: flex;
            margin-bottom: 8px;
        }}
        
        .traceability-box .label {{
            min-width: 200px;
            color: #666;
        }}
        
        .traceability-box .value {{
            color: #333;
            word-break: break-all;
        }}
        
        .qc-list {{
            list-style: none;
        }}
        
        .qc-list li {{
            padding: 8px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .qc-icon {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: white;
        }}
        
        .qc-pass {{ background: var(--success-color); }}
        .qc-warn {{ background: var(--warning-color); }}
        .qc-fail {{ background: var(--danger-color); }}
        
        .chart-container {{
            margin: 20px 0;
            padding: 15px;
            background: var(--light-bg);
            border-radius: 8px;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}

        .chart-grid .chart-cell {{
            background: var(--light-bg);
            border-radius: 8px;
            padding: 10px;
            min-height: 300px;
        }}

        .chart-grid .chart-cell h4 {{
            color: var(--primary-color);
            margin: 0 0 8px 0;
            font-size: 0.9em;
        }}

        .appendix-charts .chart-container {{
            margin: 15px 0;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .summary-card {{
            background: var(--light-bg);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        
        .summary-card .value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--primary-color);
        }}
        
        .summary-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        
        .summary-card .uncertainty {{
            font-size: 0.85em;
            color: #888;
        }}
        
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.85em;
            color: #888;
        }}
        
        @media print {{
            body {{
                max-width: none;
                padding: 0;
            }}
            .section {{
                page-break-inside: avoid;
            }}
            .chart-cell, .appendix-charts .chart-container {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
"""

HTML_FOOTER = """
    <div class="footer">
        <p>Generated by Hopper Data Studio v2.0</p>
        <p>Report generated: {timestamp}</p>
    </div>
</body>
</html>
"""


# =============================================================================
# REPORT GENERATION FUNCTIONS
# =============================================================================

def generate_measurement_table(
    measurements: Dict[str, Any],
    title: str = "Measurements with Uncertainties"
) -> str:
    """Generate HTML table for measurements with uncertainties."""
    rows = []
    
    for name, meas in measurements.items():
        if hasattr(meas, 'value') and hasattr(meas, 'uncertainty'):
            value = f"{meas.value:.4g}"
            uncertainty = f"±{meas.uncertainty:.4g}"
            rel_u = f"{meas.relative_uncertainty_percent:.1f}%"
            unit = getattr(meas, 'unit', '')
        else:
            value = f"{meas:.4g}" if isinstance(meas, (int, float)) else str(meas)
            uncertainty = "-"
            rel_u = "-"
            unit = ""
        
        rows.append(f"""
            <tr>
                <td>{name}</td>
                <td>{value} {unit}</td>
                <td>{uncertainty}</td>
                <td>{rel_u}</td>
            </tr>
        """)
    
    return f"""
    <h3>{title}</h3>
    <table class="measurement-table">
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Uncertainty (1σ)</th>
                <th>Relative</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def generate_qc_section(qc_report: Dict[str, Any]) -> str:
    """Generate HTML for QC results section."""
    passed = qc_report.get('passed', True)
    summary = qc_report.get('summary', {})
    checks = qc_report.get('checks', [])
    
    status_class = 'status-pass' if passed else 'status-fail'
    status_text = 'PASSED' if passed else 'FAILED'
    
    check_items = []
    for check in checks:
        status = check.get('status', 'PASS')
        if status == 'PASS':
            icon_class = 'qc-pass'
            icon = ''
        elif status == 'WARN':
            icon_class = 'qc-warn'
            icon = '!'
        else:
            icon_class = 'qc-fail'
            icon = ''
        
        check_items.append(f"""
            <li>
                <span class="qc-icon {icon_class}">{icon}</span>
                <span><strong>{check.get('name', 'Unknown')}</strong>: {check.get('message', '')}</span>
            </li>
        """)
    
    return f"""
    <div class="section">
        <h2>Quality Control</h2>
        <p>
            Status: <span class="status-badge {status_class}">{status_text}</span>
        </p>
        <p>
            Summary: {summary.get('passed', 0)} passed, 
            {summary.get('warnings', 0)} warnings, 
            {summary.get('failed', 0)} failed
        </p>
        <h3>Check Details</h3>
        <ul class="qc-list">
            {''.join(check_items)}
        </ul>
    </div>
    """


def generate_traceability_section(traceability: Dict[str, Any]) -> str:
    """Generate HTML for traceability section."""
    fields = []
    
    important_fields = [
        ('raw_data_hash', 'Data Hash'),
        ('raw_data_filename', 'Source File'),
        ('config_name', 'Configuration'),
        ('config_hash', 'Config Hash'),
        ('analyst_username', 'Analyst'),
        ('analyst_hostname', 'Workstation'),
        ('analysis_timestamp_utc', 'Analysis Time (UTC)'),
        ('processing_version', 'Processing Version'),
        ('steady_window_start_ms', 'Steady Window Start (ms)'),
        ('steady_window_end_ms', 'Steady Window End (ms)'),
        ('detection_method', 'Detection Method'),
    ]
    
    for key, label in important_fields:
        value = traceability.get(key, 'N/A')
        if value is not None:
            fields.append(f"""
                <div class="field">
                    <span class="label">{label}:</span>
                    <span class="value">{value}</span>
                </div>
            """)
    
    return f"""
    <div class="section">
        <h2>Traceability Record</h2>
        <div class="traceability-box">
            {''.join(fields)}
        </div>
    </div>
    """


def generate_summary_cards(metrics: Dict[str, Any]) -> str:
    """Generate summary metric cards."""
    cards = []
    
    for name, data in metrics.items():
        if hasattr(data, 'value'):
            value = f"{data.value:.4g}"
            unit = getattr(data, 'unit', '')
            uncertainty = f"±{data.uncertainty:.4g}" if hasattr(data, 'uncertainty') else ""
        elif isinstance(data, (int, float)):
            value = f"{data:.4g}"
            unit = ""
            uncertainty = ""
        else:
            value = str(data)
            unit = ""
            uncertainty = ""
        
        cards.append(f"""
            <div class="summary-card">
                <div class="value">{value}</div>
                <div class="label">{name} {unit}</div>
                <div class="uncertainty">{uncertainty}</div>
            </div>
        """)
    
    return f"""
    <div class="summary-cards">
        {''.join(cards)}
    </div>
    """


# =============================================================================
# CHART HELPERS
# =============================================================================

def _resolve_sensor_column(role: str, config: Dict[str, Any]) -> Optional[str]:
    """Resolve a logical sensor role to the actual DataFrame column name."""
    sensor_roles = config.get('sensor_roles', {})
    if not sensor_roles:
        sensor_roles = config.get('columns', {})
    return sensor_roles.get(role)


def _detect_time_col(df: pd.DataFrame) -> Optional[str]:
    """Find the time column in a DataFrame."""
    for candidate in ('time_s', 'time_ms', 'timestamp', 'Time', 't'):
        if candidate in df.columns:
            return candidate
    return None


def _create_time_series_chart(
    df: pd.DataFrame,
    time_col: str,
    data_col: str,
    title: str,
    y_label: str,
    steady_window: Optional[Tuple[float, float]] = None,
    color: str = '#18181b',
    is_first_chart: bool = False,
) -> str:
    """
    Create an embedded time-series chart as HTML.

    Args:
        df: Full test DataFrame
        time_col: Name of time column
        data_col: Name of data column to plot
        title: Chart title
        y_label: Y-axis label
        steady_window: Optional (start, end) for green overlay
        color: Line color
        is_first_chart: If True, includes plotly.js CDN

    Returns:
        HTML string with embedded chart, or empty string if unavailable
    """
    if not PLOTLY_AVAILABLE:
        return ''

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df[data_col],
        mode='lines', name=data_col,
        line=dict(width=1.5, color=color),
    ))

    if steady_window is not None:
        fig.add_vrect(
            x0=steady_window[0], x1=steady_window[1],
            fillcolor=_REPORT_COLORS['steady_fill'],
            line_width=0,
            annotation_text='steady',
            annotation_position='top left',
            annotation_font_size=10,
            annotation_font_color=_REPORT_COLORS['success'],
        )

    time_label = 'Time (s)' if time_col == 'time_s' else f'Time ({time_col})'
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title=time_label,
        yaxis_title=y_label,
        **_REPORT_LAYOUT,
    )

    include_js = 'cdn' if is_first_chart else False
    return fig.to_html(full_html=False, include_plotlyjs=include_js)


def _create_bar_chart(
    measurements: Dict[str, Any],
    metric_keys: List[str],
    title: str,
    is_first_chart: bool = False,
) -> str:
    """
    Create a bar chart for computed measurements with error bars.

    Args:
        measurements: Dict of measurement name -> MeasurementWithUncertainty
        metric_keys: List of keys to plot
        title: Chart title
        is_first_chart: If True, includes plotly.js CDN

    Returns:
        HTML string with embedded chart
    """
    if not PLOTLY_AVAILABLE:
        return ''

    names = []
    values = []
    errors = []
    units = []

    for key in metric_keys:
        if key in measurements:
            m = measurements[key]
            if hasattr(m, 'value') and hasattr(m, 'uncertainty'):
                names.append(key)
                values.append(m.value)
                errors.append(m.uncertainty)
                units.append(getattr(m, 'unit', ''))

    if not names:
        return ''

    hover_text = [
        f"{n}: {v:.4g} ± {e:.4g} {u}"
        for n, v, e, u in zip(names, values, errors, units)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=values,
        error_y=dict(type='data', array=errors, visible=True),
        marker_color=_REPORT_COLORS['accent'],
        text=hover_text,
        hoverinfo='text',
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        yaxis_title='Value',
        **_REPORT_LAYOUT,
    )

    include_js = 'cdn' if is_first_chart else False
    return fig.to_html(full_html=False, include_plotlyjs=include_js)


# =============================================================================
# CHART SECTION GENERATORS
# =============================================================================

def generate_key_charts_section(
    df: pd.DataFrame,
    test_type: str,
    config: Dict[str, Any],
    measurements: Dict[str, Any],
    steady_window: Optional[Tuple[float, float]] = None,
    is_first_chart_in_report: bool = True,
) -> str:
    """
    Generate the key charts 2x2 grid section for the report.

    Args:
        df: Full test DataFrame with time series data
        test_type: 'cold_flow' or 'hot_fire'
        config: Configuration dictionary (with sensor_roles or columns)
        measurements: Dict of measurements with uncertainties
        steady_window: (start, end) for green overlay
        is_first_chart_in_report: If True, first chart includes plotly.js CDN

    Returns:
        HTML string for the key charts section
    """
    if not PLOTLY_AVAILABLE:
        return (
            '<div class="section"><h2>Key Charts</h2>'
            '<p><em>Chart visualization requires the plotly library.</em></p></div>'
        )

    chart_specs = _KEY_CHARTS.get(test_type, _KEY_CHARTS['cold_flow'])
    time_col = _detect_time_col(df)

    if time_col is None:
        return ''

    chart_html_cells = []
    key_columns = []
    first_chart = is_first_chart_in_report

    for spec in chart_specs:
        if spec.get('type') == 'bar':
            html = _create_bar_chart(
                measurements, spec['metrics'], spec['title'],
                is_first_chart=first_chart,
            )
            if html:
                chart_html_cells.append(f'<div class="chart-cell">{html}</div>')
                if first_chart:
                    first_chart = False
        else:
            # Time series chart — resolve sensor column
            col_name = _resolve_sensor_column(spec['role'], config)

            # Try fallback roles if primary not found
            if (col_name is None or col_name not in df.columns) and 'fallbacks' in spec:
                for fb_role in spec['fallbacks']:
                    fb_col = _resolve_sensor_column(fb_role, config)
                    if fb_col and fb_col in df.columns:
                        col_name = fb_col
                        break

            if col_name is None or col_name not in df.columns:
                chart_html_cells.append(
                    f'<div class="chart-cell">'
                    f'<h4>{spec["title"]}</h4>'
                    f'<p style="color:#71717a;padding:60px 20px;text-align:center;">'
                    f'Sensor not available</p></div>'
                )
                continue

            key_columns.append(col_name)

            html = _create_time_series_chart(
                df, time_col, col_name,
                title=spec['title'],
                y_label=spec.get('y_label', ''),
                steady_window=steady_window,
                color=spec.get('color', _REPORT_COLORS['primary']),
                is_first_chart=first_chart,
            )
            if html:
                chart_html_cells.append(f'<div class="chart-cell">{html}</div>')
                if first_chart:
                    first_chart = False

    if not chart_html_cells:
        return ''

    return (
        '<div class="section">'
        '<h2>Key Charts</h2>'
        f'<div class="chart-grid">{"".join(chart_html_cells)}</div>'
        '</div>'
    )


def generate_appendix_charts_section(
    df: pd.DataFrame,
    config: Dict[str, Any],
    steady_window: Optional[Tuple[float, float]] = None,
    key_columns: Optional[List[str]] = None,
) -> str:
    """
    Generate appendix section with remaining sensor time series.

    Args:
        df: Full test DataFrame
        config: Configuration dict
        steady_window: (start, end) for green overlay
        key_columns: Columns already plotted in key charts (excluded)

    Returns:
        HTML string for appendix section
    """
    if not PLOTLY_AVAILABLE:
        return ''

    time_col = _detect_time_col(df)
    if time_col is None:
        return ''

    skip_cols = {'time', 'time_s', 'time_ms', 'timestamp', 'Time', 'TIME', 't'}
    skip_cols.update(key_columns or [])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    appendix_cols = [c for c in numeric_cols if c not in skip_cols]

    if not appendix_cols:
        return ''

    chart_parts = []
    for col in appendix_cols:
        html = _create_time_series_chart(
            df, time_col, col,
            title=col, y_label=col,
            steady_window=steady_window,
            color=_REPORT_COLORS['muted'],
            is_first_chart=False,
        )
        if html:
            chart_parts.append(f'<div class="chart-container">{html}</div>')

    if not chart_parts:
        return ''

    return (
        '<div class="section appendix-charts">'
        '<h2>Appendix: Sensor Data</h2>'
        '<p style="color:#666;font-size:0.9em;">'
        'Full time-series traces for all recorded sensor channels. '
        'Green bands indicate the steady-state analysis window.</p>'
        f'{"".join(chart_parts)}'
        '</div>'
    )


def _get_key_chart_columns(test_type: str, config: Dict[str, Any]) -> List[str]:
    """Get the list of DataFrame columns used by key charts (for appendix exclusion)."""
    chart_specs = _KEY_CHARTS.get(test_type, _KEY_CHARTS.get('cold_flow', []))
    key_cols = []
    for spec in chart_specs:
        if spec.get('type') == 'bar':
            continue
        col = _resolve_sensor_column(spec.get('role', ''), config)
        if col:
            key_cols.append(col)
        for fb_role in spec.get('fallbacks', []):
            fb_col = _resolve_sensor_column(fb_role, config)
            if fb_col:
                key_cols.append(fb_col)
    return key_cols


# =============================================================================
# MAIN REPORT GENERATORS
# =============================================================================

def generate_test_report(
    test_id: str,
    test_type: str,
    measurements: Dict[str, Any],
    traceability: Dict[str, Any],
    qc_report: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    include_config_snapshot: bool = False,
    df: Optional[pd.DataFrame] = None,
    steady_window: Optional[Tuple[float, float]] = None,
    include_charts: bool = True,
) -> str:
    """
    Generate complete HTML report for a single test.

    Args:
        test_id: Test identifier
        test_type: 'cold_flow' or 'hot_fire'
        measurements: Dictionary of measurements (with uncertainties)
        traceability: Traceability record
        qc_report: QC report dictionary
        metadata: Additional metadata
        config: Test configuration
        include_config_snapshot: Include full config JSON
        df: Processed test DataFrame for chart generation (optional)
        steady_window: (start, end) in seconds for chart overlay (optional)
        include_charts: Whether to generate charts (default True)

    Returns:
        Complete HTML report as string
    """
    metadata = metadata or {}
    
    # Build report
    report_parts = []
    
    # Header
    title = f"Test Report: {test_id}"
    report_parts.append(HTML_HEAD.format(title=title))
    
    # Report header
    test_type_display = "Cold Flow Test" if test_type == 'cold_flow' else "Hot Fire Test"
    timestamp = traceability.get('analysis_timestamp_utc', datetime.now().isoformat())
    analyst = traceability.get('analyst_username', 'Unknown')
    
    report_parts.append(f"""
    <div class="report-header">
        <h1>{title}</h1>
        <div class="report-meta">
            <span> Type: {test_type_display}</span>
            <span> Analyst: {analyst}</span>
            <span> Date: {timestamp[:10]}</span>
        </div>
    </div>
    """)
    
    # Metadata section
    if metadata:
        meta_rows = []
        for key, value in metadata.items():
            if value is not None:
                meta_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
        
        report_parts.append(f"""
        <div class="section">
            <h2>Test Information</h2>
            <table>
                <tbody>
                    {''.join(meta_rows)}
                </tbody>
            </table>
        </div>
        """)
    
    # Summary cards for key metrics
    key_metrics = {}
    if test_type == 'cold_flow':
        for key in ['Cd', 'pressure_upstream', 'mass_flow', 'delta_p']:
            if key in measurements:
                key_metrics[key] = measurements[key]
    else:
        for key in ['Isp', 'c_star', 'of_ratio', 'chamber_pressure']:
            if key in measurements:
                key_metrics[key] = measurements[key]
    
    if key_metrics:
        report_parts.append(f"""
        <div class="section">
            <h2>Key Results</h2>
            {generate_summary_cards(key_metrics)}
        </div>
        """)

    # Key charts (2x2 grid)
    if include_charts and df is not None and PLOTLY_AVAILABLE and config:
        key_chart_columns = _get_key_chart_columns(test_type, config)
        report_parts.append(generate_key_charts_section(
            df=df,
            test_type=test_type,
            config=config,
            measurements=measurements,
            steady_window=steady_window,
            is_first_chart_in_report=True,
        ))

    # Full measurements table
    report_parts.append(f"""
    <div class="section">
        <h2>All Measurements</h2>
        {generate_measurement_table(measurements)}
    </div>
    """)
    
    # QC section
    if qc_report:
        report_parts.append(generate_qc_section(qc_report))
    
    # Traceability section
    report_parts.append(generate_traceability_section(traceability))
    
    # Config snapshot
    if include_config_snapshot and config:
        config_json = json.dumps(config, indent=2, default=str)
        report_parts.append(f"""
        <div class="section">
            <h2>Configuration Snapshot</h2>
            <div class="traceability-box">
                <pre>{config_json}</pre>
            </div>
        </div>
        """)

    # Appendix charts (all remaining sensor traces)
    if include_charts and df is not None and PLOTLY_AVAILABLE:
        key_chart_cols = _get_key_chart_columns(test_type, config or {})
        report_parts.append(generate_appendix_charts_section(
            df=df,
            config=config or {},
            steady_window=steady_window,
            key_columns=key_chart_cols,
        ))

    # Footer
    report_parts.append(HTML_FOOTER.format(timestamp=datetime.now().isoformat()))
    
    return ''.join(report_parts)


def generate_campaign_report(
    campaign_name: str,
    df: pd.DataFrame,
    parameters: List[str],
    spc_analyses: Optional[Dict[str, Any]] = None,
    specs: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    Generate campaign summary report with statistics and SPC.
    
    Args:
        campaign_name: Campaign name
        df: Campaign data DataFrame
        parameters: List of parameter columns to analyze
        spc_analyses: Pre-computed SPC analyses
        specs: Specification limits
        
    Returns:
        Complete HTML report as string
    """
    specs = specs or {}
    
    report_parts = []
    
    # Header
    title = f"Campaign Report: {campaign_name}"
    report_parts.append(HTML_HEAD.format(title=title))
    
    report_parts.append(f"""
    <div class="report-header">
        <h1>{title}</h1>
        <div class="report-meta">
            <span> Tests: {len(df)}</span>
            <span> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </div>
    """)
    
    # Summary statistics
    report_parts.append('<div class="section"><h2>Summary Statistics</h2>')
    
    for param in parameters:
        if param not in df.columns:
            continue
        
        data = df[param].dropna()
        if len(data) == 0:
            continue
        
        param_specs = specs.get(param, {})
        
        stats_rows = [
            f"<tr><td>Count</td><td>{len(data)}</td></tr>",
            f"<tr><td>Mean</td><td>{data.mean():.4g}</td></tr>",
            f"<tr><td>Std Dev</td><td>{data.std():.4g}</td></tr>",
            f"<tr><td>Min</td><td>{data.min():.4g}</td></tr>",
            f"<tr><td>Max</td><td>{data.max():.4g}</td></tr>",
            f"<tr><td>Range</td><td>{data.max() - data.min():.4g}</td></tr>",
        ]
        
        if 'usl' in param_specs:
            stats_rows.append(f"<tr><td>USL</td><td>{param_specs['usl']:.4g}</td></tr>")
        if 'lsl' in param_specs:
            stats_rows.append(f"<tr><td>LSL</td><td>{param_specs['lsl']:.4g}</td></tr>")
        
        report_parts.append(f"""
        <h3>{param}</h3>
        <table>
            <tbody>
                {''.join(stats_rows)}
            </tbody>
        </table>
        """)
    
    report_parts.append('</div>')
    
    # SPC section
    if spc_analyses:
        report_parts.append('<div class="section"><h2>Statistical Process Control</h2>')
        
        for param, analysis in spc_analyses.items():
            status_class = 'status-pass' if analysis.n_violations == 0 else 'status-warn'
            status_text = 'In Control' if analysis.n_violations == 0 else f'{analysis.n_violations} Violations'
            
            capability_text = ""
            if analysis.capability and analysis.capability.cpk is not None:
                capability_text = f"Cpk = {analysis.capability.cpk:.2f}"
            
            report_parts.append(f"""
            <h3>{param}</h3>
            <p>
                Status: <span class="status-badge {status_class}">{status_text}</span>
                {f'<span style="margin-left: 20px;">{capability_text}</span>' if capability_text else ''}
            </p>
            <table>
                <tbody>
                    <tr><td>Center Line</td><td>{analysis.limits.center_line:.4g}</td></tr>
                    <tr><td>UCL (3σ)</td><td>{analysis.limits.ucl:.4g}</td></tr>
                    <tr><td>LCL (3σ)</td><td>{analysis.limits.lcl:.4g}</td></tr>
                </tbody>
            </table>
            """)
            
            if analysis.has_trend:
                report_parts.append(f"""
                <p style="color: var(--warning-color);">
                    [WARN] Trend detected: {analysis.trend_direction}
                </p>
                """)
        
        report_parts.append('</div>')
    
    # Test list
    report_parts.append('<div class="section"><h2>Test List</h2>')
    
    display_cols = ['test_id', 'test_timestamp'] + [p for p in parameters if p in df.columns]
    if 'qc_passed' in df.columns:
        display_cols.append('qc_passed')
    
    subset_df = df[display_cols].head(50)  # Limit to 50 rows
    
    headers = ''.join(f'<th>{col}</th>' for col in display_cols)
    rows = []
    for _, row in subset_df.iterrows():
        cells = []
        for col in display_cols:
            val = row[col]
            if pd.isna(val):
                cells.append('<td>-</td>')
            elif isinstance(val, float):
                cells.append(f'<td>{val:.4g}</td>')
            else:
                cells.append(f'<td>{val}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    
    report_parts.append(f"""
    <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """)
    
    if len(df) > 50:
        report_parts.append(f'<p><em>Showing 50 of {len(df)} tests</em></p>')
    
    report_parts.append('</div>')
    
    # Footer
    report_parts.append(HTML_FOOTER.format(timestamp=datetime.now().isoformat()))
    
    return ''.join(report_parts)


def save_report(html_content: str, filepath: Union[str, Path]) -> Path:
    """
    Save HTML report to file.
    
    Args:
        html_content: HTML string
        filepath: Output path
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filepath


# =============================================================================
# CHART GENERATION (requires plotly)
# =============================================================================

def generate_control_chart_html(analysis: 'SPCAnalysis') -> str:
    """
    Generate interactive control chart HTML (requires plotly).
    
    Args:
        analysis: SPCAnalysis object
        
    Returns:
        HTML string with embedded chart
    """
    if not PLOTLY_AVAILABLE:
        return '<p><em>Control chart visualization requires plotly library</em></p>'
    
    # Extract data
    x = [p.test_id for p in analysis.points]
    y = [p.value for p in analysis.points]
    colors = ['red' if not p.in_control else 'blue' for p in analysis.points]
    
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=list(range(len(x))),
        y=y,
        mode='lines+markers',
        marker=dict(color=colors, size=8),
        line=dict(color='blue', width=1),
        text=x,
        hovertemplate='%{text}<br>Value: %{y:.4f}<extra></extra>',
        name='Data'
    ))
    
    # Control limits
    fig.add_hline(y=analysis.limits.center_line, line_dash="solid", 
                  line_color="green", annotation_text="CL")
    fig.add_hline(y=analysis.limits.ucl, line_dash="dash", 
                  line_color="red", annotation_text="UCL")
    fig.add_hline(y=analysis.limits.lcl, line_dash="dash", 
                  line_color="red", annotation_text="LCL")
    
    if analysis.limits.uwl:
        fig.add_hline(y=analysis.limits.uwl, line_dash="dot", 
                      line_color="orange", annotation_text="UWL")
        fig.add_hline(y=analysis.limits.lwl, line_dash="dot", 
                      line_color="orange", annotation_text="LWL")
    
    fig.update_layout(
        title=f"Control Chart: {analysis.parameter_name}",
        xaxis_title="Test Number",
        yaxis_title=analysis.parameter_name,
        height=400,
        showlegend=False,
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# =============================================================================
# PHASE 2 REPORT SECTIONS
# =============================================================================

def generate_transient_section(phases: list, startup_metrics: Optional[Dict] = None,
                                shutdown_metrics: Optional[Dict] = None) -> str:
    """
    Generate HTML section for transient analysis results.

    Args:
        phases: List of PhaseResult objects or dicts
        startup_metrics: Dict from analyze_startup_transient()
        shutdown_metrics: Dict from analyze_shutdown_transient()

    Returns:
        HTML string for transient analysis section
    """
    html_parts = ['<div class="section"><h2>Transient Analysis</h2>']

    # Phase timeline table
    html_parts.append('<h3>Test Phase Segmentation</h3>')
    html_parts.append('''
    <table>
        <thead>
            <tr><th>Phase</th><th>Start (ms)</th><th>End (ms)</th><th>Duration (s)</th><th>Quality</th></tr>
        </thead>
        <tbody>
    ''')

    for phase in phases:
        if hasattr(phase, 'phase'):
            name = phase.phase.value if hasattr(phase.phase, 'value') else str(phase.phase)
            start = phase.start_ms
            end = phase.end_ms
            dur = phase.duration_s
            quality = getattr(phase, 'quality', 'N/A')
        else:
            name = phase.get('phase', 'unknown')
            start = phase.get('start_ms', 0)
            end = phase.get('end_ms', 0)
            dur = phase.get('duration_s', 0)
            quality = phase.get('quality', 'N/A')

        badge_class = 'status-pass' if quality in ('good', 'N/A') else 'status-warn'
        html_parts.append(f'''
            <tr>
                <td><strong>{name}</strong></td>
                <td>{start:.0f}</td>
                <td>{end:.0f}</td>
                <td>{dur:.3f}</td>
                <td><span class="status-badge {badge_class}">{quality}</span></td>
            </tr>
        ''')

    html_parts.append('</tbody></table>')

    # Startup transient metrics
    if startup_metrics:
        html_parts.append('<h3>Startup Transient Characterization</h3>')
        html_parts.append('<div class="card-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">')
        metric_labels = {
            'rise_time_s': ('Rise Time', 's'),
            'rise_time_10_90_s': ('Rise Time (10-90%)', 's'),
            'time_to_peak_s': ('Time to Peak', 's'),
            'overshoot_pct': ('Overshoot', '%'),
            'settling_time_s': ('Settling Time', 's'),
        }
        for key, (label, unit) in metric_labels.items():
            if key in startup_metrics and startup_metrics[key] is not None:
                val = startup_metrics[key]
                html_parts.append(f'''
                    <div class="summary-card">
                        <div class="summary-label">{label}</div>
                        <div class="summary-value">{val:.4f} {unit}</div>
                    </div>
                ''')
        html_parts.append('</div>')

    # Shutdown transient metrics
    if shutdown_metrics:
        html_parts.append('<h3>Shutdown Transient Characterization</h3>')
        html_parts.append('<div class="card-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">')
        metric_labels = {
            'decay_time_s': ('Decay Time', 's'),
            'decay_time_90_10_s': ('Decay Time (90-10%)', 's'),
            'tail_off_impulse': ('Tail-off Impulse', 'N*s'),
            'residual_pct': ('Residual', '%'),
        }
        for key, (label, unit) in metric_labels.items():
            if key in shutdown_metrics and shutdown_metrics[key] is not None:
                val = shutdown_metrics[key]
                html_parts.append(f'''
                    <div class="summary-card">
                        <div class="summary-label">{label}</div>
                        <div class="summary-value">{val:.4f} {unit}</div>
                    </div>
                ''')
        html_parts.append('</div>')

    html_parts.append('</div>')
    return ''.join(html_parts)


def generate_frequency_section(spectral_result=None, harmonics: Optional[list] = None,
                                resonances: Optional[list] = None,
                                band_powers: Optional[Dict] = None) -> str:
    """
    Generate HTML section for frequency analysis results.

    Args:
        spectral_result: SpectralResult object
        harmonics: List of HarmonicInfo objects
        resonances: List of resonance dicts
        band_powers: Dict of frequency band powers

    Returns:
        HTML string for frequency analysis section
    """
    html_parts = ['<div class="section"><h2>Frequency Analysis</h2>']

    # Spectral summary
    if spectral_result is not None:
        html_parts.append('<h3>Spectral Summary</h3>')
        html_parts.append('<div class="card-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">')
        html_parts.append(f'''
            <div class="summary-card">
                <div class="summary-label">Dominant Frequency</div>
                <div class="summary-value">{spectral_result.dominant_frequency:.2f} Hz</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Dominant Power</div>
                <div class="summary-value">{spectral_result.dominant_power:.4e}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Total Power</div>
                <div class="summary-value">{spectral_result.total_power:.4e}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Bandwidth</div>
                <div class="summary-value">{spectral_result.bandwidth:.2f} Hz</div>
            </div>
        ''')
        html_parts.append('</div>')

    # Harmonics table
    if harmonics:
        html_parts.append('<h3>Detected Harmonics</h3>')
        html_parts.append('''
        <table>
            <thead>
                <tr><th>#</th><th>Frequency (Hz)</th><th>Power</th><th>Relative Power (%)</th></tr>
            </thead>
            <tbody>
        ''')
        for h in harmonics:
            freq = h.frequency if hasattr(h, 'frequency') else h.get('frequency', 0)
            power = h.power if hasattr(h, 'power') else h.get('power', 0)
            num = h.harmonic_number if hasattr(h, 'harmonic_number') else h.get('harmonic_number', 0)
            rel = h.relative_power if hasattr(h, 'relative_power') else h.get('relative_power', 0)
            html_parts.append(f'''
                <tr>
                    <td>{num}</td><td>{freq:.2f}</td>
                    <td>{power:.4e}</td><td>{rel*100:.1f}%</td>
                </tr>
            ''')
        html_parts.append('</tbody></table>')

    # Resonances
    if resonances:
        html_parts.append('<h3>Detected Resonances</h3>')
        html_parts.append('''
        <table>
            <thead>
                <tr><th>Frequency (Hz)</th><th>Q-Factor</th><th>Bandwidth (Hz)</th><th>Peak Power</th></tr>
            </thead>
            <tbody>
        ''')
        for r in resonances:
            html_parts.append(f'''
                <tr>
                    <td>{r.get("frequency", 0):.2f}</td>
                    <td>{r.get("q_factor", 0):.1f}</td>
                    <td>{r.get("bandwidth", 0):.2f}</td>
                    <td>{r.get("peak_power", 0):.4e}</td>
                </tr>
            ''')
        html_parts.append('</tbody></table>')

    # Band powers
    if band_powers:
        html_parts.append('<h3>Frequency Band Power Distribution</h3>')
        total = sum(band_powers.values()) if band_powers.values() else 1
        html_parts.append('''
        <table>
            <thead><tr><th>Band</th><th>Power</th><th>% of Total</th></tr></thead>
            <tbody>
        ''')
        for band, power in band_powers.items():
            pct = (power / total * 100) if total > 0 else 0
            html_parts.append(f'<tr><td>{band}</td><td>{power:.4e}</td><td>{pct:.1f}%</td></tr>')
        html_parts.append('</tbody></table>')

    html_parts.append('</div>')
    return ''.join(html_parts)


def generate_cusum_ewma_section(cusum_result=None, ewma_result=None) -> str:
    """
    Generate HTML section for CUSUM and EWMA control chart results.

    Args:
        cusum_result: CUSUMResult object
        ewma_result: EWMAResult object

    Returns:
        HTML string for advanced SPC section
    """
    html_parts = ['<div class="section"><h2>Advanced SPC Analysis</h2>']

    if cusum_result is not None:
        html_parts.append('<h3>CUSUM Chart Analysis</h3>')
        status = 'PASS' if cusum_result.n_signals == 0 else 'FAIL'
        badge = 'status-pass' if cusum_result.n_signals == 0 else 'status-fail'
        html_parts.append(f'''
        <p>Parameter: <strong>{cusum_result.parameter_name}</strong> |
           Target: {cusum_result.target:.4f} |
           k = {cusum_result.k:.2f}, h = {cusum_result.h:.2f}</p>
        <p>Status: <span class="status-badge {badge}">{status}</span> |
           Signals: {cusum_result.n_signals}
           ({len(cusum_result.signals_upper)} upper, {len(cusum_result.signals_lower)} lower)</p>
        ''')

    if ewma_result is not None:
        html_parts.append('<h3>EWMA Chart Analysis</h3>')
        status = 'PASS' if ewma_result.n_signals == 0 else 'FAIL'
        badge = 'status-pass' if ewma_result.n_signals == 0 else 'status-fail'
        html_parts.append(f'''
        <p>Parameter: <strong>{ewma_result.parameter_name}</strong> |
           Center Line: {ewma_result.center_line:.4f} |
           Lambda = {ewma_result.lambda_param:.2f}</p>
        <p>Status: <span class="status-badge {badge}">{status}</span> |
           Signals: {ewma_result.n_signals}</p>
        ''')

    html_parts.append('</div>')
    return ''.join(html_parts)
