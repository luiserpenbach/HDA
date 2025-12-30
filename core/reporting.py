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
from typing import Dict, Any, Optional, List, Union
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
