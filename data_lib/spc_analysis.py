# Create new file: data_lib/spc_analysis.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_control_limits(data, method='3sigma', moving_range_n=2):
    """
    Calculate control limits for SPC charts.

    Args:
        data: Series or array of measurements
        method: '3sigma' or 'individuals' (X-mR chart)
        moving_range_n: Window for moving range calculation

    Returns:
        dict with: mean, ucl, lcl, usl, lsl (if applicable)
    """
    data = pd.Series(data).dropna()

    if len(data) < 2:
        return None

    mean = data.mean()

    if method == '3sigma':
        # Standard 3-sigma limits
        std = data.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        # Warning limits (2-sigma)
        uwl = mean + 2 * std
        lwl = mean - 2 * std

    elif method == 'individuals':
        # X-mR (Individuals and Moving Range) chart
        # More robust for small sample sizes or non-normal distributions

        # Calculate moving ranges
        moving_ranges = data.diff().abs().dropna()
        mr_bar = moving_ranges.mean()

        # Constants for individuals chart (n=2)
        d2 = 1.128  # Constant for moving range of 2
        D3 = 0  # Lower control limit constant
        D4 = 3.267  # Upper control limit constant

        # Process sigma estimate
        sigma_hat = mr_bar / d2

        # Control limits
        ucl = mean + 3 * sigma_hat
        lcl = mean - 3 * sigma_hat
        uwl = mean + 2 * sigma_hat
        lwl = mean - 2 * sigma_hat

    return {
        'mean': mean,
        'std': data.std(),
        'ucl': ucl,
        'lcl': lcl,
        'uwl': uwl,
        'lwl': lwl,
        'n': len(data)
    }


def detect_control_violations(data, limits, timestamps=None):
    """
    Detect Western Electric Rules violations.

    Rules:
    1. One point beyond 3-sigma (UCL/LCL)
    2. Two out of three consecutive points beyond 2-sigma (same side)
    3. Four out of five consecutive points beyond 1-sigma (same side)
    4. Eight consecutive points on same side of centerline
    5. Six points in a row steadily increasing or decreasing (trend)
    6. Fifteen points in a row within 1-sigma (over-control)
    7. Fourteen points in a row alternating up and down (stratification)

    Returns:
        DataFrame with violations
    """
    data = pd.Series(data).reset_index(drop=True)

    if timestamps is not None:
        timestamps = pd.Series(timestamps).reset_index(drop=True)
    else:
        timestamps = data.index

    mean = limits['mean']
    ucl = limits['ucl']
    lcl = limits['lcl']
    uwl = limits['uwl']
    lwl = limits['lwl']

    std = limits['std']
    sigma1_upper = mean + std
    sigma1_lower = mean - std

    violations = []

    for i in range(len(data)):
        point = data.iloc[i]
        ts = timestamps.iloc[i]

        # Rule 1: Beyond 3-sigma
        if point > ucl:
            violations.append({
                'index': i,
                'timestamp': ts,
                'value': point,
                'rule': 'Rule 1: Point above UCL',
                'severity': 'critical'
            })
        elif point < lcl:
            violations.append({
                'index': i,
                'timestamp': ts,
                'value': point,
                'rule': 'Rule 1: Point below LCL',
                'severity': 'critical'
            })

        # Rule 2: 2 out of 3 beyond 2-sigma
        if i >= 2:
            window = data.iloc[i - 2:i + 1]
            above_uwl = (window > uwl).sum()
            below_lwl = (window < lwl).sum()

            if above_uwl >= 2:
                violations.append({
                    'index': i,
                    'timestamp': ts,
                    'value': point,
                    'rule': 'Rule 2: 2/3 points above 2-sigma',
                    'severity': 'warning'
                })
            elif below_lwl >= 2:
                violations.append({
                    'index': i,
                    'timestamp': ts,
                    'value': point,
                    'rule': 'Rule 2: 2/3 points below 2-sigma',
                    'severity': 'warning'
                })

        # Rule 4: 8 consecutive on same side
        if i >= 7:
            window = data.iloc[i - 7:i + 1]
            if (window > mean).all():
                violations.append({
                    'index': i,
                    'timestamp': ts,
                    'value': point,
                    'rule': 'Rule 4: 8 consecutive above mean (trend)',
                    'severity': 'warning'
                })
            elif (window < mean).all():
                violations.append({
                    'index': i,
                    'timestamp': ts,
                    'value': point,
                    'rule': 'Rule 4: 8 consecutive below mean (trend)',
                    'severity': 'warning'
                })

        # Rule 5: 6 points trending
        if i >= 5:
            window = data.iloc[i - 5:i + 1]
            diffs = window.diff().dropna()
            if (diffs > 0).all():
                violations.append({
                    'index': i,
                    'timestamp': ts,
                    'value': point,
                    'rule': 'Rule 5: 6 points increasing (trend)',
                    'severity': 'info'
                })
            elif (diffs < 0).all():
                violations.append({
                    'index': i,
                    'timestamp': ts,
                    'value': point,
                    'rule': 'Rule 5: 6 points decreasing (trend)',
                    'severity': 'info'
                })

    return pd.DataFrame(violations)


def plot_spc_chart(data, timestamps, metric_name, limits,
                   spec_limits=None, violations=None,
                   highlight_recent=None):
    """
    Create interactive SPC chart with Plotly.

    Args:
        data: Series of measurements
        timestamps: Corresponding timestamps
        metric_name: Name of the metric (e.g., "Cd", "Mass Flow")
        limits: Dict from calculate_control_limits()
        spec_limits: Optional dict with 'usl', 'lsl' (specification limits)
        violations: DataFrame from detect_control_violations()
        highlight_recent: Number of most recent points to highlight
    """

    fig = go.Figure()

    # Main data points
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=data,
        mode='lines+markers',
        name='Measurement',
        line=dict(color='#2C3E50', width=2),
        marker=dict(size=8, color='#3498DB'),
        hovertemplate='%{x}<br>Value: %{y:.4f}<extra></extra>'
    ))

    # Highlight recent points
    if highlight_recent and highlight_recent > 0:
        recent_data = data.iloc[-highlight_recent:]
        recent_ts = timestamps.iloc[-highlight_recent:]
        fig.add_trace(go.Scatter(
            x=recent_ts,
            y=recent_data,
            mode='markers',
            name='Recent Tests',
            marker=dict(size=12, color='#F39C12', symbol='star'),
            hovertemplate='RECENT<br>%{x}<br>Value: %{y:.4f}<extra></extra>'
        ))

    # Control limits
    mean = limits['mean']
    ucl = limits['ucl']
    lcl = limits['lcl']
    uwl = limits['uwl']
    lwl = limits['lwl']

    # Mean line
    fig.add_hline(
        y=mean,
        line_dash="solid",
        line_color="green",
        line_width=2,
        annotation_text=f"Mean: {mean:.4f}",
        annotation_position="right"
    )

    # UCL/LCL (3-sigma)
    fig.add_hline(
        y=ucl,
        line_dash="dash",
        line_color="red",
        annotation_text=f"UCL (+3σ): {ucl:.4f}",
        annotation_position="right"
    )
    fig.add_hline(
        y=lcl,
        line_dash="dash",
        line_color="red",
        annotation_text=f"LCL (-3σ): {lcl:.4f}",
        annotation_position="right"
    )

    # Warning limits (2-sigma)
    fig.add_hline(
        y=uwl,
        line_dash="dot",
        line_color="orange",
        line_width=1,
        annotation_text=f"+2σ: {uwl:.4f}",
        annotation_position="left",
        annotation_font_size=10
    )
    fig.add_hline(
        y=lwl,
        line_dash="dot",
        line_color="orange",
        line_width=1,
        annotation_text=f"-2σ: {lwl:.4f}",
        annotation_position="left",
        annotation_font_size=10
    )

    # Specification limits (if provided)
    if spec_limits:
        if 'usl' in spec_limits:
            fig.add_hline(
                y=spec_limits['usl'],
                line_dash="dashdot",
                line_color="purple",
                line_width=2,
                annotation_text=f"USL: {spec_limits['usl']:.4f}",
                annotation_position="right"
            )
        if 'lsl' in spec_limits:
            fig.add_hline(
                y=spec_limits['lsl'],
                line_dash="dashdot",
                line_color="purple",
                line_width=2,
                annotation_text=f"LSL: {spec_limits['lsl']:.4f}",
                annotation_position="right"
            )

    # Mark violations
    if violations is not None and not violations.empty:
        for severity in ['critical', 'warning', 'info']:
            subset = violations[violations['severity'] == severity]
            if not subset.empty:
                color_map = {
                    'critical': 'red',
                    'warning': 'orange',
                    'info': 'yellow'
                }
                symbol_map = {
                    'critical': 'x',
                    'warning': 'diamond',
                    'info': 'triangle-up'
                }

                fig.add_trace(go.Scatter(
                    x=subset['timestamp'],
                    y=subset['value'],
                    mode='markers',
                    name=f'{severity.title()} Violation',
                    marker=dict(
                        size=15,
                        color=color_map[severity],
                        symbol=symbol_map[severity],
                        line=dict(width=2, color='white')
                    ),
                    text=subset['rule'],
                    hovertemplate='<b>VIOLATION</b><br>%{text}<br>Value: %{y:.4f}<extra></extra>'
                ))

    # Layout
    fig.update_layout(
        title=f"SPC Chart: {metric_name}",
        xaxis_title="Test Date",
        yaxis_title=metric_name,
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    # Add shaded zones
    # Red zone (beyond control limits)
    fig.add_hrect(
        y0=ucl, y1=ucl * 1.1 if ucl > 0 else ucl * 0.9,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0
    )
    fig.add_hrect(
        y0=lcl * 0.9 if lcl > 0 else lcl * 1.1, y1=lcl,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0
    )

    return fig


def calculate_process_capability(data, spec_limits):
    """
    Calculate Cp, Cpk, Pp, Ppk indices.

    Args:
        data: Series of measurements
        spec_limits: Dict with 'usl' and/or 'lsl'

    Returns:
        dict with capability indices
    """
    data = pd.Series(data).dropna()

    if len(data) < 2:
        return {}

    mean = data.mean()
    std = data.std()

    results = {
        'mean': mean,
        'std': std,
        'n': len(data)
    }

    # Process Capability (Cp, Cpk) - uses within-subgroup variation
    # For simplicity, using std (in production, use R-bar/d2 or S-bar/c4)
    if 'usl' in spec_limits and 'lsl' in spec_limits:
        usl = spec_limits['usl']
        lsl = spec_limits['lsl']

        # Cp: potential capability
        cp = (usl - lsl) / (6 * std)
        results['Cp'] = cp

        # Cpk: actual capability (accounts for centering)
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)
        results['Cpk'] = cpk
        results['CPU'] = cpu
        results['CPL'] = cpl

    elif 'usl' in spec_limits:
        usl = spec_limits['usl']
        cpu = (usl - mean) / (3 * std)
        results['CPU'] = cpu
        results['Cpk'] = cpu

    elif 'lsl' in spec_limits:
        lsl = spec_limits['lsl']
        cpl = (mean - lsl) / (3 * std)
        results['CPL'] = cpl
        results['Cpk'] = cpl

    # Interpretation
    if 'Cpk' in results:
        cpk = results['Cpk']
        if cpk >= 2.0:
            results['interpretation'] = 'Excellent (6σ capable)'
        elif cpk >= 1.67:
            results['interpretation'] = 'Very Good (5σ capable)'
        elif cpk >= 1.33:
            results['interpretation'] = 'Good (4σ capable)'
        elif cpk >= 1.0:
            results['interpretation'] = 'Adequate (3σ capable)'
        else:
            results['interpretation'] = 'Poor (< 3σ capable)'

    return results