"""
Operating Envelope Analysis

Visualizes test operating points and defines safe operating envelopes for hot fire tests.

Key features:
- O/F ratio vs Chamber Pressure scatter plots
- Rectangular operating envelope calculation
- Successful vs failed ignition visualization
- Statistical bounds (min/max with margins)

Version: 1.0.0
Created: 2026-01-16
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class OperatingEnvelope:
    """
    Rectangular operating envelope defined by min/max bounds.

    Attributes:
        of_min: Minimum O/F ratio
        of_max: Maximum O/F ratio
        pc_min: Minimum chamber pressure (bar)
        pc_max: Maximum chamber pressure (bar)
        n_tests: Number of tests used to define envelope
        margin_pct: Safety margin applied (%)
    """
    of_min: float
    of_max: float
    pc_min: float
    pc_max: float
    n_tests: int
    margin_pct: float = 0.0

    def contains_point(self, of_ratio: float, pc_bar: float) -> bool:
        """Check if a point is within the operating envelope."""
        return (self.of_min <= of_ratio <= self.of_max and
                self.pc_min <= pc_bar <= self.pc_max)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'of_min': self.of_min,
            'of_max': self.of_max,
            'pc_min': self.pc_min,
            'pc_max': self.pc_max,
            'n_tests': self.n_tests,
            'margin_pct': self.margin_pct,
        }


def calculate_operating_envelope(
    campaign_df: pd.DataFrame,
    of_column: str = 'avg_of_ratio',
    pc_column: str = 'avg_pc_bar',
    ignition_column: Optional[str] = 'ignition_successful',
    margin_pct: float = 10.0,
    filter_successful_only: bool = True
) -> OperatingEnvelope:
    """
    Calculate rectangular operating envelope from campaign data.

    Uses min/max bounds with optional safety margin. Can filter to only
    successful ignitions or include all tests.

    Args:
        campaign_df: DataFrame with test results from campaign
        of_column: Column name for O/F ratio
        pc_column: Column name for chamber pressure
        ignition_column: Column name for ignition success flag (1=success, 0=fail)
        margin_pct: Safety margin to add (percent of range), default 10%
        filter_successful_only: If True, only use successful ignitions

    Returns:
        OperatingEnvelope object with min/max bounds

    Raises:
        ValueError: If insufficient data or required columns missing
    """
    # Check required columns exist
    if of_column not in campaign_df.columns:
        raise ValueError(f"Column '{of_column}' not found in campaign data")
    if pc_column not in campaign_df.columns:
        raise ValueError(f"Column '{pc_column}' not found in campaign data")

    # Filter data
    filtered_df = campaign_df.copy()

    if filter_successful_only and ignition_column and ignition_column in campaign_df.columns:
        # Only use successful ignitions
        filtered_df = filtered_df[filtered_df[ignition_column] == 1]

    # Remove NaN values
    filtered_df = filtered_df[[of_column, pc_column]].dropna()

    if len(filtered_df) < 2:
        raise ValueError(
            f"Insufficient data to calculate envelope: {len(filtered_df)} points "
            "(need at least 2)"
        )

    # Get min/max values
    of_min = filtered_df[of_column].min()
    of_max = filtered_df[of_column].max()
    pc_min = filtered_df[pc_column].min()
    pc_max = filtered_df[pc_column].max()

    # Apply safety margin
    of_range = of_max - of_min
    pc_range = pc_max - pc_min

    of_margin = of_range * (margin_pct / 100.0)
    pc_margin = pc_range * (margin_pct / 100.0)

    # Expand envelope by margin
    of_min_safe = of_min - of_margin
    of_max_safe = of_max + of_margin
    pc_min_safe = pc_min - pc_margin
    pc_max_safe = pc_max + pc_margin

    # Don't allow negative values
    of_min_safe = max(0, of_min_safe)
    pc_min_safe = max(0, pc_min_safe)

    return OperatingEnvelope(
        of_min=of_min_safe,
        of_max=of_max_safe,
        pc_min=pc_min_safe,
        pc_max=pc_max_safe,
        n_tests=len(filtered_df),
        margin_pct=margin_pct,
    )


def plot_operating_envelope(
    campaign_df: pd.DataFrame,
    envelope: Optional[OperatingEnvelope] = None,
    of_column: str = 'avg_of_ratio',
    pc_column: str = 'avg_pc_bar',
    test_id_column: str = 'test_id',
    ignition_column: Optional[str] = 'ignition_successful',
    title: str = "Hot Fire Operating Envelope",
    show_envelope: bool = True,
    show_test_ids: bool = True,
) -> go.Figure:
    """
    Create interactive operating envelope plot with Plotly.

    Shows O/F ratio vs chamber pressure scatter plot with:
    - Color-coded successful/failed ignitions
    - Rectangular operating envelope
    - Test ID labels
    - Hover info with test details

    Args:
        campaign_df: DataFrame with test results
        envelope: Pre-calculated envelope (or None to calculate from data)
        of_column: Column name for O/F ratio
        pc_column: Column name for chamber pressure
        test_id_column: Column name for test IDs
        ignition_column: Column name for ignition success flag
        title: Plot title
        show_envelope: If True, draw the operating envelope rectangle
        show_test_ids: If True, show test IDs as text labels

    Returns:
        Plotly Figure object

    Example:
        >>> fig = plot_operating_envelope(campaign_df)
        >>> fig.show()  # Interactive plot in browser
        >>> fig.write_html("envelope.html")  # Save to file
    """
    # Calculate envelope if not provided
    if envelope is None and show_envelope:
        envelope = calculate_operating_envelope(
            campaign_df,
            of_column=of_column,
            pc_column=pc_column,
            ignition_column=ignition_column
        )

    # Prepare data
    plot_df = campaign_df[[of_column, pc_column, test_id_column]].copy()

    # Add ignition status if available
    has_ignition_data = ignition_column and ignition_column in campaign_df.columns
    if has_ignition_data:
        plot_df['ignition'] = campaign_df[ignition_column]
    else:
        plot_df['ignition'] = 1  # Assume all successful if no data

    # Remove NaN
    plot_df = plot_df.dropna(subset=[of_column, pc_column])

    # Create figure
    fig = go.Figure()

    # Plot successful ignitions (green)
    if has_ignition_data:
        successful = plot_df[plot_df['ignition'] == 1]
        if len(successful) > 0:
            fig.add_trace(go.Scatter(
                x=successful[of_column],
                y=successful[pc_column],
                mode='markers+text' if show_test_ids else 'markers',
                name='Successful Ignition',
                text=successful[test_id_column] if show_test_ids else None,
                textposition='top center',
                textfont=dict(size=8),
                marker=dict(
                    size=12,
                    color='green',
                    symbol='circle',
                    line=dict(width=2, color='darkgreen')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                              'O/F: %{x:.3f}<br>' +
                              'Pc: %{y:.2f} bar<br>' +
                              'Status: ✓ Success<extra></extra>',
            ))

        # Plot failed ignitions (red)
        failed = plot_df[plot_df['ignition'] == 0]
        if len(failed) > 0:
            fig.add_trace(go.Scatter(
                x=failed[of_column],
                y=failed[pc_column],
                mode='markers+text' if show_test_ids else 'markers',
                name='Failed Ignition',
                text=failed[test_id_column] if show_test_ids else None,
                textposition='top center',
                textfont=dict(size=8),
                marker=dict(
                    size=12,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                              'O/F: %{x:.3f}<br>' +
                              'Pc: %{y:.2f} bar<br>' +
                              'Status: ✗ Failed<extra></extra>',
            ))
    else:
        # All points same color if no ignition data
        fig.add_trace(go.Scatter(
            x=plot_df[of_column],
            y=plot_df[pc_column],
            mode='markers+text' if show_test_ids else 'markers',
            name='Test Points',
            text=plot_df[test_id_column] if show_test_ids else None,
            textposition='top center',
            textfont=dict(size=8),
            marker=dict(
                size=12,
                color='blue',
                symbol='circle',
                line=dict(width=2, color='darkblue')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                          'O/F: %{x:.3f}<br>' +
                          'Pc: %{y:.2f} bar<extra></extra>',
        ))

    # Draw operating envelope rectangle
    if show_envelope and envelope:
        # Rectangle corners
        fig.add_trace(go.Scatter(
            x=[envelope.of_min, envelope.of_max, envelope.of_max, envelope.of_min, envelope.of_min],
            y=[envelope.pc_min, envelope.pc_min, envelope.pc_max, envelope.pc_max, envelope.pc_min],
            mode='lines',
            name=f'Operating Envelope ({envelope.margin_pct:.0f}% margin)',
            line=dict(
                color='rgba(100, 100, 255, 0.8)',
                width=3,
                dash='dash'
            ),
            fill='toself',
            fillcolor='rgba(100, 100, 255, 0.1)',
            hovertemplate='<b>Operating Envelope</b><br>' +
                          f'O/F: {envelope.of_min:.3f} - {envelope.of_max:.3f}<br>' +
                          f'Pc: {envelope.pc_min:.2f} - {envelope.pc_max:.2f} bar<br>' +
                          f'Margin: {envelope.margin_pct:.0f}%<br>' +
                          f'Based on {envelope.n_tests} tests<extra></extra>',
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family='Arial, sans-serif')
        ),
        xaxis=dict(
            title=dict(text='Mixture Ratio (O/F)', font=dict(size=16)),
            tickfont=dict(size=14),
            gridcolor='lightgray',
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='Chamber Pressure (bar)', font=dict(size=16)),
            tickfont=dict(size=14),
            gridcolor='lightgray',
            zeroline=False,
        ),
        hovermode='closest',
        plot_bgcolor='white',
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1,
        ),
        width=900,
        height=700,
    )

    return fig


def create_envelope_report(
    campaign_df: pd.DataFrame,
    envelope: OperatingEnvelope,
    campaign_name: str = "Unnamed Campaign"
) -> str:
    """
    Generate HTML report with operating envelope details.

    Args:
        campaign_df: DataFrame with test results
        envelope: Calculated operating envelope
        campaign_name: Name of the campaign

    Returns:
        HTML string with envelope report
    """
    html = f"""
    <div class="operating-envelope-report">
        <h2>Operating Envelope: {campaign_name}</h2>

        <div class="envelope-summary">
            <h3>Envelope Bounds</h3>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Minimum</th>
                    <th>Maximum</th>
                </tr>
                <tr>
                    <td><b>Mixture Ratio (O/F)</b></td>
                    <td>{envelope.of_min:.3f}</td>
                    <td>{envelope.of_max:.3f}</td>
                </tr>
                <tr>
                    <td><b>Chamber Pressure (bar)</b></td>
                    <td>{envelope.pc_min:.2f}</td>
                    <td>{envelope.pc_max:.2f}</td>
                </tr>
            </table>
        </div>

        <div class="envelope-stats">
            <h3>Statistics</h3>
            <ul>
                <li><b>Tests in Envelope:</b> {envelope.n_tests}</li>
                <li><b>Safety Margin:</b> {envelope.margin_pct:.0f}%</li>
                <li><b>O/F Range:</b> {envelope.of_max - envelope.of_min:.3f}</li>
                <li><b>Pc Range:</b> {envelope.pc_max - envelope.pc_min:.2f} bar</li>
            </ul>
        </div>
    </div>
    """

    return html


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'OperatingEnvelope',
    'calculate_operating_envelope',
    'plot_operating_envelope',
    'create_envelope_report',
]
