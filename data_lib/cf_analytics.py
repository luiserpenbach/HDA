
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def analyze_cd_by_part(df, pressure_col='avg_p_up_bar', cd_col='avg_cd_CALC'):
    """
    Analyze Cd variation across different parts.

    Returns:
        dict with statistics per part
    """
    results = {}

    for part in df['part'].unique():
        part_data = df[df['part'] == part]

        results[part] = {
            'n_tests': len(part_data),
            'cd_mean': part_data[cd_col].mean(),
            'cd_std': part_data[cd_col].std(),
            'cd_min': part_data[cd_col].min(),
            'cd_max': part_data[cd_col].max(),
            'pressure_range': (part_data[pressure_col].min(), part_data[pressure_col].max())
        }

    return results


def plot_cd_vs_pressure(df, part_filter=None, fluid_filter=None):
    """
    Create Cd vs Pressure plot with trendlines.
    Grouped by part and/or fluid.
    """
    # Apply filters
    df_plot = df.copy()
    if part_filter:
        df_plot = df_plot[df_plot['part'].isin(part_filter)]
    if fluid_filter:
        df_plot = df_plot[df_plot['fluid'].isin(fluid_filter)]

    fig = go.Figure()

    # Group by part
    for part in df_plot['part'].unique():
        part_data = df_plot[df_plot['part'] == part]

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=part_data['avg_p_up_bar'],
            y=part_data['avg_cd_CALC'],
            mode='markers',
            name=part,
            text=part_data['test_id'],
            marker=dict(size=10),
            hovertemplate='<b>%{text}</b><br>P: %{x:.2f} bar<br>Cd: %{y:.4f}<extra></extra>'
        ))

        # Trendline (linear regression)
        if len(part_data) >= 2:
            x = part_data['avg_p_up_bar'].values
            y = part_data['avg_cd_CALC'].values

            # Remove NaN
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]

            if len(x) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                x_range = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_range + intercept

                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_fit,
                    mode='lines',
                    name=f'{part} Trend (R²={r_value ** 2:.3f})',
                    line=dict(dash='dash'),
                    showlegend=True,
                    hoverinfo='skip'
                ))

    fig.update_layout(
        title="Discharge Coefficient vs Pressure",
        xaxis_title="Upstream Pressure (bar)",
        yaxis_title="Cd",
        height=500,
        hovermode='closest'
    )

    return fig


def plot_cd_by_fluid(df):
    """Box plot of Cd grouped by fluid type."""
    fig = go.Figure()

    for fluid in df['fluid'].unique():
        fluid_data = df[df['fluid'] == fluid]

        fig.add_trace(go.Box(
            y=fluid_data['avg_cd_CALC'],
            name=fluid,
            boxmean='sd'  # Show mean and std dev
        ))

    fig.update_layout(
        title="Cd Distribution by Fluid",
        yaxis_title="Cd",
        height=400
    )

    return fig


def generate_cf_summary_stats(df):
    """Generate comprehensive campaign statistics."""
    summary = {
        'total_tests': len(df),
        'unique_parts': df['part'].nunique(),
        'unique_fluids': df['fluid'].nunique(),
        'date_range': (df['test_timestamp'].min(), df['test_timestamp'].max()),
        'cd_overall': {
            'mean': df['avg_cd_CALC'].mean(),
            'std': df['avg_cd_CALC'].std(),
            'min': df['avg_cd_CALC'].min(),
            'max': df['avg_cd_CALC'].max()
        },
        'pressure_range': {
            'min': df['avg_p_up_bar'].min(),
            'max': df['avg_p_up_bar'].max()
        },
        'flow_range': {
            'min': df['avg_mf_g_s'].min(),
            'max': df['avg_mf_g_s'].max()
        }
    }

    return summary