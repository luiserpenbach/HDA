# Create new file: data_lib/hf_analytics.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_operating_envelope(df, x_metric='avg_of_ratio', y_metric='avg_pc_bar'):
    """
    Plot operating envelope (e.g., O/F vs Pc).
    Shows the tested parameter space.
    """
    fig = go.Figure()

    # Color by performance (Isp or C* efficiency)
    if 'eta_isp_pct' in df.columns:
        color_metric = 'eta_isp_pct'
        color_label = 'Isp Efficiency (%)'
    else:
        color_metric = 'avg_isp_s'
        color_label = 'Isp (s)'

    fig.add_trace(go.Scatter(
        x=df[x_metric],
        y=df[y_metric],
        mode='markers',
        marker=dict(
            size=12,
            color=df[color_metric],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_label),
            line=dict(width=1, color='white')
        ),
        text=df['test_id'],
        hovertemplate='<b>%{text}</b><br>O/F: %{x:.2f}<br>Pc: %{y:.1f} bar<br>' +
                      f'{color_label}: %{{marker.color:.1f}}<extra></extra>'
    ))

    fig.update_layout(
        title="Operating Envelope",
        xaxis_title="O/F Ratio",
        yaxis_title="Chamber Pressure (bar)",
        height=500
    )

    return fig


def plot_performance_metrics(df):
    """
    Multi-panel plot showing key performance metrics over time.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Isp Efficiency', 'C* Efficiency', 'O/F Ratio', 'Chamber Pressure'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Isp Efficiency
    if 'eta_isp_pct' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['test_timestamp'],
            y=df['eta_isp_pct'],
            mode='lines+markers',
            name='η Isp',
            line=dict(color='#3498DB')
        ), row=1, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="green", row=1, col=1)

    # C* Efficiency
    if 'eta_c_star_pct' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['test_timestamp'],
            y=df['eta_c_star_pct'],
            mode='lines+markers',
            name='η C*',
            line=dict(color='#E74C3C')
        ), row=1, col=2)
        fig.add_hline(y=100, line_dash="dash", line_color="green", row=1, col=2)

    # O/F Ratio
    fig.add_trace(go.Scatter(
        x=df['test_timestamp'],
        y=df['avg_of_ratio'],
        mode='lines+markers',
        name='O/F',
        line=dict(color='#9B59B6')
    ), row=2, col=1)

    # Chamber Pressure
    fig.add_trace(go.Scatter(
        x=df['test_timestamp'],
        y=df['avg_pc_bar'],
        mode='lines+markers',
        name='Pc',
        line=dict(color='#F39C12')
    ), row=2, col=2)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="η (%)", row=1, col=1)
    fig.update_yaxes(title_text="η (%)", row=1, col=2)
    fig.update_yaxes(title_text="O/F", row=2, col=1)
    fig.update_yaxes(title_text="Pc (bar)", row=2, col=2)

    fig.update_layout(height=600, showlegend=False)

    return fig


def plot_standard_hot_fire(test_df, time_col='time_s', config=None):
    """
    Generate standard 4-panel hot fire plot:
    1. Chamber Pressure
    2. Thrust
    3. Mass Flows
    4. O/F Ratio
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Chamber Pressure', 'Thrust', 'Mass Flows', 'O/F Ratio'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    cols_cfg = config.get('columns', {}) if config else {}

    # 1. Chamber Pressure
    col_pc = cols_cfg.get('chamber_pressure')
    if col_pc and col_pc in test_df.columns:
        fig.add_trace(go.Scatter(
            x=test_df[time_col],
            y=test_df[col_pc],
            name='Pc',
            line=dict(color='#E74C3C', width=2)
        ), row=1, col=1)
        fig.update_yaxes(title_text="Pc (bar)", row=1, col=1)

    # 2. Thrust
    col_thrust = cols_cfg.get('thrust')
    if col_thrust and col_thrust in test_df.columns:
        fig.add_trace(go.Scatter(
            x=test_df[time_col],
            y=test_df[col_thrust],
            name='Thrust',
            line=dict(color='#3498DB', width=2)
        ), row=1, col=2)
        fig.update_yaxes(title_text="Thrust (N)", row=1, col=2)

    # 3. Mass Flows
    col_mf_ox = cols_cfg.get('mass_flow_ox')
    col_mf_fuel = cols_cfg.get('mass_flow_fuel')

    if col_mf_ox and col_mf_ox in test_df.columns:
        fig.add_trace(go.Scatter(
            x=test_df[time_col],
            y=test_df[col_mf_ox],
            name='Ox',
            line=dict(color='#1ABC9C', width=2)
        ), row=2, col=1)

    if col_mf_fuel and col_mf_fuel in test_df.columns:
        fig.add_trace(go.Scatter(
            x=test_df[time_col],
            y=test_df[col_mf_fuel],
            name='Fuel',
            line=dict(color='#F39C12', width=2)
        ), row=2, col=1)

    fig.update_yaxes(title_text="Mass Flow (g/s)", row=2, col=1)

    # 4. O/F Ratio
    if 'of_ratio' in test_df.columns:
        fig.add_trace(go.Scatter(
            x=test_df[time_col],
            y=test_df['of_ratio'],
            name='O/F',
            line=dict(color='#9B59B6', width=2)
        ), row=2, col=2)
        fig.update_yaxes(title_text="O/F Ratio", row=2, col=2)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)

    fig.update_layout(height=700, showlegend=True)

    return fig


def generate_hf_summary_stats(df):
    """Generate comprehensive hot fire campaign statistics."""
    summary = {
        'total_tests': len(df),
        'total_firing_time': df['duration_s'].sum(),
        'avg_duration': df['duration_s'].mean(),
        'date_range': (df['test_timestamp'].min(), df['test_timestamp'].max()),
        'performance': {
            'avg_isp': df['avg_isp_s'].mean(),
            'max_isp': df['avg_isp_s'].max(),
            'avg_c_star': df['avg_c_star_m_s'].mean() if 'avg_c_star_m_s' in df.columns else None,
            'avg_eta_isp': df['eta_isp_pct'].mean() if 'eta_isp_pct' in df.columns else None
        },
        'operating_conditions': {
            'pc_range': (df['avg_pc_bar'].min(), df['avg_pc_bar'].max()),
            'of_range': (df['avg_of_ratio'].min(), df['avg_of_ratio'].max()),
            'thrust_range': (df['avg_thrust_n'].min(), df['avg_thrust_n'].max())
        }
    }

    return summary