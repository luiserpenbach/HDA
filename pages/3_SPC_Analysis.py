"""
Statistical Process Control Page
================================
Monitor process stability with control charts, Western Electric rules,
trend detection, and capability indices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.campaign_manager_v2 import get_available_campaigns, get_campaign_data
from core.spc import (
    create_imr_chart,
    analyze_campaign_spc,
    format_spc_summary,
    calculate_capability,
    ViolationType,
)

st.set_page_config(page_title="SPC Analysis", page_icon="SPC", layout="wide")

st.title("Statistical Process Control")
st.markdown("Monitor process stability with control charts and capability analysis.")


def plot_control_chart(analysis, spec_limits=None):
    """Create interactive control chart."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"Individual Chart: {analysis.parameter_name}", "Moving Range"]
    )
    
    # Extract data
    x = list(range(len(analysis.points)))
    y = [p.value for p in analysis.points]
    test_ids = [p.test_id for p in analysis.points]
    
    # Colors based on control status
    colors = ['red' if not p.in_control else 'blue' for p in analysis.points]
    
    # Individual chart
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers+lines',
            marker=dict(color=colors, size=10),
            line=dict(color='lightblue', width=1),
            text=test_ids,
            hovertemplate='%{text}<br>Value: %{y:.4f}<extra></extra>',
            name='Data'
        ),
        row=1, col=1
    )
    
    # Control limits
    fig.add_hline(y=analysis.limits.center_line, line_dash="solid", 
                  line_color="green", row=1, col=1,
                  annotation_text=f"CL: {analysis.limits.center_line:.4f}")
    fig.add_hline(y=analysis.limits.ucl, line_dash="dash", 
                  line_color="red", row=1, col=1,
                  annotation_text=f"UCL: {analysis.limits.ucl:.4f}")
    fig.add_hline(y=analysis.limits.lcl, line_dash="dash", 
                  line_color="red", row=1, col=1,
                  annotation_text=f"LCL: {analysis.limits.lcl:.4f}")
    
    # Warning limits (2σ)
    if analysis.limits.uwl:
        fig.add_hline(y=analysis.limits.uwl, line_dash="dot", 
                      line_color="orange", row=1, col=1, opacity=0.5)
        fig.add_hline(y=analysis.limits.lwl, line_dash="dot", 
                      line_color="orange", row=1, col=1, opacity=0.5)
    
    # Spec limits if provided
    if spec_limits:
        if spec_limits.get('usl') is not None:
            fig.add_hline(y=spec_limits['usl'], line_dash="dashdot", 
                          line_color="purple", row=1, col=1,
                          annotation_text=f"USL: {spec_limits['usl']:.4f}")
        if spec_limits.get('lsl') is not None:
            fig.add_hline(y=spec_limits['lsl'], line_dash="dashdot", 
                          line_color="purple", row=1, col=1,
                          annotation_text=f"LSL: {spec_limits['lsl']:.4f}")
    
    # Moving range chart
    mr = np.abs(np.diff(y))
    mr_x = list(range(1, len(y)))
    
    fig.add_trace(
        go.Scatter(
            x=mr_x, y=mr,
            mode='markers+lines',
            marker=dict(color='blue', size=6),
            line=dict(color='lightblue', width=1),
            name='MR'
        ),
        row=2, col=1
    )
    
    # MR limits
    mr_bar = np.mean(mr)
    mr_ucl = 3.267 * mr_bar  # D4 for n=2
    
    fig.add_hline(y=mr_bar, line_dash="solid", line_color="green", row=2, col=1)
    fig.add_hline(y=mr_ucl, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Test Number", row=2, col=1)
    fig.update_yaxes(title_text=analysis.parameter_name, row=1, col=1)
    fig.update_yaxes(title_text="Moving Range", row=2, col=1)
    
    return fig


def plot_capability_histogram(values, usl=None, lsl=None, target=None):
    """Create capability histogram with spec limits."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=20,
        name='Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Spec limits
    if lsl is not None:
        fig.add_vline(x=lsl, line_dash="dash", line_color="red",
                      annotation_text=f"LSL: {lsl:.4f}")
    if usl is not None:
        fig.add_vline(x=usl, line_dash="dash", line_color="red",
                      annotation_text=f"USL: {usl:.4f}")
    if target is not None:
        fig.add_vline(x=target, line_dash="solid", line_color="green",
                      annotation_text=f"Target: {target:.4f}")
    
    # Mean
    mean = np.mean(values)
    fig.add_vline(x=mean, line_dash="dot", line_color="blue",
                  annotation_text=f"Mean: {mean:.4f}")
    
    fig.update_layout(
        title="Process Capability Distribution",
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=400,
    )
    
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Settings")
    
    # Campaign selection
    campaigns = get_available_campaigns()
    
    if campaigns:
        campaign_names = [c['name'] for c in campaigns]
        selected_campaign = st.selectbox("Select Campaign", campaign_names)
    else:
        st.warning("No campaigns found")
        selected_campaign = None
    
    if selected_campaign:
        df = get_campaign_data(selected_campaign)
        
        if df is not None and len(df) > 0:
            # Get numeric columns
            numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
            
            # Filter to likely metric columns
            metric_cols = [c for c in numeric_cols if c.startswith('avg_') or c in ['Cd', 'Isp']]
            
            if metric_cols:
                selected_parameter = st.selectbox("Parameter", metric_cols)
            else:
                selected_parameter = st.selectbox("Parameter", numeric_cols)
            
            st.divider()
            
            # Specification limits
            st.subheader("Specification Limits")
            
            use_specs = st.checkbox("Use specification limits")
            
            if use_specs:
                col_min = float(df[selected_parameter].min())
                col_max = float(df[selected_parameter].max())
                col_mean = float(df[selected_parameter].mean())
                col_range = col_max - col_min
                
                lsl = st.number_input("LSL", value=col_mean - col_range)
                usl = st.number_input("USL", value=col_mean + col_range)
                target = st.number_input("Target", value=col_mean)
            else:
                lsl, usl, target = None, None, None
        else:
            df = None
            selected_parameter = None
    else:
        df = None
        selected_parameter = None

# =============================================================================
# MAIN CONTENT
# =============================================================================

if df is not None and selected_parameter:
    # Run SPC analysis
    try:
        analysis = create_imr_chart(
            df, 
            parameter=selected_parameter,
            test_id_col='test_id' if 'test_id' in df.columns else df.columns[0],
            usl=usl if use_specs else None,
            lsl=lsl if use_specs else None,
            target=target if use_specs else None,
        )
    except Exception as e:
        st.error(f"Analysis error: {e}")
        st.stop()
    
    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status_color = "green" if analysis.n_violations == 0 else "red"
        st.markdown(f"### :{status_color}[{'In Control' if analysis.n_violations == 0 else 'Out of Control'}]")
    
    with col2:
        st.metric("Points", analysis.n_points)
    
    with col3:
        st.metric("Violations", analysis.n_violations)
    
    with col4:
        if analysis.capability and analysis.capability.cpk is not None:
            cpk = analysis.capability.cpk
            cpk_color = "green" if cpk >= 1.33 else ("orange" if cpk >= 1.0 else "red")
            st.metric("Cpk", f"{cpk:.2f}")
        else:
            st.metric("Cpk", "N/A")
    
    with col5:
        if analysis.has_trend:
            st.metric("Trend", analysis.trend_direction.title())
        else:
            st.metric("Trend", "None")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Control Chart", "Capability", "[WARN] Violations", "Summary"])
    
    # =============================================================================
    # TAB 1: Control Chart
    # =============================================================================
    
    with tab1:
        spec_limits = {'usl': usl, 'lsl': lsl} if use_specs else None
        fig = plot_control_chart(analysis, spec_limits)
        st.plotly_chart(fig, use_container_width=True)
        
        # Control limits table
        with st.expander("Control Limits"):
            limits_data = {
                'Limit': ['UCL (3σ)', 'UWL (2σ)', 'Center Line', 'LWL (2σ)', 'LCL (3σ)'],
                'Value': [
                    f"{analysis.limits.ucl:.4f}",
                    f"{analysis.limits.uwl:.4f}" if analysis.limits.uwl else "N/A",
                    f"{analysis.limits.center_line:.4f}",
                    f"{analysis.limits.lwl:.4f}" if analysis.limits.lwl else "N/A",
                    f"{analysis.limits.lcl:.4f}",
                ]
            }
            st.table(pd.DataFrame(limits_data))
    
    # =============================================================================
    # TAB 2: Capability
    # =============================================================================
    
    with tab2:
        if use_specs and analysis.capability:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                values = df[selected_parameter].dropna().values
                fig = plot_capability_histogram(values, usl, lsl, target)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Capability Indices")
                
                cap = analysis.capability
                
                # Potential Capability (Cp family)
                st.markdown("**Potential Capability**")
                
                if cap.cp is not None:
                    st.metric("Cp", f"{cap.cp:.2f}", help="(USL - LSL) / 6σ")
                if cap.cpk is not None:
                    st.metric("Cpk", f"{cap.cpk:.2f}", help="min(Cpu, Cpl)")
                if cap.cpu is not None:
                    st.metric("Cpu", f"{cap.cpu:.2f}", help="(USL - μ) / 3σ")
                if cap.cpl is not None:
                    st.metric("Cpl", f"{cap.cpl:.2f}", help="(μ - LSL) / 3σ")
                
                st.divider()
                
                # Performance (Pp family)
                st.markdown("**Overall Performance**")
                
                if cap.pp is not None:
                    st.metric("Pp", f"{cap.pp:.2f}")
                if cap.ppk is not None:
                    st.metric("Ppk", f"{cap.ppk:.2f}")
                
                # Interpretation
                st.divider()
                st.markdown("**Interpretation**")
                st.info(cap.summary())
                
                # Sigma level
                if cap.sigma_level is not None:
                    st.metric("Sigma Level", f"{cap.sigma_level:.1f}σ")
        else:
            st.info("Enable specification limits in the sidebar to see capability analysis.")
    
    # =============================================================================
    # TAB 3: Violations
    # =============================================================================
    
    with tab3:
        st.subheader("Out-of-Control Points")
        
        ooc_points = analysis.get_out_of_control_points()
        
        if ooc_points:
            violation_data = []
            for point in ooc_points:
                violation_names = [v.value for v in point.violations]
                violation_data.append({
                    'Test ID': point.test_id,
                    'Index': point.index,
                    'Value': f"{point.value:.4f}",
                    'Violations': ', '.join(violation_names)
                })
            
            st.dataframe(pd.DataFrame(violation_data), use_container_width=True, hide_index=True)
            
            # Violation type summary
            st.subheader("Violation Summary")
            
            violation_counts = {}
            for point in ooc_points:
                for v in point.violations:
                    violation_counts[v.value] = violation_counts.get(v.value, 0) + 1
            
            for vtype, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
                st.markdown(f"- **{vtype}**: {count} occurrences")
        else:
            st.success("[PASS] No out-of-control points detected")
        
        # Trend detection
        st.divider()
        st.subheader("Trend Analysis")
        
        if analysis.has_trend:
            st.warning(f"[WARN] **Trend Detected**: {analysis.trend_direction}")
            st.text(f"Slope: {analysis.trend_slope:.6f} per test")
            st.markdown("""
            A significant trend indicates the process may be drifting. 
            Investigate potential causes such as:
            - Tool wear
            - Environmental changes
            - Material variation
            - Measurement drift
            """)
        else:
            st.success("[PASS] No significant trend detected")
    
    # =============================================================================
    # TAB 4: Summary
    # =============================================================================
    
    with tab4:
        st.subheader("SPC Analysis Summary")
        
        summary_md = format_spc_summary(analysis)
        st.markdown(summary_md)
        
        # Export
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "Download Summary (Markdown)",
                summary_md,
                file_name=f"spc_summary_{selected_parameter}.md",
                mime="text/markdown"
            )
        
        with col2:
            # Export detailed data
            export_data = {
                'parameter': selected_parameter,
                'n_points': analysis.n_points,
                'n_violations': analysis.n_violations,
                'center_line': analysis.limits.center_line,
                'ucl': analysis.limits.ucl,
                'lcl': analysis.limits.lcl,
                'has_trend': analysis.has_trend,
                'trend_direction': analysis.trend_direction,
            }
            
            if analysis.capability:
                export_data['cpk'] = analysis.capability.cpk
                export_data['ppk'] = analysis.capability.ppk
            
            import json
            st.download_button(
                "Download Results (JSON)",
                json.dumps(export_data, indent=2),
                file_name=f"spc_results_{selected_parameter}.json",
                mime="application/json"
            )

else:
    st.info("Select a campaign and parameter in the sidebar to begin SPC analysis.")
    
    # Show SPC overview
    st.divider()
    st.header("About Statistical Process Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Control Charts
        
        Control charts help distinguish between:
        - **Common cause variation**: Inherent process variation
        - **Special cause variation**: Unusual events requiring investigation
        
        The I-MR (Individual-Moving Range) chart is ideal for propulsion test data
        where each test is a single measurement.
        """)
        
        st.markdown("""
        ### Western Electric Rules
        
        1. Point beyond 3σ
        2. 2 of 3 consecutive in Zone A
        3. 4 of 5 consecutive beyond 1σ
        4. 8 consecutive on same side
        5. 6 consecutive increasing/decreasing
        6. 14 consecutive alternating
        """)
    
    with col2:
        st.markdown("""
        ### Capability Indices
        
        | Index | Meaning | Good Value |
        |-------|---------|------------|
        | Cp | Potential capability | ≥ 1.33 |
        | Cpk | Actual capability | ≥ 1.33 |
        | Pp | Overall performance | ≥ 1.33 |
        | Ppk | Overall actual | ≥ 1.33 |
        
        - **Cpk < 1.0**: Process not capable
        - **1.0 ≤ Cpk < 1.33**: Marginal
        - **Cpk ≥ 1.33**: Good
        - **Cpk ≥ 1.67**: Excellent (6σ)
        """)
