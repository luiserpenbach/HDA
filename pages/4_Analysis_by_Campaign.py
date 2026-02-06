"""
Analysis by Campaign Page
=========================
Comprehensive campaign analysis with summary, plots, SPC, and reports.
Includes I-MR, CUSUM, and EWMA control chart types.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import io
import zipfile
from pathlib import Path

from core.campaign_manager_v2 import (
    get_available_campaigns,
    create_campaign,
    get_campaign_info,
    get_campaign_data,
    verify_test_data_integrity,
    get_test_traceability,
)
from core.spc import (
    create_imr_chart,
    create_cusum_chart,
    create_ewma_chart,
    analyze_campaign_spc,
    format_spc_summary,
    calculate_capability,
    ViolationType,
    ControlChartType,
)
from core.reporting import generate_campaign_report, generate_test_report
from core.uncertainty import MeasurementWithUncertainty
from pages._shared_styles import apply_custom_styles, render_page_header
from pages._shared_sidebar import render_global_context

st.set_page_config(page_title="Analysis by Campaign", page_icon="AC", layout="wide")

apply_custom_styles()

render_page_header(
    title="Analysis by Campaign",
    description="Campaign analysis with SPC control charts, reports, and qualification export",
    badge_text="P1",
    badge_type="info"
)

with st.sidebar:
    # Global context at top
    context = render_global_context()
    st.divider()

    st.header("Campaign Selection")

    campaigns = get_available_campaigns()

    if campaigns:
        campaign_names = [c['name'] for c in campaigns]
        selected_campaign = st.selectbox("Select Campaign", campaign_names)

        # Show campaign info
        for c in campaigns:
            if c['name'] == selected_campaign:
                st.caption(f"Type: {c.get('type', 'unknown')}")
                st.caption(f"Tests: {c.get('test_count', 0)}")
                created = c.get('created_date')
                st.caption(f"Created: {created[:10] if created else 'N/A'}")
    else:
        selected_campaign = None
        st.info("No campaigns found")

    st.divider()

    # Create new campaign
    st.subheader("Create New Campaign")

    new_name = st.text_input("Campaign Name")
    new_type = st.selectbox("Campaign Type", ["cold_flow", "hot_fire"])
    new_desc = st.text_area("Description", height=100)

    if st.button("Create Campaign", use_container_width=True):
        if new_name:
            try:
                create_campaign(
                    campaign_name=new_name,
                    campaign_type=new_type,
                )
                st.success(f"Created campaign: {new_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Enter a campaign name")

# =============================================================================
# MAIN CONTENT - Tabs
# =============================================================================

if selected_campaign:
    # Load campaign data
    try:
        info = get_campaign_info(selected_campaign)
        df = get_campaign_data(selected_campaign)
        campaign_type = info.get('type', 'cold_flow')
    except Exception as e:
        st.error(f"Error loading campaign: {e}")
        st.stop()

    # Part/Serial filters (in sidebar, populated after data load)
    part_filter = []
    serial_filter = []

    if df is not None and len(df) > 0:
        with st.sidebar:
            st.divider()
            st.subheader("Filters")

            # Part filter
            if 'part' in df.columns:
                available_parts = sorted([p for p in df['part'].dropna().unique().tolist() if p])
                if available_parts:
                    part_filter = st.multiselect(
                        "Part",
                        available_parts,
                        default=[],
                        key="camp_part_filter"
                    )

            # Serial number filter
            if 'serial_num' in df.columns:
                available_serials = sorted([s for s in df['serial_num'].dropna().unique().tolist() if s])
                if available_serials:
                    serial_filter = st.multiselect(
                        "Serial Number",
                        available_serials,
                        default=[],
                        key="camp_serial_filter"
                    )

        # Apply part/serial filters
        if part_filter:
            df = df[df['part'].isin(part_filter)].reset_index(drop=True)
        if serial_filter:
            df = df[df['serial_num'].isin(serial_filter)].reset_index(drop=True)

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tests", len(df) if df is not None else 0)
    with col2:
        if df is not None and 'qc_passed' in df.columns:
            qc_pass = df['qc_passed'].sum()
            st.metric("QC Passed", f"{qc_pass}/{len(df)}")
        else:
            st.metric("QC Passed", "N/A")
    with col3:
        st.metric("Type", campaign_type.replace('_', ' ').title())
    with col4:
        st.metric("Schema Version", info.get('schema_version', 'N/A'))

    st.divider()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Campaign Summary",
        "Campaign Plots",
        "SPC Analysis",
        "Reports"
    ])

    # =============================================================================
    # TAB 1: Campaign Summary (from Campaign Management)
    # =============================================================================

    with tab1:
        if df is not None and len(df) > 0:
            # Subtabs for different views
            subtab1, subtab2, subtab3, subtab4 = st.tabs(["Overview", "Data Table", "Traceability", "Export"])

            with subtab1:
                st.subheader("Campaign Overview")

                # Identify metric columns
                if campaign_type == 'cold_flow':
                    primary_metric = 'avg_cd_CALC'
                    secondary_metrics = ['avg_p_up_bar', 'avg_mf_g_s', 'avg_delta_p_bar']
                else:
                    primary_metric = 'avg_isp_s'
                    secondary_metrics = ['avg_c_star_m_s', 'avg_of_ratio', 'avg_p_c_bar']

                # Time series plot of primary metric
                if primary_metric in df.columns:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        fig = go.Figure()

                        # Add data points
                        x_vals = df['test_id'] if 'test_id' in df.columns else df.index
                        y_vals = df[primary_metric]

                        # Color by QC status if available
                        if 'qc_passed' in df.columns:
                            colors = ['green' if qc else 'red' for qc in df['qc_passed']]
                        else:
                            colors = 'blue'

                        fig.add_trace(go.Scatter(
                            x=list(range(len(x_vals))),
                            y=y_vals,
                            mode='markers+lines',
                            marker=dict(color=colors, size=10),
                            line=dict(color='lightblue', width=1),
                            text=x_vals,
                            hovertemplate='%{text}<br>Value: %{y:.4f}<extra></extra>'
                        ))

                        # Add mean line
                        mean_val = y_vals.mean()
                        fig.add_hline(y=mean_val, line_dash="dash",
                                      annotation_text=f"Mean: {mean_val:.4f}")

                        fig.update_layout(
                            title=f"{primary_metric} Trend",
                            xaxis_title="Test Number",
                            yaxis_title=primary_metric,
                            height=400,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Statistics
                        st.markdown("**Statistics**")

                        valid_data = y_vals.dropna()

                        st.metric("Mean", f"{valid_data.mean():.4f}")
                        st.metric("Std Dev", f"{valid_data.std():.4f}")
                        st.metric("Min", f"{valid_data.min():.4f}")
                        st.metric("Max", f"{valid_data.max():.4f}")
                        st.metric("CV (%)", f"{(valid_data.std() / valid_data.mean() * 100):.2f}")

                # Distribution plot
                st.subheader("Distribution")

                col1, col2 = st.columns(2)

                with col1:
                    if primary_metric in df.columns:
                        fig = px.histogram(
                            df, x=primary_metric,
                            nbins=20,
                            title=f"{primary_metric} Distribution"
                        )
                        fig.add_vline(x=df[primary_metric].mean(), line_dash="dash",
                                      annotation_text="Mean")
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Show available metrics
                    available_metrics = [m for m in secondary_metrics if m in df.columns]
                    if available_metrics:
                        selected_metric = st.selectbox("Select metric", available_metrics)

                        fig = px.histogram(
                            df, x=selected_metric,
                            nbins=20,
                            title=f"{selected_metric} Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with subtab2:
                st.subheader("Test Data")

                # Column filter
                all_columns = list(df.columns)
                default_cols = ['test_id', 'test_timestamp', 'part', 'serial_num']

                if campaign_type == 'cold_flow':
                    default_cols.extend(['avg_cd_CALC', 'u_cd_CALC', 'avg_p_up_bar', 'qc_passed'])
                else:
                    default_cols.extend(['avg_isp_s', 'u_isp_s', 'avg_c_star_m_s', 'qc_passed'])

                default_cols = [c for c in default_cols if c in all_columns]

                selected_cols = st.multiselect(
                    "Columns to display",
                    all_columns,
                    default=default_cols
                )

                if selected_cols:
                    st.dataframe(df[selected_cols], use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Download
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{selected_campaign}_data.csv",
                    mime="text/csv"
                )

            with subtab3:
                st.subheader("Data Traceability")

                if 'test_id' in df.columns:
                    # Select test to verify
                    test_ids = df['test_id'].tolist()
                    selected_test = st.selectbox("Select Test", test_ids)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Traceability Record**")

                        try:
                            trace = get_test_traceability(selected_campaign, selected_test)

                            if trace:
                                for key, value in trace.items():
                                    if value is not None:
                                        st.text(f"{key}: {value}")
                            else:
                                st.info("No traceability data found")
                        except Exception as e:
                            st.error(f"Error loading traceability: {e}")

                    with col2:
                        st.markdown("**Integrity Verification**")

                        if st.button("Verify Data Integrity"):
                            try:
                                st.info("To verify integrity, upload the original test data file.")

                                original_file = st.file_uploader("Original file", type=['csv'])

                                if original_file:
                                    is_valid = verify_test_data_integrity(
                                        selected_campaign,
                                        selected_test,
                                        original_file.name
                                    )

                                    if is_valid:
                                        st.success("[PASS] Data integrity verified")
                                    else:
                                        st.error("[FAIL] Data integrity check failed")
                            except Exception as e:
                                st.error(f"Verification error: {e}")

                    # Traceability summary
                    st.divider()
                    st.markdown("**Campaign Traceability Summary**")

                    trace_cols = ['test_id', 'raw_data_hash', 'config_hash', 'analyst_username',
                                  'analysis_timestamp_utc', 'processing_version']
                    available_trace = [c for c in trace_cols if c in df.columns]

                    if available_trace:
                        st.dataframe(df[available_trace], use_container_width=True, hide_index=True)
                    else:
                        st.info("No traceability columns in campaign data")
                else:
                    st.info("No test_id column in campaign data")

            with subtab4:
                st.subheader("Export Campaign Data")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Export Options**")

                    include_uncertainties = st.checkbox("Include uncertainty columns", value=True, key="exp_unc")
                    include_traceability = st.checkbox("Include traceability columns", value=True, key="exp_trace")
                    export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"], key="exp_fmt")

                with col2:
                    st.markdown("**Export**")

                    if export_format == "CSV":
                        cols_to_export = list(df.columns)
                        if not include_uncertainties:
                            cols_to_export = [c for c in cols_to_export if not c.startswith('u_')]
                        if not include_traceability:
                            trace_cols = ['raw_data_hash', 'config_hash', 'analyst_username',
                                          'analysis_timestamp_utc', 'processing_version']
                            cols_to_export = [c for c in cols_to_export if c not in trace_cols]

                        export_df = df[cols_to_export]

                        st.download_button(
                            "Download CSV",
                            export_df.to_csv(index=False),
                            file_name=f"{selected_campaign}_export.csv",
                            mime="text/csv",
                            key="dl_csv_summary"
                        )

                    elif export_format == "Excel":
                        from core.export import export_campaign_excel
                        import tempfile

                        if st.button("Generate Excel"):
                            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
                                export_campaign_excel(
                                    df, f.name,
                                    campaign_info={'name': selected_campaign, 'type': campaign_type}
                                )

                                with open(f.name, 'rb') as excel_file:
                                    st.download_button(
                                        "Download Excel",
                                        excel_file.read(),
                                        file_name=f"{selected_campaign}_export.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )

                    elif export_format == "JSON":
                        export_data = {
                            'campaign_name': selected_campaign,
                            'campaign_type': campaign_type,
                            'export_date': datetime.now().isoformat(),
                            'test_count': len(df),
                            'data': df.to_dict(orient='records')
                        }

                        st.download_button(
                            "Download JSON",
                            json.dumps(export_data, indent=2, default=str),
                            file_name=f"{selected_campaign}_export.json",
                            mime="application/json",
                            key="dl_json_summary"
                        )
        else:
            st.info("No test data in this campaign yet.")

    # =============================================================================
    # TAB 2: Campaign Plots
    # =============================================================================

    with tab2:
        if df is not None and len(df) > 0:
            numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
            metric_cols = [c for c in numeric_cols if c.startswith('avg_')]
            plot_cols = metric_cols if metric_cols else numeric_cols

            plot_subtab1, plot_subtab2, plot_subtab3 = st.tabs([
                "Trend Analysis", "Correlation Matrix", "Multi-Parameter"
            ])

            with plot_subtab1:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    trend_param = st.selectbox("Parameter", plot_cols, key="trend_param")
                with col2:
                    trend_type = st.selectbox("Chart Type", [
                        "Trend Line", "Box Plot", "Violin", "Histogram"
                    ], key="trend_type")
                with col3:
                    group_by = st.selectbox("Group By", ["None"] + [
                        c for c in ['part', 'serial_num', 'operator'] if c in df.columns
                    ], key="trend_group")

                if trend_param:
                    group_col = None if group_by == "None" else group_by

                    if trend_type == "Trend Line":
                        fig = go.Figure()
                        x_vals = df['test_id'] if 'test_id' in df.columns else list(range(len(df)))
                        y_vals = df[trend_param]

                        fig.add_trace(go.Scatter(
                            x=list(range(len(x_vals))), y=y_vals,
                            mode='markers+lines',
                            marker=dict(size=10, color='#3b82f6'),
                            line=dict(color='#93c5fd', width=1),
                            text=x_vals,
                            hovertemplate='%{text}<br>Value: %{y:.4f}<extra></extra>',
                            name=trend_param
                        ))

                        mean_val = y_vals.mean()
                        std_val = y_vals.std()
                        fig.add_hline(y=mean_val, line_dash="dash", line_color="#22c55e",
                                      annotation_text=f"Mean: {mean_val:.4f}")
                        fig.add_hrect(y0=mean_val - 2*std_val, y1=mean_val + 2*std_val,
                                      fillcolor="rgba(59, 130, 246, 0.08)", line_width=0)

                        # Add uncertainty bands if available
                        u_col = trend_param.replace('avg_', 'u_')
                        if u_col in df.columns:
                            fig.add_trace(go.Scatter(
                                x=list(range(len(x_vals))),
                                y=y_vals + df[u_col],
                                mode='lines', line=dict(width=0),
                                showlegend=False, hoverinfo='skip'
                            ))
                            fig.add_trace(go.Scatter(
                                x=list(range(len(x_vals))),
                                y=y_vals - df[u_col],
                                mode='lines', line=dict(width=0),
                                fill='tonexty', fillcolor='rgba(59,130,246,0.15)',
                                name='Uncertainty Band', hoverinfo='skip'
                            ))

                        fig.update_layout(
                            title=f"{trend_param} Trend",
                            xaxis_title="Test Number", yaxis_title=trend_param,
                            height=450, plot_bgcolor='white',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif trend_type == "Box Plot":
                        if group_col:
                            fig = px.box(df, x=group_col, y=trend_param,
                                         title=f"{trend_param} by {group_col}",
                                         color=group_col, points="all")
                        else:
                            fig = px.box(df, y=trend_param,
                                         title=f"{trend_param} Distribution", points="all")
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)

                    elif trend_type == "Violin":
                        if group_col:
                            fig = px.violin(df, x=group_col, y=trend_param,
                                            title=f"{trend_param} by {group_col}",
                                            color=group_col, box=True, points="all")
                        else:
                            fig = px.violin(df, y=trend_param,
                                            title=f"{trend_param} Distribution",
                                            box=True, points="all")
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)

                    elif trend_type == "Histogram":
                        fig = px.histogram(df, x=trend_param, nbins=20,
                                           title=f"{trend_param} Distribution",
                                           color=group_col if group_col else None,
                                           marginal="rug")
                        fig.add_vline(x=df[trend_param].mean(), line_dash="dash",
                                      annotation_text=f"Mean: {df[trend_param].mean():.4f}")
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)

            with plot_subtab2:
                if len(plot_cols) >= 2:
                    corr_params = st.multiselect(
                        "Parameters for correlation", plot_cols,
                        default=plot_cols[:min(6, len(plot_cols))],
                        key="corr_params"
                    )

                    if len(corr_params) >= 2:
                        corr_df = df[corr_params].dropna()
                        corr_matrix = corr_df.corr()

                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            colorscale='RdBu_r', zmin=-1, zmax=1,
                            text=np.round(corr_matrix.values, 3),
                            texttemplate='%{text}', textfont=dict(size=11),
                            hovertemplate='%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>'
                        ))
                        fig.update_layout(
                            title="Parameter Correlation Matrix",
                            height=500, plot_bgcolor='white',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Scatter plot for selected pair
                        col1, col2 = st.columns(2)
                        with col1:
                            scatter_x = st.selectbox("X axis", corr_params, key="scatter_x")
                        with col2:
                            scatter_y = st.selectbox("Y axis", corr_params,
                                                     index=min(1, len(corr_params)-1), key="scatter_y")

                        if scatter_x != scatter_y:
                            fig = px.scatter(df, x=scatter_x, y=scatter_y,
                                             title=f"{scatter_x} vs {scatter_y}",
                                             trendline="ols",
                                             hover_data=['test_id'] if 'test_id' in df.columns else None)
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation analysis.")

            with plot_subtab3:
                if len(plot_cols) >= 2:
                    multi_params = st.multiselect(
                        "Parameters to overlay", plot_cols,
                        default=plot_cols[:min(3, len(plot_cols))],
                        key="multi_params"
                    )

                    if multi_params:
                        normalize = st.checkbox("Normalize to z-scores", value=len(multi_params) > 1,
                                                key="multi_normalize")

                        fig = go.Figure()
                        for param in multi_params:
                            y = df[param].values
                            if normalize and np.std(y) > 0:
                                y = (y - np.mean(y)) / np.std(y)
                                ylabel = "Z-Score"
                            else:
                                ylabel = "Value"

                            fig.add_trace(go.Scatter(
                                x=list(range(len(y))), y=y,
                                mode='markers+lines', name=param,
                                marker=dict(size=8),
                                hovertemplate=f'{param}<br>Value: %{{y:.4f}}<extra></extra>'
                            ))

                        fig.update_layout(
                            title="Multi-Parameter Overlay",
                            xaxis_title="Test Number", yaxis_title=ylabel,
                            height=450, plot_bgcolor='white',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for multi-parameter plots.")
        else:
            st.info("No test data available for plots.")

    # =============================================================================
    # TAB 3: SPC Analysis (I-MR, CUSUM, EWMA)
    # =============================================================================

    with tab3:
        if df is not None and len(df) > 0:
            # Get numeric columns
            numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
            metric_cols = [c for c in numeric_cols if c.startswith('avg_') or c in ['Cd', 'Isp']]

            spc_subtab1, spc_subtab2, spc_subtab3 = st.tabs([
                "I-MR Chart", "CUSUM Chart", "EWMA Chart"
            ])

            # ---- I-MR Chart ----
            with spc_subtab1:
                col1, col2 = st.columns([1, 3])

                with col1:
                    imr_param = st.selectbox("Parameter", metric_cols if metric_cols else numeric_cols,
                                             key="imr_param")
                    st.divider()
                    st.markdown("**Specification Limits**")
                    imr_use_specs = st.checkbox("Use spec limits", key="imr_use_specs")
                    imr_lsl, imr_usl, imr_target = None, None, None
                    if imr_use_specs and imr_param:
                        col_mean = float(df[imr_param].mean())
                        col_range = float(df[imr_param].max() - df[imr_param].min())
                        imr_lsl = st.number_input("LSL", value=col_mean - col_range, key="imr_lsl")
                        imr_usl = st.number_input("USL", value=col_mean + col_range, key="imr_usl")
                        imr_target = st.number_input("Target", value=col_mean, key="imr_target")

                with col2:
                    if imr_param:
                        try:
                            analysis = create_imr_chart(
                                df, parameter=imr_param,
                                test_id_col='test_id' if 'test_id' in df.columns else df.columns[0],
                                usl=imr_usl if imr_use_specs else None,
                                lsl=imr_lsl if imr_use_specs else None,
                                target=imr_target if imr_use_specs else None,
                            )

                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            with mcol1:
                                status = "In Control" if analysis.n_violations == 0 else "Out of Control"
                                color = "green" if analysis.n_violations == 0 else "red"
                                st.markdown(f"### :{color}[{status}]")
                            with mcol2:
                                st.metric("Points", analysis.n_points)
                            with mcol3:
                                st.metric("Violations", analysis.n_violations)
                            with mcol4:
                                if analysis.capability and analysis.capability.cpk is not None:
                                    st.metric("Cpk", f"{analysis.capability.cpk:.2f}")
                                else:
                                    st.metric("Cpk", "N/A")

                            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                                shared_xaxes=True, vertical_spacing=0.08,
                                                subplot_titles=[f"Individual Chart: {imr_param}", "Moving Range"])

                            x = list(range(len(analysis.points)))
                            y = [p.value for p in analysis.points]
                            test_ids = [p.test_id for p in analysis.points]
                            colors = ['#ef4444' if not p.in_control else '#3b82f6' for p in analysis.points]

                            fig.add_trace(go.Scatter(
                                x=x, y=y, mode='markers+lines',
                                marker=dict(color=colors, size=9),
                                line=dict(color='#93c5fd', width=1),
                                text=test_ids,
                                hovertemplate='%{text}<br>Value: %{y:.4f}<extra></extra>',
                                name='Data'), row=1, col=1)

                            fig.add_hline(y=analysis.limits.center_line, line_dash="solid",
                                          line_color="#22c55e", row=1, col=1,
                                          annotation_text=f"CL: {analysis.limits.center_line:.4f}")
                            fig.add_hline(y=analysis.limits.ucl, line_dash="dash",
                                          line_color="#ef4444", row=1, col=1,
                                          annotation_text=f"UCL: {analysis.limits.ucl:.4f}")
                            fig.add_hline(y=analysis.limits.lcl, line_dash="dash",
                                          line_color="#ef4444", row=1, col=1,
                                          annotation_text=f"LCL: {analysis.limits.lcl:.4f}")

                            mr = np.abs(np.diff(y))
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(y))), y=mr,
                                mode='markers+lines',
                                marker=dict(color='#3b82f6', size=6),
                                line=dict(color='#93c5fd', width=1),
                                name='MR'), row=2, col=1)

                            mr_bar = np.mean(mr)
                            fig.add_hline(y=mr_bar, line_dash="solid", line_color="#22c55e", row=2, col=1)
                            fig.add_hline(y=3.267 * mr_bar, line_dash="dash", line_color="#ef4444", row=2, col=1)

                            fig.update_layout(height=500, showlegend=False, plot_bgcolor='white')
                            st.plotly_chart(fig, use_container_width=True)

                            with st.expander("SPC Summary"):
                                st.markdown(format_spc_summary(analysis))

                        except Exception as e:
                            st.error(f"I-MR Analysis error: {e}")

            # ---- CUSUM Chart ----
            with spc_subtab2:
                col1, col2 = st.columns([1, 3])

                with col1:
                    cusum_param = st.selectbox("Parameter", metric_cols if metric_cols else numeric_cols,
                                               key="cusum_param")
                    st.divider()
                    st.markdown("**CUSUM Parameters**")
                    st.caption("Tabular CUSUM (Page's procedure) detects small sustained shifts in the process mean.")
                    cusum_k = st.number_input(
                        "k (allowable slack)",
                        value=0.5, min_value=0.1, max_value=2.0, step=0.1,
                        help="Allowable slack in sigma units. Smaller k = more sensitive to small shifts.",
                        key="cusum_k"
                    )
                    cusum_h = st.number_input(
                        "h (decision interval)",
                        value=5.0, min_value=1.0, max_value=20.0, step=0.5,
                        help="Decision interval in sigma units. Larger h = fewer false alarms.",
                        key="cusum_h"
                    )

                with col2:
                    if cusum_param:
                        try:
                            values = df[cusum_param].dropna().values
                            cusum_result = create_cusum_chart(
                                values, k=cusum_k, h=cusum_h,
                                parameter_name=cusum_param
                            )

                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            with mcol1:
                                status = "In Control" if cusum_result.n_signals == 0 else "Shift Detected"
                                color = "green" if cusum_result.n_signals == 0 else "red"
                                st.markdown(f"### :{color}[{status}]")
                            with mcol2:
                                st.metric("Points", len(values))
                            with mcol3:
                                st.metric("Signals", cusum_result.n_signals)
                            with mcol4:
                                st.metric("Target", f"{cusum_result.target:.4f}")

                            fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5],
                                                shared_xaxes=True, vertical_spacing=0.08,
                                                subplot_titles=[f"Raw Data: {cusum_param}",
                                                                "CUSUM Statistics (C+ and C-)"])

                            x = list(range(len(values)))
                            fig.add_trace(go.Scatter(
                                x=x, y=values, mode='markers+lines',
                                marker=dict(size=7, color='#3b82f6'),
                                line=dict(color='#93c5fd', width=1),
                                name='Data'), row=1, col=1)
                            fig.add_hline(y=cusum_result.target, line_dash="dash",
                                          line_color="#22c55e", row=1, col=1,
                                          annotation_text=f"Target: {cusum_result.target:.4f}")

                            # CUSUM C+ and C-
                            x_cusum = list(range(len(cusum_result.c_plus)))
                            fig.add_trace(go.Scatter(
                                x=x_cusum, y=cusum_result.c_plus,
                                mode='lines', name='C+',
                                line=dict(color='#ef4444', width=2),
                                hovertemplate='C+: %{y:.3f}<extra></extra>'
                            ), row=2, col=1)
                            fig.add_trace(go.Scatter(
                                x=x_cusum, y=cusum_result.c_minus,
                                mode='lines', name='C-',
                                line=dict(color='#f97316', width=2),
                                hovertemplate='C-: %{y:.3f}<extra></extra>'
                            ), row=2, col=1)

                            fig.add_hline(y=cusum_result.h, line_dash="dash",
                                          line_color="#ef4444", row=2, col=1,
                                          annotation_text=f"H = {cusum_result.h:.1f}")

                            # Mark signal points
                            for idx in cusum_result.signals_upper:
                                fig.add_vline(x=idx, line_dash="dot", line_color="rgba(239,68,68,0.4)",
                                              row=2, col=1)
                            for idx in cusum_result.signals_lower:
                                fig.add_vline(x=idx, line_dash="dot", line_color="rgba(249,115,22,0.4)",
                                              row=2, col=1)

                            fig.update_layout(height=550, plot_bgcolor='white',
                                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
                            st.plotly_chart(fig, use_container_width=True)

                            if cusum_result.n_signals > 0:
                                st.warning(
                                    f"CUSUM detected {cusum_result.n_signals} signal(s): "
                                    f"{len(cusum_result.signals_upper)} upward shift(s), "
                                    f"{len(cusum_result.signals_lower)} downward shift(s). "
                                    f"First signal at observation {min(cusum_result.signals_upper + cusum_result.signals_lower)}."
                                )
                            else:
                                st.success("No sustained shifts detected. Process appears stable.")

                        except Exception as e:
                            st.error(f"CUSUM Analysis error: {e}")

            # ---- EWMA Chart ----
            with spc_subtab3:
                col1, col2 = st.columns([1, 3])

                with col1:
                    ewma_param = st.selectbox("Parameter", metric_cols if metric_cols else numeric_cols,
                                              key="ewma_param")
                    st.divider()
                    st.markdown("**EWMA Parameters**")
                    st.caption("EWMA smooths data to detect gradual drifts with time-varying control limits.")
                    ewma_lambda = st.slider(
                        "Lambda (smoothing)",
                        min_value=0.05, max_value=1.0, value=0.2, step=0.05,
                        help="Smoothing parameter. Smaller = more weight on history, better for small shifts.",
                        key="ewma_lambda"
                    )
                    ewma_L = st.number_input(
                        "L (control limit width)",
                        value=3.0, min_value=1.0, max_value=5.0, step=0.1,
                        help="Width of control limits in sigma units.",
                        key="ewma_L"
                    )

                with col2:
                    if ewma_param:
                        try:
                            values = df[ewma_param].dropna().values
                            ewma_result = create_ewma_chart(
                                values, lambda_param=ewma_lambda, L=ewma_L,
                                parameter_name=ewma_param
                            )

                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            with mcol1:
                                status = "In Control" if ewma_result.n_signals == 0 else "Drift Detected"
                                color = "green" if ewma_result.n_signals == 0 else "red"
                                st.markdown(f"### :{color}[{status}]")
                            with mcol2:
                                st.metric("Points", len(values))
                            with mcol3:
                                st.metric("Signals", ewma_result.n_signals)
                            with mcol4:
                                st.metric("Center Line", f"{ewma_result.center_line:.4f}")

                            fig = go.Figure()

                            x = list(range(len(ewma_result.ewma_values)))

                            # Control limit bands
                            fig.add_trace(go.Scatter(
                                x=x, y=ewma_result.ucl,
                                mode='lines', line=dict(color='rgba(239,68,68,0.4)', dash='dash', width=1),
                                name='UCL', hovertemplate='UCL: %{y:.4f}<extra></extra>'
                            ))
                            fig.add_trace(go.Scatter(
                                x=x, y=ewma_result.lcl,
                                mode='lines', line=dict(color='rgba(239,68,68,0.4)', dash='dash', width=1),
                                name='LCL', fill='tonexty',
                                fillcolor='rgba(239,68,68,0.06)',
                                hovertemplate='LCL: %{y:.4f}<extra></extra>'
                            ))

                            # Raw data (faded)
                            fig.add_trace(go.Scatter(
                                x=x, y=values[:len(x)],
                                mode='markers', name='Raw Data',
                                marker=dict(size=5, color='rgba(156,163,175,0.5)'),
                                hovertemplate='Raw: %{y:.4f}<extra></extra>'
                            ))

                            # EWMA line
                            signal_colors = ['#ef4444' if i in ewma_result.signals else '#3b82f6'
                                             for i in range(len(ewma_result.ewma_values))]
                            fig.add_trace(go.Scatter(
                                x=x, y=ewma_result.ewma_values,
                                mode='markers+lines', name='EWMA',
                                marker=dict(size=8, color=signal_colors),
                                line=dict(color='#3b82f6', width=2),
                                hovertemplate='EWMA: %{y:.4f}<extra></extra>'
                            ))

                            fig.add_hline(y=ewma_result.center_line, line_dash="solid",
                                          line_color="#22c55e",
                                          annotation_text=f"CL: {ewma_result.center_line:.4f}")

                            fig.update_layout(
                                title=f"EWMA Chart: {ewma_param} (lambda={ewma_lambda:.2f})",
                                xaxis_title="Observation", yaxis_title=ewma_param,
                                height=500, plot_bgcolor='white',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            if ewma_result.n_signals > 0:
                                first_signal = min(ewma_result.signals)
                                st.warning(
                                    f"EWMA detected {ewma_result.n_signals} out-of-control point(s). "
                                    f"First signal at observation {first_signal}. "
                                    f"This suggests a gradual drift in the process mean."
                                )
                            else:
                                st.success("No drift detected. EWMA values remain within control limits.")

                        except Exception as e:
                            st.error(f"EWMA Analysis error: {e}")
        else:
            st.info("No test data available for SPC analysis.")

    # =============================================================================
    # TAB 4: Reports (from Reports Export page)
    # =============================================================================

    with tab4:
        st.subheader("Reports & Export")

        if df is not None and len(df) > 0:
            report_subtab1, report_subtab2, report_subtab3, report_subtab4 = st.tabs([
                "Campaign Report",
                "Test Reports",
                "Data Export",
                "Qualification Package"
            ])

            with report_subtab1:
                st.markdown("**Campaign Summary Report**")

                col1, col2 = st.columns([1, 2])

                with col1:
                    numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
                    metric_cols = [c for c in numeric_cols if c.startswith('avg_')]

                    default_params = metric_cols[:4] if metric_cols else numeric_cols[:4]

                    selected_params = st.multiselect(
                        "Parameters to include",
                        numeric_cols,
                        default=default_params,
                        key="rpt_params"
                    )

                    include_spc = st.checkbox("Include SPC analysis", value=True, key="rpt_spc")

                    specs = {}
                    if include_spc and selected_params:
                        with st.expander("Specification Limits"):
                            for param in selected_params[:2]:
                                st.markdown(f"**{param}**")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    spec_lsl = st.number_input(f"LSL", key=f"rpt_lsl_{param}", value=0.0)
                                with col_b:
                                    spec_usl = st.number_input(f"USL", key=f"rpt_usl_{param}", value=1.0)

                                if spec_lsl != 0.0 or spec_usl != 1.0:
                                    specs[param] = {'lsl': spec_lsl, 'usl': spec_usl}

                    generate_report = st.button("Generate Report", type="primary", key="rpt_gen")

                with col2:
                    if generate_report and selected_params:
                        with st.spinner("Generating report..."):
                            try:
                                spc_analyses = None
                                if include_spc and specs:
                                    spc_analyses = analyze_campaign_spc(df, list(specs.keys()), specs)

                                html = generate_campaign_report(
                                    campaign_name=selected_campaign,
                                    df=df,
                                    parameters=selected_params,
                                    spc_analyses=spc_analyses,
                                )

                                st.success("Report generated!")

                                with st.expander("Preview Report"):
                                    st.components.v1.html(html, height=600, scrolling=True)

                                st.download_button(
                                    "Download HTML Report",
                                    html,
                                    file_name=f"{selected_campaign}_report.html",
                                    mime="text/html",
                                    key="dl_camp_rpt"
                                )

                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.info("Configure report options and click Generate")

            with report_subtab2:
                st.markdown("**Individual Test Reports**")

                if 'test_id' in df.columns:
                    test_ids = df['test_id'].tolist()

                    selected_test_rpt = st.selectbox("Select Test", test_ids, key="rpt_test")
                    include_config = st.checkbox("Include config snapshot", value=False, key="rpt_config")

                    if st.button("Generate Test Report", key="rpt_test_gen"):
                        test_row = df[df['test_id'] == selected_test_rpt].iloc[0]

                        measurements = {}
                        for col in df.columns:
                            if col.startswith('avg_'):
                                val = test_row[col]
                                if pd.notna(val):
                                    u_col = col.replace('avg_', 'u_')
                                    if u_col in df.columns and pd.notna(test_row[u_col]):
                                        measurements[col] = MeasurementWithUncertainty(
                                            float(val), float(test_row[u_col]), '', col
                                        )
                                    else:
                                        measurements[col] = float(val)

                        traceability = {}
                        trace_cols = ['raw_data_hash', 'config_hash', 'analyst_username',
                                      'analysis_timestamp_utc', 'processing_version']
                        for col in trace_cols:
                            if col in df.columns:
                                traceability[col] = test_row[col]

                        qc_report = {
                            'passed': bool(test_row.get('qc_passed', True)),
                            'summary': {},
                            'checks': []
                        }

                        try:
                            html = generate_test_report(
                                test_id=selected_test_rpt,
                                test_type=campaign_type,
                                measurements=measurements,
                                traceability=traceability,
                                qc_report=qc_report,
                                include_config_snapshot=include_config,
                            )

                            st.success("Report generated!")

                            st.download_button(
                                "Download Test Report",
                                html,
                                file_name=f"{selected_test_rpt}_report.html",
                                mime="text/html",
                                key="dl_test_rpt"
                            )

                            with st.expander("Preview"):
                                st.components.v1.html(html, height=500, scrolling=True)

                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("No test_id column in campaign data")

            with report_subtab3:
                st.markdown("**Export Campaign Data**")

                col1, col2 = st.columns(2)

                with col1:
                    export_format_rpt = st.selectbox("Format", ["CSV", "Excel", "JSON"], key="rpt_exp_fmt")
                    include_unc_rpt = st.checkbox("Include uncertainty columns", value=True, key="rpt_exp_unc")
                    include_trace_rpt = st.checkbox("Include traceability columns", value=True, key="rpt_exp_trace")

                with col2:
                    if st.button("Generate Export", type="primary", key="rpt_exp_gen"):
                        try:
                            export_df = df.copy()

                            if not include_unc_rpt:
                                export_df = export_df[[c for c in export_df.columns if not c.startswith('u_')]]

                            if not include_trace_rpt:
                                trace_cols = ['raw_data_hash', 'config_hash', 'analyst_username',
                                              'analysis_timestamp_utc', 'processing_version']
                                export_df = export_df[[c for c in export_df.columns if c not in trace_cols]]

                            if export_format_rpt == "CSV":
                                st.download_button(
                                    "Download CSV",
                                    export_df.to_csv(index=False),
                                    file_name=f"{selected_campaign}_export.csv",
                                    mime="text/csv",
                                    key="dl_rpt_csv"
                                )

                            elif export_format_rpt == "Excel":
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    export_df.to_excel(writer, sheet_name='Data', index=False)

                                    meta_df = pd.DataFrame([
                                        {'Field': 'Campaign', 'Value': selected_campaign},
                                        {'Field': 'Type', 'Value': campaign_type},
                                        {'Field': 'Export Date', 'Value': datetime.now().isoformat()},
                                        {'Field': 'Test Count', 'Value': len(export_df)},
                                    ])
                                    meta_df.to_excel(writer, sheet_name='Metadata', index=False)

                                st.download_button(
                                    "Download Excel",
                                    buffer.getvalue(),
                                    file_name=f"{selected_campaign}_export.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="dl_rpt_xlsx"
                                )

                            elif export_format_rpt == "JSON":
                                export_data = {
                                    'campaign': selected_campaign,
                                    'type': campaign_type,
                                    'export_date': datetime.now().isoformat(),
                                    'test_count': len(export_df),
                                    'data': export_df.to_dict(orient='records')
                                }
                                st.download_button(
                                    "Download JSON",
                                    json.dumps(export_data, indent=2, default=str),
                                    file_name=f"{selected_campaign}_export.json",
                                    mime="application/json",
                                    key="dl_rpt_json"
                                )

                            st.success("Export ready!")

                        except Exception as e:
                            st.error(f"Export error: {e}")

            with report_subtab4:
                st.markdown("**Qualification Documentation Package**")

                st.markdown("""
                Generate a complete documentation package suitable for flight qualification:
                - **Summary CSV**: Key metrics only
                - **Full Data CSV**: Complete dataset
                - **Traceability Report**: Audit trail
                - **JSON Archive**: Machine-readable data
                - **Manifest**: Package contents
                """)

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Additional Metadata"):
                        project_name = st.text_input("Project Name", value="", key="qual_proj")
                        part_number = st.text_input("Part Number", value="", key="qual_pn")
                        revision = st.text_input("Revision", value="A", key="qual_rev")
                        prepared_by = st.text_input("Prepared By", value="", key="qual_prep")

                with col2:
                    if st.button("Generate Qualification Package", type="primary", key="qual_gen"):
                        with st.spinner("Generating package..."):
                            try:
                                zip_buffer = io.BytesIO()

                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                    # Summary CSV
                                    summary_cols = ['test_id', 'test_timestamp', 'part', 'serial_num']
                                    metric_cols = [c for c in df.columns if c.startswith('avg_')]
                                    summary_cols.extend([c for c in metric_cols if c in df.columns])
                                    summary_cols.extend(['qc_passed'])
                                    available_summary = [c for c in summary_cols if c in df.columns]

                                    summary_csv = df[available_summary].to_csv(index=False)
                                    zf.writestr(f"{selected_campaign}_summary.csv", summary_csv)

                                    # Full data CSV
                                    full_csv = df.to_csv(index=False)
                                    zf.writestr(f"{selected_campaign}_full_data.csv", full_csv)

                                    # JSON archive
                                    archive_data = {
                                        'campaign_name': selected_campaign,
                                        'campaign_type': campaign_type,
                                        'export_date': datetime.now().isoformat(),
                                        'package_metadata': {
                                            'project': project_name,
                                            'part_number': part_number,
                                            'revision': revision,
                                            'prepared_by': prepared_by,
                                        },
                                        'summary': {
                                            'total_tests': len(df),
                                            'qc_passed': int(df['qc_passed'].sum()) if 'qc_passed' in df.columns else len(df),
                                        },
                                        'data': df.to_dict(orient='records')
                                    }
                                    json_content = json.dumps(archive_data, indent=2, default=str)
                                    zf.writestr(f"{selected_campaign}_archive.json", json_content)

                                    # Traceability report
                                    trace_lines = [
                                        "DATA TRACEABILITY REPORT",
                                        f"Campaign: {selected_campaign}",
                                        f"Generated: {datetime.now().isoformat()}",
                                        f"Total Tests: {len(df)}",
                                        "",
                                    ]
                                    for _, row in df.iterrows():
                                        trace_lines.append(f"Test: {row.get('test_id', 'Unknown')}")
                                        if 'raw_data_hash' in df.columns:
                                            trace_lines.append(f"  Hash: {row.get('raw_data_hash', 'N/A')}")
                                        trace_lines.append("")

                                    zf.writestr(f"{selected_campaign}_traceability.txt", "\n".join(trace_lines))

                                    # Manifest
                                    manifest = {
                                        'package_name': f"{selected_campaign}_qual_package",
                                        'created': datetime.now().isoformat(),
                                        'campaign': selected_campaign,
                                        'test_count': len(df),
                                        'files': [
                                            f"{selected_campaign}_summary.csv",
                                            f"{selected_campaign}_full_data.csv",
                                            f"{selected_campaign}_archive.json",
                                            f"{selected_campaign}_traceability.txt",
                                        ],
                                        'metadata': {
                                            'project': project_name,
                                            'part_number': part_number,
                                            'revision': revision,
                                            'prepared_by': prepared_by,
                                        }
                                    }
                                    zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

                                zip_buffer.seek(0)

                                st.success("Qualification package generated!")

                                st.download_button(
                                    "Download Qualification Package (ZIP)",
                                    zip_buffer.getvalue(),
                                    file_name=f"{selected_campaign}_qual_package.zip",
                                    mime="application/zip",
                                    key="dl_qual_pkg"
                                )

                            except Exception as e:
                                st.error(f"Error generating package: {e}")
                                import traceback
                                st.code(traceback.format_exc())
        else:
            st.info("No test data in selected campaign.")

else:
    st.info("Select or create a campaign in the sidebar to begin analysis.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### Campaign Summary
        - Overview metrics and statistics
        - Data tables with column filtering
        - Traceability information
        - Export and qualification packages
        """)

    with col2:
        st.markdown("""
        #### SPC Control Charts
        - **I-MR**: Individual-Moving Range with Western Electric rules
        - **CUSUM**: Tabular CUSUM for detecting small sustained shifts
        - **EWMA**: Exponentially weighted moving average for gradual drifts
        """)

    with col3:
        st.markdown("""
        #### Campaign Plots
        - Trend analysis with uncertainty bands
        - Correlation matrices and scatter plots
        - Multi-parameter overlay with z-score normalization
        """)
