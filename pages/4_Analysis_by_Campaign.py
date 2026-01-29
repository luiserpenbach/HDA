"""
Analysis by Campaign Page
=========================
Comprehensive campaign analysis with summary, plots, SPC, and reports.
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
    analyze_campaign_spc,
    format_spc_summary,
    calculate_capability,
    ViolationType,
)
from core.reporting import generate_campaign_report, generate_test_report
from core.uncertainty import MeasurementWithUncertainty

st.set_page_config(page_title="Analysis by Campaign", page_icon="AC", layout="wide")

st.title("Analysis by Campaign")
st.markdown("Comprehensive campaign analysis with summary, plots, SPC analysis, and reports.")

# =============================================================================
# SIDEBAR - Global Context & Campaign Selection
# =============================================================================

from pages._shared_sidebar import render_global_context

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
    # TAB 2: Campaign Plots (New - placeholder for now)
    # =============================================================================

    with tab2:
        st.subheader("Campaign Plots")
        st.info("Campaign plots coming soon. This section will include customizable visualizations for campaign data.")

        if df is not None and len(df) > 0:
            # Simple placeholder plots
            numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
            metric_cols = [c for c in numeric_cols if c.startswith('avg_')]

            if metric_cols:
                col1, col2 = st.columns(2)

                with col1:
                    plot_param = st.selectbox("Select parameter to plot", metric_cols, key="plot_param")

                with col2:
                    plot_type = st.selectbox("Plot type", ["Line", "Bar", "Scatter"], key="plot_type")

                if plot_param:
                    if plot_type == "Line":
                        fig = px.line(df, y=plot_param, title=f"{plot_param} - Line Plot")
                    elif plot_type == "Bar":
                        fig = px.bar(df, y=plot_param, title=f"{plot_param} - Bar Plot")
                    else:
                        fig = px.scatter(df, y=plot_param, title=f"{plot_param} - Scatter Plot")

                    st.plotly_chart(fig, use_container_width=True)

    # =============================================================================
    # TAB 3: SPC Analysis (from SPC Analysis page)
    # =============================================================================

    with tab3:
        st.subheader("Statistical Process Control")

        if df is not None and len(df) > 0:
            # Get numeric columns
            numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]

            # Filter to likely metric columns
            metric_cols = [c for c in numeric_cols if c.startswith('avg_') or c in ['Cd', 'Isp']]

            col1, col2 = st.columns([1, 2])

            with col1:
                if metric_cols:
                    selected_parameter = st.selectbox("Parameter", metric_cols, key="spc_param")
                else:
                    selected_parameter = st.selectbox("Parameter", numeric_cols, key="spc_param")

                st.divider()

                # Specification limits
                st.markdown("**Specification Limits**")

                use_specs = st.checkbox("Use specification limits", key="spc_use_specs")

                if use_specs and selected_parameter:
                    col_min = float(df[selected_parameter].min())
                    col_max = float(df[selected_parameter].max())
                    col_mean = float(df[selected_parameter].mean())
                    col_range = col_max - col_min

                    lsl = st.number_input("LSL", value=col_mean - col_range, key="spc_lsl")
                    usl = st.number_input("USL", value=col_mean + col_range, key="spc_usl")
                    target = st.number_input("Target", value=col_mean, key="spc_target")
                else:
                    lsl, usl, target = None, None, None

            with col2:
                if selected_parameter:
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

                        # Header metrics
                        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                        with mcol1:
                            status_color = "green" if analysis.n_violations == 0 else "red"
                            st.markdown(f"### :{status_color}[{'In Control' if analysis.n_violations == 0 else 'Out of Control'}]")

                        with mcol2:
                            st.metric("Points", analysis.n_points)

                        with mcol3:
                            st.metric("Violations", analysis.n_violations)

                        with mcol4:
                            if analysis.capability and analysis.capability.cpk is not None:
                                cpk = analysis.capability.cpk
                                st.metric("Cpk", f"{cpk:.2f}")
                            else:
                                st.metric("Cpk", "N/A")

                        # Control chart
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
                        mr_ucl = 3.267 * mr_bar

                        fig.add_hline(y=mr_bar, line_dash="solid", line_color="green", row=2, col=1)
                        fig.add_hline(y=mr_ucl, line_dash="dash", line_color="red", row=2, col=1)

                        fig.update_layout(
                            height=500,
                            showlegend=False,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Summary
                        with st.expander("SPC Summary"):
                            st.markdown(format_spc_summary(analysis))

                    except Exception as e:
                        st.error(f"SPC Analysis error: {e}")
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
    st.info("Select or create a campaign in the sidebar")

    # Show SPC overview
    st.divider()
    st.header("About Campaign Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Campaign Summary

        View test data organized by campaign with:
        - Overview metrics and statistics
        - Data tables with column filtering
        - Traceability information
        - Export capabilities
        """)

    with col2:
        st.markdown("""
        ### SPC Analysis

        Statistical Process Control features:
        - I-MR Control Charts
        - Western Electric Rules
        - Process Capability (Cpk)
        - Trend Detection
        """)
