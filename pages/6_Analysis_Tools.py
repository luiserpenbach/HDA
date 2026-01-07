"""
Analysis Tools Page
===================
Advanced analysis tools including anomaly detection and data comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

from core.advanced_anomaly import (
    run_anomaly_detection,
    format_anomaly_table,
    AnomalyType,
    AnomalySeverity,
)
from core.comparison import (
    compare_tests,
    compare_to_golden,
    create_golden_from_campaign,
    linear_regression,
    calculate_correlation_matrix,
    track_deviations,
    compare_campaigns,
    format_campaign_comparison,
    GoldenReference,
)
from core.campaign_manager_v2 import get_available_campaigns, get_campaign_data
from pages._shared_sidebar import render_global_context

st.set_page_config(page_title="Analysis Tools", page_icon="AT", layout="wide")

# =============================================================================
# SIDEBAR - Global Context
# =============================================================================

with st.sidebar:
    context = render_global_context()

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("Analysis Tools")
st.markdown("Advanced analysis tools including anomaly detection and data comparison.")

# Initialize session state
if 'anomaly_report' not in st.session_state:
    st.session_state.anomaly_report = None
if 'anomaly_df' not in st.session_state:
    st.session_state.anomaly_df = None

# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2 = st.tabs(["Anomaly Detection", "Data Comparison"])

# =============================================================================
# TAB 1: ANOMALY DETECTION
# =============================================================================

with tab1:
    st.header("Advanced Anomaly Detection")
    st.markdown("Comprehensive anomaly detection with sensor health monitoring.")

    # Sidebar-like settings in columns
    settings_col, main_col = st.columns([1, 3])

    with settings_col:
        st.subheader("Settings")

        spike_threshold = st.slider("Spike Threshold (σ)", 2.0, 6.0, 4.0, 0.5, key="ad_spike")
        sample_rate = st.number_input("Sample Rate (Hz)", 1, 10000, 100, key="ad_sample")

        st.divider()

        check_correlations = st.checkbox("Check correlations", value=False, key="ad_corr")

    with main_col:
        st.subheader("1. Upload Test Data")

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="ad_upload")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.anomaly_df = df
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")

            with st.expander("Data Preview"):
                st.dataframe(df.head(50), use_container_width=True)

        df = st.session_state.anomaly_df

        if df is not None:
            st.divider()
            st.subheader("2. Select Channels")

            numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]

            timestamp_candidates = ['timestamp', 'time', 'Time', 't', 'time_ms']
            timestamp_col = None
            for tc in timestamp_candidates:
                if tc in df.columns:
                    timestamp_col = tc
                    break

            if timestamp_col:
                numeric_cols = [c for c in numeric_cols if c != timestamp_col]
                st.info(f"Using '{timestamp_col}' as timestamp column")

            selected_channels = st.multiselect(
                "Channels to analyze",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols,
                key="ad_channels"
            )

            correlation_pairs = None
            if check_correlations and len(selected_channels) >= 2:
                with st.expander("Configure Correlation Pairs"):
                    pairs = []
                    n_pairs = st.number_input("Number of pairs", 0, 5, 1, key="ad_npairs")

                    for i in range(int(n_pairs)):
                        col1, col2 = st.columns(2)
                        with col1:
                            ch1 = st.selectbox(f"Channel 1 (pair {i+1})", selected_channels, key=f"ad_corr1_{i}")
                        with col2:
                            ch2 = st.selectbox(f"Channel 2 (pair {i+1})", selected_channels, key=f"ad_corr2_{i}")
                        if ch1 != ch2:
                            pairs.append((ch1, ch2))

                    correlation_pairs = pairs if pairs else None

            st.divider()
            st.subheader("3. Run Analysis")

            if st.button("Detect Anomalies", type="primary", key="ad_run"):
                if not selected_channels:
                    st.warning("Select at least one channel")
                else:
                    with st.spinner("Analyzing..."):
                        report = run_anomaly_detection(
                            df=df,
                            channels=selected_channels,
                            timestamp_col=timestamp_col or 'timestamp',
                            sample_rate_hz=sample_rate,
                            correlation_pairs=correlation_pairs,
                        )
                        st.session_state.anomaly_report = report
                        st.success("Analysis complete!")

            if st.session_state.anomaly_report:
                report = st.session_state.anomaly_report

                st.divider()
                st.subheader("4. Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    color = "red" if report.has_critical else ("orange" if report.warning_count > 0 else "green")
                    st.markdown(f"### :{color}[{report.total_anomalies} Anomalies]")
                with col2:
                    st.metric("Critical", report.critical_count)
                with col3:
                    st.metric("Warning", report.warning_count)
                with col4:
                    avg_health = np.mean(list(report.sensor_health.values())) if report.sensor_health else 0
                    st.metric("Avg Health", f"{avg_health:.0%}")

                result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(["Overview", "Details", "Health", "Visualization"])

                with result_tab1:
                    st.markdown("**Anomaly Summary**")

                    type_counts = {}
                    for anomaly in report.get_all_anomalies():
                        t = anomaly.anomaly_type.value
                        type_counts[t] = type_counts.get(t, 0) + 1

                    if type_counts:
                        tcol1, tcol2 = st.columns(2)
                        with tcol1:
                            st.markdown("**By Type:**")
                            for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                                st.markdown(f"- {atype}: {count}")
                        with tcol2:
                            st.markdown("**By Severity:**")
                            st.markdown(f"- Critical: {report.critical_count}")
                            st.markdown(f"- Warning: {report.warning_count}")
                            info_count = report.total_anomalies - report.critical_count - report.warning_count
                            st.markdown(f"- Info: {info_count}")
                    else:
                        st.success("[PASS] No anomalies detected!")

                    st.divider()
                    st.text(report.summary())

                with result_tab2:
                    st.markdown("**Anomaly Details**")

                    anomaly_df = format_anomaly_table(report)

                    if len(anomaly_df) > 0:
                        dcol1, dcol2, dcol3 = st.columns(3)
                        with dcol1:
                            filter_channel = st.selectbox("Filter by channel", ["All"] + list(report.channel_reports.keys()), key="ad_filter_ch")
                        with dcol2:
                            filter_type = st.selectbox("Filter by type", ["All"] + list(type_counts.keys()) if type_counts else ["All"], key="ad_filter_type")
                        with dcol3:
                            filter_severity = st.selectbox("Filter by severity", ["All", "CRITICAL", "WARNING", "INFO"], key="ad_filter_sev")

                        filtered_df = anomaly_df.copy()
                        if filter_channel != "All":
                            filtered_df = filtered_df[filtered_df['Channel'] == filter_channel]
                        if filter_type != "All":
                            filtered_df = filtered_df[filtered_df['Type'] == filter_type]
                        if filter_severity != "All":
                            filtered_df = filtered_df[filtered_df['Severity'] == filter_severity]

                        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

                        st.download_button(
                            "Download Anomaly Report",
                            anomaly_df.to_csv(index=False),
                            file_name="anomaly_report.csv",
                            mime="text/csv",
                            key="ad_dl_report"
                        )
                    else:
                        st.success("No anomalies to display")

                with result_tab3:
                    st.markdown("**Sensor Health Scores**")

                    if report.sensor_health:
                        sorted_health = sorted(report.sensor_health.items(), key=lambda x: x[1])

                        for channel, health in sorted_health:
                            if health >= 0.9:
                                color = "green"
                                status = "[PASS] Good"
                            elif health >= 0.7:
                                color = "orange"
                                status = "[WARN] Fair"
                            else:
                                color = "red"
                                status = "[FAIL] Poor"

                            hcol1, hcol2, hcol3 = st.columns([2, 1, 1])
                            with hcol1:
                                st.markdown(f"**{channel}**")
                            with hcol2:
                                st.progress(health)
                            with hcol3:
                                st.markdown(f":{color}[{health:.0%} - {status}]")

                        st.divider()
                        health_values = list(report.sensor_health.values())
                        scol1, scol2, scol3 = st.columns(3)
                        with scol1:
                            st.metric("Min Health", f"{min(health_values):.0%}")
                        with scol2:
                            st.metric("Avg Health", f"{np.mean(health_values):.0%}")
                        with scol3:
                            st.metric("Max Health", f"{max(health_values):.0%}")

                with result_tab4:
                    st.markdown("**Anomaly Visualization**")

                    viz_channel = st.selectbox("Select channel", list(report.channel_reports.keys()), key="ad_viz_ch")

                    if viz_channel and viz_channel in df.columns:
                        data = df[viz_channel].values
                        channel_anomalies = report.channel_reports.get(viz_channel, [])

                        fig = go.Figure()

                        x = df[timestamp_col].values if timestamp_col and timestamp_col in df.columns else np.arange(len(data))

                        fig.add_trace(go.Scatter(
                            x=x, y=data,
                            mode='lines',
                            name=viz_channel,
                            line=dict(color='blue', width=1),
                        ))

                        colors = {
                            AnomalySeverity.CRITICAL: 'red',
                            AnomalySeverity.WARNING: 'orange',
                            AnomalySeverity.INFO: 'yellow',
                        }

                        for anomaly in channel_anomalies:
                            color = colors.get(anomaly.severity, 'gray')
                            fig.add_vrect(
                                x0=x[anomaly.start_index],
                                x1=x[min(anomaly.end_index, len(x)-1)],
                                fillcolor=color,
                                opacity=0.3,
                                line_width=0,
                            )

                        fig.update_layout(
                            title=f"{viz_channel} with Anomalies",
                            xaxis_title="Time" if timestamp_col else "Sample",
                            yaxis_title=viz_channel,
                            height=500,
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("**Legend:** Critical | Warning | Info")

        else:
            st.info("Upload a CSV file to begin anomaly detection")

# =============================================================================
# TAB 2: DATA COMPARISON
# =============================================================================

with tab2:
    st.header("Data Comparison & Regression")
    st.markdown("Compare tests, create golden references, and analyze correlations.")

    # Get campaigns
    campaigns = get_available_campaigns()

    if not campaigns:
        st.warning("No campaigns found. Create a campaign first.")
    else:
        campaign_names = [c['name'] for c in campaigns]

        comp_settings_col, comp_main_col = st.columns([1, 3])

        with comp_settings_col:
            st.subheader("Settings")

            mode = st.radio(
                "Analysis Mode",
                ["Test Comparison", "Golden Reference", "Regression", "Correlation", "Campaign Comparison"],
                key="dc_mode"
            )

            st.divider()

            selected_campaign = st.selectbox("Primary Campaign", campaign_names, key="dc_campaign")

        with comp_main_col:
            if selected_campaign:
                df = get_campaign_data(selected_campaign)

                if df is None or len(df) == 0:
                    st.warning("No data in selected campaign")
                else:
                    # Get numeric columns
                    numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
                    metric_cols = [c for c in numeric_cols if c.startswith('avg_')]

                    # =============================================================================
                    # TEST COMPARISON
                    # =============================================================================

                    if mode == "Test Comparison":
                        st.subheader("Test-to-Test Comparison")

                        if 'test_id' not in df.columns:
                            st.error("Campaign must have test_id column")
                        else:
                            test_ids = df['test_id'].tolist()

                            tcol1, tcol2 = st.columns(2)

                            with tcol1:
                                test_a = st.selectbox("Test A", test_ids, key="dc_test_a")
                            with tcol2:
                                test_b = st.selectbox("Test B", test_ids, index=min(1, len(test_ids)-1), key="dc_test_b")

                            # Parameter selection
                            params_to_compare = st.multiselect(
                                "Parameters to compare",
                                metric_cols,
                                default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols,
                                key="dc_params"
                            )

                            # Tolerance
                            default_tol = st.slider("Default tolerance (%)", 1.0, 20.0, 5.0, 0.5, key="dc_tol")

                            if st.button("Compare Tests", type="primary", key="dc_compare"):
                                if test_a == test_b:
                                    st.warning("Select different tests")
                                else:
                                    # Get test data
                                    row_a = df[df['test_id'] == test_a].iloc[0]
                                    row_b = df[df['test_id'] == test_b].iloc[0]

                                    data_a = {p: float(row_a[p]) for p in params_to_compare if pd.notna(row_a.get(p))}
                                    data_b = {p: float(row_b[p]) for p in params_to_compare if pd.notna(row_b.get(p))}

                                    result = compare_tests(data_a, data_b, test_a, test_b, default_tolerance=default_tol)

                                    # Display results
                                    st.markdown("**Comparison Results**")

                                    status_color = "green" if result.overall_pass else "red"
                                    st.markdown(f"### :{status_color}[{'[PASS] PASS' if result.overall_pass else '[FAIL] FAIL'}]")
                                    st.caption(f"{result.n_within_tolerance}/{result.n_parameters} parameters within tolerance")

                                    # Results table
                                    st.dataframe(result.to_dataframe(), use_container_width=True, hide_index=True)

                                    # Visual comparison
                                    st.markdown("**Visual Comparison**")

                                    fig = go.Figure()

                                    params = [c.parameter for c in result.comparisons]
                                    vals_a = [c.value_a for c in result.comparisons]
                                    vals_b = [c.value_b for c in result.comparisons]

                                    fig.add_trace(go.Bar(name=test_a, x=params, y=vals_a, marker_color='blue'))
                                    fig.add_trace(go.Bar(name=test_b, x=params, y=vals_b, marker_color='orange'))

                                    fig.update_layout(barmode='group', title="Parameter Comparison")
                                    st.plotly_chart(fig, use_container_width=True)

                    # =============================================================================
                    # GOLDEN REFERENCE
                    # =============================================================================

                    elif mode == "Golden Reference":
                        st.subheader("Golden Reference")

                        golden_tab1, golden_tab2 = st.tabs(["Create Golden", "Compare to Golden"])

                        with golden_tab1:
                            st.markdown("**Create Golden Reference**")

                            golden_name = st.text_input("Reference Name", value=f"{selected_campaign}_golden", key="dc_golden_name")

                            params_for_golden = st.multiselect(
                                "Parameters to include",
                                metric_cols,
                                default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols,
                                key="dc_golden_params"
                            )

                            gcol1, gcol2 = st.columns(2)
                            with gcol1:
                                method = st.selectbox("Central value method", ["mean", "median"], key="dc_golden_method")
                            with gcol2:
                                tol_mult = st.slider("Tolerance multiplier (× σ)", 1.0, 5.0, 3.0, key="dc_golden_tol")

                            if st.button("Create Golden Reference", key="dc_create_golden"):
                                if not params_for_golden:
                                    st.warning("Select at least one parameter")
                                else:
                                    try:
                                        golden = create_golden_from_campaign(
                                            df, golden_name, params_for_golden,
                                            tolerance_multiplier=tol_mult,
                                            method=method,
                                        )

                                        st.success(f"Created golden reference from {len(df)} tests")

                                        # Show golden values
                                        golden_df = pd.DataFrame([
                                            {
                                                'Parameter': p,
                                                'Value': f"{v:.4g}",
                                                'Tolerance (%)': f"±{golden.tolerances.get(p, 5.0):.1f}",
                                                'Uncertainty': f"±{golden.uncertainties.get(p, 0):.4g}" if golden.uncertainties else "-",
                                            }
                                            for p, v in golden.parameters.items()
                                        ])
                                        st.dataframe(golden_df, use_container_width=True, hide_index=True)

                                        # Download
                                        st.download_button(
                                            "Download Golden Reference",
                                            json.dumps(golden.to_dict(), indent=2),
                                            file_name=f"{golden_name}.json",
                                            mime="application/json",
                                            key="dc_dl_golden"
                                        )

                                        # Store in session
                                        st.session_state['golden_ref'] = golden

                                    except Exception as e:
                                        st.error(f"Error: {e}")

                        with golden_tab2:
                            st.markdown("**Compare Test to Golden**")

                            # Upload or use session golden
                            golden_source = st.radio("Golden source", ["Upload JSON", "Use created golden"], key="dc_golden_src")

                            golden = None
                            if golden_source == "Upload JSON":
                                uploaded_golden = st.file_uploader("Upload golden reference", type=['json'], key="dc_golden_upload")
                                if uploaded_golden:
                                    data = json.load(uploaded_golden)
                                    golden = GoldenReference.from_dict(data)
                                    st.success(f"Loaded: {golden.name}")
                            else:
                                if 'golden_ref' in st.session_state:
                                    golden = st.session_state['golden_ref']
                                    st.info(f"Using: {golden.name}")
                                else:
                                    st.warning("Create a golden reference first")

                            if golden and 'test_id' in df.columns:
                                test_to_compare = st.selectbox("Select test", df['test_id'].tolist(), key="dc_test_golden")

                                if st.button("Compare to Golden", key="dc_compare_golden"):
                                    row = df[df['test_id'] == test_to_compare].iloc[0]
                                    test_data = {p: float(row[p]) for p in golden.parameters.keys() if pd.notna(row.get(p))}

                                    result = compare_to_golden(test_data, test_to_compare, golden)

                                    status_color = "green" if result.overall_pass else "red"
                                    st.markdown(f"### :{status_color}[{'[PASS] PASS' if result.overall_pass else '[FAIL] FAIL'}]")

                                    st.dataframe(result.to_dataframe(), use_container_width=True, hide_index=True)

                    # =============================================================================
                    # REGRESSION
                    # =============================================================================

                    elif mode == "Regression":
                        st.subheader("Regression Analysis")

                        rcol1, rcol2 = st.columns(2)

                        with rcol1:
                            x_param = st.selectbox("X Parameter (Independent)", numeric_cols, key="dc_x_param")
                        with rcol2:
                            y_param = st.selectbox("Y Parameter (Dependent)",
                                                   [c for c in numeric_cols if c != x_param], key="dc_y_param")

                        if st.button("Run Regression", key="dc_run_reg"):
                            x = df[x_param].values
                            y = df[y_param].values

                            try:
                                result = linear_regression(x, y, x_param, y_param)

                                # Results
                                rmcol1, rmcol2, rmcol3 = st.columns(3)
                                with rmcol1:
                                    st.metric("R²", f"{result.r_squared:.4f}")
                                with rmcol2:
                                    st.metric("Slope", f"{result.slope:.4g}")
                                with rmcol3:
                                    st.metric("Intercept", f"{result.intercept:.4g}")

                                st.info(f"**Equation:** {result.prediction_equation}")

                                # Plot
                                fig = go.Figure()

                                # Data points
                                fig.add_trace(go.Scatter(
                                    x=x, y=y,
                                    mode='markers',
                                    name='Data',
                                    marker=dict(size=8, color='blue'),
                                ))

                                # Regression line
                                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                                y_line = result.predict(x_line)

                                fig.add_trace(go.Scatter(
                                    x=x_line, y=y_line,
                                    mode='lines',
                                    name=f'Fit (R²={result.r_squared:.3f})',
                                    line=dict(color='red', width=2),
                                ))

                                fig.update_layout(
                                    title=f"Regression: {y_param} vs {x_param}",
                                    xaxis_title=x_param,
                                    yaxis_title=y_param,
                                    height=500,
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                st.text(result.summary())

                            except Exception as e:
                                st.error(f"Regression error: {e}")

                    # =============================================================================
                    # CORRELATION
                    # =============================================================================

                    elif mode == "Correlation":
                        st.subheader("Correlation Analysis")

                        params_for_corr = st.multiselect(
                            "Parameters to analyze",
                            metric_cols,
                            default=metric_cols[:8] if len(metric_cols) > 8 else metric_cols,
                            key="dc_corr_params"
                        )

                        threshold = st.slider("Strong correlation threshold", 0.5, 0.95, 0.7, 0.05, key="dc_corr_thresh")

                        if st.button("Calculate Correlations", key="dc_calc_corr") and params_for_corr:
                            try:
                                corr_matrix = calculate_correlation_matrix(df, params_for_corr)

                                # Heatmap
                                fig = px.imshow(
                                    corr_matrix.matrix,
                                    x=corr_matrix.parameters,
                                    y=corr_matrix.parameters,
                                    color_continuous_scale='RdBu_r',
                                    zmin=-1, zmax=1,
                                    title="Correlation Matrix"
                                )
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)

                                # Strong correlations
                                strong = corr_matrix.get_strong_correlations(threshold)

                                if strong:
                                    st.markdown(f"**Strong Correlations (|r| ≥ {threshold})**")

                                    strong_df = pd.DataFrame([
                                        {'Parameter 1': p1, 'Parameter 2': p2, 'Correlation': f"{r:.3f}"}
                                        for p1, p2, r in strong
                                    ])
                                    st.dataframe(strong_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info(f"No correlations above |{threshold}| found")

                                # Download matrix
                                st.download_button(
                                    "Download Correlation Matrix",
                                    corr_matrix.to_dataframe().to_csv(),
                                    file_name="correlation_matrix.csv",
                                    mime="text/csv",
                                    key="dc_dl_corr"
                                )

                            except Exception as e:
                                st.error(f"Error: {e}")

                    # =============================================================================
                    # CAMPAIGN COMPARISON
                    # =============================================================================

                    elif mode == "Campaign Comparison":
                        st.subheader("Campaign Comparison")

                        if len(campaigns) < 2:
                            st.warning("Need at least 2 campaigns to compare")
                        else:
                            other_campaigns = [c['name'] for c in campaigns if c['name'] != selected_campaign]
                            campaign_b = st.selectbox("Compare to campaign", other_campaigns, key="dc_campaign_b")

                            params_to_compare = st.multiselect(
                                "Parameters to compare",
                                metric_cols,
                                default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols,
                                key="dc_camp_params"
                            )

                            if st.button("Compare Campaigns", key="dc_compare_camps") and params_to_compare:
                                df_b = get_campaign_data(campaign_b)

                                if df_b is None or len(df_b) == 0:
                                    st.error(f"No data in {campaign_b}")
                                else:
                                    result = compare_campaigns(
                                        df, df_b,
                                        selected_campaign, campaign_b,
                                        params_to_compare
                                    )

                                    # Summary
                                    ccol1, ccol2 = st.columns(2)
                                    with ccol1:
                                        st.metric(f"{selected_campaign}", f"n={result['n_tests_a']}")
                                    with ccol2:
                                        st.metric(f"{campaign_b}", f"n={result['n_tests_b']}")

                                    # Results table
                                    rows = []
                                    for param, data in result['parameters'].items():
                                        rows.append({
                                            'Parameter': param,
                                            f'Mean ({selected_campaign})': f"{data['mean_a']:.4g}",
                                            f'Mean ({campaign_b})': f"{data['mean_b']:.4g}",
                                            'Δ%': f"{data['mean_diff_pct']:+.2f}%",
                                            'Status': 'Pass' if data['means_equivalent'] else 'Fail',
                                        })

                                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                                    # Visual comparison
                                    st.markdown("**Distribution Comparison**")

                                    param_to_plot = st.selectbox("Parameter to visualize", params_to_compare, key="dc_camp_viz")

                                    fig = go.Figure()
                                    fig.add_trace(go.Histogram(x=df[param_to_plot].dropna(), name=selected_campaign, opacity=0.7))
                                    fig.add_trace(go.Histogram(x=df_b[param_to_plot].dropna(), name=campaign_b, opacity=0.7))
                                    fig.update_layout(barmode='overlay', title=f"{param_to_plot} Distribution")

                                    st.plotly_chart(fig, use_container_width=True)

                                    st.text(format_campaign_comparison(result))
            else:
                st.info("Select a campaign in the sidebar")
