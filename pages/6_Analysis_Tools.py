"""
Analysis Tools Page
===================
Advanced analysis tools including anomaly detection, data comparison,
transient analysis, frequency analysis, and operating envelope visualization.

Version: 2.4.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import traceback
from typing import Optional, List, Dict

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
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
from core.transient_analysis import (
    segment_test_phases,
    analyze_startup_transient,
    analyze_shutdown_transient,
    compute_phase_metrics,
    TestPhase,
    PhaseResult,
    MultiPhaseResult,
)
from core.frequency_analysis import (
    compute_power_spectral_density,
    compute_spectrogram,
    detect_harmonics,
    compute_cross_spectrum,
    detect_resonance,
    compute_frequency_bands,
    SpectralResult,
    HarmonicInfo,
    CrossSpectralResult,
    DEFAULT_FREQUENCY_BANDS,
)
from core.operating_envelope import (
    calculate_operating_envelope,
    plot_operating_envelope,
    create_envelope_report,
    OperatingEnvelope,
)
from core.campaign_manager_v2 import get_available_campaigns, get_campaign_data

# ---------------------------------------------------------------------------
# UI imports
# ---------------------------------------------------------------------------
from pages._shared_sidebar import render_global_context
from pages._shared_styles import (
    apply_custom_styles,
    render_page_header,
    render_metric_card,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Analysis Tools", page_icon="AT", layout="wide")

# Apply the shadcn-inspired design system
apply_custom_styles()

# ---------------------------------------------------------------------------
# Sidebar - global context
# ---------------------------------------------------------------------------
with st.sidebar:
    context = render_global_context()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
render_page_header(
    "Analysis Tools",
    description="Advanced analysis tools for anomaly detection, data comparison, transient characterization, frequency analysis, and operating envelope visualization.",
    badge_text="P2",
    badge_type="info",
)

# ---------------------------------------------------------------------------
# Phase color map (used by transient analysis tab)
# ---------------------------------------------------------------------------
_PHASE_COLORS = {
    TestPhase.PRETEST: "#a1a1aa",       # zinc-400
    TestPhase.STARTUP: "#2563eb",       # blue-600
    TestPhase.TRANSIENT: "#ca8a04",     # yellow-600
    TestPhase.STEADY_STATE: "#16a34a",  # green-600
    TestPhase.SHUTDOWN: "#dc2626",      # red-600
    TestPhase.COOLDOWN: "#71717a",      # zinc-500
}

_PHASE_LABELS = {
    TestPhase.PRETEST: "Pre-test",
    TestPhase.STARTUP: "Startup",
    TestPhase.TRANSIENT: "Transient",
    TestPhase.STEADY_STATE: "Steady State",
    TestPhase.SHUTDOWN: "Shutdown",
    TestPhase.COOLDOWN: "Cooldown",
}


# =============================================================================
# Helper: auto-detect timestamp column
# =============================================================================
def _detect_time_column(df: pd.DataFrame):
    """Return the most likely timestamp column name, or None."""
    candidates = ['time_s', 'timestamp', 'time', 'Time', 't', 'time_ms']
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _numeric_columns(df: pd.DataFrame, exclude=None) -> list:
    """Return numeric column names, optionally excluding some."""
    exclude = set(exclude or [])
    return [
        c for c in df.columns
        if df[c].dtype in ['float64', 'float32', 'int64', 'int32']
        and c not in exclude
    ]


# =============================================================================
# Session state initialisation
# =============================================================================
_DEFAULT_STATE = {
    # Tab 1 - Anomaly Detection
    'anomaly_report': None,
    'anomaly_df': None,
    # Tab 3 - Transient Analysis
    'transient_df': None,
    'transient_result': None,
    'transient_startup': None,
    'transient_shutdown': None,
    # Tab 4 - Frequency Analysis
    'freq_df': None,
    'freq_psd_result': None,
    'freq_harmonics': None,
    'freq_resonances': None,
    'freq_cross_result': None,
}
for key, default in _DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Anomaly Detection",
    "Data Comparison",
    "Transient Analysis",
    "Frequency Analysis",
    "Operating Envelope",
])


# #############################################################################
# TAB 1: ANOMALY DETECTION
# #############################################################################
with tab1:
    try:
        st.subheader("Advanced Anomaly Detection")
        st.caption("Comprehensive anomaly detection with sensor health monitoring.")

        settings_col, main_col = st.columns([1, 3])

        with settings_col:
            st.markdown("**Settings**")
            spike_threshold = st.slider(
                "Spike Threshold (sigma)", 2.0, 6.0, 4.0, 0.5, key="ad_spike",
            )
            sample_rate = st.number_input(
                "Sample Rate (Hz)", 1, 10000, 100, key="ad_sample",
            )
            st.divider()
            check_correlations = st.checkbox(
                "Check correlations", value=False, key="ad_corr",
            )

        with main_col:
            # ---- Upload ----
            st.markdown("**1. Upload Test Data**")
            uploaded_file = st.file_uploader(
                "Upload CSV file", type=['csv'], key="ad_upload",
            )
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.anomaly_df = df
                st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(df.head(50), use_container_width=True)

            df = st.session_state.anomaly_df

            if df is not None:
                st.divider()
                st.markdown("**2. Select Channels**")
                timestamp_col = _detect_time_column(df)
                numeric_cols = _numeric_columns(df, exclude=[timestamp_col] if timestamp_col else [])

                if timestamp_col:
                    st.info(f"Timestamp column: **{timestamp_col}**")

                selected_channels = st.multiselect(
                    "Channels to analyze",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols,
                    key="ad_channels",
                )

                # Correlation pairs
                correlation_pairs = None
                if check_correlations and len(selected_channels) >= 2:
                    with st.expander("Configure Correlation Pairs"):
                        pairs = []
                        n_pairs = st.number_input(
                            "Number of pairs", 0, 5, 1, key="ad_npairs",
                        )
                        for i in range(int(n_pairs)):
                            c1, c2 = st.columns(2)
                            with c1:
                                ch1 = st.selectbox(
                                    f"Channel 1 (pair {i+1})", selected_channels,
                                    key=f"ad_corr1_{i}",
                                )
                            with c2:
                                ch2 = st.selectbox(
                                    f"Channel 2 (pair {i+1})", selected_channels,
                                    key=f"ad_corr2_{i}",
                                )
                            if ch1 != ch2:
                                pairs.append((ch1, ch2))
                        correlation_pairs = pairs if pairs else None

                # ---- Run ----
                st.divider()
                st.markdown("**3. Run Analysis**")
                if st.button("Detect Anomalies", type="primary", key="ad_run"):
                    if not selected_channels:
                        st.warning("Select at least one channel.")
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
                            st.success("Analysis complete.")

                # ---- Results ----
                if st.session_state.anomaly_report:
                    report = st.session_state.anomaly_report
                    st.divider()
                    st.markdown("**4. Results**")

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        color = "red" if report.has_critical else ("orange" if report.warning_count > 0 else "green")
                        st.markdown(f"### :{color}[{report.total_anomalies} Anomalies]")
                    with mc2:
                        render_metric_card("Critical", str(report.critical_count))
                    with mc3:
                        render_metric_card("Warning", str(report.warning_count))
                    with mc4:
                        avg_health = np.mean(list(report.sensor_health.values())) if report.sensor_health else 0
                        render_metric_card("Avg Health", f"{avg_health:.0%}")

                    rtab1, rtab2, rtab3, rtab4 = st.tabs(["Overview", "Details", "Health", "Visualization"])

                    # -- Overview --
                    with rtab1:
                        type_counts: dict = {}
                        for anomaly in report.get_all_anomalies():
                            t = anomaly.anomaly_type.value
                            type_counts[t] = type_counts.get(t, 0) + 1

                        if type_counts:
                            tc1, tc2 = st.columns(2)
                            with tc1:
                                st.markdown("**By Type:**")
                                for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                                    st.markdown(f"- {atype}: **{count}**")
                            with tc2:
                                st.markdown("**By Severity:**")
                                st.markdown(f"- Critical: **{report.critical_count}**")
                                st.markdown(f"- Warning: **{report.warning_count}**")
                                info_count = report.total_anomalies - report.critical_count - report.warning_count
                                st.markdown(f"- Info: **{info_count}**")
                        else:
                            st.success("No anomalies detected.")
                        st.divider()
                        st.text(report.summary())

                    # -- Details --
                    with rtab2:
                        anomaly_df = format_anomaly_table(report)
                        if len(anomaly_df) > 0:
                            dc1, dc2, dc3 = st.columns(3)
                            with dc1:
                                filter_channel = st.selectbox(
                                    "Filter by channel",
                                    ["All"] + list(report.channel_reports.keys()),
                                    key="ad_filter_ch",
                                )
                            with dc2:
                                filter_type = st.selectbox(
                                    "Filter by type",
                                    ["All"] + list(type_counts.keys()) if type_counts else ["All"],
                                    key="ad_filter_type",
                                )
                            with dc3:
                                filter_severity = st.selectbox(
                                    "Filter by severity",
                                    ["All", "CRITICAL", "WARNING", "INFO"],
                                    key="ad_filter_sev",
                                )

                            filtered = anomaly_df.copy()
                            if filter_channel != "All":
                                filtered = filtered[filtered['Channel'] == filter_channel]
                            if filter_type != "All":
                                filtered = filtered[filtered['Type'] == filter_type]
                            if filter_severity != "All":
                                filtered = filtered[filtered['Severity'] == filter_severity]

                            st.dataframe(filtered, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download Anomaly Report",
                                anomaly_df.to_csv(index=False),
                                file_name="anomaly_report.csv",
                                mime="text/csv",
                                key="ad_dl_report",
                            )
                        else:
                            st.success("No anomalies to display.")

                    # -- Health --
                    with rtab3:
                        if report.sensor_health:
                            sorted_health = sorted(report.sensor_health.items(), key=lambda x: x[1])
                            for channel, health in sorted_health:
                                if health >= 0.9:
                                    color, status = "green", "Good"
                                elif health >= 0.7:
                                    color, status = "orange", "Fair"
                                else:
                                    color, status = "red", "Poor"

                                hc1, hc2, hc3 = st.columns([2, 1, 1])
                                with hc1:
                                    st.markdown(f"**{channel}**")
                                with hc2:
                                    st.progress(health)
                                with hc3:
                                    st.markdown(f":{color}[{health:.0%} - {status}]")

                            st.divider()
                            hv = list(report.sensor_health.values())
                            sc1, sc2, sc3 = st.columns(3)
                            with sc1:
                                render_metric_card("Min Health", f"{min(hv):.0%}")
                            with sc2:
                                render_metric_card("Avg Health", f"{np.mean(hv):.0%}")
                            with sc3:
                                render_metric_card("Max Health", f"{max(hv):.0%}")

                    # -- Visualization --
                    with rtab4:
                        viz_channel = st.selectbox(
                            "Select channel",
                            list(report.channel_reports.keys()),
                            key="ad_viz_ch",
                        )
                        if viz_channel and viz_channel in df.columns:
                            data = df[viz_channel].values
                            channel_anomalies = report.channel_reports.get(viz_channel, [])

                            fig = go.Figure()
                            x = (
                                df[timestamp_col].values
                                if timestamp_col and timestamp_col in df.columns
                                else np.arange(len(data))
                            )
                            fig.add_trace(go.Scatter(
                                x=x, y=data, mode='lines',
                                name=viz_channel,
                                line=dict(color='#18181b', width=1),
                            ))

                            sev_colors = {
                                AnomalySeverity.CRITICAL: 'rgba(220,38,38,0.3)',
                                AnomalySeverity.WARNING: 'rgba(202,138,4,0.3)',
                                AnomalySeverity.INFO: 'rgba(37,99,235,0.15)',
                            }
                            for anomaly in channel_anomalies:
                                fig.add_vrect(
                                    x0=x[anomaly.start_index],
                                    x1=x[min(anomaly.end_index, len(x) - 1)],
                                    fillcolor=sev_colors.get(anomaly.severity, 'rgba(0,0,0,0.1)'),
                                    line_width=0,
                                )

                            fig.update_layout(
                                title=f"{viz_channel} with Anomalies",
                                xaxis_title="Time" if timestamp_col else "Sample",
                                yaxis_title=viz_channel,
                                height=500,
                                template="plotly_white",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Regions: red = critical, yellow = warning, blue = info.")
            else:
                st.info("Upload a CSV file to begin anomaly detection.")

    except Exception as exc:
        st.error(f"Anomaly Detection encountered an error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 2: DATA COMPARISON
# #############################################################################
with tab2:
    try:
        st.subheader("Data Comparison & Regression")
        st.caption("Compare tests, create golden references, and analyze correlations.")

        campaigns = get_available_campaigns()

        if not campaigns:
            st.warning("No campaigns found. Create a campaign first.")
        else:
            campaign_names = [c['name'] for c in campaigns]

            comp_settings_col, comp_main_col = st.columns([1, 3])

            with comp_settings_col:
                st.markdown("**Settings**")
                mode = st.radio(
                    "Analysis Mode",
                    ["Test Comparison", "Golden Reference", "Regression",
                     "Correlation", "Campaign Comparison"],
                    key="dc_mode",
                )
                st.divider()
                selected_campaign = st.selectbox(
                    "Primary Campaign", campaign_names, key="dc_campaign",
                )

            with comp_main_col:
                if selected_campaign:
                    df = get_campaign_data(selected_campaign)

                    if df is None or len(df) == 0:
                        st.warning("No data in selected campaign.")
                    else:
                        numeric_cols = _numeric_columns(df)
                        metric_cols = [c for c in numeric_cols if c.startswith('avg_')]

                        # =============================================================
                        # TEST COMPARISON
                        # =============================================================
                        if mode == "Test Comparison":
                            st.markdown("#### Test-to-Test Comparison")
                            if 'test_id' not in df.columns:
                                st.error("Campaign must have a test_id column.")
                            else:
                                test_ids = df['test_id'].tolist()
                                tc1, tc2 = st.columns(2)
                                with tc1:
                                    test_a = st.selectbox("Test A", test_ids, key="dc_test_a")
                                with tc2:
                                    test_b = st.selectbox(
                                        "Test B", test_ids,
                                        index=min(1, len(test_ids) - 1), key="dc_test_b",
                                    )

                                params_to_compare = st.multiselect(
                                    "Parameters to compare", metric_cols,
                                    default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols,
                                    key="dc_params",
                                )
                                default_tol = st.slider(
                                    "Default tolerance (%)", 1.0, 20.0, 5.0, 0.5, key="dc_tol",
                                )

                                if st.button("Compare Tests", type="primary", key="dc_compare"):
                                    if test_a == test_b:
                                        st.warning("Select different tests.")
                                    else:
                                        row_a = df[df['test_id'] == test_a].iloc[0]
                                        row_b = df[df['test_id'] == test_b].iloc[0]
                                        data_a = {p: float(row_a[p]) for p in params_to_compare if pd.notna(row_a.get(p))}
                                        data_b = {p: float(row_b[p]) for p in params_to_compare if pd.notna(row_b.get(p))}
                                        result = compare_tests(data_a, data_b, test_a, test_b, default_tolerance=default_tol)

                                        status_color = "green" if result.overall_pass else "red"
                                        st.markdown(f"### :{status_color}[{'PASS' if result.overall_pass else 'FAIL'}]")
                                        st.caption(f"{result.n_within_tolerance}/{result.n_parameters} parameters within tolerance")
                                        st.dataframe(result.to_dataframe(), use_container_width=True, hide_index=True)

                                        fig = go.Figure()
                                        params = [c.parameter for c in result.comparisons]
                                        vals_a = [c.value_a for c in result.comparisons]
                                        vals_b = [c.value_b for c in result.comparisons]
                                        fig.add_trace(go.Bar(name=test_a, x=params, y=vals_a, marker_color='#18181b'))
                                        fig.add_trace(go.Bar(name=test_b, x=params, y=vals_b, marker_color='#71717a'))
                                        fig.update_layout(barmode='group', title="Parameter Comparison", template="plotly_white")
                                        st.plotly_chart(fig, use_container_width=True)

                        # =============================================================
                        # GOLDEN REFERENCE
                        # =============================================================
                        elif mode == "Golden Reference":
                            st.markdown("#### Golden Reference")
                            golden_tab1, golden_tab2 = st.tabs(["Create Golden", "Compare to Golden"])

                            with golden_tab1:
                                golden_name = st.text_input(
                                    "Reference Name",
                                    value=f"{selected_campaign}_golden",
                                    key="dc_golden_name",
                                )
                                params_for_golden = st.multiselect(
                                    "Parameters to include", metric_cols,
                                    default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols,
                                    key="dc_golden_params",
                                )
                                gc1, gc2 = st.columns(2)
                                with gc1:
                                    method = st.selectbox("Central value method", ["mean", "median"], key="dc_golden_method")
                                with gc2:
                                    tol_mult = st.slider("Tolerance multiplier (x sigma)", 1.0, 5.0, 3.0, key="dc_golden_tol")

                                if st.button("Create Golden Reference", key="dc_create_golden"):
                                    if not params_for_golden:
                                        st.warning("Select at least one parameter.")
                                    else:
                                        try:
                                            golden = create_golden_from_campaign(
                                                df, golden_name, params_for_golden,
                                                tolerance_multiplier=tol_mult, method=method,
                                            )
                                            st.success(f"Created golden reference from {len(df)} tests.")
                                            golden_df = pd.DataFrame([
                                                {
                                                    'Parameter': p,
                                                    'Value': f"{v:.4g}",
                                                    'Tolerance (%)': f"+/-{golden.tolerances.get(p, 5.0):.1f}",
                                                    'Uncertainty': f"+/-{golden.uncertainties.get(p, 0):.4g}" if golden.uncertainties else "-",
                                                }
                                                for p, v in golden.parameters.items()
                                            ])
                                            st.dataframe(golden_df, use_container_width=True, hide_index=True)
                                            st.download_button(
                                                "Download Golden Reference",
                                                json.dumps(golden.to_dict(), indent=2),
                                                file_name=f"{golden_name}.json",
                                                mime="application/json",
                                                key="dc_dl_golden",
                                            )
                                            st.session_state['golden_ref'] = golden
                                        except Exception as e:
                                            st.error(f"Error: {e}")

                            with golden_tab2:
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
                                        st.warning("Create a golden reference first.")

                                if golden and 'test_id' in df.columns:
                                    test_to_compare = st.selectbox("Select test", df['test_id'].tolist(), key="dc_test_golden")
                                    if st.button("Compare to Golden", key="dc_compare_golden"):
                                        row = df[df['test_id'] == test_to_compare].iloc[0]
                                        test_data = {p: float(row[p]) for p in golden.parameters.keys() if pd.notna(row.get(p))}
                                        result = compare_to_golden(test_data, test_to_compare, golden)
                                        status_color = "green" if result.overall_pass else "red"
                                        st.markdown(f"### :{status_color}[{'PASS' if result.overall_pass else 'FAIL'}]")
                                        st.dataframe(result.to_dataframe(), use_container_width=True, hide_index=True)

                        # =============================================================
                        # REGRESSION
                        # =============================================================
                        elif mode == "Regression":
                            st.markdown("#### Regression Analysis")
                            rc1, rc2 = st.columns(2)
                            with rc1:
                                x_param = st.selectbox("X Parameter (Independent)", numeric_cols, key="dc_x_param")
                            with rc2:
                                y_param = st.selectbox(
                                    "Y Parameter (Dependent)",
                                    [c for c in numeric_cols if c != x_param],
                                    key="dc_y_param",
                                )

                            if st.button("Run Regression", type="primary", key="dc_run_reg"):
                                x = df[x_param].values
                                y = df[y_param].values
                                try:
                                    result = linear_regression(x, y, x_param, y_param)
                                    rm1, rm2, rm3 = st.columns(3)
                                    with rm1:
                                        render_metric_card("R-squared", f"{result.r_squared:.4f}")
                                    with rm2:
                                        render_metric_card("Slope", f"{result.slope:.4g}")
                                    with rm3:
                                        render_metric_card("Intercept", f"{result.intercept:.4g}")

                                    st.info(f"**Equation:** {result.prediction_equation}")

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=x, y=y, mode='markers', name='Data',
                                        marker=dict(size=8, color='#18181b'),
                                    ))
                                    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                                    y_line = result.predict(x_line)
                                    fig.add_trace(go.Scatter(
                                        x=x_line, y=y_line, mode='lines',
                                        name=f'Fit (R2={result.r_squared:.3f})',
                                        line=dict(color='#dc2626', width=2),
                                    ))
                                    fig.update_layout(
                                        title=f"Regression: {y_param} vs {x_param}",
                                        xaxis_title=x_param, yaxis_title=y_param,
                                        height=500, template="plotly_white",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.text(result.summary())
                                except Exception as e:
                                    st.error(f"Regression error: {e}")

                        # =============================================================
                        # CORRELATION
                        # =============================================================
                        elif mode == "Correlation":
                            st.markdown("#### Correlation Analysis")
                            params_for_corr = st.multiselect(
                                "Parameters to analyze", metric_cols,
                                default=metric_cols[:8] if len(metric_cols) > 8 else metric_cols,
                                key="dc_corr_params",
                            )
                            threshold = st.slider(
                                "Strong correlation threshold",
                                0.5, 0.95, 0.7, 0.05, key="dc_corr_thresh",
                            )
                            if st.button("Calculate Correlations", key="dc_calc_corr") and params_for_corr:
                                try:
                                    corr_matrix = calculate_correlation_matrix(df, params_for_corr)
                                    fig = px.imshow(
                                        corr_matrix.matrix,
                                        x=corr_matrix.parameters,
                                        y=corr_matrix.parameters,
                                        color_continuous_scale='RdBu_r',
                                        zmin=-1, zmax=1,
                                        title="Correlation Matrix",
                                    )
                                    fig.update_layout(height=600, template="plotly_white")
                                    st.plotly_chart(fig, use_container_width=True)

                                    strong = corr_matrix.get_strong_correlations(threshold)
                                    if strong:
                                        st.markdown(f"**Strong Correlations (|r| >= {threshold})**")
                                        strong_df = pd.DataFrame([
                                            {'Parameter 1': p1, 'Parameter 2': p2, 'Correlation': f"{r:.3f}"}
                                            for p1, p2, r in strong
                                        ])
                                        st.dataframe(strong_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info(f"No correlations above |{threshold}| found.")

                                    st.download_button(
                                        "Download Correlation Matrix",
                                        corr_matrix.to_dataframe().to_csv(),
                                        file_name="correlation_matrix.csv",
                                        mime="text/csv",
                                        key="dc_dl_corr",
                                    )
                                except Exception as e:
                                    st.error(f"Error: {e}")

                        # =============================================================
                        # CAMPAIGN COMPARISON
                        # =============================================================
                        elif mode == "Campaign Comparison":
                            st.markdown("#### Campaign Comparison")
                            if len(campaigns) < 2:
                                st.warning("Need at least 2 campaigns to compare.")
                            else:
                                other_campaigns = [c['name'] for c in campaigns if c['name'] != selected_campaign]
                                campaign_b = st.selectbox("Compare to campaign", other_campaigns, key="dc_campaign_b")
                                params_to_compare = st.multiselect(
                                    "Parameters to compare", metric_cols,
                                    default=metric_cols[:5] if len(metric_cols) > 5 else metric_cols,
                                    key="dc_camp_params",
                                )

                                if st.button("Compare Campaigns", key="dc_compare_camps") and params_to_compare:
                                    df_b = get_campaign_data(campaign_b)
                                    if df_b is None or len(df_b) == 0:
                                        st.error(f"No data in {campaign_b}.")
                                    else:
                                        result = compare_campaigns(
                                            df, df_b,
                                            selected_campaign, campaign_b,
                                            params_to_compare,
                                        )
                                        cc1, cc2 = st.columns(2)
                                        with cc1:
                                            render_metric_card(selected_campaign, f"n={result['n_tests_a']}")
                                        with cc2:
                                            render_metric_card(campaign_b, f"n={result['n_tests_b']}")

                                        rows = []
                                        for param, pdata in result['parameters'].items():
                                            rows.append({
                                                'Parameter': param,
                                                f'Mean ({selected_campaign})': f"{pdata['mean_a']:.4g}",
                                                f'Mean ({campaign_b})': f"{pdata['mean_b']:.4g}",
                                                'Delta %': f"{pdata['mean_diff_pct']:+.2f}%",
                                                'Status': 'Pass' if pdata['means_equivalent'] else 'Fail',
                                            })
                                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                                        param_to_plot = st.selectbox(
                                            "Parameter to visualize", params_to_compare, key="dc_camp_viz",
                                        )
                                        fig = go.Figure()
                                        fig.add_trace(go.Histogram(x=df[param_to_plot].dropna(), name=selected_campaign, opacity=0.7))
                                        fig.add_trace(go.Histogram(x=df_b[param_to_plot].dropna(), name=campaign_b, opacity=0.7))
                                        fig.update_layout(barmode='overlay', title=f"{param_to_plot} Distribution", template="plotly_white")
                                        st.plotly_chart(fig, use_container_width=True)

                                        st.text(format_campaign_comparison(result))
                else:
                    st.info("Select a campaign in the sidebar.")

    except Exception as exc:
        st.error(f"Data Comparison encountered an error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 3: TRANSIENT ANALYSIS
# #############################################################################
with tab3:
    try:
        st.subheader("Transient Analysis")
        st.caption(
            "Segment test data into discrete phases (pre-test, startup, transient, "
            "steady-state, shutdown, cooldown) and characterize startup/shutdown transients."
        )

        # ---- Data source ----
        ta_settings_col, ta_main_col = st.columns([1, 3])

        with ta_settings_col:
            st.markdown("**Segmentation Settings**")
            ta_threshold_pct = st.slider(
                "Activation threshold (%)", 1.0, 50.0, 10.0, 1.0,
                help="Percentage of signal range above which the test is considered active.",
                key="ta_threshold_pct",
            )
            ta_cv_threshold = st.slider(
                "CV threshold for steady state", 0.005, 0.10, 0.02, 0.005,
                help="Coefficient of variation below which a region is classified as steady-state.",
                key="ta_cv_threshold",
            )
            ta_min_phase_s = st.number_input(
                "Min phase duration (s)", 0.01, 10.0, 0.1, 0.05,
                key="ta_min_phase",
            )

        with ta_main_col:
            st.markdown("**1. Load Data**")

            ta_uploaded = st.file_uploader(
                "Upload CSV file", type=['csv'], key="ta_upload",
            )
            if ta_uploaded:
                ta_df = pd.read_csv(ta_uploaded)
                st.session_state.transient_df = ta_df
                st.success(f"Loaded {len(ta_df)} rows, {len(ta_df.columns)} columns.")
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(ta_df.head(50), use_container_width=True)

            ta_df = st.session_state.transient_df

            if ta_df is not None:
                st.divider()
                st.markdown("**2. Configure**")

                ta_time_col = _detect_time_column(ta_df)
                ta_num_cols = _numeric_columns(ta_df, exclude=[ta_time_col] if ta_time_col else [])

                tc1, tc2 = st.columns(2)
                with tc1:
                    ta_time_sel = st.selectbox(
                        "Time column",
                        [ta_time_col] + [c for c in ta_df.columns if c != ta_time_col] if ta_time_col else list(ta_df.columns),
                        key="ta_time_col",
                    )
                with tc2:
                    ta_signal_sel = st.selectbox(
                        "Signal column",
                        ta_num_cols,
                        key="ta_signal_col",
                    )

                st.divider()
                st.markdown("**3. Run Phase Segmentation**")

                if st.button("Segment Phases", type="primary", key="ta_run"):
                    with st.spinner("Segmenting test phases..."):
                        multi = segment_test_phases(
                            ta_df,
                            signal_col=ta_signal_sel,
                            time_col=ta_time_sel,
                            threshold_pct=ta_threshold_pct,
                            cv_threshold=ta_cv_threshold,
                            min_phase_duration_s=ta_min_phase_s,
                        )
                        st.session_state.transient_result = multi

                        # Run startup/shutdown transient analysis if phases exist
                        startup_metrics = None
                        shutdown_metrics = None
                        phase_types = [p.phase for p in multi.phases]

                        if TestPhase.STARTUP in phase_types:
                            startup_phase = [p for p in multi.phases if p.phase == TestPhase.STARTUP][0]
                            startup_mask = (
                                (ta_df[ta_time_sel] >= startup_phase.start_ms / 1000.0)
                                & (ta_df[ta_time_sel] <= startup_phase.end_ms / 1000.0)
                            )
                            startup_sub = ta_df.loc[startup_mask]
                            if len(startup_sub) > 2:
                                # Use steady-state mean as target if available
                                ss_phases = [p for p in multi.phases if p.phase == TestPhase.STEADY_STATE]
                                ss_val = ss_phases[0].metrics.get('mean') if ss_phases and ss_phases[0].metrics else None
                                startup_metrics = analyze_startup_transient(
                                    startup_sub, ta_signal_sel,
                                    time_col=ta_time_sel,
                                    steady_value=ss_val,
                                )
                            st.session_state.transient_startup = startup_metrics

                        if TestPhase.SHUTDOWN in phase_types:
                            shutdown_phase = [p for p in multi.phases if p.phase == TestPhase.SHUTDOWN][0]
                            shutdown_mask = (
                                (ta_df[ta_time_sel] >= shutdown_phase.start_ms / 1000.0)
                                & (ta_df[ta_time_sel] <= shutdown_phase.end_ms / 1000.0)
                            )
                            shutdown_sub = ta_df.loc[shutdown_mask]
                            if len(shutdown_sub) > 2:
                                ss_phases = [p for p in multi.phases if p.phase == TestPhase.STEADY_STATE]
                                ss_val = ss_phases[0].metrics.get('mean') if ss_phases and ss_phases[0].metrics else None
                                shutdown_metrics = analyze_shutdown_transient(
                                    shutdown_sub, ta_signal_sel,
                                    time_col=ta_time_sel,
                                    steady_value=ss_val,
                                )
                            st.session_state.transient_shutdown = shutdown_metrics

                        st.success(f"Detected {len(multi.phases)} phases in {multi.total_duration_s:.3f} s.")

                # ---- Display results ----
                multi = st.session_state.transient_result
                if multi is not None:
                    st.divider()
                    st.markdown("**4. Results**")

                    # Summary metric cards
                    sm1, sm2, sm3 = st.columns(3)
                    with sm1:
                        render_metric_card("Phases Detected", str(len(multi.phases)))
                    with sm2:
                        render_metric_card("Total Duration", f"{multi.total_duration_s:.3f} s")
                    with sm3:
                        phase_names = ", ".join(_PHASE_LABELS.get(p.phase, p.phase.value) for p in multi.phases)
                        render_metric_card("Sequence", phase_names)

                    st.markdown("")  # spacer

                    # ---- Phase timeline (horizontal bar chart) ----
                    with st.expander("Phase Timeline", expanded=True):
                        timeline_fig = go.Figure()
                        for i, phase in enumerate(multi.phases):
                            label = _PHASE_LABELS.get(phase.phase, phase.phase.value)
                            color = _PHASE_COLORS.get(phase.phase, '#71717a')
                            timeline_fig.add_trace(go.Bar(
                                x=[phase.duration_s],
                                y=["Test"],
                                orientation='h',
                                name=label,
                                marker_color=color,
                                text=f"{label} ({phase.duration_s:.3f}s)",
                                textposition='inside',
                                hovertemplate=(
                                    f"<b>{label}</b><br>"
                                    f"Start: {phase.start_ms:.0f} ms<br>"
                                    f"End: {phase.end_ms:.0f} ms<br>"
                                    f"Duration: {phase.duration_s:.3f} s<br>"
                                    f"Quality: {phase.quality}"
                                    "<extra></extra>"
                                ),
                            ))
                        timeline_fig.update_layout(
                            barmode='stack',
                            height=120,
                            margin=dict(l=10, r=10, t=10, b=10),
                            showlegend=True,
                            legend=dict(orientation='h', y=-0.4),
                            xaxis_title="Duration (s)",
                            yaxis=dict(visible=False),
                            template="plotly_white",
                        )
                        st.plotly_chart(timeline_fig, use_container_width=True)

                    # ---- Phase summary table ----
                    with st.expander("Phase Details", expanded=True):
                        phase_rows = []
                        for phase in multi.phases:
                            row = {
                                'Phase': _PHASE_LABELS.get(phase.phase, phase.phase.value),
                                'Start (ms)': f"{phase.start_ms:.1f}",
                                'End (ms)': f"{phase.end_ms:.1f}",
                                'Duration (s)': f"{phase.duration_s:.4f}",
                                'Quality': phase.quality,
                            }
                            # Include key phase metrics
                            if phase.metrics:
                                for mk, mv in phase.metrics.items():
                                    if isinstance(mv, (int, float)) and not np.isnan(mv):
                                        row[mk] = f"{mv:.4g}"
                            phase_rows.append(row)
                        st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)

                    # ---- Signal plot with phase regions ----
                    with st.expander("Signal with Phase Regions", expanded=True):
                        sig_fig = go.Figure()
                        sig_fig.add_trace(go.Scatter(
                            x=ta_df[ta_time_sel],
                            y=ta_df[ta_signal_sel],
                            mode='lines',
                            name=ta_signal_sel,
                            line=dict(color='#18181b', width=1.2),
                        ))
                        for phase in multi.phases:
                            color = _PHASE_COLORS.get(phase.phase, '#71717a')
                            label = _PHASE_LABELS.get(phase.phase, phase.phase.value)
                            sig_fig.add_vrect(
                                x0=phase.start_ms / 1000.0,
                                x1=phase.end_ms / 1000.0,
                                fillcolor=color,
                                opacity=0.12,
                                line_width=0,
                                annotation_text=label,
                                annotation_position="top left",
                                annotation_font_size=10,
                            )
                        sig_fig.update_layout(
                            title=f"{ta_signal_sel} - Phase Segmentation",
                            xaxis_title="Time (s)",
                            yaxis_title=ta_signal_sel,
                            height=500,
                            template="plotly_white",
                        )
                        st.plotly_chart(sig_fig, use_container_width=True)

                    # ---- Startup transient metrics ----
                    startup_metrics = st.session_state.transient_startup
                    if startup_metrics:
                        with st.expander("Startup Transient Metrics", expanded=True):
                            su_cols = st.columns(3)
                            key_map = [
                                ("Rise Time (10-90%)", "rise_time_10_90_s", "s"),
                                ("Time to Peak", "time_to_peak_s", "s"),
                                ("Overshoot", "overshoot_pct", "%"),
                                ("Settling Time", "settling_time_s", "s"),
                                ("Rise Time (full)", "rise_time_s", "s"),
                                ("Settling Band", "settling_band_pct", "%"),
                            ]
                            for idx, (label, mkey, unit) in enumerate(key_map):
                                val = startup_metrics.get(mkey)
                                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                    with su_cols[idx % 3]:
                                        render_metric_card(label, f"{val:.4g} {unit}")

                    # ---- Shutdown transient metrics ----
                    shutdown_metrics = st.session_state.transient_shutdown
                    if shutdown_metrics:
                        with st.expander("Shutdown Transient Metrics", expanded=True):
                            sd_cols = st.columns(3)
                            sd_map = [
                                ("Decay Time (90-10%)", "decay_time_90_10_s", "s"),
                                ("Full Decay Time", "decay_time_s", "s"),
                                ("Tail-off Impulse", "tail_off_impulse", ""),
                                ("Residual Value", "residual_value", ""),
                                ("Residual", "residual_pct", "%"),
                            ]
                            for idx, (label, mkey, unit) in enumerate(sd_map):
                                val = shutdown_metrics.get(mkey)
                                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                    with sd_cols[idx % 3]:
                                        render_metric_card(label, f"{val:.4g} {unit}")

            else:
                st.info("Upload a CSV file to begin transient analysis.")

    except Exception as exc:
        st.error(f"Transient Analysis encountered an error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 4: FREQUENCY ANALYSIS
# #############################################################################
with tab4:
    try:
        st.subheader("Frequency Analysis")
        st.caption(
            "Power spectral density, harmonic detection, spectrograms, "
            "cross-spectral coherence, and resonance identification."
        )

        # ---- Data upload ----
        fa_settings_col, fa_main_col = st.columns([1, 3])

        with fa_settings_col:
            st.markdown("**Settings**")
            fa_sample_rate = st.number_input(
                "Sample Rate (Hz)", 1, 100000, 100,
                help="Sampling frequency of the data.",
                key="fa_sample_rate",
            )
            fa_method = st.selectbox(
                "PSD Method", ["welch", "periodogram"], key="fa_method",
            )
            fa_window = st.selectbox(
                "Window Function", ["hann", "hamming", "blackman", "boxcar"],
                key="fa_window",
            )
            fa_nperseg = st.number_input(
                "Segment length (nperseg)", 32, 8192, 256,
                help="Number of samples per FFT segment.",
                key="fa_nperseg",
            )

        with fa_main_col:
            st.markdown("**1. Load Data**")
            fa_uploaded = st.file_uploader(
                "Upload CSV file", type=['csv'], key="fa_upload",
            )
            if fa_uploaded:
                fa_df = pd.read_csv(fa_uploaded)
                st.session_state.freq_df = fa_df
                st.success(f"Loaded {len(fa_df)} rows, {len(fa_df.columns)} columns.")
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(fa_df.head(50), use_container_width=True)

            fa_df = st.session_state.freq_df

            if fa_df is not None:
                fa_time_col = _detect_time_column(fa_df)
                fa_num_cols = _numeric_columns(fa_df, exclude=[fa_time_col] if fa_time_col else [])

                st.divider()

                # ---- Sub-tabs ----
                ft1, ft2, ft3, ft4, ft5 = st.tabs([
                    "PSD Analysis", "Harmonics", "Spectrogram",
                    "Cross-Spectrum", "Resonance Detection",
                ])

                # =========================================================
                # PSD Analysis
                # =========================================================
                with ft1:
                    st.markdown("**Power Spectral Density**")
                    psd_channel = st.selectbox(
                        "Select channel", fa_num_cols, key="fa_psd_ch",
                    )

                    if st.button("Compute PSD", type="primary", key="fa_psd_run"):
                        with st.spinner("Computing PSD..."):
                            sig = fa_df[psd_channel].dropna().values
                            psd_result = compute_power_spectral_density(
                                sig,
                                sample_rate_hz=fa_sample_rate,
                                method=fa_method,
                                nperseg=min(fa_nperseg, len(sig)),
                                window=fa_window,
                            )
                            st.session_state.freq_psd_result = psd_result

                    psd_result = st.session_state.freq_psd_result
                    if psd_result is not None:
                        pm1, pm2, pm3 = st.columns(3)
                        with pm1:
                            render_metric_card("Dominant Freq", f"{psd_result.dominant_frequency:.2f} Hz")
                        with pm2:
                            render_metric_card("Dominant Power", f"{psd_result.dominant_power:.4g}")
                        with pm3:
                            render_metric_card("Total Power", f"{psd_result.total_power:.4g}")

                        st.markdown("")

                        # PSD plot
                        psd_fig = go.Figure()
                        psd_fig.add_trace(go.Scatter(
                            x=psd_result.frequencies,
                            y=psd_result.power_spectral_density,
                            mode='lines',
                            name='PSD',
                            line=dict(color='#18181b', width=1.2),
                        ))
                        psd_fig.add_vline(
                            x=psd_result.dominant_frequency,
                            line_dash="dash", line_color="#dc2626",
                            annotation_text=f"Peak: {psd_result.dominant_frequency:.2f} Hz",
                        )
                        psd_fig.update_layout(
                            title="Power Spectral Density",
                            xaxis_title="Frequency (Hz)",
                            yaxis_title="PSD (unit^2/Hz)",
                            height=450,
                            template="plotly_white",
                        )
                        # Log scale toggle
                        psd_log = st.checkbox("Log scale Y-axis", value=True, key="fa_psd_log")
                        if psd_log:
                            psd_fig.update_yaxes(type="log")
                        st.plotly_chart(psd_fig, use_container_width=True)

                        # Frequency band power
                        band_power = compute_frequency_bands(psd_result)
                        st.markdown("**Frequency Band Power**")
                        band_rows = []
                        for band_name, power in band_power.items():
                            if band_name == 'total':
                                continue
                            limits = DEFAULT_FREQUENCY_BANDS.get(band_name)
                            pct = (power / band_power['total'] * 100) if band_power['total'] > 0 else 0
                            band_rows.append({
                                'Band': band_name.capitalize(),
                                'Range (Hz)': f"{limits[0]:.1f} - {limits[1]:.1f}" if limits else "-",
                                'Power': f"{power:.4g}",
                                '% of Total': f"{pct:.1f}%",
                            })
                        band_rows.append({
                            'Band': 'Total',
                            'Range (Hz)': 'All',
                            'Power': f"{band_power['total']:.4g}",
                            '% of Total': '100%',
                        })
                        st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

                # =========================================================
                # Harmonics
                # =========================================================
                with ft2:
                    st.markdown("**Harmonic Detection**")
                    st.caption("Identify the fundamental frequency and its harmonics.")

                    hc1, hc2 = st.columns(2)
                    with hc1:
                        harm_channel = st.selectbox(
                            "Select channel", fa_num_cols, key="fa_harm_ch",
                        )
                    with hc2:
                        n_harmonics = st.slider(
                            "Max harmonics", 1, 15, 5, key="fa_n_harm",
                        )
                    harm_tol = st.slider(
                        "Frequency tolerance (Hz)", 0.1, 10.0, 1.0, 0.1,
                        key="fa_harm_tol",
                    )

                    if st.button("Detect Harmonics", type="primary", key="fa_harm_run"):
                        with st.spinner("Detecting harmonics..."):
                            sig = fa_df[harm_channel].dropna().values
                            # Compute PSD first
                            psd_res = compute_power_spectral_density(
                                sig,
                                sample_rate_hz=fa_sample_rate,
                                method=fa_method,
                                nperseg=min(fa_nperseg, len(sig)),
                                window=fa_window,
                            )
                            harmonics = detect_harmonics(
                                psd_res,
                                n_harmonics=n_harmonics,
                                tolerance_hz=harm_tol,
                            )
                            st.session_state.freq_harmonics = (psd_res, harmonics)

                    harm_data = st.session_state.freq_harmonics
                    if harm_data is not None:
                        psd_res, harmonics = harm_data
                        if harmonics:
                            render_metric_card(
                                "Fundamental Frequency",
                                f"{harmonics[0].frequency:.2f} Hz" if harmonics else "N/A",
                            )
                            st.markdown("")

                            harm_rows = []
                            for h in harmonics:
                                harm_rows.append({
                                    'Harmonic #': h.harmonic_number,
                                    'Frequency (Hz)': f"{h.frequency:.2f}",
                                    'Power': f"{h.power:.4g}",
                                    'Relative Power': f"{h.relative_power:.4f}",
                                    'Relative (dB)': f"{10*np.log10(h.relative_power):.1f}" if h.relative_power > 0 else "-inf",
                                })
                            st.dataframe(pd.DataFrame(harm_rows), use_container_width=True, hide_index=True)

                            # Plot PSD with harmonic markers
                            hfig = go.Figure()
                            hfig.add_trace(go.Scatter(
                                x=psd_res.frequencies,
                                y=psd_res.power_spectral_density,
                                mode='lines', name='PSD',
                                line=dict(color='#18181b', width=1),
                            ))
                            for h in harmonics:
                                hfig.add_vline(
                                    x=h.frequency, line_dash="dot",
                                    line_color="#2563eb",
                                    annotation_text=f"H{h.harmonic_number}",
                                    annotation_font_size=10,
                                )
                            hfig.update_layout(
                                title="PSD with Detected Harmonics",
                                xaxis_title="Frequency (Hz)",
                                yaxis_title="PSD",
                                yaxis_type="log",
                                height=450,
                                template="plotly_white",
                            )
                            st.plotly_chart(hfig, use_container_width=True)
                        else:
                            st.info("No harmonics detected. The signal may lack a clear tonal component.")

                # =========================================================
                # Spectrogram
                # =========================================================
                with ft3:
                    st.markdown("**Spectrogram (Time-Frequency)**")
                    st.caption("Visualize how frequency content evolves over time.")

                    spec_channel = st.selectbox(
                        "Select channel", fa_num_cols, key="fa_spec_ch",
                    )
                    spec_nperseg = st.slider(
                        "STFT segment length", 32, 2048, 256, 32,
                        key="fa_spec_nperseg",
                    )

                    if st.button("Compute Spectrogram", type="primary", key="fa_spec_run"):
                        with st.spinner("Computing spectrogram..."):
                            sig = fa_df[spec_channel].dropna().values
                            seg = min(spec_nperseg, len(sig))
                            f_arr, t_arr, Sxx = compute_spectrogram(
                                sig,
                                sample_rate_hz=fa_sample_rate,
                                nperseg=seg,
                                window=fa_window,
                            )

                            # Use log scale for better visualization
                            Sxx_db = 10 * np.log10(np.maximum(Sxx, np.finfo(float).tiny))

                            spec_fig = go.Figure(data=go.Heatmap(
                                x=t_arr,
                                y=f_arr,
                                z=Sxx_db,
                                colorscale='Viridis',
                                colorbar=dict(title='Power (dB)'),
                            ))
                            spec_fig.update_layout(
                                title=f"Spectrogram - {spec_channel}",
                                xaxis_title="Time (s)",
                                yaxis_title="Frequency (Hz)",
                                height=500,
                                template="plotly_white",
                            )
                            st.plotly_chart(spec_fig, use_container_width=True)

                # =========================================================
                # Cross-Spectrum
                # =========================================================
                with ft4:
                    st.markdown("**Cross-Spectral Analysis**")
                    st.caption("Compute coherence and phase between two channels.")

                    if len(fa_num_cols) < 2:
                        st.warning("Need at least 2 numeric channels for cross-spectral analysis.")
                    else:
                        xc1, xc2 = st.columns(2)
                        with xc1:
                            cs_ch_a = st.selectbox(
                                "Channel A", fa_num_cols, key="fa_cs_ch_a",
                            )
                        with xc2:
                            cs_ch_b = st.selectbox(
                                "Channel B",
                                [c for c in fa_num_cols if c != cs_ch_a],
                                key="fa_cs_ch_b",
                            )

                        if st.button("Compute Cross-Spectrum", type="primary", key="fa_cs_run"):
                            with st.spinner("Computing cross-spectrum..."):
                                sig_a = fa_df[cs_ch_a].dropna().values
                                sig_b = fa_df[cs_ch_b].dropna().values
                                min_len = min(len(sig_a), len(sig_b))
                                sig_a = sig_a[:min_len]
                                sig_b = sig_b[:min_len]

                                cs_result = compute_cross_spectrum(
                                    sig_a, sig_b,
                                    sample_rate_hz=fa_sample_rate,
                                    nperseg=min(fa_nperseg, min_len),
                                )
                                st.session_state.freq_cross_result = (cs_ch_a, cs_ch_b, cs_result)

                        cs_data = st.session_state.freq_cross_result
                        if cs_data is not None:
                            cs_ch_a_stored, cs_ch_b_stored, cs_result = cs_data

                            # Peak coherence
                            peak_idx = int(np.argmax(cs_result.coherence))
                            peak_coh_freq = float(cs_result.frequencies[peak_idx])
                            peak_coh_val = float(cs_result.coherence[peak_idx])

                            csc1, csc2 = st.columns(2)
                            with csc1:
                                render_metric_card("Peak Coherence", f"{peak_coh_val:.3f}")
                            with csc2:
                                render_metric_card("At Frequency", f"{peak_coh_freq:.2f} Hz")

                            st.markdown("")

                            # Coherence plot
                            coh_fig = go.Figure()
                            coh_fig.add_trace(go.Scatter(
                                x=cs_result.frequencies,
                                y=cs_result.coherence,
                                mode='lines',
                                name='Coherence',
                                line=dict(color='#18181b', width=1.2),
                            ))
                            coh_fig.add_hline(
                                y=0.8, line_dash="dash",
                                line_color="#dc2626",
                                annotation_text="High coherence threshold",
                            )
                            coh_fig.update_layout(
                                title=f"Coherence: {cs_ch_a_stored} vs {cs_ch_b_stored}",
                                xaxis_title="Frequency (Hz)",
                                yaxis_title="Magnitude-Squared Coherence",
                                yaxis_range=[0, 1.05],
                                height=400,
                                template="plotly_white",
                            )
                            st.plotly_chart(coh_fig, use_container_width=True)

                            # Phase plot
                            phase_fig = go.Figure()
                            phase_fig.add_trace(go.Scatter(
                                x=cs_result.frequencies,
                                y=np.degrees(cs_result.phase),
                                mode='lines',
                                name='Phase',
                                line=dict(color='#2563eb', width=1.2),
                            ))
                            phase_fig.update_layout(
                                title=f"Phase: {cs_ch_a_stored} vs {cs_ch_b_stored}",
                                xaxis_title="Frequency (Hz)",
                                yaxis_title="Phase (degrees)",
                                height=350,
                                template="plotly_white",
                            )
                            st.plotly_chart(phase_fig, use_container_width=True)

                # =========================================================
                # Resonance Detection
                # =========================================================
                with ft5:
                    st.markdown("**Resonance Detection**")
                    st.caption("Find spectral peaks with Q-factor estimation.")

                    res_channel = st.selectbox(
                        "Select channel", fa_num_cols, key="fa_res_ch",
                    )
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        res_prominence = st.slider(
                            "Prominence (dB)", 1.0, 20.0, 3.0, 0.5,
                            key="fa_res_prom",
                        )
                    with rc2:
                        res_min_freq = st.number_input(
                            "Min freq (Hz)", 0.0, 10000.0, 0.5, 0.5,
                            key="fa_res_min",
                        )
                    with rc3:
                        res_max_freq = st.number_input(
                            "Max freq (Hz)", 0.0, 100000.0, 0.0,
                            help="0 = Nyquist limit",
                            key="fa_res_max",
                        )

                    if st.button("Detect Resonances", type="primary", key="fa_res_run"):
                        with st.spinner("Detecting resonances..."):
                            sig = fa_df[res_channel].dropna().values
                            psd_res = compute_power_spectral_density(
                                sig,
                                sample_rate_hz=fa_sample_rate,
                                method=fa_method,
                                nperseg=min(fa_nperseg, len(sig)),
                                window=fa_window,
                            )
                            resonances = detect_resonance(
                                psd_res,
                                prominence=res_prominence,
                                min_freq_hz=res_min_freq,
                                max_freq_hz=res_max_freq if res_max_freq > 0 else None,
                            )
                            st.session_state.freq_resonances = (psd_res, resonances)

                    res_data = st.session_state.freq_resonances
                    if res_data is not None:
                        psd_res, resonances = res_data
                        render_metric_card("Resonances Found", str(len(resonances)))
                        st.markdown("")

                        if resonances:
                            res_rows = []
                            for i, r in enumerate(resonances, 1):
                                res_rows.append({
                                    '#': i,
                                    'Frequency (Hz)': f"{r['frequency']:.2f}",
                                    'Power': f"{r['power']:.4g}",
                                    'Q-Factor': f"{r['q_factor']:.1f}" if r['q_factor'] > 0 else "N/A",
                                    'Bandwidth (Hz)': f"{r['bandwidth']:.2f}" if r['bandwidth'] > 0 else "N/A",
                                })
                            st.dataframe(pd.DataFrame(res_rows), use_container_width=True, hide_index=True)

                            # Plot PSD with resonance markers
                            rfig = go.Figure()
                            rfig.add_trace(go.Scatter(
                                x=psd_res.frequencies,
                                y=psd_res.power_spectral_density,
                                mode='lines', name='PSD',
                                line=dict(color='#18181b', width=1),
                            ))
                            for i, r in enumerate(resonances):
                                rfig.add_vline(
                                    x=r['frequency'], line_dash="dot",
                                    line_color="#dc2626",
                                    annotation_text=f"R{i+1}: {r['frequency']:.1f} Hz",
                                    annotation_font_size=9,
                                )
                            rfig.update_layout(
                                title="PSD with Resonance Peaks",
                                xaxis_title="Frequency (Hz)",
                                yaxis_title="PSD",
                                yaxis_type="log",
                                height=450,
                                template="plotly_white",
                            )
                            st.plotly_chart(rfig, use_container_width=True)
                        else:
                            st.info(
                                "No resonances detected. Try lowering the prominence threshold "
                                "or widening the frequency range."
                            )

            else:
                st.info("Upload a CSV file to begin frequency analysis.")

    except Exception as exc:
        st.error(f"Frequency Analysis encountered an error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 5: OPERATING ENVELOPE
# #############################################################################
with tab5:
    try:
        st.subheader("Operating Envelope")
        st.caption(
            "Visualize test operating points and define safe operating envelopes "
            "for hot fire campaigns."
        )

        campaigns = get_available_campaigns()

        if not campaigns:
            st.warning("No campaigns found. Run some hot fire tests and save to a campaign first.")
        else:
            campaign_names = [c['name'] for c in campaigns]

            oe_settings_col, oe_main_col = st.columns([1, 3])

            with oe_settings_col:
                st.markdown("**Configuration**")
                oe_campaign = st.selectbox(
                    "Campaign", campaign_names, key="oe_campaign",
                )

                st.divider()
                st.markdown("**Column Mapping**")
                oe_df = get_campaign_data(oe_campaign) if oe_campaign else None

                if oe_df is not None and len(oe_df) > 0:
                    all_cols = list(oe_df.columns)
                    numeric_cols = _numeric_columns(oe_df)

                    # Smart defaults
                    def _find_col(candidates, cols):
                        for c in candidates:
                            if c in cols:
                                return cols.index(c)
                        return 0

                    of_default = _find_col(['avg_of_ratio', 'of_ratio', 'avg_of'], numeric_cols)
                    pc_default = _find_col(['avg_pc_bar', 'pc_bar', 'avg_pc'], numeric_cols)

                    oe_of_col = st.selectbox(
                        "O/F Ratio column", numeric_cols,
                        index=of_default, key="oe_of_col",
                    )
                    oe_pc_col = st.selectbox(
                        "Chamber Pressure column", numeric_cols,
                        index=pc_default, key="oe_pc_col",
                    )

                    # Test ID column
                    id_default = _find_col(['test_id', 'id', 'name'], all_cols)
                    oe_id_col = st.selectbox(
                        "Test ID column", all_cols,
                        index=id_default, key="oe_id_col",
                    )

                    # Ignition column (optional)
                    oe_ign_options = ["(none)"] + all_cols
                    ign_default = 0
                    for i, c in enumerate(oe_ign_options):
                        if c in ('ignition_successful', 'ignition_success', 'ign_success'):
                            ign_default = i
                            break
                    oe_ign_col = st.selectbox(
                        "Ignition success column",
                        oe_ign_options,
                        index=ign_default,
                        key="oe_ign_col",
                    )
                    oe_ign_col_actual = oe_ign_col if oe_ign_col != "(none)" else None

                    st.divider()
                    st.markdown("**Envelope Settings**")
                    oe_margin = st.slider(
                        "Safety margin (%)", 0.0, 50.0, 10.0, 1.0,
                        key="oe_margin",
                    )
                    oe_filter_success = st.checkbox(
                        "Successful ignitions only", value=True,
                        key="oe_filter_success",
                    )
                    oe_show_ids = st.checkbox(
                        "Show test IDs on plot", value=True,
                        key="oe_show_ids",
                    )

            with oe_main_col:
                if oe_df is None or len(oe_df) == 0:
                    st.info("Select a campaign with data.")
                else:
                    # Validate columns exist and have data
                    valid = True
                    for req_col, label in [(oe_of_col, "O/F"), (oe_pc_col, "Pc")]:
                        if req_col not in oe_df.columns:
                            st.error(f"Column '{req_col}' not found.")
                            valid = False
                        elif oe_df[req_col].dropna().empty:
                            st.warning(f"Column '{req_col}' has no valid data.")
                            valid = False

                    if valid:
                        if st.button("Calculate Operating Envelope", type="primary", key="oe_run"):
                            try:
                                envelope = calculate_operating_envelope(
                                    oe_df,
                                    of_column=oe_of_col,
                                    pc_column=oe_pc_col,
                                    ignition_column=oe_ign_col_actual,
                                    margin_pct=oe_margin,
                                    filter_successful_only=oe_filter_success,
                                )

                                st.session_state['oe_envelope'] = envelope
                                st.session_state['oe_campaign_name'] = oe_campaign
                                st.success(
                                    f"Envelope calculated from {envelope.n_tests} test points "
                                    f"with {oe_margin:.0f}% safety margin."
                                )
                            except ValueError as ve:
                                st.error(f"Could not calculate envelope: {ve}")

                        # ---- Display results ----
                        if 'oe_envelope' in st.session_state and st.session_state['oe_envelope'] is not None:
                            envelope = st.session_state['oe_envelope']
                            oe_camp_name = st.session_state.get('oe_campaign_name', oe_campaign)

                            # Metric cards
                            em1, em2, em3, em4 = st.columns(4)
                            with em1:
                                render_metric_card("O/F Min", f"{envelope.of_min:.3f}")
                            with em2:
                                render_metric_card("O/F Max", f"{envelope.of_max:.3f}")
                            with em3:
                                render_metric_card("Pc Min", f"{envelope.pc_min:.2f} bar")
                            with em4:
                                render_metric_card("Pc Max", f"{envelope.pc_max:.2f} bar")

                            st.markdown("")

                            # Envelope plot
                            with st.expander("Operating Envelope Plot", expanded=True):
                                env_fig = plot_operating_envelope(
                                    oe_df,
                                    envelope=envelope,
                                    of_column=oe_of_col,
                                    pc_column=oe_pc_col,
                                    test_id_column=oe_id_col,
                                    ignition_column=oe_ign_col_actual,
                                    title=f"Operating Envelope - {oe_camp_name}",
                                    show_envelope=True,
                                    show_test_ids=oe_show_ids,
                                )
                                st.plotly_chart(env_fig, use_container_width=True)

                            # Envelope bounds table
                            with st.expander("Envelope Bounds", expanded=True):
                                bounds_data = {
                                    'Parameter': ['Mixture Ratio (O/F)', 'Chamber Pressure (bar)'],
                                    'Minimum': [f"{envelope.of_min:.4f}", f"{envelope.pc_min:.3f}"],
                                    'Maximum': [f"{envelope.of_max:.4f}", f"{envelope.pc_max:.3f}"],
                                    'Range': [
                                        f"{envelope.of_max - envelope.of_min:.4f}",
                                        f"{envelope.pc_max - envelope.pc_min:.3f} bar",
                                    ],
                                }
                                st.dataframe(
                                    pd.DataFrame(bounds_data),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.caption(
                                    f"Based on {envelope.n_tests} tests | "
                                    f"Safety margin: {envelope.margin_pct:.0f}%"
                                )

                            # HTML report
                            with st.expander("Envelope Report"):
                                report_html = create_envelope_report(
                                    oe_df, envelope, campaign_name=oe_camp_name,
                                )
                                st.markdown(report_html, unsafe_allow_html=True)

                            # Downloads
                            st.divider()
                            dl1, dl2, dl3 = st.columns(3)
                            with dl1:
                                env_fig_html = env_fig.to_html(include_plotlyjs='cdn')
                                st.download_button(
                                    "Download Plot (HTML)",
                                    env_fig_html,
                                    file_name=f"envelope_{oe_camp_name}.html",
                                    mime="text/html",
                                    key="oe_dl_plot",
                                )
                            with dl2:
                                report_html_full = create_envelope_report(
                                    oe_df, envelope, campaign_name=oe_camp_name,
                                )
                                st.download_button(
                                    "Download Report (HTML)",
                                    report_html_full,
                                    file_name=f"envelope_report_{oe_camp_name}.html",
                                    mime="text/html",
                                    key="oe_dl_report",
                                )
                            with dl3:
                                st.download_button(
                                    "Download Bounds (JSON)",
                                    json.dumps(envelope.to_dict(), indent=2),
                                    file_name=f"envelope_bounds_{oe_camp_name}.json",
                                    mime="application/json",
                                    key="oe_dl_json",
                                )

    except Exception as exc:
        st.error(f"Operating Envelope encountered an error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
