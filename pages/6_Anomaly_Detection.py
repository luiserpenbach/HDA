"""
Advanced Anomaly Detection Page (P2)
====================================
Comprehensive anomaly detection with sensor health monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.advanced_anomaly import (
    run_anomaly_detection,
    format_anomaly_table,
    AnomalyType,
    AnomalySeverity,
)

st.set_page_config(page_title="Anomaly Detection", page_icon="AD", layout="wide")

st.title("Advanced Anomaly Detection")
st.markdown("Comprehensive anomaly detection with sensor health monitoring.")

# Initialize session state
if 'anomaly_report' not in st.session_state:
    st.session_state.anomaly_report = None
if 'anomaly_df' not in st.session_state:
    st.session_state.anomaly_df = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Settings")
    
    st.subheader("Detection Parameters")
    
    spike_threshold = st.slider("Spike Threshold (σ)", 2.0, 6.0, 4.0, 0.5)
    sample_rate = st.number_input("Sample Rate (Hz)", 1, 10000, 100)
    
    st.divider()
    
    st.subheader("Correlation Checks")
    check_correlations = st.checkbox("Check correlations", value=False)

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.anomaly_df = df
    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    with st.expander("Data Preview"):
        st.dataframe(df.head(50), use_container_width=True)

df = st.session_state.anomaly_df

if df is not None:
    st.divider()
    st.header("2. Select Channels")
    
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
        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    )
    
    correlation_pairs = None
    if check_correlations and len(selected_channels) >= 2:
        with st.expander("Configure Correlation Pairs"):
            pairs = []
            n_pairs = st.number_input("Number of pairs", 0, 5, 1)
            
            for i in range(int(n_pairs)):
                col1, col2 = st.columns(2)
                with col1:
                    ch1 = st.selectbox(f"Channel 1 (pair {i+1})", selected_channels, key=f"corr1_{i}")
                with col2:
                    ch2 = st.selectbox(f"Channel 2 (pair {i+1})", selected_channels, key=f"corr2_{i}")
                if ch1 != ch2:
                    pairs.append((ch1, ch2))
            
            correlation_pairs = pairs if pairs else None
    
    st.divider()
    st.header("3. Run Analysis")
    
    if st.button("Detect Anomalies", type="primary"):
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
        st.header("4. Results")
        
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
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Details", "Health", "Visualization"])
        
        with tab1:
            st.subheader("Anomaly Summary")
            
            type_counts = {}
            for anomaly in report.get_all_anomalies():
                t = anomaly.anomaly_type.value
                type_counts[t] = type_counts.get(t, 0) + 1
            
            if type_counts:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**By Type:**")
                    for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                        st.markdown(f"- {atype}: {count}")
                with col2:
                    st.markdown("**By Severity:**")
                    st.markdown(f"- Critical: {report.critical_count}")
                    st.markdown(f"- Warning: {report.warning_count}")
                    info_count = report.total_anomalies - report.critical_count - report.warning_count
                    st.markdown(f"- Info: {info_count}")
            else:
                st.success("[PASS] No anomalies detected!")
            
            st.divider()
            st.text(report.summary())
        
        with tab2:
            st.subheader("Anomaly Details")
            
            anomaly_df = format_anomaly_table(report)
            
            if len(anomaly_df) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    filter_channel = st.selectbox("Filter by channel", ["All"] + list(report.channel_reports.keys()))
                with col2:
                    filter_type = st.selectbox("Filter by type", ["All"] + list(type_counts.keys()) if type_counts else ["All"])
                with col3:
                    filter_severity = st.selectbox("Filter by severity", ["All", "CRITICAL", "WARNING", "INFO"])
                
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
                    mime="text/csv"
                )
            else:
                st.success("No anomalies to display")
        
        with tab3:
            st.subheader("Sensor Health Scores")
            
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
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{channel}**")
                    with col2:
                        st.progress(health)
                    with col3:
                        st.markdown(f":{color}[{health:.0%} - {status}]")
                
                st.divider()
                health_values = list(report.sensor_health.values())
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Health", f"{min(health_values):.0%}")
                with col2:
                    st.metric("Avg Health", f"{np.mean(health_values):.0%}")
                with col3:
                    st.metric("Max Health", f"{max(health_values):.0%}")
        
        with tab4:
            st.subheader("Anomaly Visualization")
            
            viz_channel = st.selectbox("Select channel", list(report.channel_reports.keys()))
            
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
