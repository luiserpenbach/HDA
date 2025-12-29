# pages/03_❄️_Cold_Flow.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

from data_lib.config_loader import get_available_configs, load_config
from data_lib.sensor_data_tools import (
    resample_data, find_steady_windows, find_steady_windows_ml
)
from data_lib.propulsion_physics import calculate_cold_flow_metrics
from data_lib.campaign_manager import get_available_campaigns, save_to_campaign
from data_lib.report_generator import generate_cold_flow_test_report

st.set_page_config(page_title="Cold Flow Analysis", page_icon="❄️", layout="wide")

st.title("❄️ Cold Flow Test Analysis")
st.markdown("Single test analysis for injector and valve characterization")

# --- CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")

    configs = get_available_configs()
    if configs:
        selected_config = st.selectbox("Test Configuration", ["None"] + configs)
        if selected_config != "None":
            config = load_config(selected_config)
            st.success(f"✓ Loaded: {selected_config}")
        else:
            config = None
    else:
        st.warning("No configs found")
        config = None

    st.markdown("---")

    # Processing settings
    st.subheader("Processing")
    freq_ms = st.number_input("Resample (ms)", value=10, min_value=1)
    window_ms = st.number_input("Steady Window (ms)", value=500, min_value=100)
    cv_thresh = st.number_input("CV Threshold (%)", value=1.0, min_value=0.1)

# --- FILE UPLOAD ---
col_upload, col_meta = st.columns([1, 1])

with col_upload:
    st.subheader("📂 Data Input")

    # Option 1: Upload file
    uploaded_file = st.file_uploader("Upload Test CSV/Parquet", type=['csv', 'parquet'])

    # Option 2: Load from last ingested
    if 'last_ingested_test' in st.session_state:
        st.markdown("**Or load from last ingest:**")
        if st.button("📥 Load Last Ingested Test"):
            last_test = st.session_state['last_ingested_test']
            uploaded_file = last_test['data_file']
            st.success(f"Loaded: {last_test['test_id']}")

with col_meta:
    if uploaded_file:
        st.subheader("📋 Test Metadata")

        # Try to load from metadata if available
        default_id = os.path.splitext(uploaded_file.name)[0] if isinstance(uploaded_file, str) else uploaded_file.name

        test_id = st.text_input("Test ID", value=default_id)
        part_name = st.text_input("Part Number", value="INJ-001")
        serial_num = st.text_input("Serial Number", value="SN-001")

        if config:
            fluid = config.get('fluid', {})
            geom = config.get('geometry', {})
            st.info(f"**Fluid:** {fluid.get('name', '?')} ({fluid.get('density_kg_m3')} kg/m³)\n\n"
                    f"**Orifice:** {geom.get('orifice_area_mm2')} mm²")

# --- ANALYSIS ---
if uploaded_file:
    # Load data
    try:
        if isinstance(uploaded_file, str):
            if uploaded_file.endswith('.parquet'):
                df_raw = pd.read_parquet(uploaded_file)
            else:
                df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)

        # Apply channel mapping if config exists
        if config and 'channel_config' in config:
            mapping = config['channel_config']
            df_raw.columns = df_raw.columns.astype(str)
            df_raw.rename(columns=mapping, inplace=True)

        # Resample
        time_col = config.get('columns', {}).get('timestamp', 'timestamp') if config else 'timestamp'
        if time_col not in df_raw.columns:
            time_col = [c for c in df_raw.columns if 'time' in c.lower()][0]

        df = resample_data(df_raw, time_col=time_col, freq_ms=freq_ms)
        df['time_s'] = (df['timestamp'] - df['timestamp'].min()) / 1000.0

        st.success(f"✓ Loaded {len(df)} samples at {freq_ms}ms resolution")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # --- CHANNEL SELECTION ---
    st.subheader("1️⃣ Signal Selection")

    cols_cfg = config.get('columns', {}) if config else {}

    # Auto-detect channels
    def_p = [cols_cfg.get('upstream_pressure', cols_cfg.get('inlet_pressure'))]
    def_p = [c for c in def_p if c and c in df.columns]

    def_f = [cols_cfg.get('mass_flow', cols_cfg.get('mf'))]
    def_f = [c for c in def_f if c and c in df.columns]

    c1, c2 = st.columns(2)
    sel_p = c1.multiselect("Pressure Channels", df.columns, default=def_p)
    sel_f = c2.multiselect("Flow Channels", df.columns, default=def_f)

    # --- VISUALIZATION ---
    st.subheader("2️⃣ Data Visualization")

    fig_main = make_subplots(specs=[[{"secondary_y": True}]])

    for c in sel_p:
        fig_main.add_trace(go.Scatter(x=df['time_s'], y=df[c], name=c, line=dict(width=2)), secondary_y=False)

    for c in sel_f:
        fig_main.add_trace(go.Scatter(x=df['time_s'], y=df[c], name=c, line=dict(dash='dot', width=2)),
                           secondary_y=True)

    fig_main.update_yaxes(title_text="Pressure (bar)", secondary_y=False)
    fig_main.update_yaxes(title_text="Mass Flow (g/s)", secondary_y=True)
    fig_main.update_xaxes(title_text="Time (s)")
    fig_main.update_layout(height=400, margin=dict(t=10, b=10))

    # Temperature subplot
    t_cols = [c for c in df.columns if any(x in c.lower() for x in ['temp', 'tc'])]
    if t_cols:
        fig_temp = go.Figure()
        for c in t_cols:
            fig_temp.add_trace(go.Scatter(x=df['time_s'], y=df[c], name=c))
        fig_temp.update_layout(height=200, title="Temperature", margin=dict(t=30, b=10))
        st.plotly_chart(fig_temp, use_container_width=True)

    # --- STEADY STATE DETECTION ---
    st.subheader("3️⃣ Steady State Detection")

    col_algo, col_stab, col_thresh, col_window = st.columns([1, 2, 1, 1])

    detection_method = col_algo.selectbox("Method", ["CV-based", "ML-based"])
    stab_sigs = col_stab.multiselect("Stability Sensors", df.columns, default=sel_p)
    min_thresh = col_thresh.number_input("Min Threshold", value=1.0, step=0.5)

    windows = []
    predictions = None

    if detection_method == "CV-based":
        if stab_sigs:
            windows, max_cv = find_steady_windows(
                df, stab_sigs, 'timestamp',
                window_ms, cv_thresh, min_thresh
            )

            with st.expander("📊 CV Trace"):
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Scatter(x=df['time_s'], y=max_cv, line=dict(color='orange')))
                fig_cv.add_hline(y=cv_thresh, line_dash="dash", line_color="red")
                fig_cv.update_layout(height=200, yaxis_title="CV %")
                st.plotly_chart(fig_cv, use_container_width=True)

    else:  # ML-based
        contamination = col_window.slider("Contamination", 0.05, 0.30, 0.15, 0.05)

        if stab_sigs:
            try:
                windows, predictions = find_steady_windows_ml(
                    df, stab_sigs, 'timestamp',
                    window_ms=window_ms,
                    contamination=contamination,
                    min_duration_ms=window_ms,
                    min_val=min_thresh
                )

                with st.expander("🤖 ML Classification"):
                    fig_ml = go.Figure()

                    stable_mask = predictions == 1
                    fig_ml.add_trace(go.Scatter(
                        x=df[~stable_mask]['time_s'],
                        y=df[~stable_mask][stab_sigs[0]],
                        mode='markers',
                        name='Transient',
                        marker=dict(color='lightcoral', size=2)
                    ))
                    fig_ml.add_trace(go.Scatter(
                        x=df[stable_mask]['time_s'],
                        y=df[stable_mask][stab_sigs[0]],
                        mode='markers',
                        name='Stable',
                        marker=dict(color='lightgreen', size=2)
                    ))

                    fig_ml.update_layout(height=200)
                    st.plotly_chart(fig_ml, use_container_width=True)

                    stable_pct = (predictions == 1).sum() / len(predictions) * 100
                    st.caption(f"✅ Stable: {stable_pct:.1f}%")

            except Exception as e:
                st.error(f"ML Detection Error: {e}")
                windows = []

    # --- WINDOW SELECTION ---
    selected_window = None

    if not windows:
        st.warning("⚠️ No steady windows found")
    else:
        win_opts = []
        for i, w in enumerate(windows):
            dur_s = (w[1] - w[0]) / 1000.0
            start_s = (w[0] - df['timestamp'].min()) / 1000.0
            win_opts.append(f"Window {i + 1}: {start_s:.1f}s (Duration: {dur_s:.1f}s)")

        sel_idx = col_window.selectbox("Select Window", range(len(windows)), format_func=lambda x: win_opts[x])
        selected_window = windows[sel_idx]

        # Highlight on main plot
        s_s = (selected_window[0] - df['timestamp'].min()) / 1000.0
        e_s = (selected_window[1] - df['timestamp'].min()) / 1000.0
        fig_main.add_vrect(x0=s_s, x1=e_s, fillcolor="green", opacity=0.2)
        fig_main.add_vline(x=s_s, line_width=2, line_dash="dash", line_color="green")
        fig_main.add_vline(x=e_s, line_width=2, line_dash="dash", line_color="green")

        st.plotly_chart(fig_main, use_container_width=True)

        # --- RESULTS ---
        st.subheader("4️⃣ Test Results")

        steady_df = df[(df['timestamp'] >= selected_window[0]) & (df['timestamp'] <= selected_window[1])]
        avg = steady_df.mean().to_dict()

        # Calculate metrics
        metrics = calculate_cold_flow_metrics(avg, config or {})

        # Display
        m1, m2, m3, m4 = st.columns(4)

        val_p = avg.get(sel_p[0], 0) if sel_p else 0
        val_f = avg.get(sel_f[0], 0) if sel_f else 0
        val_t = avg.get(t_cols[0], 0) if t_cols else 20.0
        val_cd = metrics.get('Cd', 0)

        m1.metric("Pressure", f"{val_p:.2f} bar")
        m2.metric("Mass Flow", f"{val_f:.2f} g/s")
        m3.metric("Temperature", f"{val_t:.1f} °C")
        m4.metric("Cd", f"{val_cd:.4f}" if val_cd else "N/A")

        # --- SAVE OPTIONS ---
        st.markdown("---")
        st.subheader("5️⃣ Save Results")

        col_save1, col_save2 = st.columns(2)

        with col_save1:
            st.markdown("**Campaign Database**")
            campaigns = get_available_campaigns()

            if campaigns:
                target_campaign = st.selectbox("Target Campaign", campaigns)
                comments = st.text_input("Comments", placeholder="Throttle 50%, nominal")

                if st.button("💾 Save to Campaign", type="primary"):
                    save_id = test_id
                    if len(windows) > 1:
                        save_id = f"{test_id}_W{sel_idx + 1}"

                    record = {
                        'test_id': save_id,
                        'test_path': uploaded_file if isinstance(uploaded_file, str) else None,
                        'part': part_name,
                        'serial_num': serial_num,
                        'test_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'operator': 'N/A',
                        'fluid': config['fluid'].get('name') if config else 'N/A',
                        'avg_p_up_bar': val_p,
                        'avg_T_up_K': val_t + 273.15,
                        'avg_p_down_bar': 1.013,
                        'avg_mf_g_s': val_f,
                        'orifice_area_mm2': config['geometry'].get('orifice_area_mm2') if config else 0,
                        'avg_rho_kg_m3': config['fluid'].get('density_kg_m3') if config else 0,
                        'avg_cd_CALC': val_cd,
                        'dp_bar': metrics.get('dP (bar)', 0),
                        'comments': comments,
                        'config_used': selected_config if config else 'None'
                    }

                    try:
                        save_to_campaign(target_campaign, record)
                        st.success(f"✅ Saved to {target_campaign}!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error saving: {e}")
            else:
                st.info("No campaigns found. Create one in CF Campaign page.")

        with col_save2:
            st.markdown("**HTML Report**")

            if st.button("📄 Generate Report"):
                # Prepare data for report
                test_data = {
                    'test_id': test_id,
                    'part': part_name,
                    'fluid': config['fluid'].get('name') if config else 'N/A',
                    'avg_cd_CALC': val_cd,
                    'avg_p_up_bar': val_p,
                    'avg_mf_g_s': val_f,
                    'avg_T_up_K': val_t + 273.15,
                    'comments': comments if 'comments' in locals() else ''
                }

                steady_window_info = {
                    'duration_s': (selected_window[1] - selected_window[0]) / 1000.0,
                    'method': detection_method
                }

                figures = [fig_main]
                if t_cols:
                    figures.append(fig_temp)

                html_report = generate_cold_flow_test_report(
                    test_data, config, figures, steady_window_info
                )

                st.download_button(
                    "📥 Download Report",
                    html_report,
                    f"{test_id}_report.html",
                    "text/html"
                )
                st.success("Report generated!")