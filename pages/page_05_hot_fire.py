# pages/05_Hot_Fire.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import CoolProp.CoolProp as CP

from data_lib.config_loader import get_available_configs, load_config
from data_lib.sensor_data_tools import (
    resample_data, find_steady_windows, find_steady_windows_ml
)
from data_lib.propulsion_physics import calculate_hot_fire_metrics
from data_lib.campaign_manager import get_available_campaigns, save_to_campaign
from data_lib.report_generator import generate_hot_fire_test_report
from data_lib.hf_analytics import plot_standard_hot_fire

st.set_page_config(page_title="Hot Fire Analysis", page_icon="🔥", layout="wide")

st.title("Hot Fire Test Analysis")
st.markdown("Single test hot fire performance analysis and characterization")

# --- CONFIGURATION ---
with st.sidebar:
    st.header("Configuration")

    configs = get_available_configs()
    if configs:
        selected_config = st.selectbox("Test Configuration", ["None"] + configs)
        if selected_config != "None":
            config = load_config(selected_config)
            st.success(f"Loaded: {selected_config}")
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
    cv_thresh = st.number_input("CV Threshold (%)", value=2.0, min_value=0.1)

# --- FILE UPLOAD ---
col_upload, col_meta = st.columns([1, 1])

with col_upload:
    st.subheader("Data Input")

    uploaded_file = st.file_uploader("Upload Test CSV/Parquet", type=['csv', 'parquet'])

    if 'last_ingested_test' in st.session_state:
        st.markdown("**Or load from last ingest:**")
        if st.button("Load Last Ingested Test"):
            last_test = st.session_state['last_ingested_test']
            uploaded_file = last_test['data_file']
            st.success(f"Loaded: {last_test['test_id']}")

with col_meta:
    if uploaded_file:
        st.subheader("Test Metadata")

        default_id = os.path.splitext(uploaded_file.name)[0] if isinstance(uploaded_file, str) else uploaded_file.name

        test_id = st.text_input("Test ID", value=default_id)
        part_name = st.text_input("Part Number", value="ENG-001")
        serial_num = st.text_input("Serial Number", value="SN-001")

        if config:
            propellants = config.get('propellants', {})
            chamber = config.get('chamber', {})
            st.info(f"**Propellants:** {propellants.get('oxidizer', '?')} / {propellants.get('fuel', '?')}\n\n"
                    f"**Throat Area:** {chamber.get('throat_area_mm2', '?')} mm²\n\n"
                    f"**Expansion Ratio:** {chamber.get('expansion_ratio', '?')}")

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

        st.success(f"Loaded {len(df)} samples at {freq_ms}ms resolution")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # --- CHANNEL SELECTION ---
    st.subheader("1. Signal Selection")

    cols_cfg = config.get('columns', {}) if config else {}

    # Auto-detect channels
    def_pc = [cols_cfg.get('chamber_pressure', cols_cfg.get('pc'))]
    def_pc = [c for c in def_pc if c and c in df.columns]

    def_thrust = [cols_cfg.get('thrust', cols_cfg.get('force'))]
    def_thrust = [c for c in def_thrust if c and c in df.columns]

    def_mf_ox = [cols_cfg.get('mf_ox', cols_cfg.get('oxidizer_flow'))]
    def_mf_ox = [c for c in def_mf_ox if c and c in df.columns]

    def_mf_fuel = [cols_cfg.get('mf_fuel', cols_cfg.get('fuel_flow'))]
    def_mf_fuel = [c for c in def_mf_fuel if c and c in df.columns]

    c1, c2, c3, c4 = st.columns(4)

    sel_pc = c1.selectbox("Chamber Pressure", df.columns,
                          index=df.columns.tolist().index(def_pc[0]) if def_pc else 0)
    sel_thrust = c2.selectbox("Thrust", df.columns,
                              index=df.columns.tolist().index(def_thrust[0]) if def_thrust else 0)
    sel_mf_ox = c3.selectbox("Ox Mass Flow", df.columns,
                             index=df.columns.tolist().index(def_mf_ox[0]) if def_mf_ox else 0)
    sel_mf_fuel = c4.selectbox("Fuel Mass Flow", df.columns,
                               index=df.columns.tolist().index(def_mf_fuel[0]) if def_mf_fuel else 0)

    # Additional channels
    with st.expander("Additional Channels"):
        col_add1, col_add2 = st.columns(2)

        pressure_cols = [c for c in df.columns if 'pressure' in c.lower() or 'p_' in c.lower()]
        temp_cols = [c for c in df.columns if 'temp' in c.lower() or 'tc' in c.lower()]

        sel_p_ox = col_add1.selectbox("Ox Inlet Pressure (optional)", ["None"] + pressure_cols)
        sel_p_fuel = col_add2.selectbox("Fuel Inlet Pressure (optional)", ["None"] + pressure_cols)

        sel_temps = st.multiselect("Temperature Channels", temp_cols, default=temp_cols[:4] if temp_cols else [])

    # --- STANDARD PLOT ---
    st.subheader("2. Test Overview")

    # Calculate O/F ratio
    df['of_ratio'] = df[sel_mf_ox] / df[sel_mf_fuel]
    df['mf_total'] = df[sel_mf_ox] + df[sel_mf_fuel]

    # Create standard 4-panel plot
    fig_main = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Chamber Pressure', 'Thrust', 'Mass Flows', 'O/F Ratio'),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # Panel 1: Chamber Pressure
    fig_main.add_trace(
        go.Scatter(x=df['time_s'], y=df[sel_pc], name='Pc', line=dict(color='#E74C3C', width=2)),
        row=1, col=1
    )

    # Panel 2: Thrust
    fig_main.add_trace(
        go.Scatter(x=df['time_s'], y=df[sel_thrust], name='Thrust', line=dict(color='#3498DB', width=2)),
        row=2, col=1
    )

    # Panel 3: Mass Flows
    fig_main.add_trace(
        go.Scatter(x=df['time_s'], y=df[sel_mf_ox], name='Ox Flow', line=dict(color='#2ECC71', width=2)),
        row=3, col=1
    )
    fig_main.add_trace(
        go.Scatter(x=df['time_s'], y=df[sel_mf_fuel], name='Fuel Flow', line=dict(color='#F39C12', width=2)),
        row=3, col=1
    )
    fig_main.add_trace(
        go.Scatter(x=df['time_s'], y=df['mf_total'], name='Total Flow',
                   line=dict(color='#9B59B6', width=2, dash='dash')),
        row=3, col=1
    )

    # Panel 4: O/F Ratio
    fig_main.add_trace(
        go.Scatter(x=df['time_s'], y=df['of_ratio'], name='O/F', line=dict(color='#E67E22', width=2)),
        row=4, col=1
    )

    # Update axes
    fig_main.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig_main.update_yaxes(title_text="Pressure (bar)", row=1, col=1)
    fig_main.update_yaxes(title_text="Thrust (N)", row=2, col=1)
    fig_main.update_yaxes(title_text="Mass Flow (g/s)", row=3, col=1)
    fig_main.update_yaxes(title_text="O/F Ratio", row=4, col=1)

    fig_main.update_layout(height=1000, showlegend=True, hovermode='x unified')

    st.plotly_chart(fig_main, use_container_width=True)

    # Temperature subplot if available
    if sel_temps:
        fig_temp = go.Figure()
        for tc in sel_temps:
            fig_temp.add_trace(go.Scatter(x=df['time_s'], y=df[tc], name=tc, mode='lines'))
        fig_temp.update_layout(
            title="Temperature Traces",
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            height=300,
            hovermode='x unified'
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    # Injector pressure drop if available
    if sel_p_ox != "None" or sel_p_fuel != "None":
        with st.expander("Injector Pressure Drop Analysis"):
            fig_dp = make_subplots(rows=1, cols=2, subplot_titles=("Oxidizer ΔP", "Fuel ΔP"))

            if sel_p_ox != "None":
                df['dp_ox'] = df[sel_p_ox] - df[sel_pc]
                fig_dp.add_trace(
                    go.Scatter(x=df['time_s'], y=df['dp_ox'], name='ΔP Ox', line=dict(color='green')),
                    row=1, col=1
                )
                fig_dp.update_yaxes(title_text="ΔP (bar)", row=1, col=1)

            if sel_p_fuel != "None":
                df['dp_fuel'] = df[sel_p_fuel] - df[sel_pc]
                fig_dp.add_trace(
                    go.Scatter(x=df['time_s'], y=df['dp_fuel'], name='ΔP Fuel', line=dict(color='orange')),
                    row=1, col=2
                )
                fig_dp.update_yaxes(title_text="ΔP (bar)", row=1, col=2)

            fig_dp.update_xaxes(title_text="Time (s)")
            fig_dp.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_dp, use_container_width=True)

    # --- STEADY STATE DETECTION ---
    st.subheader("3. Steady State Detection")

    col_algo, col_stab, col_thresh = st.columns([1, 2, 1])

    detection_method = col_algo.selectbox("Method", ["CV-based", "ML-based"])
    stab_sigs = col_stab.multiselect(
        "Stability Sensors",
        df.columns,
        default=[sel_pc, sel_thrust, sel_mf_ox, sel_mf_fuel]
    )
    min_thresh = col_thresh.number_input("Min Threshold", value=5.0, step=1.0)

    windows = []
    predictions = None

    if detection_method == "CV-based":
        if stab_sigs:
            windows, max_cv = find_steady_windows(
                df, stab_sigs, 'timestamp',
                window_ms, cv_thresh, min_thresh
            )

            with st.expander("CV Trace"):
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Scatter(x=df['time_s'], y=max_cv, line=dict(color='orange')))
                fig_cv.add_hline(y=cv_thresh, line_dash="dash", line_color="red")
                fig_cv.update_layout(height=200, yaxis_title="CV %")
                st.plotly_chart(fig_cv, use_container_width=True)

    else:  # ML-based
        contamination = col_thresh.slider("Contamination", 0.05, 0.30, 0.15, 0.05)

        if stab_sigs:
            try:
                windows, predictions = find_steady_windows_ml(
                    df, stab_sigs, 'timestamp',
                    window_ms=window_ms,
                    contamination=contamination,
                    min_duration_ms=window_ms,
                    min_val=min_thresh,
                    active_test_only=True
                )

                with st.expander("ML Classification"):
                    fig_ml = go.Figure()

                    stable_mask = predictions == 1
                    fig_ml.add_trace(go.Scatter(
                        x=df[~stable_mask]['time_s'],
                        y=df[~stable_mask][sel_pc],
                        mode='markers',
                        name='Transient',
                        marker=dict(color='lightcoral', size=2)
                    ))
                    fig_ml.add_trace(go.Scatter(
                        x=df[stable_mask]['time_s'],
                        y=df[stable_mask][sel_pc],
                        mode='markers',
                        name='Stable',
                        marker=dict(color='lightgreen', size=2)
                    ))

                    fig_ml.update_layout(height=200)
                    st.plotly_chart(fig_ml, use_container_width=True)

                    stable_pct = (predictions == 1).sum() / len(predictions) * 100
                    st.caption(f"Stable: {stable_pct:.1f}%")

            except Exception as e:
                st.error(f"ML Detection Error: {e}")
                windows = []

    # --- WINDOW SELECTION ---
    selected_window = None

    if not windows:
        st.warning("No steady windows found")
    else:
        win_opts = []
        for i, w in enumerate(windows):
            dur_s = (w[1] - w[0]) / 1000.0
            start_s = (w[0] - df['timestamp'].min()) / 1000.0
            win_opts.append(f"Window {i + 1}: {start_s:.1f}s (Duration: {dur_s:.1f}s)")

        sel_idx = st.selectbox("Select Window", range(len(windows)), format_func=lambda x: win_opts[x])
        selected_window = windows[sel_idx]

        # Highlight on main plot
        s_s = (selected_window[0] - df['timestamp'].min()) / 1000.0
        e_s = (selected_window[1] - df['timestamp'].min()) / 1000.0

        for i in range(1, 5):
            fig_main.add_vrect(x0=s_s, x1=e_s, fillcolor="green", opacity=0.2, row=i, col=1)
            fig_main.add_vline(x=s_s, line_width=2, line_dash="dash", line_color="green", row=i, col=1)
            fig_main.add_vline(x=e_s, line_width=2, line_dash="dash", line_color="green", row=i, col=1)

        st.plotly_chart(fig_main, use_container_width=True)

        # --- RESULTS ---
        st.subheader("4. Test Results")

        steady_df = df[(df['timestamp'] >= selected_window[0]) & (df['timestamp'] <= selected_window[1])]
        avg = steady_df.mean().to_dict()

        # Calculate metrics
        metrics = calculate_hot_fire_metrics(avg, config or {})

        # Display key metrics
        st.markdown("**Primary Performance Metrics**")

        m1, m2, m3, m4, m5 = st.columns(5)

        val_pc = avg.get(sel_pc, 0)
        val_thrust = avg.get(sel_thrust, 0)
        val_isp = metrics.get('Isp (s)', 0)
        val_cstar = metrics.get('C* (m/s)', 0)
        val_of = avg.get('of_ratio', 0)

        m1.metric("Pc", f"{val_pc:.2f} bar")
        m2.metric("Thrust", f"{val_thrust:.2f} N")
        m3.metric("Isp", f"{val_isp:.1f} s" if val_isp else "N/A")
        m4.metric("C*", f"{val_cstar:.0f} m/s" if val_cstar else "N/A")
        m5.metric("O/F", f"{val_of:.2f}")

        # Additional metrics in expandable section
        with st.expander("Detailed Metrics"):
            col_det1, col_det2, col_det3 = st.columns(3)

            with col_det1:
                st.markdown("**Mass Flows**")
                st.write(f"Ox Flow: {avg.get(sel_mf_ox, 0):.2f} g/s")
                st.write(f"Fuel Flow: {avg.get(sel_mf_fuel, 0):.2f} g/s")
                st.write(f"Total Flow: {avg.get('mf_total', 0):.2f} g/s")

            with col_det2:
                st.markdown("**Derived Performance**")
                if 'Cf' in metrics:
                    st.write(f"Cf: {metrics['Cf']:.3f}")
                if 'C* efficiency (%)' in metrics:
                    st.write(f"C* η: {metrics['C* efficiency (%)']:.1f}%")
                if 'Isp efficiency (%)' in metrics:
                    st.write(f"Isp η: {metrics['Isp efficiency (%)']:.1f}%")

            with col_det3:
                st.markdown("**Burn Statistics**")
                duration = (selected_window[1] - selected_window[0]) / 1000.0
                total_impulse = val_thrust * duration
                st.write(f"Duration: {duration:.2f} s")
                st.write(f"Total Impulse: {total_impulse:.1f} N·s")

                if sel_p_ox != "None":
                    st.write(f"ΔP Ox: {avg.get('dp_ox', 0):.2f} bar")
                if sel_p_fuel != "None":
                    st.write(f"ΔP Fuel: {avg.get('dp_fuel', 0):.2f} bar")

        # Theoretical comparison if config has CEA data
        if config and 'theoretical' in config:
            st.markdown("---")
            st.markdown("**Theoretical Comparison**")

            theo = config['theoretical']

            comp_data = []

            if 'isp_vac' in theo and val_isp:
                theo_isp = theo['isp_vac']
                eta_isp = (val_isp / theo_isp) * 100
                comp_data.append({
                    'Parameter': 'Isp',
                    'Measured': f"{val_isp:.1f} s",
                    'Theoretical': f"{theo_isp:.1f} s",
                    'Efficiency': f"{eta_isp:.1f}%"
                })

            if 'c_star' in theo and val_cstar:
                theo_cstar = theo['c_star']
                eta_cstar = (val_cstar / theo_cstar) * 100
                comp_data.append({
                    'Parameter': 'C*',
                    'Measured': f"{val_cstar:.0f} m/s",
                    'Theoretical': f"{theo_cstar:.0f} m/s",
                    'Efficiency': f"{eta_cstar:.1f}%"
                })

            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # --- SAVE OPTIONS ---
        st.markdown("---")
        st.subheader("5. Save Results")

        col_save1, col_save2 = st.columns(2)

        with col_save1:
            st.markdown("**Campaign Database**")
            campaigns = [c for c in get_available_campaigns()]

            if campaigns:
                target_campaign = st.selectbox("Target Campaign", campaigns)
                comments = st.text_input("Comments", placeholder="Nominal burn, steady thrust")

                if st.button("Save to Campaign", type="primary"):
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
                        'propellants': f"{config.get('propellants', {}).get('oxidizer', '?')} / {config.get('propellants', {}).get('fuel', '?')}" if config else 'N/A',
                        'duration_s': (selected_window[1] - selected_window[0]) / 1000.0,
                        'avg_pc_bar': val_pc,
                        'avg_thrust_n': val_thrust,
                        'avg_mf_total_g_s': avg.get('mf_total', 0),
                        'avg_mf_ox_g_s': avg.get(sel_mf_ox, 0),
                        'avg_mf_fuel_g_s': avg.get(sel_mf_fuel, 0),
                        'avg_of_ratio': val_of,
                        'avg_isp_s': val_isp,
                        'avg_c_star_m_s': val_cstar,
                        'avg_cf': metrics.get('Cf', 0),
                        'eta_c_star_pct': metrics.get('C* efficiency (%)', 0),
                        'eta_isp_pct': metrics.get('Isp efficiency (%)', 0),
                        'total_impulse_ns': val_thrust * ((selected_window[1] - selected_window[0]) / 1000.0),
                        'comments': comments,
                        'config_used': selected_config if config else 'None'
                    }

                    try:
                        save_to_campaign(target_campaign, record)
                        st.success(f"Saved to {target_campaign}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error saving: {e}")
            else:
                st.info("No campaigns found. Create one in HF Campaign page.")

        with col_save2:
            st.markdown("**HTML Report**")

            if st.button("Generate Report"):
                # Prepare data for report
                test_data = {
                    'test_id': test_id,
                    'part': part_name,
                    'propellants': f"{config.get('propellants', {}).get('oxidizer', '?')} / {config.get('propellants', {}).get('fuel', '?')}" if config else 'N/A',
                    'avg_pc_bar': val_pc,
                    'avg_thrust_n': val_thrust,
                    'avg_isp_s': val_isp,
                    'avg_c_star_m_s': val_cstar,
                    'avg_of_ratio': val_of,
                    'duration_s': (selected_window[1] - selected_window[0]) / 1000.0,
                    'comments': comments if 'comments' in locals() else ''
                }

                steady_window_info = {
                    'duration_s': (selected_window[1] - selected_window[0]) / 1000.0,
                    'method': detection_method
                }

                figures = [fig_main]
                if sel_temps:
                    figures.append(fig_temp)

                html_report = generate_hot_fire_test_report(
                    test_data, config, figures, steady_window_info
                )

                st.download_button(
                    "Download Report",
                    html_report,
                    f"{test_id}_report.html",
                    "text/html"
                )
                st.success("Report generated")