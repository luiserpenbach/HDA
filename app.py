import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- CUSTOM MODULE IMPORTS ---
from data_lib.config_loader import get_available_configs, load_config, validate_columns
from data_lib.propulsion_physics import (
    calculate_performance,
    calculate_theoretical_profile,
    calculate_uncertainties,
    calculate_derived_metrics
)
from data_lib.sensor_data_tools import (
    resample_data, smooth_signal_savgol, find_steady_window,
    integrate_data, calculate_rise_time, analyze_stability_fft
)
from data_lib.transient_analysis import detect_transient_events
from data_lib.db_manager import save_test_result, get_campaign_history

# --- PAGE SETUP ---
st.set_page_config(page_title="Hopper Test Data Studio", layout="wide", page_icon="🚀")

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* Increased padding so tabs don't hide behind the header */
    .block-container {padding-top: 3rem;} 

    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    h2 {border-bottom: 1px solid #ddd; padding-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: CONFIGURATION
# ==========================================
st.sidebar.title("🚀 Hopper Data Studio")
st.sidebar.header("Test Configuration")

available_configs = get_available_configs()
current_config = None
selected_config_name = st.sidebar.selectbox(
    "Select Config",
    ["Generic / No Config"] + available_configs
)

if selected_config_name and selected_config_name != "Generic / No Config":
    current_config = load_config(selected_config_name)
    st.sidebar.success(f"Active: {selected_config_name}")

    with st.sidebar.expander("⚙️ Override Settings"):
        settings = current_config.get('settings', {})
        freq_ms = st.number_input("Resample (ms)", value=settings.get('resample_freq_ms', 10))
        window_ms = st.number_input("Window (ms)", value=settings.get('steady_window_ms', 500))
        cv_thresh = st.number_input("CV Thresh (%)", value=settings.get('cv_threshold', 1.0))

        st.subheader("Geometry")
        geom = current_config.get('geometry', {})
        throat_area = st.number_input("Throat Area (mm^2)", value=geom.get('throat_area_mm2', 0.0))
        current_config['geometry']['throat_area_mm2'] = throat_area

    if "reference_values" in current_config:
        with st.sidebar.expander("ℹ️ Reference Data"):
            st.table(current_config["reference_values"])

    with st.sidebar.expander("📄 Raw Config JSON"):
        st.json(current_config)
else:
    # Defaults if no config
    freq_ms = st.sidebar.number_input("Resample (ms)", value=10)
    window_ms = st.sidebar.number_input("Window (ms)", value=500)
    cv_thresh = st.sidebar.number_input("CV Thresh (%)", value=1.0)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

@st.cache_data
def load_and_process_data(uploaded_file, config, f_ms):
    """
    Master function to Load, Map, Resample, and enrich Physics.
    """
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    # 1. Apply Channel Mapping
    if config and 'channel_config' in config:
        mapping = config['channel_config']
        df_raw.columns = df_raw.columns.astype(str)  # Ensure string keys
        df_raw.rename(columns=mapping, inplace=True)

    # 2. Identify Time Column
    time_col = 'timestamp'
    if config:
        time_col = config.get('columns', {}).get('timestamp', 'timestamp')

    # Fallback time search
    if time_col not in df_raw.columns:
        candidates = [c for c in df_raw.columns if 'time' in c.lower() or 'ts' in c.lower()]
        if candidates:
            time_col = candidates[0]
        else:
            return None, "Time column not found. Check config mapping."

    # 3. Resample
    df = resample_data(df_raw, time_col=time_col, freq_ms=f_ms)

    # 4. Physics Calculations
    if config:
        df = calculate_performance(df, config)

    return df, None


@st.cache_data
def convert_df_to_csv(df):
    export_df = df.copy()
    if 'timestamp' in export_df.columns:
        export_df.insert(1, 'time_s', export_df['timestamp'] / 1000.0)
    if '__smooth' in export_df.columns:
        export_df.rename(columns={'__smooth': 'signal_smooth'}, inplace=True)
    return export_df.to_csv(index=False).encode('utf-8')


# ==========================================
# TAB 1: DATA VIEWER & ANALYSIS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Analysis Workbench", "📑 Batch Comparison", "📅 Campaign Trends"])

with tab1:
    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"], key="single")

    if uploaded_file:
        df, error = load_and_process_data(uploaded_file, current_config, freq_ms)

        if error:
            st.error(error)
        else:
            # --- TOP CONTROLS ---
            col_sel, col_tog = st.columns([3, 1])
            with col_sel:
                # Intelligent Default Selection
                default_cols = []
                if current_config:
                    cmap = current_config.get('columns', {})
                    # Prioritize key physics columns
                    for k in ['chamber_pressure', 'mass_flow_ox', 'thrust', 'isp']:
                        val = cmap.get(k, k)  # Use mapped name or key
                        if val in df.columns: default_cols.append(val)

                if not default_cols: default_cols = df.columns[1:3].tolist()

                plot_cols = st.multiselect("Channels to Plot", df.columns, default=default_cols)

            with col_tog:
                st.write("")  # Spacer
                enable_analysis = st.toggle("🔎 Enable Analysis", value=True)

            # --- MAIN PLOT (Dual Axis) ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            for col in plot_cols:
                # Heuristic: Pressure on Left (Y1), Flow/Thrust on Right (Y2)
                is_secondary = False
                if any(x in col.lower() for x in ['flow', 'thrust', 'isp', 'fm', 'lc']):
                    is_secondary = True

                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=col),
                    secondary_y=is_secondary
                )

            # Axis Styling
            fig.update_layout(
                title="Test Data Overview", height=500, template="plotly_white",
                xaxis_title="Time (ms)",
                legend=dict(orientation="h", y=1.1)
            )
            fig.update_yaxes(title_text="Pressure / Generic", secondary_y=False)
            fig.update_yaxes(title_text="Flow / Thrust", secondary_y=True)

            # --- ANALYSIS LOGIC ---
            avg_stats = {}  # Initialize empty dicts to prevent scope errors later
            derived_metrics = {}
            bounds = None
            steady_df = pd.DataFrame()

            if enable_analysis:
                st.markdown("---")
                st.header("🔬 Deep Dive Analysis")

                # ==========================================
                # SECTION 1: STEADY STATE ID
                # ==========================================
                st.subheader("1. Steady State Identification")

                # Configuration Controls
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    target_default = plot_cols[0] if plot_cols else df.columns[1]
                    if current_config:
                        target_default = current_config.get('columns', {}).get('chamber_pressure', target_default)
                    target_col = st.selectbox("Signal for Stability", df.columns,
                                              index=list(df.columns).index(
                                                  target_default) if target_default in df.columns else 0)
                with c2:
                    clip_view = st.checkbox("✂️ Auto-Zoom Plot (±5s)", value=True)
                with c3:
                    show_theory = st.checkbox("📐 Theoretical Overlay", value=False)

                # Run Detection
                df['__smooth'] = smooth_signal_savgol(df, target_col)
                bounds, cv_trace = find_steady_window(df, '__smooth', 'timestamp', window_ms, cv_thresh)

                # Manual Override UI
                t_min, t_max = float(df['timestamp'].min()), float(df['timestamp'].max())
                s_def, e_def = bounds if bounds else (t_min, t_max)

                # Display Status
                if bounds:
                    st.success(f"✅ Stable Window Detected: {s_def:.0f} - {e_def:.0f} ms")
                else:
                    st.warning("⚠️ No automatic steady state detected. Please adjust window manually.")

                with st.expander("🛠 Window Fine-Tuning", expanded=(bounds is None)):
                    s_start, s_end = st.slider("Select Window (ms)", t_min, t_max, (float(s_def), float(e_def)), 100.0)

                # Update Main Plot with Window
                fig.add_vrect(x0=s_start, x1=s_end, fillcolor="green", opacity=0.1, annotation_text="Steady",
                              secondary_y=False)
                if clip_view:
                    view_s = max(t_min, s_start - 3000)
                    view_e = min(t_max, s_end + 3000)
                    fig.update_xaxes(range=[view_s, view_e])

                # Theoretical Overlay
                if show_theory and current_config:
                    df_theo = calculate_theoretical_profile(df, current_config)
                    if df_theo is not None and 'thrust_ideal' in df_theo:
                        fig.add_trace(go.Scatter(x=df['timestamp'], y=df_theo['thrust_ideal'],
                                                 mode='lines', name='Ideal Thrust',
                                                 line=dict(dash='dot', color='black')), secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)

                # ==========================================
                # SECTION 2: PERFORMANCE METRICS
                # ==========================================
                mask = (df['timestamp'] >= s_start) & (df['timestamp'] <= s_end)
                steady_df = df[mask]

                if not steady_df.empty:
                    st.markdown("---")
                    st.subheader("2. Performance Report (Steady State)")

                    # Calculate
                    avg_stats = steady_df.mean().to_dict()
                    derived_metrics = calculate_derived_metrics(avg_stats, current_config or {})
                    uncertainties = calculate_uncertainties(avg_stats, current_config or {})


                    # Error Formatting Helper
                    def fmt_err(key, val, unit=""):
                        u = uncertainties.get(key, 0.0)
                        if u == 0: return f"{val:.2f} {unit}"
                        if u < 0.1: return f"{val:.3f} ± {u:.3f} {unit}"
                        if u < 10: return f"{val:.2f} ± {u:.2f} {unit}"
                        return f"{val:.0f} ± {u:.0f} {unit}"


                    # Primary Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    cols_cfg = current_config.get('columns', {}) if current_config else {}

                    with m1:
                        p_col = cols_cfg.get('chamber_pressure')
                        if p_col in avg_stats: st.metric("Avg Pc", fmt_err('chamber_pressure', avg_stats[p_col], "bar"))
                    with m2:
                        t_col = cols_cfg.get('thrust')
                        if t_col in avg_stats: st.metric("Thrust", fmt_err('thrust', avg_stats[t_col], "N"))
                    with m3:
                        tot_flow = avg_stats.get('mass_flow_total', 0)
                        if tot_flow > 0: st.metric("Total Flow", fmt_err('mass_flow_total', tot_flow, "g/s"))
                    with m4:
                        if 'isp' in avg_stats: st.metric("Isp", fmt_err('isp', avg_stats['isp'], "s"))

                    st.divider()

                    # Derived Metrics
                    d_cols = st.columns(5)
                    metrics_list = []
                    if 'c_star' in avg_stats: metrics_list.append(("C*", fmt_err('c_star', avg_stats['c_star'], "m/s")))
                    if 'of_ratio' in avg_stats: metrics_list.append(("O/F", fmt_err('of_ratio', avg_stats['of_ratio'])))
                    for k, v in derived_metrics.items(): metrics_list.append((k, f"{v:.1f}"))

                    for i, (label, val) in enumerate(metrics_list):
                        with d_cols[i % 5]: st.metric(label, val)

                    # ==========================================
                    # SECTION 3: STABILITY (FFT)
                    # ==========================================
                    st.markdown("---")
                    st.subheader("3. Combustion Stability (Frequency Domain)")

                    if len(steady_df) > 50:
                        freqs, psd, peak = analyze_stability_fft(steady_df, target_col)
                        f_fft = go.Figure()
                        f_fft.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='purple')))
                        f_fft.update_layout(
                            title=f"PSD Spectrum: {target_col} (Peak: {peak:.1f} Hz)",
                            xaxis_title="Frequency (Hz)",
                            yaxis_title="Power Spectral Density",
                            yaxis_type="log",
                            height=400
                        )
                        st.plotly_chart(f_fft, use_container_width=True)
                    else:
                        st.info("Not enough steady-state data for FFT analysis.")

                # ==========================================
                # SECTION 4: TRANSIENT ANALYSIS
                # ==========================================
                st.markdown("---")
                st.subheader("4. Transient Analysis (Timing)")

                # Config hack to allow manual column selection
                cols_cfg = current_config.get('columns', {}) if current_config else {}
                cmd_col = cols_cfg.get('fire_command')

                if not cmd_col or cmd_col not in df.columns:
                    cmd_col = st.selectbox("Select Valve/Fire Command Signal", ["None"] + list(df.columns))

                if cmd_col and cmd_col != "None":
                    t_cfg = current_config.copy() if current_config else {'columns': {}}
                    if 'columns' not in t_cfg: t_cfg['columns'] = {}
                    t_cfg['columns']['fire_command'] = cmd_col

                    events = detect_transient_events(df, t_cfg, steady_bounds=(s_start, s_end))

                    if events:
                        c_t1, c_t2, c_t3 = st.columns(3)
                        c_t1.metric("T-0 (Valve Open)", f"{events.get('t_zero', 0):.0f} ms")
                        c_t2.metric("Ignition Delay", f"{events.get('ignition_delay_ms', 0):.0f} ms")
                        c_t3.metric("Shutdown Impulse", f"{events.get('shutdown_impulse_ns', 0):.2f} Ns")

                        # Transient Zoom Plot
                        if events.get('t_zero') and events.get('t_ignition'):
                            fig_trans = go.Figure()
                            # Zoom window
                            z_s = events['t_zero'] - 200
                            z_e = events['t_ignition'] + 500
                            z_df = df[(df['timestamp'] >= z_s) & (df['timestamp'] <= z_e)]

                            # Plot Pressure
                            pc_col = cols_cfg.get('chamber_pressure', target_col)
                            if pc_col in z_df:
                                fig_trans.add_trace(go.Scatter(x=z_df['timestamp'], y=z_df[pc_col], name="Pressure"))
                                # Normalize Command to match scale
                                max_p = z_df[pc_col].max()
                                cmd_norm = z_df[cmd_col] / z_df[cmd_col].max() * max_p
                                fig_trans.add_trace(
                                    go.Scatter(x=z_df['timestamp'], y=cmd_norm, name="Command", line=dict(dash='dot')))

                                # Markers
                                fig_trans.add_vline(x=events['t_zero'], line_color="green", annotation_text="T0")
                                fig_trans.add_vline(x=events['t_ignition'], line_color="red",
                                                    annotation_text="Ignition")
                                fig_trans.update_layout(height=350, title="Start-up Transient Zoom",
                                                        xaxis_title="Time (ms)")
                                st.plotly_chart(fig_trans, use_container_width=True)
                    else:
                        st.info("No transient events detected. Ensure command signal has a clear rising edge.")

            else:
                st.plotly_chart(fig, use_container_width=True)

            # --- ACTION BAR (Save & Export) ---
            st.markdown("### 💾 Actions & Export")
            a1, a2 = st.columns([1, 1])

            with a1:
                comments = st.text_input("Test Notes", placeholder="e.g. Test 001, Cold Flow")
                if st.button("Save to Campaign History", disabled=steady_df.empty):
                    # Re-calc rise time for DB
                    rise_col = current_config.get('columns', {}).get('mass_flow_ox') or plot_cols[0]
                    t10, t90, rise_val = calculate_rise_time(df, rise_col, s_start, avg_stats.get(rise_col, 0))

                    db_stats = {
                        'duration': (s_end - s_start) / 1000.0,
                        'avg_cv': float(cv_trace[mask].mean() if 'cv_trace' in locals() else 0),
                        'rise_time': float(rise_val) if rise_val else 0.0,
                        'avg_pressure': avg_stats.get(current_config.get('columns', {}).get('chamber_pressure')),
                        'avg_thrust': avg_stats.get(current_config.get('columns', {}).get('thrust')),
                        'mass_flow_total': avg_stats.get('mass_flow_total'),
                        'isp': avg_stats.get('isp'),
                        'c_star': avg_stats.get('c_star'),
                    }

                    save_test_result(
                        filename=uploaded_file.name,
                        config_name=selected_config_name,
                        stats=db_stats,
                        derived=derived_metrics,
                        comments=comments
                    )
                    st.success("Saved to Database!")

            with a2:
                st.write("")  # Alignment spacer
                st.write("")
                csv_data = convert_df_to_csv(df)
                st.download_button(
                    label="💾 Download Processed CSV",
                    data=csv_data,
                    file_name=f"Processed_{uploaded_file.name}",
                    mime='text/csv',
                )

# ==========================================
# TAB 2: BATCH COMPARISON
# ==========================================
with tab2:
    st.header("Batch Overlay")
    files = st.file_uploader("Upload Multiple CSVs", type=["csv"], accept_multiple_files=True, key="batch")

    if files:
        first_df, _ = load_and_process_data(files[0], current_config, freq_ms)
        if first_df is not None:
            overlay_col = st.selectbox("Channel to Overlay", first_df.columns, index=1)
            align_mode = st.radio("Alignment", ["Raw Time", "Align to Steady Start"], horizontal=True)

            fig_over = go.Figure()

            for f in files:
                d, _ = load_and_process_data(f, current_config, freq_ms)
                if d is not None and overlay_col in d.columns:
                    x_axis = d['timestamp']
                    if align_mode == "Align to Steady Start":
                        d['__sm'] = smooth_signal_savgol(d, overlay_col)
                        b, _ = find_steady_window(d, '__sm', 'timestamp', window_ms, cv_thresh)
                        if b:
                            x_axis = (d['timestamp'] - b[0]) / 1000.0
                        else:
                            continue
                    else:
                        x_axis = d['timestamp'] / 1000.0

                    fig_over.add_trace(go.Scatter(x=x_axis, y=d[overlay_col], mode='lines', name=f.name))

            fig_over.update_layout(height=600, title=f"Comparison: {overlay_col}", xaxis_title="Time (s)")
            st.plotly_chart(fig_over, use_container_width=True)

# ==========================================
# TAB 3: CAMPAIGN TRENDS
# ==========================================
with tab3:
    st.header("Campaign Trends")
    history_df = get_campaign_history()

    if not history_df.empty:
        with st.expander("Raw Database", expanded=False):
            st.dataframe(history_df)

        c_filt, c_plot = st.columns([1, 3])
        with c_filt:
            configs = history_df['config_name'].unique()
            sel_conf = st.multiselect("Filter Config", configs, default=configs)
            filtered_df = history_df[history_df['config_name'].isin(sel_conf)]

        with c_plot:
            metric_y = st.selectbox("Metric to Trend",
                                    ['isp_s', 'c_star_ms', 'avg_thrust_n', 'eta_c_star_pct', 'rise_time_s'])

        if not filtered_df.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=filtered_df['timestamp'], y=filtered_df[metric_y],
                mode='lines+markers', text=filtered_df['filename'],
                marker=dict(size=10, color='royalblue')
            ))
            fig_trend.update_layout(title=f"Trend: {metric_y}", height=500, template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)

            st.info(f"Mean {metric_y}: {filtered_df[metric_y].mean():.2f} | Std: {filtered_df[metric_y].std():.2f}")
    else:
        st.info("No history found. Save a test result in Tab 1 to start tracking.")