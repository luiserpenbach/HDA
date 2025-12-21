import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- CUSTOM MODULE IMPORTS ---
from data_lib.config_loader import get_available_configs, load_config
from data_lib.propulsion_physics import (
    calculate_performance,
    calculate_theoretical_profile,
    calculate_uncertainties,
    calculate_derived_metrics
)
from data_lib.sensor_data_tools import (
    resample_data, smooth_signal_savgol, find_steady_window,
    calculate_rise_time, analyze_stability_fft
)
from data_lib.reporting import generate_html_report
from data_lib.transient_analysis import detect_transient_events
from data_lib.db_manager import save_test_result, get_campaign_history

# --- PAGE SETUP ---
st.set_page_config(page_title="Hopper Data Studio", layout="wide", page_icon="🚀")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h2 {border-bottom: 2px solid #f0f2f6; padding-bottom: 10px; margin-top: 20px;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: GLOBAL CONFIGURATION
# ==========================================
st.sidebar.title("🚀 Hopper Studio")
st.sidebar.header("Global Settings")

# Config Loader
available_configs = get_available_configs()
current_config = None
selected_config_name = st.sidebar.selectbox(
    "Test Configuration",
    ["Generic / No Config"] + available_configs
)

if selected_config_name and selected_config_name != "Generic / No Config":
    current_config = load_config(selected_config_name)
    st.sidebar.success(f"Active: {selected_config_name}")

    # Global Overrides (Apply to all tabs)
    with st.sidebar.expander("⚙️ Processing Settings"):
        settings = current_config.get('settings', {})
        freq_ms = st.number_input("Resample (ms)", value=settings.get('resample_freq_ms', 10))
        window_ms = st.number_input("Steady Window (ms)", value=settings.get('steady_window_ms', 500))
        cv_thresh = st.number_input("CV Thresh (%)", value=settings.get('cv_threshold', 1.0))

        st.caption("Geometry")
        geom = current_config.get('geometry', {})
        throat_area = st.number_input("Throat Area (mm^2)", value=geom.get('throat_area_mm2', 0.0))
        current_config['geometry']['throat_area_mm2'] = throat_area

    with st.sidebar.expander("📄 View Raw Config"):
        st.json(current_config)
else:
    # Defaults
    freq_ms = st.sidebar.number_input("Resample (ms)", value=10)
    window_ms = st.sidebar.number_input("Steady Window (ms)", value=500)
    cv_thresh = st.sidebar.number_input("CV Thresh (%)", value=1.0)


# ==========================================
# HELPER: UNIFIED DATA LOADER
# ==========================================
@st.cache_data
def load_and_prep_file(uploaded_file, config, f_ms):
    """
    Standardizes loading, mapping, and resampling for all tabs.
    """
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    # 1. Apply Channel Mapping
    if config and 'channel_config' in config:
        mapping = config['channel_config']
        df_raw.columns = df_raw.columns.astype(str)
        df_raw.rename(columns=mapping, inplace=True)

    # 2. Find Time Column
    time_col = 'timestamp'
    if config:
        time_col = config.get('columns', {}).get('timestamp', 'timestamp')

    if time_col not in df_raw.columns:
        # Fuzzy Search
        candidates = [c for c in df_raw.columns if 'time' in c.lower() or 'ts' in c.lower()]
        if candidates:
            time_col = candidates[0]
        else:
            return None, "Time column not found. Check config mapping."

    # 3. Resample
    df = resample_data(df_raw, time_col=time_col, freq_ms=f_ms)

    # 4. Basic Physics (Standard Calculations)
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
# MAIN TABS (THE PLUG-IN ARCHITECTURE)
# ==========================================
tab_insp, tab_cold, tab_hot, tab_db = st.tabs([
    "General Inspector",
    "Cold Flow / Component",
    "Hot Fire Analysis",
    "Campaign Record"
])

# -----------------------------------------------------------------------------
# TAB 1: GENERAL INSPECTOR (Multi-Plot)
# -----------------------------------------------------------------------------
with tab_insp:
    st.header("General Data Inspector")
    files = st.file_uploader("Upload CSVs to Inspect", accept_multiple_files=True, key="insp_up")

    if files:
        # File Selector
        file_names = [f.name for f in files]
        active_name = st.selectbox("Select File", file_names)
        active_file = next(f for f in files if f.name == active_name)

        df, err = load_and_prep_file(active_file, current_config, freq_ms)

        if df is not None:
            col_ctrl, col_plot = st.columns([1, 4])

            with col_ctrl:
                st.subheader("Display Settings")
                num_plots = st.number_input("Number of Subplots", 1, 4, 1)

                plot_configs = []
                # Intelligent Defaults
                all_cols = [c for c in df.columns if c != 'timestamp']

                for i in range(num_plots):
                    st.caption(f"**Plot Area {i + 1}**")
                    # Try to pre-select something different for each plot
                    def_sel = []
                    if i < len(all_cols):
                        def_sel = [all_cols[i]]

                    sels = st.multiselect(f"Signals {i + 1}", all_cols, default=def_sel, key=f"p_sel_{i}")
                    plot_configs.append(sels)

            with col_plot:
                if any(plot_configs):
                    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05)

                    for i, signals in enumerate(plot_configs):
                        for sig in signals:
                            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[sig], name=sig), row=i + 1, col=1)

                    fig.update_layout(height=300 + (200 * (num_plots - 1)), margin=dict(t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select signals to plot.")

            # --- ANALYSIS: CORRELATION & STATS ---
            col_corr, col_stats = st.columns([1, 1])

            with col_corr:
                st.subheader("Correlation Matrix")
                # Drop time columns for clean sensor correlation
                numerics = df.select_dtypes(include=[np.number])
                drop_cols = [c for c in numerics.columns if 'time' in c.lower()]
                corr_df = numerics.drop(columns=drop_cols)

                if not corr_df.empty:
                    corr = corr_df.corr()
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale='RdBu',
                        zmin=-1, zmax=1,
                        colorbar=dict(title="Corr")
                    ))
                    fig_corr.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("No numeric columns available for correlation.")

            with col_stats:
                st.subheader("Descriptive Statistics")
                st.dataframe(df.describe().T, height=400)

# -----------------------------------------------------------------------------
# TAB 2: COLD FLOW / COMPONENT (Batch Processor)
# -----------------------------------------------------------------------------
with tab_cold:
    st.header("Cold Flow & Component Characterization")

    col_cf_1, col_cf_2 = st.columns([1, 2])

    with col_cf_1:
        st.info("Upload multiple test files to calculate Cd vs Pressure Drop.")
        cf_files = st.file_uploader("Upload Batch CSVs", accept_multiple_files=True, key="cf_batch")

        # Settings for Batch Logic
        st.subheader("Analysis Logic")
        target_col_guess = "IG-PT-01"
        if current_config:
            target_col_guess = current_config.get('columns', {}).get('chamber_pressure',
                                                                     current_config.get('columns', {}).get(
                                                                         'inlet_pressure_ox', ''))

        target_col_cf = st.text_input("Pressure Signal for Steady State", value=target_col_guess,
                                      help="Column name to detect window on.")

        run_batch = st.button("Run Batch Analysis", disabled=not cf_files)

    with col_cf_2:
        if run_batch and cf_files and current_config:
            results = []
            progress_bar = st.progress(0)

            for i, f in enumerate(cf_files):
                # 1. Load
                df_cf, _ = load_and_prep_file(f, current_config, freq_ms)

                if df_cf is not None and target_col_cf in df_cf.columns:
                    # 2. Find Window
                    df_cf['__sm'] = smooth_signal_savgol(df_cf, target_col_cf)
                    bounds, _ = find_steady_window(df_cf, '__sm', 'timestamp', window_ms, cv_thresh)

                    row = {'Filename': f.name, 'Status': 'Skipped'}

                    if bounds:
                        s, e = bounds
                        steady = df_cf[(df_cf['timestamp'] >= s) & (df_cf['timestamp'] <= e)]

                        # 3. Calculate Averages
                        avg = steady.mean().to_dict()

                        # 4. Calculate Cd (using physics lib)
                        metrics = calculate_derived_metrics(avg, current_config)

                        # Pack Result
                        row['Status'] = 'OK'
                        row['Avg Pressure'] = avg.get(target_col_cf, 0)

                        # Try to find flow for the table
                        f_col = current_config.get('columns', {}).get('mass_flow_ox')
                        if f_col: row['Avg Flow'] = avg.get(f_col, 0)

                        # Add Calculated Metrics (Cd, etc.)
                        row.update(metrics)

                    results.append(row)

                progress_bar.progress((i + 1) / len(cf_files))

            # Display Results
            res_df = pd.DataFrame(results)

            # Filter to successful
            valid_res = res_df[res_df['Status'] == 'OK']

            if not valid_res.empty:
                st.success(f"Processed {len(valid_res)}/{len(cf_files)} files successfully.")

                # Plotting Cd
                cd_cols = [c for c in valid_res.columns if 'Cd' in c]
                if cd_cols:
                    cd_target = cd_cols[0]  # Pick first Cd found (Ox or Fuel)

                    fig_cd = go.Figure()
                    fig_cd.add_trace(go.Scatter(
                        x=valid_res['Avg Upstream Pressure'],
                        y=valid_res[cd_target],
                        mode='markers+text',
                        text=valid_res['Filename'],
                        marker=dict(size=12, color='blue'),
                        name=cd_target
                    ))

                    # Add Trendline (Average Cd)
                    avg_cd_val = valid_res[cd_target].mean()
                    fig_cd.add_hline(y=avg_cd_val, line_dash="dash", annotation_text=f"Avg: {avg_cd_val:.4f}")

                    fig_cd.update_layout(
                        title=f"{cd_target} vs Pressure",
                        xaxis_title="Pressure (bar)",
                        yaxis_title="Discharge Coefficient (-)",
                        height=500
                    )
                    st.plotly_chart(fig_cd, use_container_width=True)

                    st.subheader("Results Table")
                    st.dataframe(valid_res)

                    # Export
                    csv = valid_res.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results CSV", csv, "batch_results.csv", "text/csv")

                else:
                    st.warning(
                        "Steady states found, but Cd could not be calculated. Check Config mapping for Flow and Inlet Pressure.")
            else:
                st.error("No valid steady states found in batch.")
                st.dataframe(res_df)

# -----------------------------------------------------------------------------
# TAB 3: HOT FIRE ANALYSIS (Deep Dive)
# -----------------------------------------------------------------------------
with tab_hot:
    st.header("Hot Fire Engineering Analysis")

    hf_file = st.file_uploader("Upload Hot Fire CSV", key="hf_single")

    if hf_file:
        df, error = load_and_prep_file(hf_file, current_config, freq_ms)

        if error:
            st.error(error)
        else:
            # --- 1. PLOTTING ---
            c_sel, c_view = st.columns([3, 1])
            with c_sel:
                # Defaults based on config
                def_cols = []
                if current_config:
                    cmap = current_config.get('columns', {})
                    for k in ['chamber_pressure', 'mass_flow_ox', 'thrust']:
                        if cmap.get(k) in df.columns: def_cols.append(cmap[k])
                if not def_cols: def_cols = df.columns[1:3].tolist()

                plot_cols = st.multiselect("Channels", df.columns, default=def_cols, key="hf_cols")

            with c_view:
                st.write("")
                enable_ana = st.toggle("Enable Analysis", value=True, key="hf_tog")

            # Main Plot (Dual Axis)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for col in plot_cols:
                is_sec = any(x in col.lower() for x in ['flow', 'thrust', 'isp'])
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col), secondary_y=is_sec)

            fig.update_layout(height=450, title="Test Trace", margin=dict(t=30, b=0))

            # --- 2. ANALYSIS LOGIC ---
            if enable_ana:
                # Controls
                c1, c2, c3 = st.columns(3)
                with c1:
                    # Target Col
                    def_t = plot_cols[0] if plot_cols else df.columns[1]
                    if current_config:
                        def_t = current_config.get('columns', {}).get('chamber_pressure', def_t)
                    target_col = st.selectbox("Steady Signal", df.columns,
                                              index=list(df.columns).index(def_t) if def_t in df.columns else 0)
                with c2:
                    show_theo = st.checkbox("Show Theoretical", value=False)
                with c3:
                    auto_zoom = st.checkbox("Auto-Zoom Window", value=True)

                # Steady State
                df['__sm'] = smooth_signal_savgol(df, target_col)
                bounds, cv_trace = find_steady_window(df, '__sm', 'timestamp', window_ms, cv_thresh)

                t_min, t_max = float(df['timestamp'].min()), float(df['timestamp'].max())
                s_def, e_def = bounds if bounds else (t_min, t_max)

                if bounds:
                    st.success(f"Steady: {bounds[0]:.0f} - {bounds[1]:.0f} ms")
                    fig.add_vrect(x0=bounds[0], x1=bounds[1], fillcolor="green", opacity=0.1, secondary_y=False)

                with st.expander("Adjust Window", expanded=(bounds is None)):
                    s_start, s_end = st.slider("Window", t_min, t_max, (float(s_def), float(e_def)))

                # Theory Overlay
                if show_theo and current_config:
                    df_theo = calculate_theoretical_profile(df, current_config)
                    if df_theo is not None and 'thrust_ideal' in df_theo:
                        fig.add_trace(go.Scatter(x=df['timestamp'], y=df_theo['thrust_ideal'], name="Ideal Thrust",
                                                 line=dict(dash='dot', color='black')), secondary_y=True)

                if auto_zoom:
                    fig.update_xaxes(range=[max(t_min, s_start - 2000), min(t_max, s_end + 2000)])

                st.plotly_chart(fig, use_container_width=True)

                # --- 3. METRICS ---
                mask = (df['timestamp'] >= s_start) & (df['timestamp'] <= s_end)
                steady = df[mask]

                if not steady.empty:
                    st.subheader("Performance Report")
                    avg = steady.mean().to_dict()
                    derived = calculate_derived_metrics(avg, current_config or {})
                    uncert = calculate_uncertainties(avg, current_config or {})


                    # Formatter
                    def fmt(k, v, u=""):
                        err = uncert.get(k, 0)
                        if err == 0: return f"{v:.2f} {u}"
                        return f"{v:.1f} ± {err:.1f} {u}"


                    # Grid
                    m1, m2, m3, m4 = st.columns(4)
                    cols_cfg = current_config.get('columns', {}) if current_config else {}

                    with m1:
                        k = cols_cfg.get('chamber_pressure')
                        if k in avg: st.metric("Avg Pc", fmt('chamber_pressure', avg[k], "bar"))
                    with m2:
                        k = cols_cfg.get('thrust')
                        if k in avg: st.metric("Thrust", fmt('thrust', avg[k], "N"))
                    with m3:
                        if 'mass_flow_total' in avg: st.metric("Flow",
                                                               fmt('mass_flow_total', avg['mass_flow_total'], "g/s"))
                    with m4:
                        if 'isp' in avg: st.metric("Isp", fmt('isp', avg['isp'], "s"))

                    st.divider()
                    d_cols = st.columns(5)
                    met_list = []
                    if 'c_star' in avg: met_list.append(("C*", fmt('c_star', avg['c_star'], "m/s")))
                    if 'of_ratio' in avg: met_list.append(("O/F", fmt('of_ratio', avg['of_ratio'])))
                    for k, v in derived.items(): met_list.append((k, f"{v:.2f}"))

                    for i, (l, v) in enumerate(met_list):
                        with d_cols[i % 5]: st.metric(l, v)

                    # --- 4. TRANSIENT & FFT ---
                    c_fft, c_trans = st.columns(2)
                    with c_fft:
                        st.markdown("**Stability (FFT)**")
                        freqs, psd, peak = analyze_stability_fft(steady, target_col)
                        f_fft = go.Figure(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='purple')))
                        f_fft.update_layout(title=f"Peak: {peak:.1f} Hz", yaxis_type="log", height=300,
                                            margin=dict(t=30, b=0))
                        st.plotly_chart(f_fft, use_container_width=True)

                    with c_trans:
                        st.markdown("**Transients**")
                        # Transient Logic
                        cmd = cols_cfg.get('fire_command')
                        if not cmd or cmd not in df:
                            cmd = st.selectbox("Fire Command", ["None"] + list(df.columns))

                        if cmd and cmd != "None":
                            # Temp config for transient func
                            t_cfg = current_config.copy() if current_config else {'columns': {}}
                            if 'columns' not in t_cfg: t_cfg['columns'] = {}
                            t_cfg['columns']['fire_command'] = cmd

                            ev = detect_transient_events(df, t_cfg, (s_start, s_end))
                            if ev:
                                c1, c2 = st.columns(2)
                                c1.metric("Ign. Delay", f"{ev.get('ignition_delay_ms', 0):.0f} ms")
                                c2.metric("Shut. Impulse", f"{ev.get('shutdown_impulse_ns', 0):.2f} Ns")
                            else:
                                st.caption("No clear edges found.")

                    # --- 5. SAVE ---
                    st.markdown("Actions & Reporting")
                    col_act_1, col_act_2, col_act_3 = st.columns([2, 1, 1])

                    with col_act_1:
                        notes = st.text_input("Test Notes", placeholder="e.g. Nominal run, new injector...")

                        if st.button("Save to Test Point to Campaign DB"):
                            # (Existing DB Save Logic...)
                            rise_k = cols_cfg.get('mass_flow_ox') or target_col
                            _, _, r_time = calculate_rise_time(df, rise_k, s_start, avg.get(rise_k, 0))

                            db_stats = {
                                'duration': (s_end - s_start) / 1000.0,
                                'avg_cv': float(cv_trace[mask].mean()),
                                'rise_time': float(r_time) if r_time else 0.0,
                                'avg_pressure': avg.get(cols_cfg.get('chamber_pressure')),
                                'avg_thrust': avg.get(cols_cfg.get('thrust')),
                                'mass_flow_total': avg.get('mass_flow_total'),
                                'isp': avg.get('isp'),
                                'c_star': avg.get('c_star')
                            }
                            save_test_result(hf_file.name, selected_config_name, db_stats, derived, notes)
                            st.success("Saved!")

                    with col_act_2:
                        st.write("")  # Spacer to align buttons
                        st.write("")
                        # Generate Report Button
                        # We gather the string-formatted metrics for the report
                        report_stats = {k: fmt(k, v) for k, v in avg.items() if k in [
                            cols_cfg.get('chamber_pressure'), cols_cfg.get('thrust'), 'mass_flow_total', 'isp'
                        ]}
                        report_derived = {k: f"{v:.2f}" for k, v in derived.items()}

                        # Gather Figures (Main Plot + FFT)
                        # Note: We use the figure objects 'fig' and 'f_fft' created earlier in the script
                        report_figs = [fig]
                        if 'f_fft' in locals(): report_figs.append(f_fft)
                        if 'fig_trans' in locals(): report_figs.append(fig_trans)

                        html_bytes = generate_html_report(
                            test_metadata={'Filename': hf_file.name, 'Config': selected_config_name},
                            stats=report_stats,
                            derived=report_derived,
                            figures=report_figs,
                            notes=notes
                        )

                        st.download_button(
                            label="📄 Download Report",
                            data=html_bytes,
                            file_name=f"Report_{hf_file.name.replace('.csv', '.html')}",
                            mime='text/html'
                        )

                    with col_act_3:
                        st.write("")  # Spacer
                        st.write("")
                        csv_dl = convert_df_to_csv(df)
                        st.download_button("Download CSV", csv_dl, f"Proc_{hf_file.name}", "text/csv")
            else:
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: CAMPAIGN DB
# -----------------------------------------------------------------------------
with tab_db:
    st.header("Campaign History")
    hist = get_campaign_history()

    if not hist.empty:
        st.dataframe(hist)

        c_filter, c_trend = st.columns([1, 3])
        with c_filter:
            confs = hist['config_name'].unique()
            sel = st.multiselect("Filter Config", confs, default=confs)
            df_filt = hist[hist['config_name'].isin(sel)]

        with c_trend:
            metric = st.selectbox("Trend Metric",
                                  ['isp_s', 'c_star_ms', 'avg_pressure_bar', 'rise_time_s', 'eta_c_star_pct'])
            if not df_filt.empty:
                f_trend = go.Figure(go.Scatter(
                    x=df_filt['timestamp'], y=df_filt[metric],
                    mode='lines+markers', text=df_filt['filename']
                ))
                f_trend.update_layout(height=400, title=f"Trend: {metric}")
                st.plotly_chart(f_trend, use_container_width=True)
    else:
        st.info("No history yet.")