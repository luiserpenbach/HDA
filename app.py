import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
from datetime import datetime

# --- CUSTOM IMPORTS ---
from data_lib.config_loader import get_available_configs, load_config
from data_lib.propulsion_physics import (
    calculate_hot_fire_series, calculate_cold_flow_metrics, calculate_hot_fire_metrics,
    calculate_theoretical_profile
)
from data_lib.sensor_data_tools import (
    resample_data, smooth_signal_savgol, find_steady_windows, find_steady_window,
    calculate_rise_time, analyze_stability_fft, find_steady_windows_ml
)
from data_lib.transient_analysis import detect_transient_events
from data_lib.db_manager import save_test_result, get_campaign_history, save_cold_flow_record, get_cold_flow_history, \
    init_db
from data_lib.reporting import generate_html_report
from data_lib.spc_analysis import (
    calculate_control_limits,
    detect_control_violations,
    plot_spc_chart,
    calculate_process_capability
)

st.set_page_config(page_title="Hopper Data Studio", layout="wide")
init_db()

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h2 {border-bottom: 2px solid #f0f2f6; padding-bottom: 10px; margin-top: 20px;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_all_columns_from_session():
    all_cols = set()

    def extract(f):
        try:
            f.seek(0)
            c = pd.read_csv(f, nrows=0).columns.tolist()
            f.seek(0)
            return c
        except:
            return []

    for key in ['insp_up', 'cf_single', 'cf_batch', 'hf_single', 'hf_ref']:
        if st.session_state.get(key):
            files = st.session_state[key]
            if isinstance(files, list):
                for f in files: all_cols.update(extract(f))
            else:
                all_cols.update(extract(files))
    return sorted(list(all_cols))


def guess_unit(col_name):
    c = col_name.lower()
    if 'press' in c or 'pt' in c: return "Pressure (bar)"
    if 'thrust' in c or 'force' in c or 'lc' in c: return "Thrust (N)"
    if 'flow' in c or 'fm' in c: return "Mass Flow (g/s)"
    if 'temp' in c or 'tc' in c: return "Temperature (C)"
    return "Value"


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Hopper Studio")
st.sidebar.header("Global Settings")

available_configs = get_available_configs()
current_config = None
selected_config_name = st.sidebar.selectbox(
    "Test Configuration",
    ["Generic / No Config"] + available_configs
)

if selected_config_name and selected_config_name != "Generic / No Config":
    current_config = load_config(selected_config_name)
    st.sidebar.success(f"Active: {selected_config_name}")

    with st.sidebar.expander("Processing Settings"):
        settings = current_config.get('settings', {})
        freq_ms = st.number_input("Resample (ms)", value=settings.get('resample_freq_ms', 10))
        window_ms = st.number_input("Steady Window (ms)", value=settings.get('steady_window_ms', 500))
        cv_thresh = st.number_input("CV Thresh (%)", value=settings.get('cv_threshold', 1.0))

        st.caption("Geometry")
        geom = current_config.get('geometry', {})
        throat_area = st.number_input("Throat Area (mm^2)", value=geom.get('throat_area_mm2', 0.0))
        current_config['geometry']['throat_area_mm2'] = throat_area

    with st.sidebar.expander("Channel Map Overrides"):
        st.caption("Re-assign physics roles to sensors.")
        if 'columns' not in current_config: current_config['columns'] = {}

        roles = list(current_config.get('columns', {}).keys())
        raw_cols = get_all_columns_from_session()
        available_cols = raw_cols.copy()
        if 'channel_config' in current_config:
            mapping = current_config['channel_config']
            mapped = [mapping.get(c, c) for c in raw_cols]
            available_cols = sorted(list(set(available_cols + mapped)))

        for role in roles:
            def_val = current_config['columns'].get(role, "")
            if available_cols:
                options = available_cols.copy()
                if def_val and def_val not in options:
                    options.insert(0, def_val)
                elif not def_val:
                    options.insert(0, "")
                idx = options.index(def_val) if def_val in options else 0
                new_val = st.selectbox(role, options, index=idx, key=f"map_{role}")
            else:
                new_val = st.text_input(role, value=def_val, key=f"map_{role}")
            if new_val != def_val:
                current_config['columns'][role] = new_val.strip()

    st.sidebar.markdown("---")
    config_json = json.dumps(current_config, indent=4)
    st.sidebar.download_button("Download Active Config", config_json, "config.json", "application/json")

    with st.sidebar.expander("🔍 Quick Compare"):
        compare_ids = st.multiselect("Select Tests to Compare",
                                     get_cold_flow_history()['test_id'].tolist())
        if compare_ids:
            comparison_df = get_cold_flow_history()
            comparison_df = comparison_df[comparison_df['test_id'].isin(compare_ids)]

            fig = go.Figure()
            for idx, row in comparison_df.iterrows():
                fig.add_trace(go.Bar(name=row['test_id'],
                                     x=['Cd', 'Flow', 'Pressure'],
                                     y=[row['avg_cd_CALC'], row['avg_mf_g_s'], row['avg_p_up_bar']]))
            st.plotly_chart(fig)

else:
    freq_ms = st.sidebar.number_input("Resample (ms)", value=10)
    window_ms = st.sidebar.number_input("Steady Window (ms)", value=500)
    cv_thresh = st.sidebar.number_input("CV Thresh (%)", value=1.0)


@st.cache_data
def load_and_prep_file(uploaded_file, config, f_ms):
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    if config and 'channel_config' in config:
        mapping = config['channel_config']
        df_raw.columns = df_raw.columns.astype(str)
        df_raw.rename(columns=mapping, inplace=True)

    time_col = 'timestamp'
    if config: time_col = config.get('columns', {}).get('timestamp', 'timestamp')
    if time_col not in df_raw.columns:
        candidates = [c for c in df_raw.columns if 'time' in c.lower() or 'ts' in c.lower()]
        if candidates:
            time_col = candidates[0]
        else:
            return None, "Time column not found."

    df = resample_data(df_raw, time_col=time_col, freq_ms=f_ms)

    # NOTE: Physics calculation moved to specific tabs to save performance

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
# MAIN TABS
# ==========================================
tab_config, tab_insp, tab_cfs, tab_cfb, tab_hot, tab_db = st.tabs([
    "Test Config Manager",
    "Inspector",
    "Cold Flow Analysis",
    "Cold Flow Batch",
    "Hot Fire Analysis",
    "Campaign DB"
])

with tab_config:
    st.header("⚙ Configuration Manager")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Available Configs")
        configs = get_available_configs()

        for cfg in configs:
            with st.expander(cfg):
                if st.button("Load", key=f"load_{cfg}"):
                    current_config = load_config(cfg)
                if st.button("Edit", key=f"edit_{cfg}"):
                    st.session_state['editing_config'] = cfg
                if st.button("Delete", key=f"del_{cfg}"):
                    os.remove(f"test_configs/{cfg}.json")
                    st.rerun()

    with col2:
        st.subheader("Config Editor")

        if 'editing_config' in st.session_state:
            cfg = load_config(st.session_state['editing_config'])

            # JSON editor
            edited_json = st.text_area("Configuration JSON",
                                       json.dumps(cfg, indent=2),
                                       height=400)

            if st.button("Save Changes"):
                try:
                    new_cfg = json.loads(edited_json)
                    with open(f"test_configs/{st.session_state['editing_config']}.json", 'w') as f:
                        json.dump(new_cfg, f, indent=2)
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")


# -----------------------------------------------------------------------------
# TAB 2: INSPECTOR
# -----------------------------------------------------------------------------
with tab_insp:
    st.header("General Data Inspector")
    files = st.file_uploader("Upload CSVs", accept_multiple_files=True, key="insp_up")

    if files:
        f_map = {f.name: f for f in files}
        act_name = st.selectbox("Active File", list(f_map.keys()))
        df, err = load_and_prep_file(f_map[act_name], current_config, freq_ms)

        if df is not None:
            t_start = df['timestamp'].min()
            df['time_s'] = (df['timestamp'] - t_start) / 1000.0

            c_set, c_plot = st.columns([1, 4])
            with c_set:
                n_plots = st.number_input("Subplots", 1, 4, 1)
                configs = []
                cols = [c for c in df.columns if c not in ['timestamp', 'time_s']]
                for i in range(n_plots):
                    def_s = [cols[i]] if i < len(cols) else []
                    s = st.multiselect(f"Area {i + 1}", cols, default=def_s, key=f"p{i}")
                    configs.append(s)

            with c_plot:
                if any(configs):
                    fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                    for i, sigs in enumerate(configs):
                        for s in sigs:
                            fig.add_trace(go.Scatter(x=df['time_s'], y=df[s], name=s), row=i + 1, col=1)
                        if sigs: fig.update_yaxes(title_text=guess_unit(sigs[0]), row=i + 1, col=1)
                    fig.update_xaxes(title_text="Time (s)", row=n_plots, col=1)
                    fig.update_layout(height=300 + (200 * (n_plots - 1)), margin=dict(t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df.describe().T, height=300)

# -----------------------------------------------------------------------------
# TAB 2: COLD FLOW ANALYSIS (SINGLE)
# -----------------------------------------------------------------------------
with tab_cfs:
    st.header("Cold Flow Analysis")

    col_load, col_info = st.columns([1, 1])
    with col_load:
        cfs_file = st.file_uploader("Upload Single Test CSV", key="cf_single")

    with col_info:
        if cfs_file:
            st.subheader("Test Metadata")
            default_id = os.path.splitext(cfs_file.name)[0]

            c_meta_1, c_meta_2 = st.columns(2)
            test_id = c_meta_1.text_input("Test ID", value=default_id)
            part_name = c_meta_2.text_input("Part Name", value="Injector-01")
            serial_num = c_meta_1.text_input("Serial Number", value="SN001")

            if current_config:
                fluid = current_config.get('fluid', {})
                geom = current_config.get('geometry', {})
                st.info(
                    f"Fluid: {fluid.get('name', '?')} ({fluid.get('density_kg_m3')} kg/m³) | Orifice: {geom.get('orifice_area_mm2')} mm²")

    if cfs_file:
        df, err = load_and_prep_file(cfs_file, current_config, freq_ms)
        if df is not None:
            df['time_s'] = (df['timestamp'] - df['timestamp'].min()) / 1000.0

            # Auto-Select Channels based on Config
            cols_cfg = current_config.get('columns', {}) if current_config else {}

            def_p = [cols_cfg.get('upstream_pressure', cols_cfg.get('inlet_pressure'))]
            def_p = [c for c in def_p if c and c in df.columns]

            def_f = [cols_cfg.get('mass_flow', cols_cfg.get('mf'))]
            def_f = [c for c in def_f if c and c in df.columns]

            st.subheader("1. Hydraulic Performance")
            c_sel_p, c_sel_f = st.columns(2)
            sel_p = c_sel_p.multiselect("Pressure Channels", df.columns, default=def_p)
            sel_f = c_sel_f.multiselect("Flow Channels", df.columns, default=def_f)

            # --- PLOT SETUP ---
            # Create a placeholder to render the plot LATER, after we add the window
            plot_placeholder = st.empty()

            fig_pf = make_subplots(specs=[[{"secondary_y": True}]])
            for c in sel_p: fig_pf.add_trace(go.Scatter(x=df['time_s'], y=df[c], name=c), secondary_y=False)
            for c in sel_f: fig_pf.add_trace(go.Scatter(x=df['time_s'], y=df[c], name=c, line=dict(dash='dot')),
                                             secondary_y=True)
            fig_pf.update_yaxes(title_text="Pressure (bar)", secondary_y=False)
            fig_pf.update_yaxes(title_text="Mass Flow (g/s)", secondary_y=True)
            fig_pf.update_layout(height=400, margin=dict(t=10, b=10))

            # Temperature Plot (Can remain static as it's separate)
            t_cols = [c for c in df.columns if any(x in c.lower() for x in ['temp', 'tc'])]
            if t_cols:
                fig_t = go.Figure()
                for c in t_cols: fig_t.add_trace(go.Scatter(x=df['time_s'], y=df[c], name=c))
                fig_t.update_layout(title="Temperature", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_t, use_container_width=True)

            # Analysis Controls
            st.subheader("3. Steady State Detection")

            # Add algorithm selector
            ac_algo, ac1, ac2, ac3 = st.columns([1, 2, 1, 1])

            detection_method = ac_algo.selectbox(
                "Method",
                ["CV-based (Classic)", "ML-based (Advanced)"],
                help="CV: Fast, threshold-based. ML: Slower but more robust to noise."
            )

            # A. Stability Sensors
            stab_sigs = ac1.multiselect(
                "Stability Criteria (Sensors must be stable)",
                df.columns,
                default=sel_p,
                help="Select sensors that must ALL be stable simultaneously"
            )

            # B. Threshold
            min_thresh = ac2.number_input(
                "Min Value Threshold",
                value=1.0,
                help="Signal must be above this value to count as a test point."
            )

            # C. Detection Parameters
            windows = []
            predictions = None

            if detection_method == "CV-based (Classic)":
                # Original CV-based detection
                if stab_sigs:
                    windows, max_cv = find_steady_windows(
                        df, stab_sigs, 'timestamp',
                        window_ms, cv_thresh, min_thresh
                    )

                    # Display CV plot in expander
                    with st.expander("📊 Stability Trace (CV Method)"):
                        fig_cv = go.Figure()
                        fig_cv.add_trace(go.Scatter(
                            x=df['time_s'],
                            y=max_cv,
                            name='Max CV %',
                            line=dict(color='orange')
                        ))
                        fig_cv.add_hline(
                            y=cv_thresh,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Threshold: {cv_thresh}%"
                        )
                        fig_cv.update_layout(
                            height=250,
                            yaxis_title="CV %",
                            xaxis_title="Time (s)",
                            margin=dict(t=20, b=20)
                        )
                        st.plotly_chart(fig_cv, use_container_width=True)

            else:  # ML-based
                contamination = ac3.slider(
                    "Contamination",
                    0.05, 0.30, 0.15, 0.05,
                    help="Expected % of TRANSIENT data within the test. Lower = stricter."  # UPDATED
                )

                if stab_sigs:
                    try:
                        windows, predictions = find_steady_windows_ml(
                            df, stab_sigs, 'timestamp',
                            window_ms=window_ms,
                            contamination=contamination,
                            min_duration_ms=window_ms,  # At least one window duration
                            min_val=min_thresh
                        )

                        # Display ML predictions plot
                        with st.expander("🤖 ML Stability Classification"):
                            # Create visual of predictions
                            pred_visual = predictions.map({1: 'Stable', -1: 'Transient'})

                            fig_ml = go.Figure()

                            # Plot first stability signal with color coding
                            if stab_sigs[0] in df.columns:
                                # Stable regions in green
                                stable_mask = predictions == 1
                                stable_data = df[stable_mask]
                                transient_data = df[~stable_mask]

                                fig_ml.add_trace(go.Scatter(
                                    x=transient_data['time_s'],
                                    y=transient_data[stab_sigs[0]],
                                    mode='markers',
                                    name='Transient',
                                    marker=dict(color='lightcoral', size=3, opacity=0.6)
                                ))

                                fig_ml.add_trace(go.Scatter(
                                    x=stable_data['time_s'],
                                    y=stable_data[stab_sigs[0]],
                                    mode='markers',
                                    name='Stable',
                                    marker=dict(color='lightgreen', size=3)
                                ))

                            fig_ml.update_layout(
                                height=250,
                                yaxis_title=f"{stab_sigs[0]}",
                                xaxis_title="Time (s)",
                                margin=dict(t=20, b=20)
                            )
                            st.plotly_chart(fig_ml, use_container_width=True)

                            # Statistics
                            stable_pct = (predictions == 1).sum() / len(predictions) * 100
                            test_pct = ((predictions == 1) | (predictions == -1)).sum() / len(predictions) * 100

                            st.caption(
                                f"✅ Stable: {stable_pct:.1f}% of total | "
                                f"⚠️ Test Region: {test_pct:.1f}% | "
                                f"⬜ Baseline: {100 - test_pct:.1f}%"
                            )

                    except Exception as e:
                        st.error(f"ML Detection Error: {e}")
                        windows = []

            selected_window = None

            if not windows:
                ac3.warning("No steady windows found meeting criteria.")
            else:
                # D. Window Selector
                win_opts = []
                for i, w in enumerate(windows):
                    dur_s = (w[1] - w[0]) / 1000.0
                    start_s = (w[0] - df['timestamp'].min()) / 1000.0
                    win_opts.append(f"Window {i + 1}: {start_s:.1f}s (Dur: {dur_s:.1f}s)")

                sel_idx = ac3.selectbox("Select Window", range(len(windows)), format_func=lambda x: win_opts[x])
                selected_window = windows[sel_idx]

                # Highlight on plot (This updates fig_pf)
                s_s = (selected_window[0] - df['timestamp'].min()) / 1000.0
                e_s = (selected_window[1] - df['timestamp'].min()) / 1000.0
                fig_pf.add_vrect(x0=s_s, x1=e_s, fillcolor="green", opacity=0.2)
                # Visible Vertical Lines
                fig_pf.add_vline(x=s_s, line_width=2, line_dash="dash", line_color="green")
                fig_pf.add_vline(x=e_s, line_width=2, line_dash="dash", line_color="green")

                # Metrics
                st.subheader("3. Results")
                steady_df = df[(df['timestamp'] >= selected_window[0]) & (df['timestamp'] <= selected_window[1])]
                avg = steady_df.mean().to_dict()

                # Use Cold Flow Metrics Function
                metrics = calculate_cold_flow_metrics(avg, current_config or {})

                r1, r2, r3, r4 = st.columns(4)

                val_p = avg.get(sel_p[0], 0) if sel_p else 0
                val_f = avg.get(sel_f[0], 0) if sel_f else 0
                val_t = avg.get(t_cols[0], 0) if t_cols else 0
                val_cd = metrics.get('Cd', 0)

                r1.metric("Avg Pressure", f"{val_p:.2f} bar")
                r2.metric("Avg Flow", f"{val_f:.2f} g/s")
                r3.metric("Avg Temp", f"{val_t:.1f} °C")
                r4.metric("Cd", f"{val_cd:.4f}" if val_cd else "N/A")

                # Save
                comments = st.text_input("Comments", placeholder="e.g. Throttle 50%")
                if st.button("Save to Campaign DB"):
                    if current_config:
                        # Append window suffix if multiple windows exist
                        save_id = test_id
                        if len(windows) > 1:
                            save_id = f"{test_id}_W{sel_idx + 1}"

                        record = {
                            'test_id': save_id,
                            'part': part_name,
                            'serial_num': serial_num,
                            'test_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'fluid': current_config['fluid'].get('name'),
                            'avg_p_up_bar': val_p,
                            'avg_T_up_K': val_t + 273.15,
                            'avg_p_down_bar': 1.013,
                            'avg_mf_g_s': val_f,
                            'orfifice_area_mm2': current_config['geometry'].get('ox_injector_area_mm2'),
                            'avg_rho_CALC': current_config['fluid'].get('ox_density_kg_m3'),
                            'avg_cd_CALC': val_cd,
                            'comments': comments
                        }
                        save_cold_flow_record(record)
                        st.success(f"Saved {save_id} to Database!")
                    else:
                        st.error("Please load a configuration to save results.")

            # --- FINAL PLOT RENDER ---
            # We call this AT THE END so the rectangle (vrect) added above is included.
            plot_placeholder.plotly_chart(fig_pf, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: COLD FLOW BATCH
# -----------------------------------------------------------------------------
with tab_cfb:
    st.header("Cold Flow Batch Processor")

    col_setup, col_run = st.columns([1, 2])
    with col_setup:
        st.info("Batch process multiple files for one part.")
        cf_files = st.file_uploader("Batch CSVs", accept_multiple_files=True, key="cf_batch")

        st.subheader("Batch Metadata")
        b_part = st.text_input("Part Name", value="Injector-01", key="b_part")
        b_serial = st.text_input("Serial Number", value="SN001", key="b_serial")

        # Config Helper
        def_p = "IG-PT-01"
        if current_config:
            def_p = current_config.get('columns', {}).get('chamber_pressure',
                                                          current_config.get('columns', {}).get('inlet_pressure_ox',
                                                                                                ''))

        # Dropdown
        batch_cols = set()
        if cf_files:
            try:
                cf_files[0].seek(0)
                batch_cols = set(pd.read_csv(cf_files[0], nrows=0).columns)
                cf_files[0].seek(0)
            except:
                pass

        cf_options = list(batch_cols)
        if current_config and 'channel_config' in current_config:
            mapping = current_config['channel_config']
            cf_options = sorted(list(set([mapping.get(c, c) for c in batch_cols])))

        if cf_options:
            idx = cf_options.index(def_p) if def_p in cf_options else 0
            target_p = st.selectbox("Steady Pressure Signal", cf_options, index=idx)
        else:
            target_p = st.text_input("Steady Pressure Signal", value=def_p)

        do_batch = st.button("Run Batch", disabled=not cf_files)

    with col_run:
        if do_batch and cf_files and current_config:
            res_list = []
            db_records = []

            bar = st.progress(0)
            for i, f in enumerate(cf_files):
                d, _ = load_and_prep_file(f, current_config, freq_ms)
                if d is not None and target_p in d.columns:
                    # Use multi-window detector with pressure as the criterion
                    # Hardcode thresh for batch to be robust
                    windows, _ = find_steady_windows(d, [target_p], 'timestamp', window_ms, cv_thresh, 1.0)

                    tid = os.path.splitext(f.name)[0]
                    r = {'Test ID': tid, 'Status': 'Skipped'}

                    if windows:
                        # For batch, pick the longest window
                        best_win = max(windows, key=lambda x: x[1] - x[0])

                        steady = d[(d['timestamp'] >= best_win[0]) & (d['timestamp'] <= best_win[1])]
                        avg = steady.mean().to_dict()

                        # Use Cold Flow Physics
                        mets = calculate_cold_flow_metrics(avg, current_config)

                        r.update({'Status': 'OK', 'Pressure': avg.get(target_p, 0)})
                        r.update(mets)  # Adds Cd

                        # Prepare DB Record
                        cols_cfg = current_config.get('columns', {})
                        f_col = cols_cfg.get('mass_flow_ox')
                        val_f = avg.get(f_col, 0) if f_col else 0
                        t_col = next((c for c in d.columns if 'temp' in c.lower()), None)
                        val_t = avg.get(t_col, 0) if t_col else 20.0

                        val_cd = mets.get('Cd', 0)

                        db_rec = {
                            'test_id': tid,
                            'part': b_part,
                            'serial_num': b_serial,
                            'test_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'fluid': current_config['fluid'].get('name'),
                            'avg_p_up_bar': avg.get(target_p, 0),
                            'avg_T_up_K': val_t + 273.15,
                            'avg_p_down_bar': 1.013,
                            'avg_mf_g_s': val_f,
                            'orfifice_area_mm2': current_config['geometry'].get('ox_injector_area_mm2'),
                            'avg_rho_CALC': current_config['fluid'].get('ox_density_kg_m3'),
                            'avg_cd_CALC': val_cd,
                            'comments': 'Batch Processed'
                        }
                        db_records.append(db_rec)

                    res_list.append(r)
                bar.progress((i + 1) / len(cf_files))

            # Show Results
            rdf = pd.DataFrame(res_list)
            st.dataframe(rdf)

            # SAVE ALL BUTTON
            if db_records:
                if st.button(f"Save {len(db_records)} Records to DB"):
                    for rec in db_records:
                        save_cold_flow_record(rec)
                    st.success("All records saved to database!")

                    # Plot
                    cd_c = [c for c in rdf.columns if 'Cd' in c]
                    if cd_c:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=rdf['Pressure'], y=rdf[cd_c[0]], mode='markers', text=rdf['Test ID'],
                                                 marker=dict(size=10)))
                        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: HOT FIRE ANALYSIS
# -----------------------------------------------------------------------------
with tab_hot:
    st.header("Hot Fire Analysis")

    c_main, c_ref = st.columns([1, 1])
    with c_main:
        hf_file = st.file_uploader("Primary Test CSV", key="hf_single")
    with c_ref:
        ref_file = st.file_uploader("Reference / Golden Run CSV (Optional)", key="hf_ref")

    if hf_file:
        df, err = load_and_prep_file(hf_file, current_config, freq_ms)
        df_ref, _ = load_and_prep_file(ref_file, current_config, freq_ms) if ref_file else (None, None)

        if err:
            st.error(err)
        else:
            # Apply Hot Fire Physics Here
            if current_config:
                df = calculate_hot_fire_series(df, current_config)

            cols_cfg = current_config.get('columns', {}) if current_config else {}

            # Auto-detect steady
            def_t = df.columns[1]
            if current_config: def_t = cols_cfg.get('chamber_pressure', def_t)
            steady_target_col = def_t if def_t in df.columns else df.columns[1]

            # Use legacy detection for hot fire
            df['__sm'] = smooth_signal_savgol(df, steady_target_col)
            bounds_ms, cv_tr = find_steady_window(df, '__sm', 'timestamp', window_ms, cv_thresh)

            # Auto-detect T0
            t0_ms = None
            cmd = cols_cfg.get('fire_command')
            if cmd and cmd in df:
                t_cfg = current_config.copy() if current_config else {'columns': {}}
                t_cfg['columns']['fire_command'] = cmd
                ev = detect_transient_events(df, t_cfg, bounds_ms)
                t0_ms = ev.get('t_zero')

            # Controls
            c_sel, c_align, c_opt = st.columns([2, 1, 1])
            with c_sel:
                def_c = []
                if current_config:
                    for k in ['chamber_pressure', 'thrust', 'mass_flow_ox']:
                        if cols_cfg.get(k) in df.columns: def_c.append(cols_cfg[k])
                p_cols = st.multiselect("Channels", df.columns, default=def_c)

            with c_align:
                align_opts = ["File Start"]
                if t0_ms: align_opts.append("Valve Open (T-0)")
                if bounds_ms: align_opts.append("Steady Start")
                align_mode = st.selectbox("Align Time (0s)", align_opts)

                t_offset = df['timestamp'].min()
                if align_mode == "Valve Open (T-0)" and t0_ms:
                    t_offset = t0_ms
                elif align_mode == "Steady Start" and bounds_ms:
                    t_offset = bounds_ms[0]

            with c_opt:
                do_ana = st.toggle("Analysis Mode", value=True)

            # Shift
            df['time_s'] = (df['timestamp'] - t_offset) / 1000.0
            if df_ref is not None: df_ref['time_s'] = (df_ref['timestamp'] - df_ref['timestamp'].min()) / 1000.0

            # Plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for c in p_cols:
                sec = any(x in c.lower() for x in ['flow', 'thrust', 'isp'])
                fig.add_trace(go.Scatter(x=df['time_s'], y=df[c], name=c), secondary_y=sec)

            if df_ref is not None:
                for c in p_cols:
                    if c in df_ref.columns:
                        sec = any(x in c.lower() for x in ['flow', 'thrust', 'isp'])
                        fig.add_trace(go.Scatter(x=df_ref['time_s'], y=df_ref[c], name=f"REF: {c}",
                                                 line=dict(dash='dot', width=1), opacity=0.6), secondary_y=sec)

            fig.update_layout(height=500, title=f"Test: {hf_file.name}", xaxis_title="Time (s)", margin=dict(t=30, b=0))

            if do_ana:
                # Logic
                t_col = st.selectbox("Steady Signal", df.columns, index=list(df.columns).index(
                    steady_target_col) if steady_target_col in df.columns else 0)

                t_min_s = df['time_s'].min()
                t_max_s = df['time_s'].max()
                sd_s, ed_s = t_min_s, t_max_s
                if bounds_ms:
                    sd_s = (bounds_ms[0] - t_offset) / 1000.0
                    ed_s = (bounds_ms[1] - t_offset) / 1000.0

                with st.expander("Window Adjust", expanded=(bounds_ms is None)):
                    c1, c2 = st.columns(2)
                    s_start_s = c1.number_input("Start (s)", value=float(sd_s), step=0.1)
                    s_end_s = c2.number_input("End (s)", value=float(ed_s), step=0.1)

                s_start_ms = s_start_s * 1000.0 + t_offset
                s_end_ms = s_end_s * 1000.0 + t_offset

                fig.add_vrect(x0=s_start_s, x1=s_end_s, fillcolor="green", opacity=0.1, secondary_y=False)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                mask = (df['timestamp'] >= s_start_ms) & (df['timestamp'] <= s_end_ms)
                steady = df[mask]

                if not steady.empty:
                    avg = steady.mean().to_dict()

                    # Hot Fire Specific Metrics
                    der = calculate_hot_fire_metrics(avg, current_config or {})
                    # Also include standard derived ones
                    der.update(calculate_derived_metrics(avg, current_config or {}))

                    st.subheader("Results")
                    m1, m2, m3, m4 = st.columns(4)
                    if cols_cfg.get('chamber_pressure') in avg: m1.metric("Pc",
                                                                          f"{avg[cols_cfg['chamber_pressure']]:.2f}")
                    if cols_cfg.get('thrust') in avg: m2.metric("Thrust", f"{avg[cols_cfg['thrust']]:.2f}")
                    if 'isp' in avg: m3.metric("Isp", f"{avg['isp']:.1f}")

                    # Save
                    if st.button("Save Hot Fire to DB"):
                        # Re-use existing generic save for HF
                        stats = {
                            'duration': (s_end_ms - s_start_ms) / 1000.0,
                            'avg_cv': float(cv_tr[mask].mean()),
                            'avg_pressure': avg.get(cols_cfg.get('chamber_pressure')),
                            'avg_thrust': avg.get(cols_cfg.get('thrust')),
                            'mass_flow_total': avg.get('mass_flow_total'),
                            'isp': avg.get('isp'),
                            'c_star': avg.get('c_star')
                        }
                        save_test_result(hf_file.name, selected_config_name, stats, der, "Hot Fire Analysis")
                        st.toast("Saved!")

            else:
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: CAMPAIGN DB WITH SPC
# -----------------------------------------------------------------------------
with tab_db:
    st.header("Campaign History & SPC Analysis")

    type_view = st.radio(
        "View Database",
        ["Cold Flow Campaign", "Hot Fire Results"],
        horizontal=True
    )

    if type_view == "Cold Flow Campaign":
        cf_df = get_cold_flow_history()

        if not cf_df.empty:
            # Ensure timestamps are datetime
            cf_df['test_timestamp'] = pd.to_datetime(cf_df['test_timestamp'])
            cf_df = cf_df.sort_values('test_timestamp')

            # --- DATA TABLE ---
            with st.expander("📊 Raw Data Table", expanded=False):
                st.dataframe(cf_df, use_container_width=True)

            # --- SPC CONFIGURATION ---
            st.subheader("📈 Statistical Process Control")

            col_config1, col_config2, col_config3 = st.columns(3)

            with col_config1:
                # Metric selection
                available_metrics = [
                    'avg_cd_CALC',
                    'avg_mf_g_s',
                    'avg_p_up_bar',
                    'avg_T_up_K'
                ]
                metric = st.selectbox(
                    "Metric to Monitor",
                    available_metrics,
                    format_func=lambda x: {
                        'avg_cd_CALC': 'Discharge Coefficient (Cd)',
                        'avg_mf_g_s': 'Mass Flow (g/s)',
                        'avg_p_up_bar': 'Pressure (bar)',
                        'avg_T_up_K': 'Temperature (K)'
                    }.get(x, x)
                )

            with col_config2:
                # Part filter
                all_parts = cf_df['part'].unique().tolist()
                part_filter = st.multiselect(
                    "Filter by Part",
                    all_parts,
                    default=all_parts
                )

            with col_config3:
                # SPC method
                spc_method = st.selectbox(
                    "Control Limit Method",
                    ['3sigma', 'individuals'],
                    format_func=lambda x: {
                        '3sigma': '3-Sigma (Standard)',
                        'individuals': 'X-mR (Robust)'
                    }.get(x, x)
                )

            # Filter data
            df_plot = cf_df[cf_df['part'].isin(part_filter)].copy()

            if len(df_plot) < 2:
                st.warning("Need at least 2 data points for SPC analysis.")
            else:
                # --- SPECIFICATION LIMITS ---
                with st.expander("⚙️ Specification Limits (Optional)", expanded=False):
                    st.caption("Define customer/design requirements (USL/LSL)")

                    col_spec1, col_spec2, col_spec3 = st.columns(3)

                    enable_specs = col_spec1.checkbox("Enable Spec Limits")
                    spec_limits = None

                    if enable_specs:
                        usl = col_spec2.number_input(
                            "Upper Spec Limit (USL)",
                            value=float(df_plot[metric].max() * 1.1),
                            format="%.4f"
                        )
                        lsl = col_spec3.number_input(
                            "Lower Spec Limit (LSL)",
                            value=float(df_plot[metric].min() * 0.9),
                            format="%.4f"
                        )
                        spec_limits = {'usl': usl, 'lsl': lsl}

                # --- CALCULATE SPC ---
                from data_lib.spc_analysis import (
                    calculate_control_limits,
                    detect_control_violations,
                    plot_spc_chart,
                    calculate_process_capability
                )

                limits = calculate_control_limits(
                    df_plot[metric].values,
                    method=spc_method
                )

                violations = detect_control_violations(
                    df_plot[metric].values,
                    limits,
                    timestamps=df_plot['test_timestamp']
                )

                # --- SPC CHART ---
                fig_spc = plot_spc_chart(
                    data=df_plot[metric],
                    timestamps=df_plot['test_timestamp'],
                    metric_name=metric,
                    limits=limits,
                    spec_limits=spec_limits,
                    violations=violations,
                    highlight_recent=5  # Highlight last 5 tests
                )

                st.plotly_chart(fig_spc, use_container_width=True)

                # --- METRICS SUMMARY ---
                st.subheader("Process Metrics")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                col_m1.metric(
                    "Process Mean",
                    f"{limits['mean']:.4f}",
                    delta=None
                )

                col_m2.metric(
                    "Std Deviation",
                    f"{limits['std']:.4f}",
                    delta=None
                )

                col_m3.metric(
                    "Sample Size",
                    f"{limits['n']}",
                    delta=None
                )

                # Violations count
                n_violations = len(violations) if not violations.empty else 0
                col_m4.metric(
                    "Control Violations",
                    f"{n_violations}",
                    delta=None,
                    delta_color="inverse"
                )

                # --- VIOLATIONS TABLE ---
                if not violations.empty:
                    st.subheader("⚠️ Control Chart Violations")


                    # Color code by severity
                    def highlight_severity(row):
                        colors = {
                            'critical': 'background-color: #ffcccc',
                            'warning': 'background-color: #fff4cc',
                            'info': 'background-color: #cce5ff'
                        }
                        return [colors.get(row['severity'], '')] * len(row)


                    styled_violations = violations.style.apply(
                        highlight_severity,
                        axis=1
                    )

                    st.dataframe(
                        styled_violations,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Export violations
                    csv_violations = violations.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Violations Report",
                        csv_violations,
                        f"spc_violations_{metric}.csv",
                        "text/csv"
                    )

                # --- PROCESS CAPABILITY ---
                if spec_limits:
                    st.subheader("Process Capability Analysis")

                    capability = calculate_process_capability(
                        df_plot[metric],
                        spec_limits
                    )

                    col_cap1, col_cap2, col_cap3 = st.columns(3)

                    if 'Cp' in capability:
                        col_cap1.metric(
                            "Cp (Potential)",
                            f"{capability['Cp']:.2f}",
                            help="Process capability if perfectly centered"
                        )

                    if 'Cpk' in capability:
                        cpk_value = capability['Cpk']
                        col_cap2.metric(
                            "Cpk (Actual)",
                            f"{cpk_value:.2f}",
                            delta="Good" if cpk_value >= 1.33 else "Poor",
                            delta_color="normal" if cpk_value >= 1.33 else "inverse",
                            help="Actual capability accounting for process centering"
                        )

                    if 'interpretation' in capability:
                        col_cap3.info(f"**Assessment:** {capability['interpretation']}")

                # --- TREND ANALYSIS ---
                with st.expander("📉 Trend Analysis", expanded=False):
                    # Rolling statistics
                    window_size = st.slider(
                        "Rolling Window Size",
                        min_value=3,
                        max_value=min(20, len(df_plot)),
                        value=5
                    )

                    df_plot['rolling_mean'] = df_plot[metric].rolling(
                        window=window_size,
                        center=True
                    ).mean()

                    df_plot['rolling_std'] = df_plot[metric].rolling(
                        window=window_size,
                        center=True
                    ).std()

                    fig_trend = go.Figure()

                    # Raw data
                    fig_trend.add_trace(go.Scatter(
                        x=df_plot['test_timestamp'],
                        y=df_plot[metric],
                        mode='markers',
                        name='Data',
                        marker=dict(size=6, color='lightblue')
                    ))

                    # Rolling mean
                    fig_trend.add_trace(go.Scatter(
                        x=df_plot['test_timestamp'],
                        y=df_plot['rolling_mean'],
                        mode='lines',
                        name=f'Rolling Mean ({window_size})',
                        line=dict(color='red', width=2)
                    ))

                    fig_trend.update_layout(
                        title=f"Trend: {metric}",
                        xaxis_title="Date",
                        yaxis_title=metric,
                        height=350
                    )

                    st.plotly_chart(fig_trend, use_container_width=True)

        else:
            st.info("No Cold Flow records found. Run some tests and save to database!")

    else:  # Hot Fire Results
        h = get_campaign_history()
        if not h.empty:
            st.dataframe(h, use_container_width=True)

            # TODO: Add SPC for hot fire metrics (Isp, C*, etc.)
            st.info("SPC for Hot Fire metrics coming soon!")
        else:
            st.info("No Hot Fire records found.")