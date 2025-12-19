import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io
from data_lib.config_loader import get_available_configs, load_config, validate_columns
from data_lib.propulsion_physics import calculate_performance

# Import your library
from data_lib.sensor_data_tools import (
    resample_data, smooth_signal_savgol, find_steady_window,
    integrate_data, calculate_rise_time, analyze_stability_fft
)

CONFIG_DIR = 'test_configs'


st.set_page_config(page_title="Hopper Test Data Studio", layout="wide", page_icon="🚀") # CHANGE TO HOPPER ICON

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")

# 1. Load Config List
available_configs = get_available_configs()
if not available_configs:
    st.sidebar.error(f"No configs found in /{CONFIG_DIR}!")
    selected_config_name = None
else:
    selected_config_name = st.sidebar.selectbox("Select Test Config", available_configs)

# 2. Load the Config Object
current_config = {}
if selected_config_name:
    current_config = load_config(selected_config_name)

    # Display details (Read-only verification)
    st.sidebar.info(f"Loaded: {current_config.get('config_name')}")
    st.sidebar.caption(current_config.get('description'))

    # Allow temporary overrides in UI (Optional but useful)
    with st.sidebar.expander("Override Settings"):
        settings_title = st.header("Sampling Settings")
        # We pre-fill these with values from the JSON
        settings = current_config.get('settings', {})
        freq_ms = st.number_input("Resample (ms)", value=settings.get('resample_freq_ms', 10))
        window_ms = st.number_input("Window (ms)", value=settings.get('steady_window_ms', 500))
        cv_thresh = st.number_input("CV Thresh (%)", value=settings.get('cv_threshold', 1.005))

        geom_title = st.header("Geometry Settings")
        # Geometry overrides
        geom = current_config.get('geometry', {})
        throat_area = st.number_input("Throat Area (mm^2)", value=geom.get('throat_area_mm2', 0.0))
        # Update config object with overrides
        current_config['geometry']['throat_area_mm2'] = throat_area

        ref_val_title = st.header("Reference Values")
        ref_val_table = st.table(current_config["reference_values"])


else:
    # Fallback defaults if no config exists yet
    freq_ms, window_ms, cv_thresh = 10, 500, 1.0


# --- HELPER: PROCESS FILE ---
@st.cache_data
def process_file(uploaded_file, config):
    # Load with header=0 to get column names
    df_raw = pd.read_csv(uploaded_file)

    # ---------------------------------------------------------
    # STEP 0: APPLY CHANNEL MAPPING (Renaming)
    # ---------------------------------------------------------
    if config and 'channel_config' in config:
        mapping = config['channel_config']

        # Safety: Ensure DataFrame column names are strings to match JSON keys
        # (Pandas might read "10001" as integer)
        df_raw.columns = df_raw.columns.astype(str)

        # Rename columns: "10001" -> "IG-PT-01"
        df_raw.rename(columns=mapping, inplace=True)

        # Verify: Did we miss any required mappings?
        # (Optional: Print warning for unmapped columns)

    # ---------------------------------------------------------
    # STEP 1: VALIDATION & SETUP
    # ---------------------------------------------------------
    if config:
        # Validate columns (Now checks for Sensor IDs like 'IG-PT-01')
        missing_cols = validate_columns(df_raw, config)
        if missing_cols:
            st.error(f"⚠️ Column Mismatch! Missing: {missing_cols}")
            return None, None, None

        # Get settings
        col_map = config.get('columns', {})
        settings = config.get('settings', {})

        # Get Sensor IDs for key variables
        time_col = col_map.get('timestamp', 'timestamp')

        # Note: col_map.get('chamber_pressure') now returns "IG-PT-01"
        # Since we renamed the DF, df["IG-PT-01"] works instantly!
        target_col = col_map.get('chamber_pressure', col_map.get('mass_flow_ox'))

        # 2. Resample
        f_ms = settings.get('resample_freq_ms', 10)

        # Safety: Ensure time column exists (mapping might have failed if timestamp name changed)
        if time_col not in df_raw.columns:
            # Fallback: Try to find 'timestamp' or 'Time'
            possible_times = [c for c in df_raw.columns if 'time' in c.lower()]
            if possible_times:
                time_col = possible_times[0]

        df = resample_data(df_raw, time_col=time_col, freq_ms=f_ms)

        # 3. Physics Calculations
        df = calculate_performance(df, config)

        # 4. Smoothing & Window
        if target_col and target_col in df:
            df['signal_smooth'] = smooth_signal_savgol(df, target_col)

            bounds, cv_trace = find_steady_window(
                df, 'signal_smooth', 'timestamp',
                settings.get('steady_window_ms', 500),
                settings.get('cv_threshold', 1.0)
            )
            return df, bounds, cv_trace

    return None, None, None


def get_report_html(df, filename, stats, bounds, cv_trace, col_flow, col_press):
    """
    Generates the HTML report string in memory for downloading.
    """
    # Create Summary Text
    summary_html = f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px;">
        <h2 style="color: #333;">Test Report: {filename}</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
            <div><strong>Duration:</strong> {stats['duration']:.2f} s</div>
            <div><strong>Total Mass:</strong> {stats['total_mass']:.2f} g</div>
            <div><strong>Steady Flow:</strong> {stats['avg_flow']:.3f} g/s</div>
            <div><strong>Avg Pressure:</strong> {stats['avg_press']:.2f} bar</div>
            <div><strong>Rise Time:</strong> {stats.get('rise_time', 0):.3f} s</div>
            <div><strong>Stability (CV):</strong> {stats['avg_cv']:.3f} %</div>
        </div>
    </div>
    """

    # --- FIGURE 1: Overview ---
    fig1 = go.Figure()
    # Raw Flow
    fig1.add_trace(go.Scatter(x=df['timestamp'], y=df[col_flow],
                              mode='markers', name='Raw Flow',
                              marker=dict(color='lightgray', size=4)))
    # Smooth Flow
    if 'flow_smooth' in df:
        fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['flow_smooth'],
                                  mode='lines', name='Smoothed', line=dict(color='blue')))
    # Steady Window Highlight
    if bounds:
        s_start, s_end = bounds
        fig1.add_vrect(x0=s_start, x1=s_end, fillcolor="green", opacity=0.1,
                       annotation_text="Steady Window", annotation_position="top left")

    fig1.update_layout(title="Flow Analysis", height=400, template="plotly_white")

    # --- FIGURE 2: Stability & FFT ---
    # We'll make a 2-column subplot for CV trace and FFT
    from plotly.subplots import make_subplots
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Stability Trace (CV)", "Combustion Instability (FFT)"))

    # CV Trace
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=cv_trace,
                              mode='lines', name='CV %', line=dict(color='orange')), row=1, col=1)
    fig2.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=1)

    # FFT (Recalculate based on current bounds)
    if bounds:
        s_start, s_end = bounds
        steady_df = df[(df['timestamp'] >= s_start) & (df['timestamp'] <= s_end)]
        if len(steady_df) > 50:
            freqs, psd, peak = analyze_stability_fft(steady_df, col_press)
            fig2.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD', line=dict(color='purple')), row=1,
                           col=2)
            fig2.update_yaxes(type="log", row=1, col=2)
            fig2.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)

    fig2.update_layout(height=400, showlegend=False, template="plotly_white")

    # Combine HTML
    html_content = f"""
    <html>
        <head><title>Report: {filename}</title></head>
        <body>
            {summary_html}
            {fig1.to_html(full_html=False, include_plotlyjs='cdn')}
            {fig2.to_html(full_html=False, include_plotlyjs='cdn')}
        </body>
    </html>
    """
    return html_content


# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Single Test Analysis", "Batch Comparison", "Performance Analysis"])

# ==========================================
# TAB 1: SINGLE TEST DEEP DIVE
# ==========================================
with tab1:
    st.header("Single Test Workbench")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="single")

    if uploaded_file and current_config:
        # --- FIX: Extract standard columns from Config for Plotting ---
        # The plotting code needs to know which CSV column corresponds to "Flow" and "Pressure"
        col_map = current_config.get('columns', {})
        col_flow = col_map.get('mass_flow_ox')  # Default to Oxidizer flow for plots
        col_press = col_map.get('chamber_pressure')
        col_thrust = col_map.get('thrust')

        # 1. Process File using CONFIG
        df, bounds, cv_trace = process_file(uploaded_file, current_config)

        if df is not None:
            # 2. Interface for Cutting
            col_main, col_stats = st.columns([3, 1])

            with col_stats:
                st.subheader("Metrics")
                use_manual = st.checkbox("Manual Override", value=False)

                # Default bounds
                t_min_file = float(df['timestamp'].min())
                t_max_file = float(df['timestamp'].max())

                if bounds:
                    start_def, end_def = bounds
                else:
                    start_def, end_def = t_min_file, t_max_file
                    if not use_manual:
                        st.warning("No stable window detected.")

                # MANUAL CUTTING SLIDER
                if use_manual:
                    m_start, m_end = st.slider(
                        "Select Steady Window (ms)",
                        min_value=t_min_file, max_value=t_max_file,
                        value=(float(start_def), float(end_def)), step=100.0
                    )
                    current_bounds = (m_start, m_end)
                else:
                    current_bounds = (start_def, end_def)
                    if bounds:
                        st.info(f"Auto: {start_def:.0f} - {end_def:.0f} ms")

                # CALCULATE LIVE STATS
                s_start, s_end = current_bounds
                steady_mask = (df['timestamp'] >= s_start) & (df['timestamp'] <= s_end)
                steady_df = df[steady_mask]

                if not steady_df.empty:
                    # Use Physics Columns if available (Calculated in process_file)
                    # Fallback to raw columns if physics failed
                    val_press = steady_df[col_press].mean() if col_press in steady_df else 0.0
                    val_thrust = steady_df[col_thrust].mean() if col_thrust and col_thrust in steady_df else 0.0

                    # Display Metrics
                    st.metric("Avg Pressure", f"{val_press:.2f} bar")
                    if val_thrust > 0:
                        st.metric("Avg Thrust", f"{val_thrust:.2f} N")

                    # Performance Metrics (if they exist)
                    if 'isp' in steady_df:
                        st.metric("Specific Impulse (Isp)", f"{steady_df['isp'].mean():.1f} s")
                    if 'c_star' in steady_df:
                        st.metric("C*", f"{steady_df['c_star'].mean():.0f} m/s")

                # DOWNLOAD BUTTON (Optional - requires the helper function from before)
                # st.download_button(...)

            with col_main:
                # Clip Toggle
                clip_view = st.checkbox("Focus View (±5s)", value=True)

                # PLOTLY CHART
                fig = go.Figure()

                plot_df = df.copy()
                if clip_view:
                    view_start = max(t_min_file, s_start - 5000)
                    view_end = min(t_max_file, s_end + 5000)
                    plot_df = df[(df['timestamp'] >= view_start) & (df['timestamp'] <= view_end)]

                # Plot configured columns
                if col_flow and col_flow in plot_df:
                    fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df[col_flow],
                                             mode='markers', name='Ox Flow', marker=dict(color='silver', size=3)))
                if col_press and col_press in plot_df:
                    # Plot pressure on secondary axis or just overlay normalized?
                    # For now, let's just plot Flow + Pressure on left axis or let user choose.
                    fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df[col_press],
                                             mode='lines', name='Pressure', line=dict(color='red')))

                # Highlight Selected Window
                fig.add_vrect(x0=s_start, x1=s_end, fillcolor="green", opacity=0.1, annotation_text="Active Window")

                fig.update_layout(title="Test Overview", height=500, xaxis_title="Time (ms)")
                st.plotly_chart(fig, use_container_width=True)

                # FFT Analysis (on Pressure)
                if not steady_df.empty and col_press in steady_df and len(steady_df) > 50:
                    freqs, psd, peak = analyze_stability_fft(steady_df, col_press)
                    fig_fft = go.Figure()
                    fig_fft.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='purple')))
                    fig_fft.update_layout(title=f"Combustion Stability (Peak: {peak:.1f} Hz)", height=300,
                                          yaxis_type="log")
                    st.plotly_chart(fig_fft, use_container_width=True)

    elif uploaded_file and not current_config:
        st.warning("Please select a Configuration from the sidebar first!")

# ==========================================
# TAB 2: BATCH COMPARISON
# ==========================================
with tab2:
    st.header("Batch Analysis & Overlay")
    files = st.file_uploader("Upload Multiple CSVs", type=["csv"], accept_multiple_files=True, key="batch")

    if files:
        results = []
        dfs_for_plot = {}

        # Align Options
        align_mode = st.radio("Alignment Mode", ["Raw Time", "Align to Steady Start"], horizontal=True)

        with st.spinner("Processing Batch..."):
            for f in files:
                # Process
                df_b, bounds_b, _ = process_file(f, col_flow, col_press, freq_ms, window_ms, cv_thresh)

                # Stats
                stats = {'Filename': f.name, 'Status': 'No Steady State'}
                if bounds_b:
                    bs, be = bounds_b
                    mask = (df_b['timestamp'] >= bs) & (df_b['timestamp'] <= be)
                    sub = df_b[mask]
                    stats = {
                        'Filename': f.name,
                        'Status': 'OK',
                        'Avg_Press': sub[col_press].mean(),
                        'Avg_Flow': sub[col_flow].mean(),
                        'Duration_s': (be - bs) / 1000.0,
                        'Steady_Start': bs
                    }

                    # Store for plotting
                    # Add Relative Time column if needed
                    if align_mode == "Align to Steady Start":
                        df_b['plot_time'] = (df_b['timestamp'] - bs) / 1000.0
                    else:
                        df_b['plot_time'] = df_b['timestamp'] / 1000.0

                    dfs_for_plot[f.name] = df_b

                results.append(stats)

        # 1. Summary Table
        st.subheader("Summary Table")
        res_df = pd.DataFrame(results)
        st.dataframe(res_df.style.highlight_max(axis=0))

        # 2. Overlay Plot
        st.subheader("Test Overlay")
        tests_to_plot = st.multiselect("Select Tests to Overlay", list(dfs_for_plot.keys()),
                                       default=list(dfs_for_plot.keys()))

        fig_over = go.Figure()
        for name in tests_to_plot:
            d = dfs_for_plot[name]
            # Plot Flow
            fig_over.add_trace(go.Scatter(x=d['plot_time'], y=d['flow_smooth'], mode='lines', name=name))

        fig_over.update_layout(height=600, xaxis_title="Time (s)", yaxis_title="Mass Flow")

        if align_mode == "Align to Steady Start":
            fig_over.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Steady Start")

        st.plotly_chart(fig_over, use_container_width=True)