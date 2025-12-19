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

# 1. Config Loader
available_configs = get_available_configs()
current_config = None
selected_config_name = st.sidebar.selectbox(
    "Test Configuration",
    ["Generic / No Config"] + available_configs
)

if selected_config_name and selected_config_name != "Generic / No Config":
    current_config = load_config(selected_config_name)
    st.sidebar.success(f"Loaded: {selected_config_name}")

    # Allow overrides
    with st.sidebar.expander("Override Settings"):
        settings = current_config.get('settings', {})
        freq_ms = st.number_input("Resample (ms)", value=settings.get('resample_freq_ms', 10))
        window_ms = st.number_input("Window (ms)", value=settings.get('steady_window_ms', 500))
        cv_thresh = st.number_input("CV Thresh (%)", value=settings.get('cv_threshold', 1.0))
        geom_title = st.header("Geometry Settings")
        # Geometry overrides
        geom = current_config.get('geometry', {})
        throat_area = st.number_input("Throat Area (mm^2)", value=geom.get('throat_area_mm2', 0.0))
        # Update config object with overrides
        current_config['geometry']['throat_area_mm2'] = throat_area

        ref_val_title = st.header("Reference Values")
        ref_val_table = st.table(current_config["reference_values"])
else:
    # Generic Defaults
    freq_ms = st.sidebar.number_input("Resample (ms)", value=10)
    window_ms = st.sidebar.number_input("Window (ms)", value=500)
    cv_thresh = st.sidebar.number_input("CV Thresh (%)", value=1.0)


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


# --- CORE PROCESSING FUNCTION ---
@st.cache_data
def load_and_process(uploaded_file, config, f_ms):
    """
    Loads, Renames, Resamples, and Enriches data.
    Does NOT perform steady state analysis (that comes later).
    """
    # 1. Load Raw
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    # 2. Rename Columns (Apply Channel Map)
    if config and 'channel_config' in config:
        mapping = config['channel_config']
        # Ensure raw columns are strings for mapping
        df_raw.columns = df_raw.columns.astype(str)
        df_raw.rename(columns=mapping, inplace=True)

    # 3. Identify Time Column
    time_col = 'timestamp'  # Default
    if config:
        time_col = config.get('columns', {}).get('timestamp', 'timestamp')

    # Fallback search for time
    if time_col not in df_raw.columns:
        candidates = [c for c in df_raw.columns if 'time' in c.lower() or 'ts' in c.lower()]
        if candidates:
            time_col = candidates[0]
        else:
            return None, "Could not identify Time column. Check config or CSV."

    # 4. Resample
    df = resample_data(df_raw, time_col=time_col, freq_ms=f_ms)

    # 5. Physics Calculations (Optional)
    if config:
        df = calculate_performance(df, config)

    return df, None


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
tab1, tab2 = st.tabs(["Data Viewer & Analysis", "Batch Comparison"])

# ==========================================
# TAB 1: VIEWER & ANALYSIS
# ==========================================
with tab1:
    uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"], key="single")

    if uploaded_file:
        df, error = load_and_process(uploaded_file, current_config, freq_ms)

        if error:
            st.error(error)
        else:
            # --- SELECTION BAR ---
            col_sel, col_act = st.columns([3, 1])

            with col_sel:
                # Default selection: Look for config columns, else pick first few
                default_cols = []
                if current_config:
                    cmap = current_config.get('columns', {})
                    # prioritize pressure and flow
                    for k in ['chamber_pressure', 'mass_flow_ox', 'thrust']:
                        if k in cmap and cmap[k] in df.columns:
                            default_cols.append(cmap[k])

                if not default_cols:
                    default_cols = df.columns[1:3].tolist()  # Skip time, take next 2

                plot_cols = st.multiselect("Select Channels to Plot", df.columns, default=default_cols)

            with col_act:
                enable_analysis = st.toggle("🔎 Enable Steady State Analysis", value=False)

            # --- MAIN PLOT ---
            fig = go.Figure()
            for col in plot_cols:
                # Normalize? Maybe later. For now, raw values.
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=col))

            fig.update_layout(height=500, xaxis_title="Time (ms)", title="Sensor Data View", template="plotly_white")

            # --- ANALYSIS LOGIC ---
            if enable_analysis:
                st.markdown("---")
                st.subheader("Steady State Analysis")

                # 1. Select Signal for Stability
                # Default to the first plotted column or config pressure
                target_default = plot_cols[0] if plot_cols else df.columns[1]
                if current_config:
                    target_default = current_config.get('columns', {}).get('chamber_pressure', target_default)

                target_col = st.selectbox("Signal for Stability Detection", df.columns, index=list(df.columns).index(
                    target_default) if target_default in df.columns else 0)

                # 2. Run Detection
                # Smooth first
                df['__smooth'] = smooth_signal_savgol(df, target_col)
                bounds, cv_trace = find_steady_window(df, '__smooth', 'timestamp', window_ms, cv_thresh)

                # 3. Display Results
                col_res, col_fft = st.columns([1, 1])

                with col_res:
                    # Manual Override
                    t_min = float(df['timestamp'].min())
                    t_max = float(df['timestamp'].max())

                    if bounds:
                        s_def, e_def = bounds
                        st.success(f"Detected: {s_def:.0f} - {e_def:.0f} ms")
                    else:
                        s_def, e_def = t_min, t_max
                        st.warning("No steady state detected.")

                    use_manual = st.checkbox("Manual Cut", value=(bounds is None))

                    if use_manual:
                        s_start, s_end = st.slider("Steady Window", t_min, t_max, (float(s_def), float(e_def)), 100.0)
                    else:
                        s_start, s_end = s_def, e_def

                    # Add Window to Plot
                    fig.add_vrect(x0=s_start, x1=s_end, fillcolor="green", opacity=0.1,
                                  annotation_text="Analysis Window")



                    # --- CALCULATE & DISPLAY METRICS ---
                    st.subheader("🏁 Performance Report")
                    # Stats Table
                    mask = (df['timestamp'] >= s_start) & (df['timestamp'] <= s_end)
                    steady_df = df[mask]

                    if not steady_df.empty:
                        # 1. Calculate Averages for ALL columns (mapped names)
                        avg_stats = steady_df.mean().to_dict()

                        # 2. Get Derived Metrics (Cd, Efficiencies)
                        from data_lib.propulsion_physics import calculate_derived_metrics

                        derived_metrics = calculate_derived_metrics(avg_stats, current_config or {})

                        # 3. Create a clean display grid
                        # PRIMARY METRICS (Pressure, Flow, Thrust)
                        m1, m2, m3 = st.columns(3)

                        # We try to find the standard columns to show prominently
                        cols_cfg = current_config.get('columns', {}) if current_config else {}

                        with m1:
                            p_col = cols_cfg.get('chamber_pressure', 'chamber_pressure')
                            if p_col in avg_stats:
                                st.metric("Avg Pc", f"{avg_stats[p_col]:.2f} bar")
                            elif 'pressure' in plot_cols[0].lower():  # Fallback guess
                                st.metric("Avg Signal", f"{avg_stats[plot_cols[0]]:.2f}")

                        with m2:
                            # Try to sum flows if they exist
                            total_flow = avg_stats.get('mass_flow_total', 0)
                            if total_flow > 0:
                                st.metric("Total Flow", f"{total_flow:.3f} g/s")
                            else:
                                # Show single flow if only one exists
                                f_col = cols_cfg.get('mass_flow_ox', plot_cols[0])
                                if f_col in avg_stats:
                                    st.metric("Avg Flow", f"{avg_stats[f_col]:.3f} g/s")

                        with m3:
                            t_col = cols_cfg.get('thrust', 'thrust')
                            if t_col in avg_stats:
                                st.metric("Thrust", f"{avg_stats[t_col]:.1f} N")

                        # SECONDARY / DERIVED METRICS (Isp, C*, Cd)
                        st.divider()
                        d_cols = st.columns(4)

                        # Helper to place metrics in the grid
                        metric_list = []

                        # Add Time-Series Averages (if calculated in Step 3 of process_file)
                        if 'isp' in avg_stats: metric_list.append(("Isp", f"{avg_stats['isp']:.1f} s"))
                        if 'c_star' in avg_stats: metric_list.append(("C*", f"{avg_stats['c_star']:.0f} m/s"))
                        if 'of_ratio' in avg_stats: metric_list.append(("O/F", f"{avg_stats['of_ratio']:.2f}"))

                        # Add Single-Value Derived Metrics (Cd, Eta)
                        for k, v in derived_metrics.items():
                            val_str = f"{v:.2f}" if v < 10 else f"{v:.1f}"  # Format logic
                            metric_list.append((k, val_str))

                        # Render the grid
                        for i, (label, val) in enumerate(metric_list):
                            with d_cols[i % 4]:
                                st.metric(label, val)

                    else:
                        st.info("Select a valid window to see metrics.")

                with col_fft:
                    # FFT Analysis
                    if len(steady_df) > 50:
                        freqs, psd, peak = analyze_stability_fft(steady_df, target_col)
                        fig_fft = go.Figure()
                        fig_fft.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='purple')))
                        fig_fft.update_layout(title=f"FFT: {target_col} (Peak: {peak:.1f} Hz)", yaxis_type="log",
                                              height=350)
                        st.plotly_chart(fig_fft, use_container_width=True)

            # Show the main plot (updated with window if analysis is on)
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: BATCH COMPARISON
# ==========================================
with tab2:
    st.header("Batch Overlay")
    files = st.file_uploader("Upload Multiple CSVs", type=["csv"], accept_multiple_files=True, key="batch")

    if files:
        # Select what to plot (e.g. comparing Chamber Pressure across tests)
        # We need to guess available columns from the first file
        first_df, _ = load_and_process(files[0], current_config, freq_ms)
        if first_df is not None:
            overlay_col = st.selectbox("Select Channel to Overlay", first_df.columns, index=1)
            align_mode = st.radio("Alignment", ["Raw Time", "Align to Steady Start"], horizontal=True)

            fig_over = go.Figure()

            for f in files:
                d, _ = load_and_process(f, current_config, freq_ms)
                if d is not None and overlay_col in d.columns:
                    # For alignment, we need to run detection quickly
                    x_axis = d['timestamp']

                    if align_mode == "Align to Steady Start":
                        # Quick detection on the overlay column
                        d['__sm'] = smooth_signal_savgol(d, overlay_col)
                        b, _ = find_steady_window(d, '__sm', 'timestamp', window_ms, cv_thresh)
                        if b:
                            offset = b[0]
                            x_axis = (d['timestamp'] - offset) / 1000.0  # Seconds
                        else:
                            continue  # Skip or plot raw? Let's skip for cleaner plot
                    else:
                        x_axis = d['timestamp'] / 1000.0  # Seconds

                    fig_over.add_trace(go.Scatter(x=x_axis, y=d[overlay_col], mode='lines', name=f.name))

            fig_over.update_layout(height=600, title=f"Batch Comparison: {overlay_col}", xaxis_title="Time (s)")
            st.plotly_chart(fig_over, use_container_width=True)