import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io

# Import your library
from sensor_data_tools import (
    resample_data, smooth_signal_savgol, find_steady_window,
    integrate_data, calculate_rise_time, analyze_stability_fft
)

st.set_page_config(page_title="Hopper Data Studio", layout="wide")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔧 Sensor Config")
col_flow = st.sidebar.text_input("Mass Flow Column", "10009")
col_press = st.sidebar.text_input("Pressure Column", "10003")
freq_ms = st.sidebar.number_input("Resample Step (ms)", value=10)
window_ms = st.sidebar.number_input("Stability Window (ms)", value=500)
cv_thresh = st.sidebar.slider("CV Threshold (%)", 0.1, 5.0, 1.0)


# --- HELPER: PROCESS FILE ---
@st.cache_data
def process_file(uploaded_file, c_flow, c_press, f_ms, w_ms, cv_th):
    # Load and Preprocess
    df_raw = pd.read_csv(uploaded_file)
    df = resample_data(df_raw, time_col='timestamp', freq_ms=f_ms)
    df['flow_smooth'] = smooth_signal_savgol(df, c_flow)

    # Auto-Detect Window
    bounds, cv_trace = find_steady_window(df, 'flow_smooth', 'timestamp', w_ms, cv_th)

    return df, bounds, cv_trace


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

    if uploaded_file:
        # 1. Process
        df, bounds, cv_trace = process_file(uploaded_file, col_flow, col_press, freq_ms, window_ms, cv_thresh)

        # 2. Interface for Cutting
        col_main, col_stats = st.columns([3, 1])

        with col_stats:
            st.subheader("Metrics")
            use_manual = st.checkbox("🛠 Manual Override", value=False)

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
                    st.info(f"Auto-Detected: {start_def:.0f} - {end_def:.0f} ms")

            # CALCULATE LIVE STATS
            s_start, s_end = current_bounds
            steady_mask = (df['timestamp'] >= s_start) & (df['timestamp'] <= s_end)
            steady_df = df[steady_mask]

            # Initialize default stats
            report_stats = {
                'duration': (t_max_file - t_min_file) / 1000.0,
                'total_mass': 0.0, 'avg_flow': 0.0, 'avg_press': 0.0,
                'avg_cv': 0.0, 'rise_time': 0.0
            }

            if not steady_df.empty:
                # Update stats with actual data
                avg_flow = steady_df[col_flow].mean()
                avg_press = steady_df[col_press].mean()
                total_mass = integrate_data(df, col_flow, 'timestamp', s_start, s_end)
                avg_cv = cv_trace[steady_mask].mean()

                # Display Metrics
                st.metric("Avg Pressure", f"{avg_press:.2f} bar")
                st.metric("Avg Flow", f"{avg_flow:.2f} g/s")
                st.metric("Steady Mass", f"{total_mass:.2f} g")

                t10, t90, rise_s = calculate_rise_time(df, 'flow_smooth', s_start, avg_flow)
                if rise_s:
                    st.metric("Rise Time", f"{rise_s:.3f} s")

                # Update report dictionary
                report_stats.update({
                    'total_mass': total_mass,
                    'avg_flow': avg_flow,
                    'avg_press': avg_press,
                    'avg_cv': avg_cv,
                    'rise_time': rise_s if rise_s else 0.0
                })

            # --- DOWNLOAD BUTTON ---
            st.markdown("---")
            if st.button("Generate Report Preview"):
                # We generate it on click to save resources
                html_data = get_report_html(df, uploaded_file.name, report_stats,
                                            current_bounds, cv_trace, col_flow, col_press)

                # The actual download button needs to be rendered
                st.download_button(
                    label="💾 Download HTML Report",
                    data=html_data,
                    file_name=f"Report_{uploaded_file.name}.html",
                    mime="text/html"
                )

        with col_main:
            # Clip Toggle
            clip_view = st.checkbox("✂️ Focus View (±5s)", value=True)

            # PLOTLY CHART
            fig = go.Figure()

            plot_df = df.copy()
            if clip_view:
                view_start = max(t_min_file, s_start - 5000)
                view_end = min(t_max_file, s_end + 5000)
                plot_df = df[(df['timestamp'] >= view_start) & (df['timestamp'] <= view_end)]

            fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df[col_flow],
                                     mode='markers', name='Raw', marker=dict(color='silver', size=3)))
            fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df['flow_smooth'],
                                     mode='lines', name='Smooth', line=dict(color='blue')))

            # Highlight Selected Window
            fig.add_vrect(x0=s_start, x1=s_end, fillcolor="green", opacity=0.1, annotation_text="Active Window")

            fig.update_layout(title="Flow Analysis", height=500, xaxis_title="Time (ms)", yaxis_title="Flow")
            st.plotly_chart(fig, use_container_width=True)

            # FFT
            if not steady_df.empty and len(steady_df) > 50:
                freqs, psd, peak = analyze_stability_fft(steady_df, col_press)
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='purple')))
                fig_fft.update_layout(title=f"Instability (Peak: {peak:.1f} Hz)", height=300, yaxis_type="log")
                st.plotly_chart(fig_fft, use_container_width=True)

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