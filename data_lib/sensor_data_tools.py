import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, welch


def calculate_rise_time(df, col_name, t_steady_start, steady_mean, time_col='timestamp'):
    """
    Calculates the 10% to 90% rise time relative to the steady state value.
    Returns: (t_10, t_90, rise_time_seconds)
    """
    # 1. Isolate data BEFORE the steady state
    # We assume the rise happens before the stable window starts
    pre_steady = df[df[time_col] < t_steady_start].copy()

    if pre_steady.empty:
        return None, None, None

    # 2. Determine Baseline (0% reference)
    # We take the median of the first 10 points or the min value to be safe
    base_val = np.percentile(pre_steady[col_name], 5)

    # 3. Calculate Thresholds
    total_rise = steady_mean - base_val
    p10 = base_val + 0.10 * total_rise
    p90 = base_val + 0.90 * total_rise

    try:
        # Find first time it crosses 10%
        t_10 = pre_steady[pre_steady[col_name] >= p10].iloc[0][time_col]

        # Find first time it crosses 90% (must be after t_10)
        subset_90 = pre_steady[(pre_steady[col_name] >= p90) &
                               (pre_steady[time_col] > t_10)]

        if subset_90.empty:
            return t_10, None, None

        t_90 = subset_90.iloc[0][time_col]

        rise_time_s = (t_90 - t_10) / 1000.0
        return t_10, t_90, rise_time_s

    except (IndexError, ValueError):
        return None, None, None

def integrate_data(df, col_name, time_col='timestamp', t_start=None, t_end=None):
    """
    Calculates integral (Area under curve). Expects time_col in MILLISECONDS.
    """
    # Optimization: Filter using boolean indexing (faster than copying/subsetting first)
    mask = np.ones(len(df), dtype=bool)
    if t_start is not None:
        mask &= (df[time_col] >= t_start)
    if t_end is not None:
        mask &= (df[time_col] <= t_end)

    subset = df[mask].dropna(subset=[col_name, time_col])

    if subset.empty:
        return 0.0

    y = subset[col_name].values
    x_ms = subset[time_col].values

    # Convert ms to seconds for physical integration units (e.g. g/s * s = g)
    x_seconds = x_ms / 1000.0

    return np.trapz(y, x=x_seconds)


def resample_data(df, time_col='timestamp', freq_ms=10):
    """
    Resamples data to fixed grid. freq_ms is the time step in milliseconds.
    """
    df_temp = df.copy()

    # 1. Create temporary time index
    df_temp.index = pd.to_timedelta(df_temp[time_col], unit='ms')
    df_temp = df_temp.drop(columns=[time_col])  # Prevent collision

    # 2. Resample
    df_clean = df_temp.resample(f'{freq_ms}ms').mean().interpolate(method='time')

    # 3. Restore time column
    df_clean = df_clean.reset_index()
    time_idx_name = df_clean.columns[0]  # Usually 'index'

    # Convert timedelta back to float ms
    df_clean[time_col] = df_clean[time_idx_name].dt.total_seconds() * 1000.0

    if time_idx_name != time_col:
        df_clean = df_clean.drop(columns=[time_idx_name])

    return df_clean


def smooth_signal_savgol(df, col_name, window=21, polyorder=3):
    y = df[col_name].ffill().bfill()
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def find_steady_window(df, col_name, time_col='timestamp', window_ms=1000, threshold_pct=0.5):
    """
    Finds stable window. Returns bounds in MILLISECONDS.
    """
    df_clean = df.dropna(subset=[col_name, time_col]).sort_values(time_col).copy()

    # Calculate sampling rate from data to determine window size
    dt_ms = df_clean[time_col].diff().median()
    if np.isnan(dt_ms) or dt_ms <= 0: dt_ms = 10.0

    window_samples = max(1, int(window_ms / dt_ms))

    # Rolling Stats
    rolling = df_clean[col_name].rolling(window=window_samples, center=True)
    r_mean = rolling.mean()
    r_std = rolling.std()

    # CV Calculation
    safe_mean = r_mean.copy()
    safe_mean[safe_mean.abs() < 1e-4] = np.nan
    cv_trace = (r_std / safe_mean) * 100

    # Logic to find longest stable group
    is_stable = cv_trace < threshold_pct
    group_id = (is_stable != is_stable.shift()).cumsum()

    stable_rows = df_clean[is_stable].copy()
    stable_rows['group'] = group_id[is_stable]

    if stable_rows.empty:
        return None, cv_trace

    best_group_id = stable_rows['group'].value_counts().idxmax()
    best_window = stable_rows[stable_rows['group'] == best_group_id]

    # Return ms bounds
    return (best_window[time_col].min(), best_window[time_col].max()), cv_trace


def analyze_stability_fft(df, col_name, time_col='timestamp'):
    subset = df.dropna(subset=[col_name, time_col]).sort_values(time_col)
    data = subset[col_name].values
    t_ms = subset[time_col].values

    if len(t_ms) < 2: return np.array([]), np.array([]), 0.0

    dt_ms = np.nanmean(np.diff(t_ms))
    fs = 1000.0 / dt_ms if dt_ms > 0 else 100.0

    freqs, psd = welch(data - np.mean(data), fs=fs, nperseg=min(256, len(data)))
    peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0

    return freqs, psd, peak_freq


def generate_pdf_report(df, filename, output_filename, analysis_results, time_col='timestamp'):
    """
    Generates PDF. corrected to use time_col for slicing.
    """
    bounds = analysis_results.get('bounds')
    cv_trace = analysis_results.get('cv_trace')
    stats = analysis_results.get('stats')
    col_flow = analysis_results.get('col_flow')
    col_press = analysis_results.get('col_press')

    # Create relative time for X-axis plotting (0 to end)
    t_start_abs = df[time_col].min()
    x_axis = (df[time_col] - t_start_abs) / 1000.0  # Convert ms -> s for plot labels

    with PdfPages(output_filename) as pdf:
        # PAGE 1
        fig1 = plt.figure(figsize=(11, 8))
        plt.suptitle(f"Test Report: {filename}", fontsize=18, weight='bold')

        txt = (f"Duration:     {stats['duration']:.2f} s\n"
               f"Total Mass:   {stats['total_mass']:.2f} g\n"
               f"Steady Flow:  {stats['avg_flow']:.3f} g/s")
        plt.figtext(0.05, 0.7, txt, fontfamily='monospace', fontsize=12)

        ax1 = fig1.add_axes([0.4, 0.6, 0.55, 0.3])
        ax1.plot(x_axis, df[col_flow], label='Flow')
        ax1.set_ylabel('Flow')
        ax1.legend(loc='upper right')

        # FFT
        ax3 = fig1.add_axes([0.1, 0.1, 0.8, 0.4])
        if bounds:
            t_s, t_e = bounds
            # FIX: Filter using the column, not the index
            mask = (df[time_col] >= t_s) & (df[time_col] <= t_e)
            steady_data = df[mask]

            f, p, peak = analyze_stability_fft(steady_data, col_press, time_col)
            ax3.semilogy(f, p, color='purple')
            ax3.set_title(f"Stability (Peak: {peak:.1f} Hz)")

        pdf.savefig(fig1)
        plt.close()

        # PAGE 2
        fig2, (ax_f, ax_cv) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        ax_f.plot(x_axis, df[col_flow], '.', color='silver')
        if 'flow_smooth' in df:
            ax_f.plot(x_axis, df['flow_smooth'], color='blue')

        if bounds:
            t_s, t_e = bounds
            # Convert absolute bounds to relative seconds for the plot
            ts_rel = (t_s - t_start_abs) / 1000.0
            te_rel = (t_e - t_start_abs) / 1000.0
            ax_f.axvspan(ts_rel, te_rel, color='green', alpha=0.2)

        ax_cv.plot(x_axis, cv_trace, color='orange')
        ax_cv.set_ylim(0, 5)
        ax_cv.axhline(0.5, color='red', linestyle='--')

        pdf.savefig(fig2)
        plt.close()

    print(f"Report Generated: {output_filename}")



def generate_html_report(df, filename, output_filename, analysis_results):
    """
    Generates a standalone HTML report with interactive plots.
    """
    bounds = analysis_results.get('bounds')
    cv_trace = analysis_results.get('cv_trace')
    stats = analysis_results.get('stats')
    col_flow = analysis_results.get('col_flow')
    col_press = analysis_results.get('col_press')

    # Create the Summary Text
    summary_html = f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #f0f0f0;">
        <h2>Test Report: {filename}</h2>
        <pre>
            DURATION:     {stats['duration']:.2f} s
            TOTAL MASS:   {stats['total_mass']:.2f} g
            STEADY FLOW:  {stats['avg_flow']:.3f} g/s
            RISE TIME:    {stats.get('rise_time', 0):.3f} s
            STABILITY:    {stats['avg_cv']:.3f} % (CV)
        </pre>
    </div>
    """

    # --- FIGURE 1: Overview & Steady State ---
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Flow Analysis", "Stability (CV)"))

    # Flow Traces
    fig1.add_trace(go.Scatter(x=df['timestamp'], y=df[col_flow],
                              mode='markers', name='Raw Flow',
                              marker=dict(color='lightgray', size=4)), row=1, col=1)

    if 'flow_smooth' in df:
        fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['flow_smooth'],
                                  mode='lines', name='Smoothed', line=dict(color='blue')), row=1, col=1)

    # Stability Trace
    fig1.add_trace(go.Scatter(x=df['timestamp'], y=cv_trace,
                              mode='lines', name='CV %', line=dict(color='orange')), row=2, col=1)

    # Add Threshold & Steady Window
    if bounds:
        t_s, t_e = bounds
        # Green window for steady state
        fig1.add_vrect(x0=t_s, x1=t_e, fillcolor="green", opacity=0.1,
                       layer="below", line_width=0, annotation_text="Steady", annotation_position="top left")

    fig1.add_hline(y=1.0, line_dash="dash", line_color="red", row=2, col=1)
    fig1.update_layout(height=600, title_text="Time Domain Analysis")

    # --- FIGURE 2: FFT ---
    fig2 = go.Figure()
    if bounds:
        # Re-run FFT for the plot (or pass it in if you prefer)
        from scipy.signal import welch
        steady_df = df[(df['timestamp'] >= t_s) & (df['timestamp'] <= t_e)]
        if len(steady_df) > 10:
            data = steady_df[col_press].values
            dt_ms = (steady_df['timestamp'].diff().median())
            fs = 1000.0 / dt_ms
            freqs, psd = welch(data - data.mean(), fs=fs, nperseg=256)

            fig2.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='purple')))
            fig2.update_layout(title="Combustion Stability (FFT)", yaxis_type="log",
                               xaxis_title="Frequency (Hz)", yaxis_title="PSD")
        else:
            fig2.add_annotation(text="Not enough data for FFT", showarrow=False)

    # Combine into one HTML file
    with open(output_filename, 'w') as f:
        f.write(summary_html)
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))

    print(f"Interactive Report generated: {output_filename}")