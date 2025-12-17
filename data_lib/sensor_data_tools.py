import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from scipy.signal import welch

def integrate_data(df, col_name, time_col='timestamp', t_start=None, t_end=None, time_unit='ms'):
    """
    Calculates the integral of a column over time

    Args:
        df (pd.DataFrame): Source dataframe.
        col_name (str): Column to integrate (y-axis).
        time_col (str): Time column (x-axis).
        t_start (float, optional): Start time (in same units as time_col).
        t_end (float, optional): End time (in same units as time_col).
        time_unit (str): 'ms' or 's'. scales the time delta to Seconds.

    Returns:
        float: The integrated total.
    """


    # 1. Handle Index vs Column
    # If time_col is not a column, check if it's the index
    subset = df.copy()
    if time_col not in subset.columns and subset.index.name == time_col:
        subset = subset.reset_index()
    elif time_col not in subset.columns and isinstance(subset.index, pd.DatetimeIndex):
        # Fallback for resampled data where index might not have a name
        subset = subset.reset_index()
        time_col = subset.columns[0]  # Assume first column is the former index

    # 2. Filter by time window  (if provided)
    if t_start is not None:
        subset = subset[subset[time_col] >= t_start]
    if t_end is not None:
        subset = subset[subset[time_col] <= t_end]

    # 3. Clean Data
    # Drops NaNs so we don't integrate empty space or get NaN result
    subset = subset.dropna().sort_values(time_col)

    if subset.empty:
        print(f"Warning: No valid data found for {col_name} in the given window.")
        return 0.0

    # 4. Extract Vectors
    y = subset[col_name].values
    x = subset[time_col].values

    # 5. Handle Unit Conversion (Get dt in Seconds)
    if np.issubdtype(x.dtype, np.datetime64):
        # Convert Datetime to seconds (float)
        x_seconds = x.astype('datetime64[ns]').astype(np.int64) / 1e9
    else:
        # Convert Numeric to seconds
        scale_factor = 1.0
        if time_unit == 'ms':
            scale_factor = 1000.0
        elif time_unit == 'us':
            scale_factor = 1000000.0
        elif time_unit == 'ns':
            scale_factor = 1e9
        x_seconds = x / scale_factor

    total = np.trapz(y, x=x_seconds)
    return total



def analyze_stability_fft(df, col_name, fs=100):
    """
    Performs FFT/PSD analysis to detect combustion instability.
    fs: Sampling frequency (Hz).
    """
    # Extract clean data
    data = df[col_name].dropna().values

    # Detrend (remove the DC offset/average pressure)
    data_ac = data - np.mean(data)

    # Welch's Method (Standard for sensor noise analysis)
    freqs, psd = welch(data_ac, fs=fs, nperseg=256)

    # Find dominant frequency
    peak_freq = freqs[np.argmax(psd)]

    return freqs, psd, peak_freq



def smooth_signal_savgol(df, col_name, window=21, polyorder=3):
    """
    Smooths noise without destroying signal peaks.
    window: Must be odd (e.g. 11, 21, 51). Larger = smoother.
    """
    y = df[col_name].ffill() # Safety fill
    print("Smoothing signal done.")
    return savgol_filter(y, window_length=window, polyorder=polyorder)

def resample_data(df, time_col='timestamp', freq='10ms'):
    """
    Converts sparse/irregular data to a fixed frequency (100Hz = 10ms).
    """
    # Convert to datetime objects
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], unit='ms')

    # Set time as index and resample
    df = df.set_index(time_col)

    # 'mean' handles duplicate timestamps; 'interpolate' fills the gaps
    df_clean = df.resample(freq).mean().interpolate(method='time')

    return df_clean


def calculate_rate(df, col_name):
    """
    Calculates rate of change per second (e.g. bar/s or g/s^2).
    """
    # 1. Calculate delta Y
    dy = df[col_name].diff()

    # 2. Calculate delta T in seconds
    dt = df.index.to_series().diff().dt.total_seconds()

    # 3. Rate = dy / dt
    return dy / dt

def find_steady_window_robust(data, col_name, window_sec=0.5, threshold_pct=0.5):
    # ---------------------------------------------------------
    # STEP 1: PRE-PROCESSING
    # ---------------------------------------------------------
    # Drop rows where the target column is NaN. This packs the data so 'rolling' works on consecutive measurements.
    clean_data = data[['timestamp', col_name]].dropna().sort_values('timestamp').copy()

    # Create relative time in Seconds
    start_time = data['timestamp'].min()  # Keep original start reference
    clean_data['time_rel_s'] = (clean_data['timestamp'] - start_time) / 1000.0

    # ---------------------------------------------------------
    # STEP 2: DETECT SAMPLING RATE & WINDOW
    # ---------------------------------------------------------
    # Calculate dt based on the CLEANED data
    dt_s = clean_data['time_rel_s'].diff().median()
    if dt_s <= 0 or np.isnan(dt_s): dt_s = 0.01  # Fallback

    # Calculate window size in samples
    window_samples = int(window_sec / dt_s)
    window_samples = max(1, window_samples)

    print(f"DEBUG: Sampling Rate = {1 / dt_s:.1f} Hz | Window = {window_samples} samples")

    # ---------------------------------------------------------
    # STEP 3: STABILITY CALCULATION
    # ---------------------------------------------------------
    # Calculate Rolling Stats
    rolling = clean_data[col_name].rolling(window=window_samples, center=True)
    r_mean = rolling.mean()
    r_std = rolling.std()

    # Calculate CV (Coefficient of Variation)
    # Filter out near-zero mean to avoid Infinity
    safe_mean = r_mean.copy()
    safe_mean[safe_mean.abs() < 1e-4] = np.nan

    # Add the CV trace back to the dataframe for plotting
    clean_data['cv_pct'] = (r_std / safe_mean) * 100

    # ---------------------------------------------------------
    # STEP 4: FIND STABLE REGION
    # ---------------------------------------------------------
    is_stable = clean_data['cv_pct'] < threshold_pct

    # Group consecutive stable points
    group_id = (is_stable != is_stable.shift()).cumsum()
    clean_data['group'] = group_id

    # Filter only the stable rows
    stable_rows = clean_data[is_stable]

    if stable_rows.empty:
        print(f"No stable window found. Min CV was {clean_data['cv_pct'].min():.3f}%")
        return None, clean_data

    # Find longest group
    largest_group = stable_rows['group'].value_counts().idxmax()
    best_window = stable_rows[stable_rows['group'] == largest_group]

    t_start = best_window['time_rel_s'].min()
    t_end = best_window['time_rel_s'].max()

    return (t_start, t_end), clean_data


def find_steady_window_dense(df, col_name, window_sec=1.0, threshold_pct=0.5, freq_ms=10):
    """
    Finds the longest stable window on DENSE (resampled) data.
    """
    # Calculate window size in samples
    dt_s = freq_ms / 1000.0
    window_samples = max(1, int(window_sec / dt_s))

    # Rolling Stats
    rolling = df[col_name].rolling(window=window_samples, center=True)
    r_mean = rolling.mean()
    r_std = rolling.std()

    # CV Calculation (protect against divide by zero)
    safe_mean = r_mean.copy()
    safe_mean[safe_mean.abs() < 1e-4] = np.nan
    cv_trace = (r_std / safe_mean) * 100

    # Identify Stable Regions
    is_stable = cv_trace < threshold_pct

    # Group consecutive stable points
    group_id = (is_stable != is_stable.shift()).cumsum()
    stable_groups = df[is_stable].copy()
    stable_groups['group'] = group_id[is_stable]

    if stable_groups.empty:
        return None, cv_trace

    # Find largest group
    best_group_id = stable_groups['group'].value_counts().idxmax()
    best_window = stable_groups[stable_groups['group'] == best_group_id]

    return (best_window.index.min(), best_window.index.max()), cv_trace


def generate_pdf_report(df, filename, output_filename, analysis_results):
    """
    Generates a multi-page PDF report.
    """
    # Unpack results
    bounds = analysis_results.get('bounds')
    cv_trace = analysis_results.get('cv_trace')
    stats = analysis_results.get('stats')
    col_flow = analysis_results.get('col_flow')
    col_press = analysis_results.get('col_press')

    time_rel = (df.index - df.index[0]).total_seconds()

    with PdfPages(output_filename) as pdf:
        # --- PAGE 1: SUMMARY DASHBOARD ---
        fig1 = plt.figure(figsize=(11.69, 8.27))  # A4 Landscape
        plt.suptitle(f"Test Report: {filename}", fontsize=20, weight='bold')

        # Text Box with Stats
        text_str = (
            f"ANALYSIS SUMMARY\n"
            f"----------------------------------------\n"
            f"Total Active Mass:  {stats['total_mass']:.2f} g\n\n"
            f"STEADY STATE METRICS\n"
            f"----------------------------------------\n"
            f"Status:             {'DETECTED' if bounds else 'NOT FOUND'}\n"
            f"Duration:           {stats['duration']:.2f} s\n"
            f"Avg Mass Flow:      {stats['avg_flow']:.3f} g/s\n"
            f"Avg Pressure:       {stats['avg_press']:.3f} bar\n"
            f"Steady Mass:        {stats['steady_mass']:.2f} g\n"
            f"Stability (CV):     {stats['avg_cv']:.3f} %\n"
        )

        # Add text to figure
        plt.figtext(0.05, 0.6, text_str, fontsize=12, fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))

        # Small Overview Plot (Flow & Pressure)
        ax1 = fig1.add_axes([0.45, 0.55, 0.5, 0.35])  # [left, bottom, width, height]
        ax1.plot(time_rel, df[col_flow], color='blue', label='Flow')
        ax1.set_ylabel('Mass Flow', color='blue')
        ax2 = ax1.twinx()
        ax2.plot(time_rel, df[col_press], color='green', label='Pressure', alpha=0.7)
        ax2.set_ylabel('Pressure', color='green')
        ax1.set_title("Test Overview")
        ax1.grid(True, alpha=0.3)

        # Stability/FFT Plot (Bottom Half)
        ax3 = fig1.add_axes([0.1, 0.1, 0.8, 0.35])
        if bounds:
            t_start, t_end = bounds
            steady_data = df[(df.index >= t_start) & (df.index <= t_end)]
            freqs, psd, peak = analyze_stability_fft(steady_data, col_press)

            ax3.semilogy(freqs, psd, color='purple')
            ax3.set_title(f"Combustion Stability (Pressure PSD) - Peak: {peak:.1f} Hz")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("PSD")
            ax3.grid(True, which="both", alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No Steady State for FFT Analysis", ha='center')

        pdf.savefig(fig1)
        plt.close()

        # --- PAGE 2: DETAILED TIME SERIES ---
        fig2, (ax_flow, ax_stab) = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)

        # Flow Plot
        ax_flow.plot(time_rel, df[col_flow], '.', color='lightgray', label='Raw')
        ax_flow.plot(time_rel, df['flow_smooth'], color='blue', label='Smoothed')
        if bounds:
            t_start, t_end = bounds
            s_rel = (t_start - df.index[0]).total_seconds()
            e_rel = (t_end - df.index[0]).total_seconds()
            ax_flow.axvspan(s_rel, e_rel, color='green', alpha=0.2, label='Steady Window')
        ax_flow.set_ylabel('Mass Flow')
        ax_flow.set_title('Detailed Flow Analysis')
        ax_flow.legend()
        ax_flow.grid(True)

        # Stability Trace
        ax_stab.plot(time_rel, cv_trace, color='orange')
        ax_stab.axhline(0.5, color='red', linestyle='--', label='Threshold')
        ax_stab.set_ylabel('CV (%)')
        ax_stab.set_xlabel('Time (s)')
        ax_stab.set_ylim(0, 2.0)
        ax_stab.grid(True)
        ax_stab.legend()

        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close()

    print(f"Report generated: {output_filename}")