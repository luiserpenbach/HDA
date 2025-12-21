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

