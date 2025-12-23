import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, welch


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
    time_idx_name = df_clean.columns[0]  # Usually 'index' or 'timestamp'

    # Convert timedelta back to float ms
    df_clean[time_col] = df_clean[time_idx_name].dt.total_seconds() * 1000.0

    if time_idx_name != time_col:
        df_clean = df_clean.drop(columns=[time_idx_name])

    return df_clean


def smooth_signal_savgol(df, col_name, window=21, polyorder=3):
    """
    Apply Savitzky-Golay smoothing to a column.
    """
    y = df[col_name].ffill().bfill()
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def integrate_data(df, col_name, time_col='timestamp', t_start=None, t_end=None):
    """
    Calculates integral (Area under curve). Expects time_col in MILLISECONDS.
    """
    mask = np.ones(len(df), dtype=bool)
    if t_start is not None: mask &= (df[time_col] >= t_start)
    if t_end is not None: mask &= (df[time_col] <= t_end)

    subset = df[mask].dropna(subset=[col_name, time_col])
    if subset.empty: return 0.0

    y = subset[col_name].values
    x_s = subset[time_col].values / 1000.0  # Convert ms to seconds

    return np.trapz(y, x=x_s)


def calculate_rise_time(df, col_name, t_steady_start, steady_mean, time_col='timestamp'):
    """
    Calculates the 10% to 90% rise time relative to the steady state value.
    Returns: (t_10, t_90, rise_time_seconds)
    """
    pre_steady = df[df[time_col] < t_steady_start].copy()
    if pre_steady.empty: return None, None, None

    # Baseline is 5th percentile to avoid outliers
    base_val = np.percentile(pre_steady[col_name], 5)
    rise = steady_mean - base_val

    p10 = base_val + 0.10 * rise
    p90 = base_val + 0.90 * rise

    try:
        t_10 = pre_steady[pre_steady[col_name] >= p10].iloc[0][time_col]

        # Find 90% crossing AFTER t_10
        subset_90 = pre_steady[(pre_steady[col_name] >= p90) & (pre_steady[time_col] > t_10)]
        if subset_90.empty: return t_10, None, None

        t_90 = subset_90.iloc[0][time_col]
        return t_10, t_90, (t_90 - t_10) / 1000.0
    except (IndexError, ValueError):
        return None, None, None


def analyze_stability_fft(df, col_name, time_col='timestamp'):
    """
    Performs FFT to find dominant frequency and PSD.
    """
    subset = df.dropna(subset=[col_name]).sort_values(time_col)
    if len(subset) < 10: return np.array([]), np.array([]), 0.0

    dt_ms = np.nanmean(np.diff(subset[time_col]))
    fs = 1000.0 / dt_ms if dt_ms > 0 else 100.0

    data = subset[col_name].values
    f, p = welch(data - np.mean(data), fs=fs, nperseg=min(256, len(data)))

    peak_freq = f[np.argmax(p)] if len(p) > 0 else 0.0
    return f, p, peak_freq


# ==========================================
# STEADY STATE DETECTORS
# ==========================================

def find_steady_window(df, col_name, time_col='timestamp', window_ms=1000, threshold_pct=0.5):
    """
    LEGACY: Finds the single longest stable window for one column.
    Used by Hot Fire analysis.
    """
    df_clean = df.dropna(subset=[col_name, time_col]).sort_values(time_col).copy()

    dt_ms = df_clean[time_col].diff().median()
    if np.isnan(dt_ms) or dt_ms <= 0: dt_ms = 10.0
    win_samples = max(1, int(window_ms / dt_ms))

    rolling = df_clean[col_name].rolling(window=win_samples, center=True)
    r_mean = rolling.mean()
    r_std = rolling.std()

    safe_mean = r_mean.copy()
    safe_mean[safe_mean.abs() < 1e-4] = np.nan
    cv_trace = (r_std / safe_mean) * 100

    is_stable = cv_trace < threshold_pct
    group_id = (is_stable != is_stable.shift()).cumsum()

    stable_rows = df_clean[is_stable].copy()
    stable_rows['group'] = group_id[is_stable]

    if stable_rows.empty: return None, cv_trace

    best_group = stable_rows['group'].value_counts().idxmax()
    best_win = stable_rows[stable_rows['group'] == best_group]

    return (best_win[time_col].min(), best_win[time_col].max()), cv_trace


def find_steady_windows(df, col_names, time_col='timestamp', window_ms=1000, cv_thresh=1.0, min_val=0.0):
    """
    ADVANCED: Identifies MULTIPLE stable windows where ALL columns meet criteria.
    Args:
        col_names: list of column names to check for stability.
        min_val: Signal must be above this value (e.g. Pressure > 5 bar) to be considered.
    Returns:
        windows: List of tuples [(start_ms, end_ms), ...]
        combined_cv: Series of the 'worst' CV among columns at each point.
    """
    if isinstance(col_names, str): col_names = [col_names]

    # Prepare Data
    sub = df[[time_col] + col_names].sort_values(time_col).copy()
    dt = sub[time_col].diff().median()
    if np.isnan(dt) or dt <= 0: dt = 10.0
    win_samples = max(1, int(window_ms / dt))

    # Initialize Logic Masks
    is_stable_global = pd.Series(True, index=sub.index)
    max_cv_trace = pd.Series(0.0, index=sub.index)

    for col in col_names:
        rolling = sub[col].rolling(window=win_samples, center=True)
        rmean = rolling.mean()
        rstd = rolling.std()

        # Calculate CV
        safe_mean = rmean.copy()
        safe_mean[safe_mean.abs() < 1e-6] = np.nan
        cv = (rstd / safe_mean).abs() * 100.0

        # Track worst stability (highest CV is the limiting factor)
        max_cv_trace = np.maximum(max_cv_trace, cv.fillna(0))

        # Check constraints: Stable AND above minimum value
        col_stable = (cv < cv_thresh) & (rmean.abs() > min_val)
        is_stable_global &= col_stable

    # Identify Segments
    sub['stable'] = is_stable_global.fillna(False)
    sub['group'] = (sub['stable'] != sub['stable'].shift()).cumsum()

    # Extract Windows
    windows = []
    valid_groups = sub[sub['stable'] == True]

    if not valid_groups.empty:
        for _, grp in valid_groups.groupby('group'):
            ts = grp[time_col].min()
            te = grp[time_col].max()

            # Filter out micro-segments shorter than the window itself
            if (te - ts) >= window_ms:
                windows.append((ts, te))

    return windows, max_cv_trace