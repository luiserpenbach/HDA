"""
Steady-State Detection Methods
===============================
Comprehensive steady-state detection algorithms for test data analysis.

This module consolidates all steady-state detection methods used across
the application, eliminating duplication between pages.

Available methods:
- CV-based: Coefficient of variation threshold
- ML-based: Isolation Forest anomaly detection
- Derivative-based: Rate of change threshold
- Simple: Hardcoded middle 50% (fallback)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


def detect_steady_state_cv(
    df: pd.DataFrame,
    signal_col: str,
    window_size: int = 50,
    cv_threshold: float = 0.02,
    time_col: str = 'time_s'
) -> Tuple[Optional[float], Optional[float]]:
    """
    CV-based steady-state detection.

    Detects steady state by finding regions where the coefficient of variation
    (CV = std/mean) falls below a threshold.

    Args:
        df: DataFrame with test data
        signal_col: Column to analyze for stability
        window_size: Rolling window size in samples
        cv_threshold: Maximum CV for steady state
        time_col: Time column name

    Returns:
        Tuple of (start_time, end_time) in the units of time_col, or (None, None)

    Example:
        >>> start, end = detect_steady_state_cv(df, 'P_upstream', window_size=50, cv_threshold=0.02)
        >>> if start and end:
        >>>     steady_data = df[(df['time_s'] >= start) & (df['time_s'] <= end)]
    """
    if signal_col not in df.columns:
        return None, None

    signal = df[signal_col].values
    times = df[time_col].values if time_col in df.columns else np.arange(len(signal))

    if len(signal) < window_size:
        return None, None

    # Calculate rolling CV
    rolling_mean = pd.Series(signal).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(signal).rolling(window=window_size, center=True).std()

    cv = rolling_std / rolling_mean.abs()
    cv = cv.fillna(1.0)  # Fill NaN with high CV

    # Find stable regions
    stable_mask = cv < cv_threshold

    if not stable_mask.any():
        return None, None

    stable_indices = np.where(stable_mask)[0]

    if len(stable_indices) == 0:
        return None, None

    # Find longest continuous stable region
    breaks = np.where(np.diff(stable_indices) > 1)[0]

    if len(breaks) == 0:
        start_idx = stable_indices[0]
        end_idx = stable_indices[-1]
    else:
        segments = []
        prev = 0
        for b in breaks:
            segments.append((prev, b))
            prev = b + 1
        segments.append((prev, len(stable_indices) - 1))

        longest = max(segments, key=lambda x: x[1] - x[0])
        start_idx = stable_indices[longest[0]]
        end_idx = stable_indices[longest[1]]

    return float(times[start_idx]), float(times[end_idx])


def detect_steady_state_ml(
    df: pd.DataFrame,
    signal_cols: List[str],
    time_col: str = 'time_s',
    contamination: float = 0.3
) -> Tuple[Optional[float], Optional[float]]:
    """
    ML-based steady-state detection using Isolation Forest.

    Uses anomaly detection to find the "normal" operating region,
    which typically corresponds to steady state.

    Args:
        df: DataFrame with test data
        signal_cols: Columns to use for detection
        time_col: Time column name
        contamination: Expected fraction of non-steady-state data

    Returns:
        Tuple of (start_time, end_time) in the units of time_col, or (None, None)

    Raises:
        ImportError: If scikit-learn is not available
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError(
            "scikit-learn not available for ML detection. "
            "Install with: pip install scikit-learn"
        )

    # Prepare features
    available_cols = [c for c in signal_cols if c in df.columns]
    if not available_cols:
        return None, None

    times = df[time_col].values if time_col in df.columns else np.arange(len(df))

    # Create feature matrix with rolling statistics
    features = []
    for col in available_cols:
        signal = df[col].values
        features.append(signal)

        # Add rolling mean and std as features
        rolling_mean = pd.Series(signal).rolling(window=20, center=True).mean()
        rolling_mean = rolling_mean.bfill().ffill()
        rolling_std = pd.Series(signal).rolling(window=20, center=True).std().fillna(0)
        features.append(rolling_mean.values)
        features.append(rolling_std.values)

    X = np.column_stack(features)

    # Remove rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(X_valid) < 50:
        return None, None

    # Fit Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    predictions = clf.fit_predict(X_valid)

    # Find longest inlier region (predictions == 1)
    inlier_mask = predictions == 1
    inlier_indices = valid_indices[inlier_mask]

    if len(inlier_indices) == 0:
        return None, None

    # Find longest continuous inlier region
    breaks = np.where(np.diff(inlier_indices) > 5)[0]  # Allow small gaps

    if len(breaks) == 0:
        start_idx = inlier_indices[0]
        end_idx = inlier_indices[-1]
    else:
        segments = []
        prev = 0
        for b in breaks:
            segments.append((inlier_indices[prev], inlier_indices[b]))
            prev = b + 1
        segments.append((inlier_indices[prev], inlier_indices[-1]))

        longest = max(segments, key=lambda x: x[1] - x[0])
        start_idx, end_idx = longest

    return float(times[start_idx]), float(times[end_idx])


def detect_steady_state_derivative(
    df: pd.DataFrame,
    signal_col: str,
    time_col: str = 'time_s',
    derivative_threshold: float = 0.1
) -> Tuple[Optional[float], Optional[float]]:
    """
    Derivative-based steady-state detection.

    Finds regions where the derivative is close to zero, indicating
    no significant change in the signal.

    Args:
        df: DataFrame with test data
        signal_col: Column to analyze
        time_col: Time column name
        derivative_threshold: Maximum normalized derivative for steady state

    Returns:
        Tuple of (start_time, end_time), or (None, None)
    """
    if signal_col not in df.columns:
        return None, None

    signal = df[signal_col].values
    times = df[time_col].values if time_col in df.columns else np.arange(len(signal))

    # Calculate derivative
    dt = np.diff(times)
    ds = np.diff(signal)

    # Avoid division by zero
    dt[dt == 0] = 1e-10
    derivative = ds / dt

    # Normalize by signal range
    signal_range = np.ptp(signal)
    if signal_range > 0:
        normalized_deriv = np.abs(derivative) / signal_range
    else:
        normalized_deriv = np.abs(derivative)

    # Smooth the derivative
    smoothed = pd.Series(normalized_deriv).rolling(window=20, center=True).mean().fillna(1.0)

    # Find stable regions
    stable_mask = smoothed.values < derivative_threshold

    if not stable_mask.any():
        return None, None

    stable_indices = np.where(stable_mask)[0]

    if len(stable_indices) < 10:
        return None, None

    # Find longest continuous stable region
    breaks = np.where(np.diff(stable_indices) > 1)[0]

    if len(breaks) == 0:
        start_idx = stable_indices[0]
        end_idx = stable_indices[-1]
    else:
        segments = []
        prev = 0
        for b in breaks:
            segments.append((prev, b))
            prev = b + 1
        segments.append((prev, len(stable_indices) - 1))

        longest = max(segments, key=lambda x: x[1] - x[0])
        start_idx = stable_indices[longest[0]]
        end_idx = stable_indices[longest[1]]

    # Adjust indices since derivative is shorter
    return float(times[start_idx]), float(times[min(end_idx + 1, len(times) - 1)])


def detect_steady_state_simple(
    df: pd.DataFrame,
    config: dict,
    time_col: str = 'time_s',
    middle_fraction: float = 0.5
) -> Tuple[Optional[float], Optional[float]]:
    """
    Simple steady-state detection (fallback method).

    Returns the middle portion of the data, assuming steady state occurs
    in the middle of the test after initial transients and before shutdown.

    Args:
        df: DataFrame with test data
        config: Configuration dict (not used, for API compatibility)
        time_col: Time column name
        middle_fraction: Fraction of data to use (centered)

    Returns:
        Tuple of (start_time, end_time)

    Example:
        >>> start, end = detect_steady_state_simple(df, config)
        >>> # Returns middle 50% of data by default
    """
    if time_col not in df.columns:
        n = len(df)
        start_idx = int(n * (1 - middle_fraction) / 2)
        end_idx = int(n * (1 + middle_fraction) / 2)
        return float(start_idx), float(end_idx)

    times = df[time_col].values
    n = len(times)
    start_idx = int(n * (1 - middle_fraction) / 2)
    end_idx = int(n * (1 + middle_fraction) / 2)

    return float(times[start_idx]), float(times[end_idx])


def detect_steady_state_auto(
    df: pd.DataFrame,
    config: dict,
    preferred_method: str = 'cv',
    time_col: str = 'time_s'
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Automatic steady-state detection with fallback.

    Tries methods in order of preference until one succeeds.

    Args:
        df: DataFrame with test data
        config: Configuration dict
        preferred_method: Preferred method ('cv', 'ml', 'derivative', 'simple')
        time_col: Time column name

    Returns:
        Tuple of (start_time, end_time, method_used)

    Example:
        >>> start, end, method = detect_steady_state_auto(df, config, preferred_method='cv')
        >>> print(f"Detected using {method}: {start:.2f} - {end:.2f}")
    """
    # Get appropriate signal column for detection
    # Support both sensor_roles (v2.4.0+) and columns (legacy)
    columns = config.get('sensor_roles', config.get('columns', {}))
    signal_col = (
        columns.get('upstream_pressure') or
        columns.get('chamber_pressure') or
        columns.get('pressure') or
        next((col for col in df.columns if 'pressure' in col.lower()), None)
    )

    methods = [preferred_method, 'cv', 'derivative', 'simple']
    methods = list(dict.fromkeys(methods))  # Remove duplicates while preserving order

    for method in methods:
        try:
            if method == 'cv' and signal_col:
                start, end = detect_steady_state_cv(df, signal_col, time_col=time_col)
                if start is not None and end is not None:
                    return start, end, 'cv'

            elif method == 'ml':
                # Use multiple signal columns for ML
                signal_cols = [
                    col for col in df.columns
                    if col != time_col and df[col].dtype in ['float64', 'int64']
                ][:5]  # Limit to 5 sensors
                if signal_cols:
                    start, end = detect_steady_state_ml(df, signal_cols, time_col=time_col)
                    if start is not None and end is not None:
                        return start, end, 'ml'

            elif method == 'derivative' and signal_col:
                start, end = detect_steady_state_derivative(df, signal_col, time_col=time_col)
                if start is not None and end is not None:
                    return start, end, 'derivative'

            elif method == 'simple':
                start, end = detect_steady_state_simple(df, config, time_col=time_col)
                return start, end, 'simple'

        except Exception:
            # Continue to next method on failure
            continue

    # Final fallback
    start, end = detect_steady_state_simple(df, config, time_col=time_col)
    return start, end, 'simple'


def validate_steady_window(
    df: pd.DataFrame,
    start_time: float,
    end_time: float,
    time_col: str = 'time_s',
    min_samples: int = 10
) -> Tuple[bool, str]:
    """
    Validate that a steady-state window is reasonable.

    Args:
        df: DataFrame with test data
        start_time: Start time of window
        end_time: End time of window
        time_col: Time column name
        min_samples: Minimum number of samples required

    Returns:
        Tuple of (is_valid, error_message)
    """
    if time_col not in df.columns:
        return False, f"Time column '{time_col}' not found"

    if start_time >= end_time:
        return False, "Start time must be less than end time"

    times = df[time_col].values
    window_mask = (times >= start_time) & (times <= end_time)
    n_samples = window_mask.sum()

    if n_samples < min_samples:
        return False, f"Window contains only {n_samples} samples (minimum: {min_samples})"

    if n_samples < 0.05 * len(df):
        return False, f"Window is too small (< 5% of data)"

    if n_samples > 0.95 * len(df):
        return False, f"Window is too large (> 95% of data)"

    return True, ""
