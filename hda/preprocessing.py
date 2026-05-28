"""Test data preprocessing pipeline for the HDA Qt UI.

Mirrors the Streamlit Setup-tab pipeline: time conversion, gap filling,
resampling, and time-window trimming. All internal time columns use seconds
(``time_s``) for compatibility with ``core`` analysis.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

TIME_DERIVED = frozenset({"time_s", "time_ms"})

# Magnitude thresholds for auto-detecting epoch vs relative timestamps.
_UNIX_US_MIN = 1e14
_UNIX_MS_MIN = 1e11
_UNIX_S_MIN = 1e8
_RELATIVE_MS_MIN = 1e3


def normalize_time_unit(unit: str) -> str:
    """Normalize UI time-unit strings."""
    u = (unit or "unix_ms").strip().lower()
    if u in {"unix_ms", "unix-ms", "epoch_ms", "epoch ms"}:
        return "unix_ms"
    if u in {"unix_s", "unix-s", "epoch_s", "epoch s", "unix", "epoch"}:
        return "unix_s"
    if u in {"unix_us", "unix-us", "epoch_us", "epoch us"}:
        return "unix_us"
    if u in {"us", "μs", "µs"}:
        return "us"
    if u == "s":
        return "s"
    if u == "ms":
        return "ms"
    return "unix_ms"


def detect_time_unit(values: pd.Series) -> str:
    """Guess how to interpret a raw timestamp column from value magnitude."""
    v = values.dropna().astype(float)
    if len(v) == 0:
        return "unix_ms"
    med = float(np.median(np.abs(v)))
    if med >= _UNIX_US_MIN:
        return "unix_us"
    if med >= _UNIX_MS_MIN:
        return "unix_ms"
    if med >= _UNIX_S_MIN:
        return "unix_s"
    if med >= _RELATIVE_MS_MIN:
        return "ms"
    return "s"


def raw_time_to_seconds(values: pd.Series, time_unit: str) -> pd.Series:
    """Convert a raw timestamp column to absolute seconds."""
    unit = normalize_time_unit(time_unit)
    vals = values.astype(float)
    if unit == "unix_ms":
        return vals / 1000.0
    if unit == "unix_us":
        return vals / 1_000_000.0
    if unit == "unix_s":
        return vals
    if unit == "ms":
        return vals / 1000.0
    if unit == "us":
        return vals / 1_000_000.0
    return vals


def preview_time_seconds(
    df: pd.DataFrame,
    time_col: str,
    time_unit: str,
    *,
    shift_to_zero: bool = True,
) -> np.ndarray:
    """Build a seconds time vector for plotting (processed or raw preview)."""
    if df is None or len(df) == 0:
        return np.array([])
    if "time_s" in df.columns:
        return df["time_s"].values.astype(float)
    if not time_col or time_col not in df.columns:
        return np.array([])
    t = raw_time_to_seconds(df[time_col], time_unit).values.astype(float)
    if shift_to_zero and len(t) > 0:
        t = t - t[0]
    return t


def seconds_to_display(values: np.ndarray, time_unit: str) -> Tuple[np.ndarray, str]:
    """Map ``time_s`` values to plot axis values and label (legacy helper)."""
    unit = normalize_time_unit(time_unit)
    if unit in {"unix_ms", "ms"}:
        return values * 1000.0, "Time (ms)"
    if unit in {"unix_us", "us"}:
        return values * 1_000_000.0, "Time (μs)"
    return values, "Time (s)"


def display_to_seconds(value: float, time_unit: str) -> float:
    """Convert a plot-axis time value to seconds (legacy helper)."""
    unit = normalize_time_unit(time_unit)
    if unit in {"unix_ms", "ms"}:
        return value / 1000.0
    if unit in {"unix_us", "us"}:
        return value / 1_000_000.0
    return value


def apply_channel_mapping(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """Rename DAQ channel IDs to sensor names using ``channel_config``."""
    channel_config = config.get("channel_config") or config.get("columns", {})
    if not channel_config:
        return df, {"applied": False, "mappings_found": 0}

    stats: Dict[str, Any] = {"applied": True, "mappings_found": 0, "mappings_applied": []}
    df_mapped = df.copy()
    rename_map: Dict[Any, str] = {}

    for raw_id, sensor_name in channel_config.items():
        for candidate in [raw_id, str(raw_id)]:
            if candidate in df_mapped.columns:
                rename_map[candidate] = sensor_name
                stats["mappings_applied"].append(f"{raw_id} -> {sensor_name}")
                stats["mappings_found"] += 1
                break
        else:
            try:
                int_id = int(raw_id)
                if int_id in df_mapped.columns:
                    rename_map[int_id] = sensor_name
                    stats["mappings_applied"].append(f"{raw_id} -> {sensor_name}")
                    stats["mappings_found"] += 1
            except (TypeError, ValueError):
                pass

    if rename_map:
        df_mapped = df_mapped.rename(columns=rename_map)
    return df_mapped, stats


def preprocess_time(
    df: pd.DataFrame,
    time_col: str,
    time_unit: str = "unix_ms",
    shift_to_zero: bool = True,
) -> pd.DataFrame:
    """Sort, deduplicate, and add ``time_s`` / ``time_ms`` from a raw time column."""
    df_proc = df.copy()
    if not time_col or time_col not in df_proc.columns:
        return df_proc

    df_proc = df_proc.sort_values(time_col).reset_index(drop=True)
    df_proc = df_proc.drop_duplicates(subset=[time_col], keep="first")
    df_proc["time_s"] = raw_time_to_seconds(df_proc[time_col], time_unit)
    if shift_to_zero:
        df_proc["time_s"] = df_proc["time_s"] - df_proc["time_s"].iloc[0]
    df_proc["time_ms"] = df_proc["time_s"] * 1000.0
    return df_proc


def handle_nan_values(
    df: pd.DataFrame,
    method: str = "interpolate+ffill",
    max_gap: int = 5,
) -> Tuple[pd.DataFrame, dict]:
    """Handle NaN values with interpolation, drop, or forward-fill."""
    stats: Dict[str, Any] = {
        "original_rows": len(df),
        "nan_counts": {},
        "method": method,
    }
    df_clean = df.copy()

    for col in df_clean.columns:
        nan_count = int(df_clean[col].isna().sum())
        if nan_count > 0:
            stats["nan_counts"][col] = nan_count
    stats["rows_affected"] = int(df_clean.isna().any(axis=1).sum())

    if method in ("interpolate", "interpolate+ffill"):
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method="linear", limit=max_gap)
        if method == "interpolate+ffill":
            df_clean = df_clean.ffill().bfill()
    elif method == "drop":
        df_clean = df_clean.dropna()
    elif method == "ffill":
        df_clean = df_clean.ffill().bfill()

    stats["final_rows"] = len(df_clean)
    return df_clean, stats


def resample_data(
    df: pd.DataFrame,
    target_rate_hz: float,
    time_col: str = "time_s",
) -> Tuple[pd.DataFrame, dict]:
    """Resample data to a uniform sample rate via linear interpolation."""
    if time_col not in df.columns:
        return df, {"error": f"Time column {time_col} not found"}

    stats: Dict[str, Any] = {
        "original_rows": len(df),
        "original_duration_s": float(df[time_col].max() - df[time_col].min()),
        "target_rate_hz": target_rate_hz,
    }

    dt = np.diff(df[time_col].values)
    if len(dt) > 0 and np.mean(dt) > 0:
        stats["original_mean_rate_hz"] = float(1.0 / np.mean(dt))

    t_min = float(df[time_col].min())
    t_max = float(df[time_col].max())
    if t_max <= t_min:
        return df, {**stats, "error": "Invalid time range"}

    step = 1.0 / target_rate_hz
    new_time = np.arange(t_min, t_max, step)
    if len(new_time) == 0:
        new_time = np.array([t_min])
    stats["resampled_rows"] = len(new_time)

    out: Dict[str, Any] = {time_col: new_time}
    if time_col == "time_s":
        out["time_ms"] = new_time * 1000.0

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in TIME_DERIVED or col == time_col:
            continue
        out[col] = np.interp(new_time, df[time_col].values, df[col].values)

    return pd.DataFrame(out), stats


def trim_time_window(
    df: pd.DataFrame,
    start_s: float,
    end_s: float,
    time_col: str = "time_s",
) -> Tuple[pd.DataFrame, dict]:
    """Keep only rows within ``[start_s, end_s]`` on the processed time axis."""
    if time_col not in df.columns:
        return df, {"error": f"Time column {time_col} not found"}

    lo = min(start_s, end_s)
    hi = max(start_s, end_s)
    mask = (df[time_col] >= lo) & (df[time_col] <= hi)
    trimmed = df.loc[mask].copy().reset_index(drop=True)

    stats = {
        "original_rows": len(df),
        "trimmed_rows": len(trimmed),
        "start_s": lo,
        "end_s": hi,
    }
    if "time_s" in trimmed.columns and len(trimmed) > 0:
        stats["duration_s"] = float(trimmed["time_s"].max() - trimmed["time_s"].min())
    return trimmed, stats


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    *,
    time_col: str,
    time_unit: str = "unix_ms",
    shift_to_zero: bool = True,
    nan_method: str = "interpolate+ffill",
    resample_hz: Optional[float] = None,
    trim_start_s: Optional[float] = None,
    trim_end_s: Optional[float] = None,
    config: Optional[dict] = None,
    apply_mapping: bool = False,
) -> Tuple[pd.DataFrame, dict, Optional[pd.DataFrame]]:
    """Run the full preprocessing pipeline and return processed data + stats.

    Returns:
        (processed_df, stats, df_before_trim) — ``df_before_trim`` is set when
        trimming was applied (full series for plot highlighting).
    """
    stats: Dict[str, Any] = {}
    df_proc = df.copy()
    df_before_trim: Optional[pd.DataFrame] = None

    if apply_mapping and config:
        df_proc, map_stats = apply_channel_mapping(df_proc, config)
        stats["mapping"] = map_stats

    df_proc = preprocess_time(df_proc, time_col, time_unit, shift_to_zero)
    stats["rows_after_time"] = len(df_proc)
    stats["detected_time_unit"] = normalize_time_unit(time_unit)

    if nan_method and nan_method != "none":
        df_proc, nan_stats = handle_nan_values(df_proc, nan_method)
        stats["nan"] = nan_stats

    if resample_hz and resample_hz > 0:
        df_proc, rs_stats = resample_data(df_proc, resample_hz)
        stats["resample"] = rs_stats

    if trim_start_s is not None and trim_end_s is not None:
        df_before_trim = df_proc.copy()
        df_proc, trim_stats = trim_time_window(df_proc, trim_start_s, trim_end_s)
        stats["trim"] = trim_stats

    stats["final_rows"] = len(df_proc)
    if "time_s" in df_proc.columns and len(df_proc) > 0:
        stats["duration_s"] = float(df_proc["time_s"].max() - df_proc["time_s"].min())
    return df_proc, stats, df_before_trim
