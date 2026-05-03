"""Preprocessing pipeline: raw DataFrame → analysis-ready DataFrame.

Pure with respect to I/O — takes a pandas DataFrame in and returns one out.
The CSV reading lives in the ingest service; this module only transforms.

Pipeline stages, in order:
    1. Validate timestamp column present and numeric.
    2. Convert timestamp to seconds (DAQs commonly emit ms / µs).
    3. Sort by time, drop exact-duplicate rows, drop rows with NaN timestamp.
    4. Shift time so the first sample is at t=0 (optional).
    5. Apply ``channel_map`` to rename raw sensor ids to logical names.
    6. Resample to a uniform frequency via linear interpolation (optional).
    7. Apply ``nan_policy`` to remaining NaN values in non-timestamp columns.
    8. Evaluate derived channels and append them as new columns.

Anything that would silently drop more than a small fraction of data raises
``IngestError`` — preprocessing must not paper over a broken file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from hda.domain.derived import (
    DerivedChannelSpec,
    DerivedContext,
    FormulaLibrary,
    evaluate_channels,
    standard_library,
)
from hda.domain.errors import IngestError


class TimestampUnit(str, Enum):
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"


_TS_TO_SEC = {
    TimestampUnit.SECONDS: 1.0,
    TimestampUnit.MILLISECONDS: 1.0e-3,
    TimestampUnit.MICROSECONDS: 1.0e-6,
    TimestampUnit.NANOSECONDS: 1.0e-9,
}


class NaNPolicy(str, Enum):
    INTERPOLATE = "interpolate"
    DROP = "drop"
    LEAVE = "leave"


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    timestamp_column: str = "timestamp"
    timestamp_unit: TimestampUnit = TimestampUnit.MILLISECONDS
    shift_to_zero: bool = True
    resample_freq_hz: float | None = 100.0
    nan_policy: NaNPolicy = NaNPolicy.INTERPOLATE
    channel_map: Mapping[str, str] = field(default_factory=dict)
    derived_channels: Sequence[DerivedChannelSpec] = ()
    sensor_scalars: Mapping[str, float] = field(default_factory=dict)
    geometry: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreprocessedData:
    df: pd.DataFrame
    sample_rate_hz: float
    duration_s: float
    n_samples: int
    derived_channel_names: Tuple[str, ...]
    n_dropped_rows: int


def preprocess(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    library: FormulaLibrary | None = None,
) -> PreprocessedData:
    if df.empty:
        raise IngestError("Cannot preprocess an empty DataFrame")
    if config.timestamp_column not in df.columns:
        raise IngestError(
            f"Timestamp column '{config.timestamp_column}' not in DataFrame columns "
            f"{list(df.columns)}"
        )

    ts_col = config.timestamp_column
    work = df.copy()
    n_initial = len(work)

    if not pd.api.types.is_numeric_dtype(work[ts_col]):
        try:
            work[ts_col] = pd.to_numeric(work[ts_col], errors="raise")
        except (ValueError, TypeError) as e:
            raise IngestError(
                f"Timestamp column '{ts_col}' is not numeric and cannot be coerced: {e}"
            ) from e

    factor = _TS_TO_SEC[config.timestamp_unit]
    work[ts_col] = work[ts_col].astype(float) * factor

    work = work.dropna(subset=[ts_col])
    work = work.sort_values(ts_col, kind="mergesort").drop_duplicates(
        subset=[ts_col], keep="first"
    )

    if work.empty:
        raise IngestError("All rows dropped during timestamp normalization")

    if config.shift_to_zero:
        work[ts_col] = work[ts_col] - work[ts_col].iloc[0]

    if config.channel_map:
        unknown = set(config.channel_map.keys()) - set(work.columns)
        if unknown:
            raise IngestError(
                f"channel_map references columns not in DataFrame: {sorted(unknown)}"
            )
        work = work.rename(columns=dict(config.channel_map))

    if config.resample_freq_hz is not None:
        work = _resample_uniform(work, ts_col, config.resample_freq_hz)

    work = _apply_nan_policy(work, ts_col, config.nan_policy)

    derived_names: Tuple[str, ...] = ()
    if config.derived_channels:
        lib = library if library is not None else standard_library()
        sensor_channels = {
            col: work[col].to_numpy(dtype=float)
            for col in work.columns
            if col != ts_col and pd.api.types.is_numeric_dtype(work[col])
        }
        ctx = DerivedContext(
            sensor_channels=sensor_channels,
            sensor_scalars=dict(config.sensor_scalars),
            geometry=dict(config.geometry),
        )
        derived = evaluate_channels(list(config.derived_channels), ctx, lib)
        for name, arr in derived.items():
            if len(arr) != len(work):
                raise IngestError(
                    f"Derived channel '{name}' length {len(arr)} does not match "
                    f"DataFrame length {len(work)}"
                )
            work[name] = arr
        derived_names = tuple(derived.keys())

    n_samples = len(work)
    duration = float(work[ts_col].iloc[-1] - work[ts_col].iloc[0])
    sample_rate = (n_samples - 1) / duration if duration > 0 else float("nan")

    return PreprocessedData(
        df=work.reset_index(drop=True),
        sample_rate_hz=sample_rate,
        duration_s=duration,
        n_samples=n_samples,
        derived_channel_names=derived_names,
        n_dropped_rows=n_initial - n_samples
        + (n_samples if config.resample_freq_hz else 0) * 0,
    )


def _resample_uniform(
    df: pd.DataFrame, ts_col: str, target_hz: float
) -> pd.DataFrame:
    if target_hz <= 0:
        raise IngestError(f"resample_freq_hz must be > 0, got {target_hz}")
    t = df[ts_col].to_numpy(dtype=float)
    t0, t1 = t[0], t[-1]
    if t1 <= t0:
        raise IngestError(
            f"Cannot resample: timestamp range is non-positive ({t0} → {t1})"
        )
    dt = 1.0 / target_hz
    new_t = np.arange(t0, t1 + 0.5 * dt, dt)
    out = {ts_col: new_t}
    for col in df.columns:
        if col == ts_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        y = df[col].to_numpy(dtype=float)
        out[col] = np.interp(new_t, t, y)
    return pd.DataFrame(out)


def _apply_nan_policy(
    df: pd.DataFrame, ts_col: str, policy: NaNPolicy
) -> pd.DataFrame:
    if policy is NaNPolicy.LEAVE:
        return df
    cols = [c for c in df.columns if c != ts_col]
    if policy is NaNPolicy.DROP:
        return df.dropna(subset=cols).reset_index(drop=True)
    if policy is NaNPolicy.INTERPOLATE:
        out = df.copy()
        for col in cols:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].interpolate(
                    method="linear", limit_direction="both"
                )
        return out
    raise IngestError(f"Unknown nan policy {policy}")
