"""Steady-state detection.

CV-based detector: finds the longest contiguous window where the rolling
coefficient of variation of the chosen signal stays below a threshold.
Returns a SteadyWindow with a confidence score so the dashboard can decide
whether to auto-confirm or surface the test for operator review.

A `simple` fallback returns the middle 50% of the trace; the analysis
service uses it when the CV detector returns no window (so the operator
can still tweak it manually rather than the test failing outright).

``window_stats`` is the live-preview primitive: takes a DataFrame, a
timestamp column, and a [start_s, end_s] window; returns per-channel
mean / std / n / cv. The interactive drag-handle widget calls this on
every drag tick, so it stays pure-numpy with no allocations beyond what
the slice itself requires.

Pure: numpy in, SteadyWindow out. No I/O, no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from hda.domain.errors import AnalysisError
from hda.domain.types import SteadyWindow


@dataclass(frozen=True, slots=True)
class ChannelStats:
    mean: float
    std: float
    n: int
    cv: float

    @property
    def is_finite(self) -> bool:
        import math

        return math.isfinite(self.mean) and math.isfinite(self.std)


def window_stats(
    df: "pd.DataFrame",
    start_s: float,
    end_s: float,
    timestamp_column: str = "timestamp",
) -> Mapping[str, ChannelStats]:
    """Per-channel mean/std/n/cv over the closed slice [start_s, end_s].

    Channels with zero finite samples in the window get NaN mean and std
    and n=0 (rather than raising) so the live preview stays fluid even
    when the operator drags both handles together.
    """
    if end_s <= start_s:
        raise AnalysisError(
            f"window_stats: end_s ({end_s}) must exceed start_s ({start_s})"
        )
    if timestamp_column not in df.columns:
        raise AnalysisError(
            f"window_stats: timestamp column '{timestamp_column}' not in DataFrame"
        )
    ts = df[timestamp_column].to_numpy(dtype=float)
    mask = (ts >= start_s) & (ts <= end_s)
    out: dict[str, ChannelStats] = {}
    for col in df.columns:
        if col == timestamp_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        arr = df[col].to_numpy(dtype=float)[mask]
        finite = arr[np.isfinite(arr)]
        n = int(finite.size)
        if n == 0:
            out[col] = ChannelStats(
                mean=float("nan"), std=float("nan"), n=0, cv=float("nan")
            )
            continue
        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
        cv = float("nan") if mean == 0.0 else abs(std / mean)
        out[col] = ChannelStats(mean=mean, std=std, n=n, cv=cv)
    return out


def detect_cv(
    signal: np.ndarray,
    time_s: np.ndarray,
    cv_threshold: float = 0.02,
    window_s: float = 1.0,
    min_duration_s: float = 2.0,
) -> Optional[SteadyWindow]:
    """Return the longest window where rolling CV ≤ ``cv_threshold``.

    Confidence is a function of how stable the window is and how long it is
    relative to the test:

        confidence = clamp01(1 - mean_cv/threshold) * clamp01(duration/test/0.5)

    so a window covering ≥50% of the trace at a CV well below threshold
    scores near 1.0; a marginal window scores near 0.

    Returns ``None`` when no window of at least ``min_duration_s`` qualifies.
    """
    _validate_signal_and_time(signal, time_s)
    if cv_threshold <= 0:
        raise AnalysisError(f"cv_threshold must be > 0, got {cv_threshold}")
    n = signal.size
    dt = float(np.median(np.diff(time_s)))
    if dt <= 0:
        raise AnalysisError("Cannot compute steady-state on non-monotonic time")

    window_n = max(2, int(round(window_s / dt)))
    if window_n >= n:
        return None

    rolling_mean = _rolling_mean(signal, window_n)
    rolling_std = _rolling_std(signal, window_n, rolling_mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(rolling_mean != 0.0, rolling_std / np.abs(rolling_mean), np.inf)

    mask = cv <= cv_threshold
    start_i, end_i = _longest_true_run(mask)
    if start_i is None:
        return None

    start_s = float(time_s[start_i])
    end_s = float(time_s[end_i])
    duration = end_s - start_s
    if duration < min_duration_s:
        return None

    in_window_cv = cv[start_i : end_i + 1]
    finite = in_window_cv[np.isfinite(in_window_cv)]
    mean_cv = float(np.mean(finite)) if finite.size else cv_threshold
    test_duration = float(time_s[-1] - time_s[0])
    coverage = duration / test_duration if test_duration > 0 else 0.0

    stability = max(0.0, 1.0 - mean_cv / cv_threshold)
    coverage_score = min(1.0, coverage / 0.5)
    confidence = max(0.0, min(1.0, stability * coverage_score))

    return SteadyWindow(
        start_s=start_s,
        end_s=end_s,
        method="cv",
        confidence=confidence,
    )


def detect_simple(
    time_s: np.ndarray,
    fraction: float = 0.5,
) -> SteadyWindow:
    """Return the centered ``fraction`` of the trace as a fallback window."""
    if not 0.0 < fraction <= 1.0:
        raise AnalysisError(f"fraction must be in (0,1], got {fraction}")
    if time_s.size < 2:
        raise AnalysisError("Need at least 2 samples")
    t0, t1 = float(time_s[0]), float(time_s[-1])
    span = t1 - t0
    if span <= 0:
        raise AnalysisError("Non-monotonic time")
    margin = (1.0 - fraction) * 0.5 * span
    return SteadyWindow(
        start_s=t0 + margin,
        end_s=t1 - margin,
        method="simple",
        confidence=0.0,
    )


def _validate_signal_and_time(signal: np.ndarray, time_s: np.ndarray) -> None:
    if signal.ndim != 1 or time_s.ndim != 1:
        raise AnalysisError("signal and time must be 1-D")
    if signal.size != time_s.size:
        raise AnalysisError(
            f"signal length {signal.size} != time length {time_s.size}"
        )
    if signal.size < 4:
        raise AnalysisError("Need at least 4 samples for steady-state detection")


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(arr, kernel, mode="same")


def _rolling_std(arr: np.ndarray, w: int, mean: np.ndarray) -> np.ndarray:
    sq = arr.astype(float) ** 2
    kernel = np.ones(w, dtype=float) / w
    mean_sq = np.convolve(sq, kernel, mode="same")
    var = np.maximum(mean_sq - mean**2, 0.0)
    return np.sqrt(var)


def _longest_true_run(mask: np.ndarray) -> tuple[Optional[int], Optional[int]]:
    if not mask.any():
        return None, None
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    if starts.size == 0:
        return None, None
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best])
