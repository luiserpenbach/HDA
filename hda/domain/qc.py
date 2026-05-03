"""Quality-control checks.

Each check is a pure function: numpy in, ``QCFinding`` out. ``run_qc``
orchestrates a configurable suite and returns a ``QCReport``.

The legacy app's ``skip_qc=True`` escape hatch is gone — the analysis
service must consult ``QCReport.passed`` and refuse to advance to
``ANALYZED`` when blocking findings exist. NEEDS_REVIEW is the explicit
state for non-blocking warnings the operator must clear.

Findings flagged with ``blocking=True`` block the run; ``blocking=False``
findings produce warnings the dashboard surfaces. This is the only knob
that decides "do we hard-fail vs. ask the operator" — no thresholds
hidden in the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from hda.domain.types import QCFinding, QCReport, QCStatus


@dataclass(frozen=True, slots=True)
class SensorRange:
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass(frozen=True, slots=True)
class QCConfig:
    expected_sample_rate_hz: Optional[float] = None
    sample_rate_tolerance: float = 0.05
    nan_max_ratio: float = 0.05
    flatline_window_n: int = 100
    flatline_std_threshold: float = 1e-6
    sensor_ranges: Mapping[str, SensorRange] = field(default_factory=dict)


def check_timestamp_monotonic(time_s: np.ndarray) -> QCFinding:
    if time_s.size < 2:
        return QCFinding(
            "timestamp_monotonic",
            QCStatus.FAIL,
            "Fewer than 2 samples — cannot evaluate monotonicity",
            blocking=True,
        )
    diffs = np.diff(time_s)
    if np.any(diffs <= 0):
        n_bad = int(np.sum(diffs <= 0))
        return QCFinding(
            "timestamp_monotonic",
            QCStatus.FAIL,
            f"{n_bad} non-monotonic timestamp(s) detected",
            blocking=True,
        )
    return QCFinding("timestamp_monotonic", QCStatus.PASS, "", blocking=True)


def check_sample_rate_stability(
    time_s: np.ndarray, expected_hz: float, tolerance: float = 0.05
) -> QCFinding:
    if expected_hz <= 0:
        return QCFinding(
            "sample_rate_stability",
            QCStatus.FAIL,
            f"Invalid expected_hz {expected_hz}",
            blocking=False,
        )
    diffs = np.diff(time_s)
    if diffs.size == 0:
        return QCFinding(
            "sample_rate_stability",
            QCStatus.FAIL,
            "Cannot evaluate: fewer than 2 samples",
            blocking=False,
        )
    actual_hz = 1.0 / float(np.median(diffs))
    rel = abs(actual_hz - expected_hz) / expected_hz
    if rel <= tolerance:
        return QCFinding(
            "sample_rate_stability",
            QCStatus.PASS,
            f"{actual_hz:.2f} Hz vs expected {expected_hz:.2f} Hz",
            blocking=False,
        )
    return QCFinding(
        "sample_rate_stability",
        QCStatus.WARN,
        f"{actual_hz:.2f} Hz off from expected {expected_hz:.2f} Hz "
        f"by {rel*100:.1f}%",
        blocking=False,
    )


def check_nan_ratio(
    series: np.ndarray, channel: str, max_ratio: float = 0.05
) -> QCFinding:
    if series.size == 0:
        return QCFinding(
            f"nan_ratio:{channel}",
            QCStatus.FAIL,
            "Empty series",
            blocking=True,
        )
    ratio = float(np.mean(np.isnan(series)))
    if ratio == 0.0:
        return QCFinding(
            f"nan_ratio:{channel}", QCStatus.PASS, "", blocking=False
        )
    if ratio <= max_ratio:
        return QCFinding(
            f"nan_ratio:{channel}",
            QCStatus.WARN,
            f"{ratio*100:.2f}% NaN (≤ {max_ratio*100:.0f}% allowed)",
            blocking=False,
        )
    return QCFinding(
        f"nan_ratio:{channel}",
        QCStatus.FAIL,
        f"{ratio*100:.2f}% NaN exceeds {max_ratio*100:.0f}% threshold",
        blocking=True,
    )


def check_flatline(
    series: np.ndarray,
    channel: str,
    window_n: int = 100,
    std_threshold: float = 1e-6,
) -> QCFinding:
    if series.size < window_n:
        return QCFinding(
            f"flatline:{channel}",
            QCStatus.PASS,
            "Series shorter than flatline window — skipped",
            blocking=False,
        )
    finite = series[np.isfinite(series)]
    if finite.size < window_n:
        return QCFinding(
            f"flatline:{channel}",
            QCStatus.WARN,
            "Too few finite samples to evaluate flatline",
            blocking=False,
        )
    rolling_std = _rolling_std(finite, window_n)
    if np.any(rolling_std < std_threshold):
        return QCFinding(
            f"flatline:{channel}",
            QCStatus.FAIL,
            f"{channel} stuck at constant value within a {window_n}-sample window",
            blocking=True,
        )
    return QCFinding(
        f"flatline:{channel}", QCStatus.PASS, "", blocking=False
    )


def check_sensor_range(
    series: np.ndarray, channel: str, sensor_range: SensorRange
) -> QCFinding:
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return QCFinding(
            f"range:{channel}",
            QCStatus.FAIL,
            "All values are NaN",
            blocking=True,
        )
    smin, smax = float(np.nanmin(series)), float(np.nanmax(series))
    if sensor_range.min_value is not None and smin < sensor_range.min_value:
        return QCFinding(
            f"range:{channel}",
            QCStatus.FAIL,
            f"{channel} min {smin:.4g} < allowed {sensor_range.min_value:.4g}",
            blocking=True,
        )
    if sensor_range.max_value is not None and smax > sensor_range.max_value:
        return QCFinding(
            f"range:{channel}",
            QCStatus.FAIL,
            f"{channel} max {smax:.4g} > allowed {sensor_range.max_value:.4g}",
            blocking=True,
        )
    return QCFinding(
        f"range:{channel}",
        QCStatus.PASS,
        f"{channel} in [{smin:.4g}, {smax:.4g}]",
        blocking=False,
    )


def run_qc(
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    config: QCConfig,
) -> QCReport:
    """Run the standard QC suite and return a single QCReport."""
    findings: list[QCFinding] = [check_timestamp_monotonic(time_s)]

    if config.expected_sample_rate_hz is not None:
        findings.append(
            check_sample_rate_stability(
                time_s,
                config.expected_sample_rate_hz,
                config.sample_rate_tolerance,
            )
        )

    for name, series in channels.items():
        findings.append(check_nan_ratio(series, name, config.nan_max_ratio))
        findings.append(
            check_flatline(
                series,
                name,
                config.flatline_window_n,
                config.flatline_std_threshold,
            )
        )
        sr = config.sensor_ranges.get(name)
        if sr is not None:
            findings.append(check_sensor_range(series, name, sr))

    return QCReport(findings=tuple(findings))


def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    kernel = np.ones(w, dtype=float) / w
    mean = np.convolve(arr, kernel, mode="valid")
    sq_mean = np.convolve(arr.astype(float) ** 2, kernel, mode="valid")
    var = np.maximum(sq_mean - mean**2, 0.0)
    return np.sqrt(var)
