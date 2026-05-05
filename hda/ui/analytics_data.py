"""Pure helpers for the analytics screen.

The analytics window translates a raw cross-campaign DataFrame into
plotted points and a one-line summary. Keeping the transformation pure
lets it be unit-tested without a Qt event loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Mapping, Sequence

import math

import pandas as pd


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    test_run_id: str
    campaign_id: str
    serial_number: str
    timestamp_unix: float  # seconds since epoch — ready for pyqtgraph DateAxisItem
    timestamp_iso: str
    value: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class HistorySummary:
    n: int
    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def empty(cls) -> "HistorySummary":
        nan = float("nan")
        return cls(n=0, mean=nan, std=nan, min=nan, max=nan)


def history_to_points(df: pd.DataFrame) -> List[HistoryPoint]:
    """Convert a hardware_history DataFrame into a list of HistoryPoints,
    sorted by time. Rows with missing or unparseable timestamps are dropped.
    """
    if df.empty:
        return []
    out: list[HistoryPoint] = []
    for _, row in df.iterrows():
        ts_iso = row.get("persisted_at") or row.get("discovered_at")
        if not ts_iso:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_iso))
        except ValueError:
            continue
        out.append(
            HistoryPoint(
                test_run_id=str(row["test_run_id"]),
                campaign_id=str(row["campaign_id"]),
                serial_number=str(row["serial_number"]),
                timestamp_unix=ts.timestamp(),
                timestamp_iso=str(ts_iso),
                value=float(row["value"]),
                uncertainty=float(row["uncertainty"]),
            )
        )
    out.sort(key=lambda p: p.timestamp_unix)
    return out


def summarize(points: Sequence[HistoryPoint]) -> HistorySummary:
    """Mean/std/min/max over the values. ``std`` uses ddof=1 (sample std)."""
    if not points:
        return HistorySummary.empty()
    values = [p.value for p in points]
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return HistorySummary(
        n=n, mean=mean, std=std, min=min(values), max=max(values)
    )


def format_summary(summary: HistorySummary, unit: str = "") -> str:
    if summary.n == 0:
        return "No matching tests."
    cv = (
        100.0 * summary.std / abs(summary.mean)
        if summary.mean != 0
        else float("nan")
    )
    cv_str = "—" if not math.isfinite(cv) else f"{cv:.2f}%"
    suffix = f" {unit}" if unit else ""
    return (
        f"n={summary.n}   mean={summary.mean:.6g}{suffix}   "
        f"std={summary.std:.4g}   cv={cv_str}   "
        f"range=[{summary.min:.6g}, {summary.max:.6g}]{suffix}"
    )
