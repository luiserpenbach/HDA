"""Pure data-transformation helpers for the analytics screen."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from hda.ui.analytics_data import (
    HistoryPoint,
    HistorySummary,
    format_summary,
    history_to_points,
    summarize,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_empty_df_yields_empty_points_and_zero_summary():
    assert history_to_points(pd.DataFrame()) == []
    s = summarize([])
    assert s.n == 0
    assert math.isnan(s.mean) and math.isnan(s.std)
    assert math.isnan(s.min) and math.isnan(s.max)
    assert "No matching" in format_summary(s)


def test_history_to_points_sorts_by_time():
    df = _df(
        [
            {
                "test_run_id": "b",
                "campaign_id": "C1",
                "serial_number": "S",
                "persisted_at": "2026-02-01T00:00:00",
                "discovered_at": "2026-02-01T00:00:00",
                "value": 0.7,
                "uncertainty": 0.01,
                "unit": "",
            },
            {
                "test_run_id": "a",
                "campaign_id": "C1",
                "serial_number": "S",
                "persisted_at": "2026-01-01T00:00:00",
                "discovered_at": "2026-01-01T00:00:00",
                "value": 0.6,
                "uncertainty": 0.01,
                "unit": "",
            },
        ]
    )
    pts = history_to_points(df)
    assert [p.test_run_id for p in pts] == ["a", "b"]
    assert pts[0].timestamp_unix < pts[1].timestamp_unix


def test_history_to_points_falls_back_to_discovered_at():
    df = _df(
        [
            {
                "test_run_id": "x",
                "campaign_id": "C1",
                "serial_number": "S",
                "persisted_at": None,
                "discovered_at": "2026-03-15T12:00:00",
                "value": 1.0,
                "uncertainty": 0.0,
                "unit": "",
            }
        ]
    )
    pts = history_to_points(df)
    assert len(pts) == 1
    assert pts[0].timestamp_iso == "2026-03-15T12:00:00"


def test_history_to_points_drops_unparseable_timestamp():
    df = _df(
        [
            {
                "test_run_id": "x",
                "campaign_id": "C1",
                "serial_number": "S",
                "persisted_at": "not-a-date",
                "discovered_at": None,
                "value": 1.0,
                "uncertainty": 0.0,
                "unit": "",
            }
        ]
    )
    assert history_to_points(df) == []


def test_summarize_basic_stats():
    pts = [
        HistoryPoint("a", "C1", "S", 1.0, "2026-01-01T00:00:00", 0.6, 0.01),
        HistoryPoint("b", "C1", "S", 2.0, "2026-01-02T00:00:00", 0.7, 0.01),
        HistoryPoint("c", "C1", "S", 3.0, "2026-01-03T00:00:00", 0.8, 0.01),
    ]
    s = summarize(pts)
    assert s.n == 3
    assert s.mean == pytest.approx(0.7)
    assert s.std == pytest.approx(0.1, rel=1e-6)
    assert s.min == 0.6
    assert s.max == 0.8


def test_summarize_single_point_zero_std():
    pts = [HistoryPoint("a", "C1", "S", 1.0, "ts", 5.0, 0.1)]
    s = summarize(pts)
    assert s.n == 1
    assert s.mean == 5.0
    assert s.std == 0.0


def test_format_summary_with_unit():
    s = HistorySummary(n=3, mean=0.65, std=0.012, min=0.63, max=0.68)
    text = format_summary(s, unit="")
    assert "n=3" in text
    assert "mean=0.65" in text
    assert "std=0.012" in text


def test_format_summary_cv_nan_when_mean_zero():
    s = HistorySummary(n=3, mean=0.0, std=0.5, min=-0.5, max=0.5)
    text = format_summary(s)
    # cv unreliable when mean=0 — should not blow up
    assert "cv=—" in text
