"""Tests for hda.preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hda.preprocessing import (
    detect_time_unit,
    display_to_seconds,
    preprocess_time,
    preview_time_seconds,
    resample_data,
    run_preprocessing_pipeline,
    seconds_to_display,
    trim_time_window,
)


def test_preprocess_time_ms_to_seconds():
    df = pd.DataFrame({"timestamp": [0, 500, 1000, 1780], "P": [1, 2, 3, 4]})
    out = preprocess_time(df, "timestamp", "ms", shift_to_zero=True)
    assert np.isclose(out["time_s"].iloc[-1], 1.78)
    assert np.isclose(out["time_ms"].iloc[-1], 1780.0)


def test_unix_ms_epoch_to_relative_seconds():
    t0 = 1_716_835_200_000.0
    df = pd.DataFrame({"timestamp": [t0, t0 + 1000, t0 + 5000], "P": [1, 2, 3]})
    out = preprocess_time(df, "timestamp", "unix_ms", shift_to_zero=True)
    assert np.isclose(out["time_s"].iloc[0], 0.0)
    assert np.isclose(out["time_s"].iloc[-1], 5.0)


def test_detect_time_unit_unix_ms():
    s = pd.Series([1.7168352e12, 1.716835201e12, 1.716835205e12])
    assert detect_time_unit(s) == "unix_ms"


def test_detect_time_unit_unix_s():
    s = pd.Series([1.7168352e9, 1.716835201e9, 1.716835205e9])
    assert detect_time_unit(s) == "unix_s"


def test_preview_time_seconds_unix_ms():
    t0 = 1_716_835_200_000.0
    df = pd.DataFrame({"timestamp": [t0, t0 + 2500], "P": [1, 2]})
    t = preview_time_seconds(df, "timestamp", "unix_ms", shift_to_zero=True)
    assert np.isclose(t[0], 0.0)
    assert np.isclose(t[-1], 2.5)


def test_seconds_to_display_keeps_seconds_for_unix_s():
    t = np.array([0.0, 1.0, 1.78])
    vals, label = seconds_to_display(t, "unix_s")
    assert label == "Time (s)"
    assert np.isclose(vals[-1], 1.78)


def test_resample_produces_uniform_grid():
    t = np.linspace(0, 1, 11)
    df = pd.DataFrame({"time_s": t, "time_ms": t * 1000, "P": np.sin(t)})
    out, stats = resample_data(df, 100.0)
    assert stats["resampled_rows"] == len(out)
    assert len(out) > 50
    diffs = np.diff(out["time_s"].values)
    assert np.allclose(diffs, diffs[0], rtol=1e-6)


def test_trim_time_window():
    df = pd.DataFrame({"time_s": [0, 0.5, 1.0, 1.5, 2.0], "P": [1, 2, 3, 4, 5]})
    out, stats = trim_time_window(df, 0.4, 1.6)
    assert stats["trimmed_rows"] == 3
    assert list(out["P"]) == [2, 3, 4]


def test_run_preprocessing_pipeline_with_resample_and_trim():
    df = pd.DataFrame({"t": np.arange(0, 2000, 10), "sig": np.random.default_rng(0).random(200)})
    out, stats, _ = run_preprocessing_pipeline(
        df,
        time_col="t",
        time_unit="ms",
        resample_hz=100.0,
        trim_start_s=0.2,
        trim_end_s=1.5,
    )
    assert "time_s" in out.columns
    assert stats["final_rows"] == len(out)
    assert out["time_s"].min() >= 0.2 - 1e-9
    assert out["time_s"].max() <= 1.5 + 1e-9


def test_display_to_seconds_roundtrip():
    assert display_to_seconds(1780.0, "ms") == 1.78
