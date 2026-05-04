"""window_stats: pure live-preview primitive."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hda.domain.errors import AnalysisError
from hda.domain.steady_state import window_stats


def _df(n: int = 1001) -> pd.DataFrame:
    t = np.linspace(0.0, 10.0, n)
    return pd.DataFrame(
        {
            "timestamp": t,
            "PT-up": 10.0 + 0.0 * t,  # exactly constant
            "PT-down": np.linspace(0.0, 5.0, n),  # linear ramp
        }
    )


def test_returns_per_channel_stats_for_constant_signal():
    s = window_stats(_df(), 2.0, 4.0)
    assert "PT-up" in s and "PT-down" in s and "timestamp" not in s
    assert s["PT-up"].mean == pytest.approx(10.0)
    assert s["PT-up"].std == pytest.approx(0.0)
    assert s["PT-up"].n > 0
    assert s["PT-up"].cv == 0.0


def test_returns_per_channel_stats_for_linear_ramp():
    s = window_stats(_df(), 0.0, 10.0)
    # mean of [0..5] is 2.5
    assert s["PT-down"].mean == pytest.approx(2.5, abs=0.01)
    assert s["PT-down"].std > 0


def test_window_outside_data_yields_zero_n():
    s = window_stats(_df(), 100.0, 200.0)
    assert s["PT-up"].n == 0
    assert math.isnan(s["PT-up"].mean)


def test_invalid_window_raises():
    with pytest.raises(AnalysisError):
        window_stats(_df(), 5.0, 5.0)
    with pytest.raises(AnalysisError):
        window_stats(_df(), 5.0, 4.0)


def test_missing_timestamp_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(AnalysisError):
        window_stats(df, 0.0, 1.0)


def test_skips_non_numeric_columns():
    df = pd.DataFrame(
        {
            "timestamp": np.linspace(0.0, 1.0, 11),
            "PT-up": np.full(11, 10.0),
            "label": ["a"] * 11,
        }
    )
    s = window_stats(df, 0.0, 1.0)
    assert "label" not in s
    assert "PT-up" in s


def test_cv_is_nan_when_mean_is_zero():
    df = pd.DataFrame(
        {
            "timestamp": np.linspace(0.0, 1.0, 11),
            "x": np.linspace(-5.0, 5.0, 11),  # mean ~ 0
        }
    )
    s = window_stats(df, 0.0, 1.0)
    assert math.isnan(s["x"].cv) or abs(s["x"].mean) < 1e-12
