"""Steady-state detection."""

from __future__ import annotations

import numpy as np
import pytest

from hda.domain.errors import AnalysisError
from hda.domain.steady_state import detect_cv, detect_simple


def _time(n: int, dt: float = 0.01) -> np.ndarray:
    return np.arange(n) * dt


def test_detect_cv_finds_long_steady_segment():
    # 1s ramp -> 5s flat -> 1s ramp at 100 Hz
    n_ramp, n_flat = 100, 500
    t = _time(2 * n_ramp + n_flat)
    sig = np.concatenate(
        [
            np.linspace(0.0, 10.0, n_ramp),
            np.full(n_flat, 10.0) + 0.001 * np.random.default_rng(0).standard_normal(n_flat),
            np.linspace(10.0, 0.0, n_ramp),
        ]
    )
    win = detect_cv(sig, t, cv_threshold=0.005, window_s=0.2, min_duration_s=2.0)
    assert win is not None
    assert win.duration_s >= 2.0
    # Found window should sit inside the flat segment.
    assert win.start_s >= (n_ramp - 5) * 0.01
    assert win.end_s <= (n_ramp + n_flat + 5) * 0.01
    assert 0.0 < win.confidence <= 1.0


def test_detect_cv_returns_none_on_pure_ramp():
    t = _time(500)
    sig = np.linspace(0.0, 100.0, 500)
    win = detect_cv(sig, t, cv_threshold=0.001, min_duration_s=2.0)
    assert win is None


def test_detect_cv_high_confidence_when_window_dominates():
    # 90% flat
    n = 1000
    t = _time(n)
    sig = np.full(n, 5.0) + 1e-6 * np.random.default_rng(1).standard_normal(n)
    sig[:50] = np.linspace(0.0, 5.0, 50)
    sig[-50:] = np.linspace(5.0, 0.0, 50)
    win = detect_cv(sig, t, cv_threshold=0.01, window_s=0.2, min_duration_s=2.0)
    assert win is not None
    assert win.confidence > 0.8


def test_detect_cv_rejects_invalid_threshold():
    t = _time(100)
    with pytest.raises(AnalysisError):
        detect_cv(np.zeros(100), t, cv_threshold=0.0)


def test_detect_cv_rejects_short_signal():
    t = _time(3)
    with pytest.raises(AnalysisError):
        detect_cv(np.zeros(3), t)


def test_detect_simple_returns_centered_half():
    t = _time(1000)
    win = detect_simple(t, fraction=0.5)
    assert win.confidence == 0.0
    assert win.method == "simple"
    assert win.start_s == pytest.approx(t[-1] * 0.25, abs=0.005)
    assert win.end_s == pytest.approx(t[-1] * 0.75, abs=0.005)


def test_detect_simple_invalid_fraction():
    t = _time(100)
    with pytest.raises(AnalysisError):
        detect_simple(t, fraction=0.0)
    with pytest.raises(AnalysisError):
        detect_simple(t, fraction=1.5)
