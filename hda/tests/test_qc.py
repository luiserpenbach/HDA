"""QC checks."""

from __future__ import annotations

import numpy as np

from hda.domain.qc import (
    QCConfig,
    SensorRange,
    check_flatline,
    check_nan_ratio,
    check_sample_rate_stability,
    check_sensor_range,
    check_timestamp_monotonic,
    run_qc,
)
from hda.domain.types import QCStatus


def test_monotonic_passes_strictly_increasing():
    res = check_timestamp_monotonic(np.array([0.0, 0.1, 0.2, 0.3]))
    assert res.status is QCStatus.PASS


def test_monotonic_fails_on_duplicate_or_backward():
    res = check_timestamp_monotonic(np.array([0.0, 0.1, 0.1, 0.2]))
    assert res.status is QCStatus.FAIL
    assert res.blocking is True

    res = check_timestamp_monotonic(np.array([0.0, 0.2, 0.1]))
    assert res.status is QCStatus.FAIL


def test_sample_rate_within_tolerance():
    t = np.linspace(0.0, 1.0, 101)  # 100 Hz
    res = check_sample_rate_stability(t, expected_hz=100.0, tolerance=0.05)
    assert res.status is QCStatus.PASS


def test_sample_rate_warns_when_off():
    t = np.linspace(0.0, 1.0, 51)  # 50 Hz
    res = check_sample_rate_stability(t, expected_hz=100.0, tolerance=0.05)
    assert res.status is QCStatus.WARN


def test_nan_ratio_passes_clean():
    res = check_nan_ratio(np.array([1.0, 2.0, 3.0]), "x")
    assert res.status is QCStatus.PASS


def test_nan_ratio_warns_below_threshold():
    arr = np.concatenate([np.ones(98), np.array([np.nan, np.nan])])
    res = check_nan_ratio(arr, "x", max_ratio=0.05)
    assert res.status is QCStatus.WARN


def test_nan_ratio_fails_above_threshold():
    arr = np.concatenate([np.ones(80), np.full(20, np.nan)])
    res = check_nan_ratio(arr, "x", max_ratio=0.05)
    assert res.status is QCStatus.FAIL
    assert res.blocking is True


def test_flatline_passes_normal_signal():
    rng = np.random.default_rng(0)
    arr = 10.0 + rng.standard_normal(500)
    res = check_flatline(arr, "PT-01", window_n=100)
    assert res.status is QCStatus.PASS


def test_flatline_fails_on_constant_segment():
    arr = np.concatenate([np.full(200, 5.0), 5.0 + np.random.default_rng(0).standard_normal(200) * 0.01])
    res = check_flatline(arr, "PT-01", window_n=100, std_threshold=1e-6)
    assert res.status is QCStatus.FAIL
    assert res.blocking is True


def test_sensor_range_pass_inside():
    res = check_sensor_range(
        np.array([1.0, 2.0, 3.0]), "x", SensorRange(min_value=0.0, max_value=10.0)
    )
    assert res.status is QCStatus.PASS


def test_sensor_range_fail_below_min():
    res = check_sensor_range(
        np.array([-1.0, 2.0, 3.0]), "x", SensorRange(min_value=0.0)
    )
    assert res.status is QCStatus.FAIL


def test_sensor_range_fail_above_max():
    res = check_sensor_range(
        np.array([1.0, 2.0, 99.0]), "x", SensorRange(max_value=50.0)
    )
    assert res.status is QCStatus.FAIL


def test_run_qc_aggregates_findings_and_blocking():
    t = np.linspace(0.0, 1.0, 101)
    chans = {
        "PT-up": np.full(101, 10.0) + 0.01 * np.random.default_rng(0).standard_normal(101),
        "PT-down": np.full(101, np.nan),  # all NaN -> nan ratio fails
    }
    cfg = QCConfig(expected_sample_rate_hz=100.0, nan_max_ratio=0.05)
    rep = run_qc(t, chans, cfg)
    assert rep.passed is False
    assert any(f.check_name.startswith("nan_ratio:PT-down") for f in rep.blocking_failures)
