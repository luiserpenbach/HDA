"""Uncertainty propagation: SensorUncertainty + analytical Jacobian + Monte Carlo."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hda.domain.errors import AnalysisError, ConfigError
from hda.domain.uncertainty import (
    SensorUncertainty,
    UncertaintyKind,
    propagate_analytical,
    propagate_monte_carlo,
)


def test_sensor_absolute_uncertainty():
    s = SensorUncertainty(UncertaintyKind.ABSOLUTE, 0.1)
    assert s.standard_uncertainty(reading=12.0) == 0.1
    assert s.standard_uncertainty(reading=-5.0) == 0.1


def test_sensor_relative_uncertainty_uses_magnitude_of_reading():
    s = SensorUncertainty(UncertaintyKind.RELATIVE, 0.005)
    assert s.standard_uncertainty(reading=200.0) == pytest.approx(1.0)
    assert s.standard_uncertainty(reading=-10.0) == pytest.approx(0.05)


def test_sensor_percent_fs_requires_full_scale():
    with pytest.raises(ConfigError):
        SensorUncertainty(UncertaintyKind.PERCENT_FS, 0.001)


def test_sensor_percent_fs_uses_full_scale():
    s = SensorUncertainty(UncertaintyKind.PERCENT_FS, 0.001, full_scale=100.0)
    # 0.1% of 100 -> 0.1, regardless of reading
    assert s.standard_uncertainty(reading=10.0) == pytest.approx(0.1)
    assert s.standard_uncertainty(reading=99.0) == pytest.approx(0.1)


def test_sensor_negative_value_rejected():
    with pytest.raises(ConfigError):
        SensorUncertainty(UncertaintyKind.ABSOLUTE, -0.1)


def test_analytical_addition_independent_inputs():
    # f(a,b) = a + b ; sigma_y = sqrt(sigma_a^2 + sigma_b^2)
    val, u = propagate_analytical(
        lambda a, b: a + b, {"a": 10.0, "b": 5.0}, {"a": 0.3, "b": 0.4}
    )
    assert val == pytest.approx(15.0)
    assert u == pytest.approx(0.5, rel=1e-6)


def test_analytical_subtraction_uncertainty_adds_in_quadrature():
    val, u = propagate_analytical(
        lambda a, b: a - b, {"a": 10.0, "b": 4.0}, {"a": 0.3, "b": 0.4}
    )
    assert val == pytest.approx(6.0)
    assert u == pytest.approx(math.hypot(0.3, 0.4), rel=1e-6)


def test_analytical_product_relative_uncertainty():
    # Y = a * b ; (sigma_y/y)^2 = (sigma_a/a)^2 + (sigma_b/b)^2
    a, b, sa, sb = 4.0, 3.0, 0.04, 0.06
    val, u = propagate_analytical(
        lambda a, b: a * b, {"a": a, "b": b}, {"a": sa, "b": sb}
    )
    expected_rel = math.hypot(sa / a, sb / b)
    assert val == pytest.approx(12.0)
    assert u / val == pytest.approx(expected_rel, rel=1e-5)


def test_analytical_ratio_relative_uncertainty():
    a, b, sa, sb = 12.0, 4.0, 0.3, 0.1
    val, u = propagate_analytical(
        lambda a, b: a / b, {"a": a, "b": b}, {"a": sa, "b": sb}
    )
    expected_rel = math.hypot(sa / a, sb / b)
    assert val == pytest.approx(3.0)
    assert u / val == pytest.approx(expected_rel, rel=1e-5)


def test_analytical_ignores_zero_uncertainty_inputs():
    # b has 0 uncertainty -> contributes nothing
    _, u = propagate_analytical(
        lambda a, b: a * b, {"a": 4.0, "b": 3.0}, {"a": 0.04, "b": 0.0}
    )
    assert u == pytest.approx((0.04) * 3.0, rel=1e-6)


def test_analytical_constant_via_fixed_arg():
    val, u = propagate_analytical(
        lambda x, c: c * x,
        inputs={"x": 5.0},
        uncertainties={"x": 0.1},
        fixed={"c": 7.0},
    )
    assert val == pytest.approx(35.0)
    assert u == pytest.approx(0.7, rel=1e-6)


def test_analytical_function_failure_raises_analysis_error():
    def f(x):
        if x < 0:
            raise ValueError("negative")
        return math.sqrt(x)
    with pytest.raises(AnalysisError):
        propagate_analytical(f, {"x": -1.0}, {"x": 0.1})


def test_monte_carlo_matches_analytical_for_linear_function():
    fn = lambda a, b: 3.0 * a + 2.0 * b
    inputs = {"a": 10.0, "b": 5.0}
    uncs = {"a": 0.5, "b": 0.3}
    val_a, u_a = propagate_analytical(fn, inputs, uncs)
    val_m, u_m = propagate_monte_carlo(
        fn, inputs, uncs, n_samples=20_000, seed=42
    )
    assert val_a == pytest.approx(val_m, abs=0.05)
    assert u_a == pytest.approx(u_m, rel=0.03)


def test_monte_carlo_handles_nonlinear_function():
    # Y = a^2 ; for small sigma_a/a, sigma_y ~ 2|a|*sigma_a
    fn = lambda a: a * a
    val_m, u_m = propagate_monte_carlo(
        fn, {"a": 5.0}, {"a": 0.1}, n_samples=20_000, seed=1
    )
    assert val_m == pytest.approx(25.0, rel=0.005)
    assert u_m == pytest.approx(2 * 5.0 * 0.1, rel=0.05)


def test_monte_carlo_rejects_too_few_samples():
    with pytest.raises(ConfigError):
        propagate_monte_carlo(lambda x: x, {"x": 1.0}, {"x": 0.1}, n_samples=10)


def test_monte_carlo_falls_back_when_function_not_vectorized():
    # Force scalar-only path: function rejects arrays.
    def scalar_only(x):
        if hasattr(x, "__len__"):
            raise TypeError("scalar only")
        return x * 2.0
    val, u = propagate_monte_carlo(
        scalar_only, {"x": 5.0}, {"x": 0.5}, n_samples=2000, seed=0
    )
    assert val == pytest.approx(10.0, rel=0.02)
    assert u == pytest.approx(1.0, rel=0.05)
