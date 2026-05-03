"""Standard formula library: numerical correctness + edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hda.domain.derived.standard_library import (
    cd_orifice_incompressible,
    c_star,
    integrate_trapezoid,
    isp,
    of_ratio,
    peak,
    ratio,
    standard_library,
    subtract,
)


def test_ratio_scalar():
    assert ratio(10.0, 4.0) == pytest.approx(2.5)


def test_ratio_returns_nan_on_zero_divisor():
    arr = ratio(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 3.0]))
    assert math.isnan(arr[1])
    assert arr[0] == pytest.approx(1.0)
    assert arr[2] == pytest.approx(1.0)


def test_subtract_array_array():
    np.testing.assert_allclose(
        subtract(np.array([10.0, 11.0]), np.array([1.0, 2.0])),
        np.array([9.0, 9.0]),
    )


def test_cd_orifice_known_value():
    # cd=1, A=1 mm^2 = 1e-6 m^2, rho=1000, dp=5e5 Pa.
    # m_dot = 1 * 1e-6 * sqrt(2*1000*5e5) = 1e-6 * sqrt(1e9) = 1e-6 * 31622.776...
    # in g/s = 31.6227766... g/s
    out = cd_orifice_incompressible(
        p_up_bar=10.0, p_down_bar=5.0, density_kg_m3=1000.0, area_mm2=1.0, cd=1.0
    )
    assert out == pytest.approx(math.sqrt(1.0e9) * 1.0e-6 * 1.0e3, rel=1e-9)


def test_cd_orifice_backflow_yields_nan():
    out = cd_orifice_incompressible(
        p_up_bar=np.array([5.0, 10.0]),
        p_down_bar=np.array([10.0, 5.0]),
        density_kg_m3=1000.0,
        area_mm2=1.0,
        cd=1.0,
    )
    assert math.isnan(out[0])
    assert not math.isnan(out[1])


def test_c_star_known_value():
    # pc=10 bar = 1e6 Pa, mf=1 kg/s = 1000 g/s, At=100 mm^2 = 1e-4 m^2
    # c* = 1e6 * 1e-4 / 1 = 100 m/s
    out = c_star(pc_bar=10.0, mf_total_g_s=1000.0, throat_area_mm2=100.0)
    assert out == pytest.approx(100.0)


def test_isp_known_value():
    # F=1000 N, mf=1 kg/s = 1000 g/s; Isp = 1000 / (1*9.80665) ~= 101.97 s
    out = isp(thrust_n=1000.0, mf_total_g_s=1000.0)
    assert out == pytest.approx(1000.0 / 9.80665, rel=1e-9)


def test_isp_zero_flow_yields_nan():
    out = isp(thrust_n=1000.0, mf_total_g_s=0.0)
    assert math.isnan(out)


def test_of_ratio():
    assert of_ratio(120.0, 40.0) == pytest.approx(3.0)


def test_peak():
    assert peak(np.array([1.0, 5.0, 3.0, 4.0, 2.0])) == pytest.approx(5.0)


def test_peak_rejects_2d():
    with pytest.raises(ValueError):
        peak(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_peak_empty_raises():
    with pytest.raises(ValueError):
        peak(np.array([]))


def test_integrate_trapezoid_constant():
    # constant 5 over 0..1s with dt=0.1 -> integral = 5
    arr = np.full(11, 5.0)
    assert integrate_trapezoid(arr, dt_s=0.1) == pytest.approx(5.0)


def test_integrate_trapezoid_linear():
    # 0..1 over 11 samples, dt=0.1 -> integral = 0.5
    arr = np.linspace(0.0, 1.0, 11)
    assert integrate_trapezoid(arr, dt_s=0.1) == pytest.approx(0.5)


def test_integrate_trapezoid_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        integrate_trapezoid(np.array([0.0, 1.0]), dt_s=0.0)


def test_standard_library_registers_all_functions():
    lib = standard_library()
    expected = {
        "ratio",
        "subtract",
        "cd_orifice_incompressible",
        "c_star",
        "isp",
        "of_ratio",
        "peak",
        "integrate_trapezoid",
    }
    assert expected.issubset(set(lib.names()))


def test_standard_library_versions_are_set():
    lib = standard_library()
    for name in lib.names():
        assert lib.version(name)
