"""Derived-channel and derived-measurement evaluator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hda.domain.derived import (
    DerivedChannelSpec,
    DerivedContext,
    DerivedMeasurementSpec,
    FormulaLibrary,
    UncertaintyMethod,
    evaluate_channels,
    evaluate_measurements,
    standard_library,
)
from hda.domain.errors import AnalysisError, ConfigError
from hda.domain.types import Provenance


def _lib() -> FormulaLibrary:
    return standard_library()


def test_topo_orders_dependencies_first():
    spec_a = DerivedChannelSpec(
        name="dp",
        unit="bar",
        formula="subtract",
        inputs={"a": "PT-up", "b": "PT-down"},
    )
    spec_b = DerivedChannelSpec(
        name="dp_doubled",
        unit="bar",
        formula="ratio",
        inputs={"num": "dp", "den": "half"},
    )
    p_up = np.array([10.0, 11.0, 12.0])
    p_down = np.array([5.0, 6.0, 7.0])
    ctx = DerivedContext(
        sensor_channels={"PT-up": p_up, "PT-down": p_down},
        sensor_scalars={"half": 0.5},
    )
    out = evaluate_channels([spec_b, spec_a], ctx, _lib())
    np.testing.assert_allclose(out["dp"], np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(out["dp_doubled"], np.array([10.0, 10.0, 10.0]))


def test_topo_detects_cycle():
    a = DerivedChannelSpec(
        name="a",
        unit="",
        formula="ratio",
        inputs={"num": "b", "den": "k"},
    )
    b = DerivedChannelSpec(
        name="b",
        unit="",
        formula="ratio",
        inputs={"num": "a", "den": "k"},
    )
    ctx = DerivedContext(sensor_scalars={"k": 1.0})
    with pytest.raises(ConfigError, match="Cyclic"):
        evaluate_channels([a, b], ctx, _lib())


def test_topo_reports_unresolved_source():
    spec = DerivedChannelSpec(
        name="x",
        unit="",
        formula="ratio",
        inputs={"num": "PT-missing", "den": "k"},
    )
    ctx = DerivedContext(sensor_scalars={"k": 1.0})
    with pytest.raises(ConfigError, match="unknown sources"):
        evaluate_channels([spec], ctx, _lib())


def test_topo_rejects_duplicate_names():
    a = DerivedChannelSpec(
        name="dup", unit="", formula="ratio", inputs={"num": "x", "den": "y"}
    )
    b = DerivedChannelSpec(
        name="dup", unit="", formula="ratio", inputs={"num": "x", "den": "y"}
    )
    ctx = DerivedContext(sensor_scalars={"x": 1.0, "y": 2.0})
    with pytest.raises(ConfigError, match="Duplicate"):
        evaluate_channels([a, b], ctx, _lib())


def test_evaluate_channel_with_params():
    spec = DerivedChannelSpec(
        name="mf_fuel",
        unit="g/s",
        formula="cd_orifice_incompressible",
        inputs={
            "p_up_bar": "PT-fuel-up",
            "p_down_bar": "PT-fuel-down",
            "density_kg_m3": "rho_fuel",
        },
        params={"area_mm2": 3.14159, "cd": 0.7},
    )
    p_up = np.array([20.0, 20.0, 20.0])
    p_down = np.array([15.0, 15.0, 15.0])
    ctx = DerivedContext(
        sensor_channels={"PT-fuel-up": p_up, "PT-fuel-down": p_down},
        sensor_scalars={"rho_fuel": 800.0},
    )
    out = evaluate_channels([spec], ctx, _lib())
    arr = out["mf_fuel"]
    assert arr.shape == (3,)
    # m_dot = cd * A * sqrt(2 * rho * dp); dp = 5 bar = 5e5 Pa
    expected_kg_s = 0.7 * 3.14159e-6 * math.sqrt(2.0 * 800.0 * 5.0e5)
    np.testing.assert_allclose(arr, np.full(3, expected_kg_s * 1000.0), rtol=1e-6)


def test_formula_failure_surfaces_as_analysis_error():
    spec = DerivedChannelSpec(
        name="bad",
        unit="",
        formula="integrate_trapezoid",  # needs >= 2 samples
        inputs={"series": "x"},
        params={"dt_s": 0.01},
    )
    ctx = DerivedContext(sensor_channels={"x": np.array([1.0])})
    with pytest.raises(AnalysisError):
        evaluate_channels([spec], ctx, _lib())


def test_evaluate_measurement_with_none_uncertainty():
    spec = DerivedMeasurementSpec(
        name="of_ratio",
        unit="",
        formula="of_ratio",
        inputs={"mf_ox_g_s": "mf_ox_mean", "mf_fuel_g_s": "mf_fuel_mean"},
        uncertainty_method=UncertaintyMethod.NONE,
    )
    ctx = DerivedContext(
        sensor_scalars={"mf_ox_mean": 100.0, "mf_fuel_mean": 50.0}
    )
    out = evaluate_measurements([spec], ctx, _lib())
    m = out["of_ratio"]
    assert m.value == pytest.approx(2.0)
    assert m.uncertainty == 0.0
    assert m.provenance is Provenance.DERIVED


def test_evaluate_measurement_chains_dependencies():
    a = DerivedMeasurementSpec(
        name="dp_mean",
        unit="bar",
        formula="subtract",
        inputs={"a": "p_up_mean", "b": "p_down_mean"},
        uncertainty_method=UncertaintyMethod.NONE,
    )
    b = DerivedMeasurementSpec(
        name="dp_half",
        unit="bar",
        formula="ratio",
        inputs={"num": "dp_mean", "den": "two"},
        uncertainty_method=UncertaintyMethod.NONE,
    )
    ctx = DerivedContext(
        sensor_scalars={"p_up_mean": 10.0, "p_down_mean": 4.0, "two": 2.0}
    )
    out = evaluate_measurements([b, a], ctx, _lib())
    assert out["dp_mean"].value == pytest.approx(6.0)
    assert out["dp_half"].value == pytest.approx(3.0)


def test_evaluate_measurement_rejects_unimplemented_uncertainty():
    spec = DerivedMeasurementSpec(
        name="x",
        unit="",
        formula="ratio",
        inputs={"num": "a", "den": "b"},
        uncertainty_method=UncertaintyMethod.ANALYTICAL,
    )
    ctx = DerivedContext(sensor_scalars={"a": 1.0, "b": 2.0})
    with pytest.raises(NotImplementedError):
        evaluate_measurements([spec], ctx, _lib())


def test_evaluate_uses_geometry_and_metadata_as_scalars():
    spec = DerivedMeasurementSpec(
        name="thrust_per_area",
        unit="bar",
        formula="ratio",
        inputs={"num": "thrust", "den": "area"},
        uncertainty_method=UncertaintyMethod.NONE,
    )
    ctx = DerivedContext(
        sensor_scalars={"thrust": 1000.0},
        geometry={"area": 100.0},
    )
    out = evaluate_measurements([spec], ctx, _lib())
    assert out["thrust_per_area"].value == pytest.approx(10.0)
