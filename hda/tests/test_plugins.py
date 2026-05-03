"""v3 plugin Protocol + registry + basic_means plugin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hda.domain.errors import ConfigError
from hda.domain.plugin_modules import BasicMeansPlugin
from hda.domain.plugins import AnalysisContext, PluginRegistry
from hda.domain.types import (
    Hardware,
    MeasurementWithUncertainty,
    Provenance,
    SteadyWindow,
    TestMetadata,
)
from hda.domain.uncertainty import SensorUncertainty, UncertaintyKind


def _ctx(steady_df: pd.DataFrame, sensor_calibrations: dict | None = None) -> AnalysisContext:
    md = TestMetadata(
        hardware=Hardware(part_number="PN-1", serial_number="SN-1"),
        fluid="N2",
        operator="alice",
        test_id="T-001",
    )
    return AnalysisContext(
        df=steady_df,
        steady_df=steady_df,
        steady_window=SteadyWindow(
            start_s=0.0, end_s=float(steady_df["timestamp"].iloc[-1]),
            method="cv", confidence=0.9
        ),
        metadata=md,
        sensor_calibrations=sensor_calibrations or {},
        geometry={},
    )


def test_registry_register_and_get():
    reg = PluginRegistry()
    p = BasicMeansPlugin()
    reg.register(p)
    assert "basic_means" in reg.names()
    assert reg.get("basic_means") is p


def test_registry_rejects_duplicate():
    reg = PluginRegistry()
    reg.register(BasicMeansPlugin())
    with pytest.raises(ConfigError, match="already registered"):
        reg.register(BasicMeansPlugin())


def test_registry_get_unknown_raises():
    reg = PluginRegistry()
    with pytest.raises(ConfigError, match="No plugin"):
        reg.get("missing")


def test_registry_rejects_non_protocol():
    class NotAPlugin:
        pass
    with pytest.raises(ConfigError):
        PluginRegistry().register(NotAPlugin())  # type: ignore[arg-type]


def test_basic_means_emits_avg_per_channel():
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 0.01, 0.02, 0.03],
            "PT-up": [10.0, 10.1, 10.0, 9.9],
            "PT-down": [5.0, 5.0, 5.0, 5.0],
        }
    )
    plugin = BasicMeansPlugin()
    out = plugin.compute(_ctx(df))
    assert set(out.keys()) == {"avg_PT-up", "avg_PT-down"}
    assert out["avg_PT-up"].value == pytest.approx(10.0, abs=0.05)
    assert out["avg_PT-down"].value == pytest.approx(5.0)
    assert out["avg_PT-down"].uncertainty == 0.0  # constant -> SEM = 0
    for m in out.values():
        assert isinstance(m, MeasurementWithUncertainty)
        assert m.provenance is Provenance.SENSOR


def test_basic_means_combines_sem_and_calibration():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "timestamp": np.linspace(0.0, 1.0, 101),
        "PT-up": 10.0 + rng.standard_normal(101) * 0.5,  # sigma ~0.5
    })
    plugin = BasicMeansPlugin()
    cal = {"PT-up": SensorUncertainty(UncertaintyKind.ABSOLUTE, 0.1)}
    out = plugin.compute(_ctx(df, sensor_calibrations=cal))
    m = out["avg_PT-up"]
    # SEM ~ 0.5/sqrt(101) ~ 0.05; combined with 0.1 cal -> sqrt(0.05^2 + 0.1^2) ~ 0.112
    assert 0.05 < m.uncertainty < 0.2


def test_basic_means_relative_calibration_scales_with_reading():
    df = pd.DataFrame({
        "timestamp": np.linspace(0.0, 1.0, 101),
        "PT-up": np.full(101, 200.0),  # constant -> SEM ~ 0
    })
    cal = {"PT-up": SensorUncertainty(UncertaintyKind.RELATIVE, 0.005)}  # 0.5%
    out = BasicMeansPlugin().compute(_ctx(df, sensor_calibrations=cal))
    # 0.5% of 200 -> 1.0
    assert out["avg_PT-up"].uncertainty == pytest.approx(1.0, rel=1e-3)


def test_basic_means_skips_non_numeric_columns():
    df = pd.DataFrame({
        "timestamp": [0.0, 0.01],
        "PT-up": [1.0, 2.0],
        "label": ["a", "b"],
    })
    out = BasicMeansPlugin().compute(_ctx(df))
    assert "avg_PT-up" in out
    assert "avg_label" not in out
