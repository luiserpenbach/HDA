"""Cold-flow plugin: Cd via the incompressible-orifice formula with full
uncertainty propagation through mass flow, both pressures, area, and density."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hda.domain.errors import AnalysisError, ConfigError
from hda.domain.plugin_modules import (
    ColdFlowChannelMap,
    ColdFlowConfig,
    ColdFlowPlugin,
)
from hda.domain.plugins import AnalysisContext
from hda.domain.types import (
    Hardware,
    Provenance,
    SteadyWindow,
    TestMetadata,
)
from hda.domain.uncertainty import SensorUncertainty, UncertaintyKind


def _df(n: int = 1001, p_up: float = 10.0, p_down: float = 5.0, mf: float = 30.0):
    t = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "timestamp": t,
            "PT-up": np.full(n, p_up),
            "PT-down": np.full(n, p_down),
            "MF-01": np.full(n, mf),
        }
    )


def _channel_map():
    return ColdFlowChannelMap(
        upstream_pressure="PT-up",
        downstream_pressure="PT-down",
        mass_flow="MF-01",
    )


def _ctx(
    df: pd.DataFrame,
    *,
    sensor_calibrations: dict | None = None,
    geometry: dict | None = None,
    geometry_uncertainties: dict | None = None,
    metadata_extra: dict | None = None,
) -> AnalysisContext:
    md = TestMetadata(
        hardware=Hardware(part_number="PN-1", serial_number="SN-1"),
        fluid="water",
        operator="alice",
        test_id="T-1",
        geometry={"orifice_area_mm2": 1.0} if geometry is None else geometry,
        extra={"density_kg_m3": 1000.0} if metadata_extra is None else metadata_extra,
    )
    return AnalysisContext(
        df=df,
        steady_df=df,
        steady_window=SteadyWindow(
            start_s=0.0, end_s=float(df["timestamp"].iloc[-1]),
            method="cv", confidence=0.95,
        ),
        metadata=md,
        sensor_calibrations=sensor_calibrations or {},
        geometry=md.geometry,
        geometry_uncertainties=geometry_uncertainties or {},
    )


def test_required_channels_returns_mapped_columns():
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    assert set(plugin.required_channels()) == {"PT-up", "PT-down", "MF-01"}


def test_cd_value_matches_closed_form():
    """Cd from the inverted incompressible-flow equation:
        m_dot = Cd * A * sqrt(2 * rho * dp)
    With m_dot=30 g/s, A=1 mm^2, rho=1000 kg/m^3, dp=5 bar:
        Cd = (30e-3) / (1e-6 * sqrt(2 * 1000 * 5e5))
           = 0.03 / (1e-6 * 31622.776...)
           = 0.948683...
    """
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    out = plugin.compute(_ctx(_df()))
    expected_cd = 0.030 / (1.0e-6 * math.sqrt(2.0 * 1000.0 * 5.0e5))
    assert out["cd"].value == pytest.approx(expected_cd, rel=1e-9)
    assert out["cd"].uncertainty == 0.0  # no uncertainties supplied
    assert out["cd"].provenance is Provenance.SENSOR


def test_cd_uncertainty_propagates_from_sensor_calibrations():
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    cal = {
        "PT-up": SensorUncertainty(UncertaintyKind.RELATIVE, 0.005),     # 0.5%
        "PT-down": SensorUncertainty(UncertaintyKind.RELATIVE, 0.005),
        "MF-01": SensorUncertainty(UncertaintyKind.RELATIVE, 0.01),       # 1%
    }
    out = plugin.compute(_ctx(_df(), sensor_calibrations=cal))
    cd = out["cd"]
    # Cd ~ m_dot / sqrt(dp); sensor uncertainties translate roughly to
    # relative cd uncertainty bounded below by the mass-flow term (1%).
    assert cd.uncertainty > 0.0
    assert cd.rel_uncertainty_pct is not None
    assert 0.5 < cd.rel_uncertainty_pct < 5.0


def test_cd_uncertainty_includes_area_and_density():
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    out = plugin.compute(
        _ctx(
            _df(),
            geometry_uncertainties={"orifice_area_mm2": 0.01},  # 1% of 1 mm^2
            metadata_extra={
                "density_kg_m3": 1000.0,
                "density_uncertainty_kg_m3": 5.0,  # 0.5%
            },
        )
    )
    # No sensor calibrations -> uncertainty comes only from area + density.
    # Closed form: rel_u(Cd) = sqrt((rel_u(A))^2 + 0.25 * (rel_u(rho))^2)
    expected = math.sqrt(0.01**2 + 0.25 * 0.005**2)
    assert out["cd"].rel_uncertainty_pct == pytest.approx(expected * 100, rel=0.05)


def test_cd_density_priority_metadata_over_default():
    plugin = ColdFlowPlugin(
        ColdFlowConfig(channel_map=_channel_map(), default_density_kg_m3=999.0)
    )
    out = plugin.compute(_ctx(_df(), metadata_extra={"density_kg_m3": 1000.0}))
    expected_cd = 0.030 / (1.0e-6 * math.sqrt(2.0 * 1000.0 * 5.0e5))
    assert out["cd"].value == pytest.approx(expected_cd, rel=1e-9)


def test_cd_uses_default_density_when_metadata_omits_it():
    plugin = ColdFlowPlugin(
        ColdFlowConfig(
            channel_map=_channel_map(),
            default_density_kg_m3=1000.0,
            default_density_uncertainty_kg_m3=10.0,
        )
    )
    out = plugin.compute(_ctx(_df(), metadata_extra={}))
    assert math.isfinite(out["cd"].value)
    assert out["cd"].uncertainty > 0.0  # density uncertainty contributes


def test_density_missing_with_no_default_raises_config_error():
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    with pytest.raises(ConfigError, match="density"):
        plugin.compute(_ctx(_df(), metadata_extra={}))


def test_density_non_numeric_raises_config_error():
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    with pytest.raises(ConfigError, match="density_kg_m3"):
        plugin.compute(_ctx(_df(), metadata_extra={"density_kg_m3": "wet"}))


def test_geometry_missing_raises_config_error():
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    with pytest.raises(ConfigError, match="orifice_area_mm2"):
        plugin.compute(_ctx(_df(), geometry={}))


def test_required_channel_missing_raises_config_error():
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    df = pd.DataFrame({
        "timestamp": np.linspace(0, 1, 11),
        "PT-up": np.full(11, 10.0),
        # MF-01 missing
        "PT-down": np.full(11, 5.0),
    })
    with pytest.raises(ConfigError, match="MF-01"):
        plugin.compute(_ctx(df))


def test_avg_channels_emitted_with_sem():
    rng = np.random.default_rng(0)
    n = 1001
    df = pd.DataFrame({
        "timestamp": np.linspace(0.0, 1.0, n),
        "PT-up": 10.0 + 0.5 * rng.standard_normal(n),
        "PT-down": np.full(n, 5.0),
        "MF-01": np.full(n, 30.0),
    })
    plugin = ColdFlowPlugin(ColdFlowConfig(channel_map=_channel_map()))
    out = plugin.compute(_ctx(df))
    assert "avg_PT-up" in out and "avg_PT-down" in out and "avg_MF-01" in out
    # SEM ~ 0.5 / sqrt(1001) ~ 0.016
    assert 0.005 < out["avg_PT-up"].uncertainty < 0.05
    assert out["avg_PT-down"].uncertainty == 0.0
