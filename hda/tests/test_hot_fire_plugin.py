"""Hot-fire plugin: avg_<channel> + mf_total + of_ratio + c_star + isp,
all with full Jacobian uncertainty propagation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hda.domain.errors import ConfigError
from hda.domain.plugin_modules import (
    HotFireChannelMap,
    HotFireConfig,
    HotFirePlugin,
)
from hda.domain.plugins import AnalysisContext
from hda.domain.types import (
    Hardware,
    Provenance,
    SteadyWindow,
    TestMetadata,
)
from hda.domain.uncertainty import SensorUncertainty, UncertaintyKind


PC_BAR = 30.0
THRUST_N = 4500.0
MF_OX_G_S = 1500.0
MF_FUEL_G_S = 600.0
THROAT_AREA_MM2 = 250.0


def _df(n: int = 1001) -> pd.DataFrame:
    t = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "timestamp": t,
            "PC-01": np.full(n, PC_BAR),
            "LC-01": np.full(n, THRUST_N),
            "MF-OX": np.full(n, MF_OX_G_S),
            "MF-FUEL": np.full(n, MF_FUEL_G_S),
        }
    )


def _channel_map() -> HotFireChannelMap:
    return HotFireChannelMap(
        chamber_pressure="PC-01",
        thrust="LC-01",
        mass_flow_ox="MF-OX",
        mass_flow_fuel="MF-FUEL",
    )


def _ctx(
    df: pd.DataFrame,
    *,
    sensor_calibrations: dict | None = None,
    geometry: dict | None = None,
    geometry_uncertainties: dict | None = None,
) -> AnalysisContext:
    md = TestMetadata(
        hardware=Hardware(part_number="IGN-1", serial_number="SN-1"),
        fluid="LOX/RP-1",
        operator="alice",
        test_id="HF-001",
        geometry={"throat_area_mm2": THROAT_AREA_MM2}
        if geometry is None
        else geometry,
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
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    assert set(plugin.required_channels()) == {"PC-01", "LC-01", "MF-OX", "MF-FUEL"}


def test_avg_channels_emitted():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    out = plugin.compute(_ctx(_df()))
    assert {"avg_PC-01", "avg_LC-01", "avg_MF-OX", "avg_MF-FUEL"}.issubset(out.keys())


def test_mf_total_value_and_uncertainty():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    cal = {
        "MF-OX": SensorUncertainty(UncertaintyKind.ABSOLUTE, 5.0),
        "MF-FUEL": SensorUncertainty(UncertaintyKind.ABSOLUTE, 3.0),
    }
    out = plugin.compute(_ctx(_df(), sensor_calibrations=cal))
    mf = out["mf_total"]
    assert mf.value == pytest.approx(MF_OX_G_S + MF_FUEL_G_S)
    # u(a+b) = sqrt(u_a^2 + u_b^2)
    assert mf.uncertainty == pytest.approx(math.hypot(5.0, 3.0), rel=1e-4)
    assert mf.unit == "g/s"


def test_of_ratio_value_and_uncertainty():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    cal = {
        "MF-OX": SensorUncertainty(UncertaintyKind.RELATIVE, 0.01),     # 1%
        "MF-FUEL": SensorUncertainty(UncertaintyKind.RELATIVE, 0.01),
    }
    out = plugin.compute(_ctx(_df(), sensor_calibrations=cal))
    of = out["of_ratio"]
    assert of.value == pytest.approx(MF_OX_G_S / MF_FUEL_G_S)
    # rel_u(of) = sqrt((1%)^2 + (1%)^2) = sqrt(2)%
    expected_rel = math.hypot(0.01, 0.01)
    assert of.uncertainty / of.value == pytest.approx(expected_rel, rel=0.05)


def test_c_star_value_matches_closed_form():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    out = plugin.compute(_ctx(_df()))
    cstar = out["c_star"]
    pc_pa = PC_BAR * 1.0e5
    a_t_m2 = THROAT_AREA_MM2 * 1.0e-6
    mf_kg_s = (MF_OX_G_S + MF_FUEL_G_S) * 1.0e-3
    expected = pc_pa * a_t_m2 / mf_kg_s
    assert cstar.value == pytest.approx(expected, rel=1e-9)
    assert cstar.uncertainty == 0.0
    assert cstar.unit == "m/s"


def test_isp_value_matches_closed_form():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    out = plugin.compute(_ctx(_df()))
    isp = out["isp"]
    mf_kg_s = (MF_OX_G_S + MF_FUEL_G_S) * 1.0e-3
    expected = THRUST_N / (mf_kg_s * 9.80665)
    assert isp.value == pytest.approx(expected, rel=1e-9)
    assert isp.uncertainty == 0.0
    assert isp.unit == "s"


def test_c_star_uncertainty_includes_throat_area():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    out = plugin.compute(
        _ctx(
            _df(),
            geometry_uncertainties={"throat_area_mm2": 2.5},  # 1% of 250
        )
    )
    cstar = out["c_star"]
    # No sensor calibrations -> only A_throat contributes; rel_u(c*) = rel_u(A) = 1%
    assert cstar.rel_uncertainty_pct == pytest.approx(1.0, rel=0.05)


def test_isp_uncertainty_propagates_from_thrust_and_flows():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    cal = {
        "LC-01": SensorUncertainty(UncertaintyKind.RELATIVE, 0.005),    # 0.5%
        "MF-OX": SensorUncertainty(UncertaintyKind.RELATIVE, 0.01),     # 1%
        "MF-FUEL": SensorUncertainty(UncertaintyKind.RELATIVE, 0.01),
    }
    out = plugin.compute(_ctx(_df(), sensor_calibrations=cal))
    isp = out["isp"]
    # Isp = F / (mf_total * g0); rel_u^2 = rel_u(F)^2 + rel_u(mf_total)^2
    # rel_u(mf_total) = sqrt((u_ox*frac_ox)^2 + (u_fuel*frac_fuel)^2) where frac = mf/mf_total
    frac_ox = MF_OX_G_S / (MF_OX_G_S + MF_FUEL_G_S)
    frac_fuel = MF_FUEL_G_S / (MF_OX_G_S + MF_FUEL_G_S)
    rel_mf_total = math.hypot(0.01 * frac_ox, 0.01 * frac_fuel)
    expected_rel = math.hypot(0.005, rel_mf_total)
    assert isp.uncertainty / isp.value == pytest.approx(expected_rel, rel=0.05)


def test_throat_area_missing_raises_config_error():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    with pytest.raises(ConfigError, match="throat_area_mm2"):
        plugin.compute(_ctx(_df(), geometry={}))


def test_required_channel_missing_raises_config_error():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    df = pd.DataFrame({
        "timestamp": np.linspace(0, 1, 11),
        "PC-01": np.full(11, PC_BAR),
        "LC-01": np.full(11, THRUST_N),
        "MF-OX": np.full(11, MF_OX_G_S),
        # MF-FUEL missing
    })
    with pytest.raises(ConfigError, match="MF-FUEL"):
        plugin.compute(_ctx(df))


def test_zero_fuel_flow_emits_nan_of_ratio_but_other_metrics_survive():
    """A degenerate scalar (OF when fuel flow is zero) must not abort the
    whole hot-fire analysis. The detail panel still gets pc, thrust,
    mf_total, c*, isp; OF lands as NaN with a logged warning."""
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    df = _df()
    df["MF-FUEL"] = 0.0
    out = plugin.compute(_ctx(df))
    assert math.isnan(out["of_ratio"].value)
    assert math.isnan(out["of_ratio"].uncertainty)
    # The rest still computed normally.
    assert out["avg_PC-01"].value == pytest.approx(PC_BAR)
    assert out["mf_total"].value == pytest.approx(MF_OX_G_S + 0.0)
    assert math.isfinite(out["isp"].value)


def test_provenance_is_sensor_for_all_outputs():
    plugin = HotFirePlugin(HotFireConfig(channel_map=_channel_map()))
    out = plugin.compute(_ctx(_df()))
    for m in out.values():
        assert m.provenance is Provenance.SENSOR
