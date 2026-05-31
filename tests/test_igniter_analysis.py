"""Unit tests for igniter post-test analysis (core/igniter_analysis.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.igniter_analysis import (
    IgniterHardware,
    IgniterTestInputs,
    column_means,
    inputs_from_steady_state,
    steady_window_slice,
)


def test_steady_window_slice_seconds():
    df = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "IG-PT-01": [1, 1, 20, 20, 20, 1],
        }
    )
    sub = steady_window_slice(df, (2.0, 4.0), "time_s")
    assert len(sub) == 3
    assert sub["IG-PT-01"].mean() == pytest.approx(20.0)


def test_steady_window_slice_milliseconds():
    df = pd.DataFrame(
        {
            "timestamp": [0, 1000, 2000, 3000, 4000, 5000],
            "OX-FM-01": [0, 0, 30, 30, 30, 0],
        }
    )
    sub = steady_window_slice(df, (2.0, 4.0))
    assert len(sub) == 3
    assert sub["OX-FM-01"].mean() == pytest.approx(30.0)


def test_column_means():
    df = pd.DataFrame({"A": [10.0, 12.0], "B": [1.0, 3.0]})
    avgs = column_means(df, {"p": "A", "mf": "B"})
    assert avgs["p"] == pytest.approx(11.0)
    assert avgs["mf"] == pytest.approx(2.0)


def test_hardware_from_metadata():
    meta = {
        "geometry": {"throat_diameter_mm": 5.0},
        "igniter_hardware": {"cd_n2o": 0.8, "d_n2o_orifice_mm": 0.7},
    }
    hw = IgniterHardware.from_metadata(meta)
    assert hw.throat_diameter_mm == pytest.approx(5.0)
    assert hw.cd_n2o == pytest.approx(0.8)
    assert hw.d_n2o_orifice_mm == pytest.approx(0.7)


def test_inputs_from_steady_state():
    t = np.linspace(0, 5, 501)
    df = pd.DataFrame(
        {
            "time_s": t,
            "IG-PT-01": np.where((t >= 2) & (t <= 4), 20.0, 0.0),
            "OX-FM-01": np.where((t >= 2) & (t <= 4), 30.0, 0.0),
            "FU-FM-01": np.where((t >= 2) & (t <= 4), 12.0, 0.0),
            "FU-PT-01": np.where((t >= 2) & (t <= 4), 55.0, 0.0),
        }
    )
    roles = {
        "chamber_pressure": "IG-PT-01",
        "mass_flow_ox": "OX-FM-01",
        "mass_flow_fuel": "FU-FM-01",
        "upstream_pressure": "FU-PT-01",
    }
    hw = IgniterHardware()
    inputs = inputs_from_steady_state(df, roles, (2.0, 4.0), hw)
    assert inputs.pc_bar == pytest.approx(20.0, rel=0.01)
    assert inputs.mdot_n2o_g_s == pytest.approx(30.0, rel=0.01)
    assert inputs.mdot_eth_g_s == pytest.approx(12.0, rel=0.01)
    assert inputs.p_eth_upstream_bar == pytest.approx(55.0, rel=0.01)
    assert inputs.input_source == "steady_state"


def test_inputs_from_steady_state_missing_required():
    df = pd.DataFrame({"time_s": [0, 1], "IG-PT-01": [1, 1]})
    with pytest.raises(ValueError, match="mass_flow_ox"):
        inputs_from_steady_state(
            df,
            {"chamber_pressure": "IG-PT-01"},
            (0.0, 1.0),
            IgniterHardware(),
        )


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("CoolProp"),
    reason="CoolProp not installed",
)
def test_analyze_with_measured_ethanol():
    from core.igniter_analysis import analyze_igniter_post_test

    hw = IgniterHardware(throat_diameter_mm=4.4)
    inputs = IgniterTestInputs(
        pc_bar=20.0,
        mdot_n2o_g_s=30.0,
        mdot_eth_g_s=12.0,
        p_n2o_upstream_bar=55.0,
        d_n2o_orifice_mm=0.6,
    )
    result = analyze_igniter_post_test(inputs, hw)
    assert result.of_ratio == pytest.approx(2.5, rel=0.01)
    assert result.mdot_eth_source == "measured"
    expected_cstar = 20e5 * hw.throat_area_m2 / ((30 + 12) * 1e-3)
    assert result.cstar_actual_m_s == pytest.approx(expected_cstar, rel=0.01)


def test_igniter_plugin_registered():
    from core.plugins import PluginRegistry

    PluginRegistry.clear()
    plugin = PluginRegistry.get_plugin("igniter_hot_fire")
    assert plugin.metadata.slug == "igniter_hot_fire"
    assert plugin.metadata.test_type == "hot_fire"
