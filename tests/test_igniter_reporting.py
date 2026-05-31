"""Unit tests for igniter advanced report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.igniter_analysis import IgniterAnalysisResult
from core.igniter_reporting import (
    generate_igniter_hotfire_report,
    save_igniter_hotfire_report,
)


def _sample_result() -> IgniterAnalysisResult:
    return IgniterAnalysisResult(
        pc_bar=24.0,
        mdot_n2o_g_s=30.0,
        mdot_eth_g_s=12.0,
        mdot_eth_source="measured",
        mdot_total_g_s=42.0,
        of_ratio=2.5,
        cstar_actual_m_s=1320.0,
        cstar_theoretical_m_s=1560.0,
        eta_cstar=0.846,
        pc_predicted_bar=23.5,
        tc_theo_k=3030.0,
        cd_n2o_back=0.78,
        cd_n2o_delta=0.01,
        oxidizer_name="N2O",
        fuel_name="Ethanol",
        warnings=["example warning"],
        n2o_flow_diagnostics={"regime": "CHOKED", "choked": True, "Pcrit_bar": 20.2},
    )


def test_generate_igniter_hotfire_report_contains_sections():
    df = pd.DataFrame(
        {
            "time_s": [0.0, 0.5, 1.0, 1.5],
            "pc_bar_sensor": [5.0, 10.0, 12.0, 11.0],
            "ox_flow_g_s": [10.0, 20.0, 21.0, 20.5],
            "fuel_flow_g_s": [4.0, 8.0, 8.2, 8.1],
        }
    )
    html = generate_igniter_hotfire_report(
        test_id="IGN-HF-UNIT-001",
        result=_sample_result(),
        metadata={"operator": "unit-test"},
        config={"test_type": "hot_fire"},
        traceability={"raw_data_hash": "sha256:abc123"},
        steady_window_s=(1.5, 3.0),
        df=df,
        sensor_roles={
            "chamber_pressure": "pc_bar_sensor",
            "mass_flow_ox": "ox_flow_g_s",
            "mass_flow_fuel": "fuel_flow_g_s",
        },
    )
    assert "Igniter Hot-Fire Engineering Report" in html
    assert "Combustion Performance Assessment" in html
    assert "Traceability and Metadata" in html
    assert "Standard Plot: Chamber Pressure" in html
    assert "Standard Plot: Mass Flow" in html
    assert "N2O / Ethanol" in html
    assert "example warning" in html


def test_save_igniter_hotfire_report_writes_file(tmp_path):
    out = tmp_path / "igniter_report.html"
    html = generate_igniter_hotfire_report(
        test_id="IGN-HF-UNIT-002",
        result=_sample_result(),
    )
    saved = save_igniter_hotfire_report(html, out)
    assert saved == Path(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "IGN-HF-UNIT-002" in text
