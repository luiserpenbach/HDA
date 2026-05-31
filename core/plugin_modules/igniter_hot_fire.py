"""
Igniter Hot Fire Analysis Plugin — N₂O / Ethanol torch igniter (IGN-CP01).

Post-test analysis: NHNE N₂O flow, ethanol SPI estimate, c* efficiency, Cd back-calc.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from core.plugins import (
    PluginMetadata,
    PluginRegistry,
    ColumnSpec,
    InputSpec,
    UncertaintySpec,
    UncertaintySourceType,
)
from core.qc_checks import QCReport, run_qc_checks, run_quick_qc
from core.igniter_analysis import (
    IgniterHardware,
    IgniterTestInputs,
    analyze_igniter_post_test,
    steady_window_slice,
    column_means,
)


class IgniterHotFirePlugin:
    """Post-test analysis for N₂O/Ethanol torch igniter hot fires."""

    def __init__(self) -> None:
        self.metadata = PluginMetadata(
            name="Igniter Hot Fire (N₂O/Ethanol)",
            slug="igniter_hot_fire",
            version="1.0.0",
            test_type="hot_fire",
            description=(
                "Torch igniter post-test analysis: NHNE N₂O flow, ethanol SPI estimate, "
                "c* efficiency, Cd back-calculation (IGN-CP01 class)"
            ),
            author="HDA Core Team",
            required_hda_version=">=2.4.0",
            required_sensors=[
                "chamber_pressure",
                "mass_flow_ox",
            ],
            optional_sensors=[
                "mass_flow_fuel",
                "upstream_pressure",
                "thrust",
            ],
            database_columns=[
                ColumnSpec("avg_pc_bar", "REAL", nullable=False),
                ColumnSpec("avg_mdot_n2o_g_s", "REAL", nullable=False),
                ColumnSpec("avg_mdot_eth_g_s", "REAL", nullable=True),
                ColumnSpec("avg_of_ratio", "REAL", nullable=True),
                ColumnSpec("avg_cstar_actual_m_s", "REAL", nullable=True),
                ColumnSpec("avg_cstar_theo_m_s", "REAL", nullable=True),
                ColumnSpec("avg_eta_cstar", "REAL", nullable=True),
                ColumnSpec("avg_cd_n2o_back", "REAL", nullable=True),
            ],
            ui_inputs=[
                InputSpec(
                    name="throat_diameter_mm",
                    label="Throat diameter [mm]",
                    input_type="number",
                    default=4.4,
                ),
                InputSpec(
                    name="d_n2o_orifice_mm",
                    label="N₂O orifice [mm]",
                    input_type="select",
                    options=["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                    default="0.6",
                ),
                InputSpec(
                    name="d_eth_orifice_mm",
                    label="Ethanol orifice [mm]",
                    input_type="select",
                    options=["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                    default="0.5",
                ),
                InputSpec(
                    name="eta_cstar_target",
                    label="η_c* target",
                    input_type="number",
                    default=0.85,
                    min_value=0.5,
                    max_value=1.0,
                ),
            ],
            uncertainty_specs=[
                UncertaintySpec(
                    name="eta_cstar",
                    formula="η_c* = c*_actual / c*_theoretical",
                    sources=[
                        ("cstar_actual", UncertaintySourceType.CALCULATION),
                        ("cstar_theoretical", UncertaintySourceType.CALCULATION),
                    ],
                    propagation_method="analytical",
                ),
            ],
        )

    @staticmethod
    def _sensor_roles(config: Dict[str, Any]) -> Dict[str, str]:
        roles = config.get("sensor_roles") or config.get("columns") or {}
        return dict(roles)

    def validate_config(self, config: Dict[str, Any]) -> None:
        roles = self._sensor_roles(config)
        missing = [
            s for s in self.metadata.required_sensors if not roles.get(s)
        ]
        if missing:
            raise ValueError(
                f"Igniter plugin missing sensor roles: {', '.join(missing)}"
            )
        hw = IgniterHardware.from_metadata(config)
        if hw.throat_diameter_mm <= 0:
            raise ValueError("throat_diameter_mm must be positive")

    def run_qc_checks(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        quick: bool = False,
    ) -> QCReport:
        time_col = "time_s" if "time_s" in df.columns else "timestamp"
        if quick:
            return run_quick_qc(df, time_col=time_col)
        return run_qc_checks(df, config, time_col=time_col)

    def extract_steady_state(
        self,
        df: pd.DataFrame,
        steady_window: Tuple[float, float],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        time_col = "time_s" if "time_s" in df.columns else None
        return steady_window_slice(df, steady_window, time_col)

    def compute_raw_metrics(
        self,
        steady_df: pd.DataFrame,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        meta = {**(metadata or {}), **config}
        hardware = IgniterHardware.from_metadata(meta)
        roles = self._sensor_roles(config)

        avgs = column_means(steady_df, roles)
        pc = avgs.get("chamber_pressure")
        mdot_ox = avgs.get("mass_flow_ox")
        if pc is None or mdot_ox is None:
            raise ValueError("Steady averages missing chamber_pressure or mass_flow_ox")

        inputs = IgniterTestInputs(
            pc_bar=float(pc),
            mdot_n2o_g_s=float(mdot_ox),
            mdot_eth_g_s=float(avgs["mass_flow_fuel"])
            if avgs.get("mass_flow_fuel") is not None
            else None,
            p_eth_upstream_bar=float(avgs["upstream_pressure"])
            if avgs.get("upstream_pressure") is not None
            else None,
            d_eth_orifice_mm=hardware.d_eth_orifice_mm,
            cd_eth_override=hardware.cd_eth,
            d_n2o_orifice_mm=hardware.d_n2o_orifice_mm,
            input_source="steady_state",
        )
        hw_meta = meta.get("igniter_hardware") or {}
        if hw_meta.get("p_n2o_upstream_bar"):
            inputs.p_n2o_upstream_bar = float(hw_meta["p_n2o_upstream_bar"])

        result = analyze_igniter_post_test(inputs, hardware)

        metrics: Dict[str, float] = {
            "pc_bar": float(result.pc_bar),
            "mdot_n2o_g_s": float(result.mdot_n2o_g_s),
            "mdot_eth_g_s": float(result.mdot_eth_g_s),
            "mdot_total_g_s": float(result.mdot_total_g_s),
            "of_ratio": float(result.of_ratio),
            "cstar_actual_m_s": float(result.cstar_actual_m_s),
        }
        if result.cstar_theoretical_m_s is not None:
            metrics["cstar_theoretical_m_s"] = float(result.cstar_theoretical_m_s)
        if result.eta_cstar is not None:
            metrics["eta_cstar"] = float(result.eta_cstar)
        if result.cd_n2o_back is not None:
            metrics["cd_n2o_back"] = float(result.cd_n2o_back)
        if result.pc_predicted_bar is not None:
            metrics["pc_predicted_bar"] = float(result.pc_predicted_bar)
        return metrics

    def get_uncertainty_specs(self) -> Dict[str, UncertaintySpec]:
        return {spec.name: spec for spec in self.metadata.uncertainty_specs}

    def generate_report_sections(self, result: Any) -> Dict[str, str]:
        measurements = getattr(result, "measurements", {}) or {}
        eta = measurements.get("eta_cstar")
        if not eta:
            return {}
        val = getattr(eta, "value", eta)
        if val is None:
            return {}
        return {
            "igniter_performance": (
                f"<div class='metric-highlight'>"
                f"<h3>Igniter c* Efficiency</h3>"
                f"<p class='value'>η_c* = {float(val) * 100:.1f}%</p>"
                f"</div>"
            )
        }

    def get_display_order(self) -> List[str]:
        return [
            "pc_bar",
            "mdot_n2o_g_s",
            "mdot_eth_g_s",
            "of_ratio",
            "cstar_actual_m_s",
            "eta_cstar",
            "cd_n2o_back",
        ]


def register_plugin(registry: Optional[PluginRegistry] = None) -> IgniterHotFirePlugin:
    plugin = IgniterHotFirePlugin()
    reg = registry or PluginRegistry()
    reg.register(plugin)
    return plugin


register_plugin()
