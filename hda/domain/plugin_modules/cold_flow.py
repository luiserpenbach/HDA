"""Cold-flow plugin.

Computes the canonical cold-flow metrics from a steady-state slice:

    avg_<channel>     mean of each declared sensor channel, with
                      uncertainty = sqrt(SEM^2 + cal^2) using the
                      SensorUncertainty supplied in
                      ``ctx.sensor_calibrations``.

    cd                discharge coefficient via the incompressible-flow
                      orifice equation, with full uncertainty propagation
                      through mass flow, both pressures, fluid density,
                      and orifice area:

                          m_dot = Cd * A * sqrt(2 * rho * dp)

                      Inverted for Cd, then the analytical Jacobian is
                      taken over the five inputs. Uncertainty in any
                      one input is reflected in u(Cd) without the user
                      having to write derivatives.

Configuration:

    ``ColdFlowConfig`` carries the channel-role mapping
    (which DataFrame column is upstream pressure, downstream pressure,
    mass flow) and a default fluid density + uncertainty. A test can
    override the density via ``metadata.extra["density_kg_m3"]`` and
    ``metadata.extra["density_uncertainty_kg_m3"]``; if neither is
    supplied and no default is configured, the plugin raises
    ``ConfigError`` rather than silently picking a placeholder value.

The geometry parameter ``orifice_area_mm2`` is read from
``metadata.geometry``; its uncertainty from
``ctx.geometry_uncertainties`` (defaults to 0 when not provided).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from hda.domain.errors import AnalysisError, ConfigError
from hda.domain.plugins import AnalysisContext, AnalysisPlugin
from hda.domain.types import MeasurementWithUncertainty, Provenance
from hda.domain.uncertainty import propagate_analytical


_BAR_TO_PA = 1.0e5
_MM2_TO_M2 = 1.0e-6
_KG_TO_G = 1.0e3


@dataclass(frozen=True, slots=True)
class ColdFlowChannelMap:
    upstream_pressure: str
    downstream_pressure: str
    mass_flow: str


@dataclass(frozen=True, slots=True)
class ColdFlowConfig:
    channel_map: ColdFlowChannelMap
    default_density_kg_m3: Optional[float] = None
    default_density_uncertainty_kg_m3: float = 0.0


def _cd_from_inputs(
    *,
    mf_g_s: float,
    p_up_bar: float,
    p_down_bar: float,
    area_mm2: float,
    density_kg_m3: float,
) -> float:
    """Discharge coefficient from incompressible-flow orifice mass flow."""
    mf_kg_s = mf_g_s / _KG_TO_G
    a_m2 = area_mm2 * _MM2_TO_M2
    dp_pa = (p_up_bar - p_down_bar) * _BAR_TO_PA
    if dp_pa <= 0.0 or a_m2 <= 0.0 or density_kg_m3 <= 0.0:
        return float("nan")
    return mf_kg_s / (a_m2 * math.sqrt(2.0 * density_kg_m3 * dp_pa))


class ColdFlowPlugin(AnalysisPlugin):
    name: str = "cold_flow"
    version: str = "1.0.0"

    def __init__(self, config: ColdFlowConfig) -> None:
        self._cfg = config

    def required_channels(self) -> Sequence[str]:
        cm = self._cfg.channel_map
        return (cm.upstream_pressure, cm.downstream_pressure, cm.mass_flow)

    def compute(
        self, ctx: AnalysisContext
    ) -> Mapping[str, MeasurementWithUncertainty]:
        cm = self._cfg.channel_map
        avg = self._steady_means(ctx, [cm.upstream_pressure, cm.downstream_pressure, cm.mass_flow])
        out: dict[str, MeasurementWithUncertainty] = dict(avg)

        density, density_u = self._resolve_density(ctx)
        area_mm2 = self._resolve_geometry_value(ctx, "orifice_area_mm2")
        area_u = float(ctx.geometry_uncertainties.get("orifice_area_mm2", 0.0))

        cd_inputs = {
            "mf_g_s": avg[f"avg_{cm.mass_flow}"].value,
            "p_up_bar": avg[f"avg_{cm.upstream_pressure}"].value,
            "p_down_bar": avg[f"avg_{cm.downstream_pressure}"].value,
            "area_mm2": area_mm2,
            "density_kg_m3": density,
        }
        cd_uncs = {
            "mf_g_s": avg[f"avg_{cm.mass_flow}"].uncertainty,
            "p_up_bar": avg[f"avg_{cm.upstream_pressure}"].uncertainty,
            "p_down_bar": avg[f"avg_{cm.downstream_pressure}"].uncertainty,
            "area_mm2": area_u,
            "density_kg_m3": density_u,
        }
        try:
            cd_value, cd_u = propagate_analytical(
                _cd_from_inputs, cd_inputs, cd_uncs
            )
        except AnalysisError as e:
            raise AnalysisError(f"cold_flow Cd propagation failed: {e}") from e
        out["cd"] = MeasurementWithUncertainty(
            name="cd",
            value=cd_value,
            uncertainty=cd_u,
            unit="",
            provenance=Provenance.SENSOR,
        )
        return out

    def _steady_means(
        self, ctx: AnalysisContext, columns: Sequence[str]
    ) -> dict[str, MeasurementWithUncertainty]:
        ts = ctx.timestamp_column
        out: dict[str, MeasurementWithUncertainty] = {}
        for col in columns:
            if col not in ctx.steady_df.columns:
                raise ConfigError(
                    f"cold_flow: required channel '{col}' is not in steady_df "
                    f"(have: {sorted(ctx.steady_df.columns)})"
                )
            if col == ts:
                continue
            arr = ctx.steady_df[col].to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                raise AnalysisError(
                    f"cold_flow: channel '{col}' has no finite samples in the steady window"
                )
            mean = float(np.mean(finite))
            sem = (
                float(np.std(finite, ddof=1) / math.sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
            cal = ctx.sensor_calibrations.get(col)
            cal_u = cal.standard_uncertainty(mean) if cal is not None else 0.0
            u = math.sqrt(sem * sem + cal_u * cal_u)
            name = f"avg_{col}"
            out[name] = MeasurementWithUncertainty(
                name=name, value=mean, uncertainty=u, unit="",
                provenance=Provenance.SENSOR,
            )
        return out

    def _resolve_density(self, ctx: AnalysisContext) -> tuple[float, float]:
        extra = ctx.metadata.extra or {}
        if "density_kg_m3" in extra:
            try:
                value = float(extra["density_kg_m3"])
            except (TypeError, ValueError) as e:
                raise ConfigError(
                    f"cold_flow: metadata.extra['density_kg_m3'] is not numeric: {e}"
                ) from e
            u = float(extra.get("density_uncertainty_kg_m3", self._cfg.default_density_uncertainty_kg_m3))
            return value, u
        if self._cfg.default_density_kg_m3 is not None:
            return (
                float(self._cfg.default_density_kg_m3),
                float(self._cfg.default_density_uncertainty_kg_m3),
            )
        raise ConfigError(
            "cold_flow: density_kg_m3 must be supplied either in "
            "metadata.extra['density_kg_m3'] or via "
            "ColdFlowConfig.default_density_kg_m3"
        )

    @staticmethod
    def _resolve_geometry_value(ctx: AnalysisContext, key: str) -> float:
        if key not in ctx.geometry:
            raise ConfigError(
                f"cold_flow: required geometry parameter '{key}' missing "
                "(check metadata.geometry)"
            )
        try:
            return float(ctx.geometry[key])
        except (TypeError, ValueError) as e:
            raise ConfigError(
                f"cold_flow: geometry['{key}'] is not numeric: {e}"
            ) from e
