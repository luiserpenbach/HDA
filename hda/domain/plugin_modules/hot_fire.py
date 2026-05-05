"""Hot-fire plugin.

Computes the canonical hot-fire scalars from a steady-state slice with
full Jacobian uncertainty propagation:

    avg_<channel>   mean of each declared sensor channel ± sqrt(SEM^2 + cal^2).

    mf_total        m_dot_ox + m_dot_fuel.
                    Propagation: input set {mf_ox, mf_fuel}.

    of_ratio        m_dot_ox / m_dot_fuel.
                    Propagation: same input set.

    c_star          p_c * A_t / m_dot_total.
                    Propagation done over the *primary* sensors
                    {pc, mf_ox, mf_fuel, A_throat} rather than (mf_total,
                    pc, A_t) — the Jacobian over the actual measurements
                    correctly captures the contribution of each, since
                    mf_total is a deterministic function of the two flow
                    sensors.

    isp             F / (m_dot_total * g0).
                    Propagation over {thrust, mf_ox, mf_fuel}.

Configuration:

    ``HotFireConfig.channel_map`` declares which DataFrame column carries
    chamber pressure, thrust, oxidizer mass flow, and fuel mass flow.

    Throat area is read from ``metadata.geometry["throat_area_mm2"]`` and
    its calibration uncertainty from ``ctx.geometry_uncertainties``
    (defaults to 0). Missing throat area raises ``ConfigError`` with a
    clear message about where to put it — no silent placeholder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from hda.domain.errors import AnalysisError, ConfigError
from hda.domain.plugins import AnalysisContext, AnalysisPlugin
from hda.domain.types import MeasurementWithUncertainty, Provenance
from hda.domain.uncertainty import propagate_analytical


_log = logging.getLogger("hda.plugins.hot_fire")


_BAR_TO_PA = 1.0e5
_MM2_TO_M2 = 1.0e-6
_G_TO_KG = 1.0e-3
_G0_M_S2 = 9.80665


@dataclass(frozen=True, slots=True)
class HotFireChannelMap:
    chamber_pressure: str
    thrust: str
    mass_flow_ox: str
    mass_flow_fuel: str


@dataclass(frozen=True, slots=True)
class HotFireConfig:
    channel_map: HotFireChannelMap


def _mf_total_from_inputs(*, mf_ox_g_s: float, mf_fuel_g_s: float) -> float:
    return mf_ox_g_s + mf_fuel_g_s


def _of_ratio_from_inputs(*, mf_ox_g_s: float, mf_fuel_g_s: float) -> float:
    if mf_fuel_g_s == 0.0:
        return float("nan")
    return mf_ox_g_s / mf_fuel_g_s


def _c_star_from_inputs(
    *,
    pc_bar: float,
    mf_ox_g_s: float,
    mf_fuel_g_s: float,
    throat_area_mm2: float,
) -> float:
    """c* = p_c · A_t / m_dot_total."""
    mf_total_kg_s = (mf_ox_g_s + mf_fuel_g_s) * _G_TO_KG
    a_t_m2 = throat_area_mm2 * _MM2_TO_M2
    if mf_total_kg_s <= 0.0 or a_t_m2 <= 0.0:
        return float("nan")
    return (pc_bar * _BAR_TO_PA) * a_t_m2 / mf_total_kg_s


def _isp_from_inputs(
    *, thrust_n: float, mf_ox_g_s: float, mf_fuel_g_s: float
) -> float:
    """Isp = F / (m_dot · g_0)."""
    mf_total_kg_s = (mf_ox_g_s + mf_fuel_g_s) * _G_TO_KG
    if mf_total_kg_s <= 0.0:
        return float("nan")
    return thrust_n / (mf_total_kg_s * _G0_M_S2)


class HotFirePlugin(AnalysisPlugin):
    name: str = "hot_fire"
    version: str = "1.0.0"

    def __init__(self, config: HotFireConfig) -> None:
        self._cfg = config

    def required_channels(self) -> Sequence[str]:
        cm = self._cfg.channel_map
        return (
            cm.chamber_pressure,
            cm.thrust,
            cm.mass_flow_ox,
            cm.mass_flow_fuel,
        )

    def compute(
        self, ctx: AnalysisContext
    ) -> Mapping[str, MeasurementWithUncertainty]:
        cm = self._cfg.channel_map
        avg = self._steady_means(
            ctx,
            (
                cm.chamber_pressure,
                cm.thrust,
                cm.mass_flow_ox,
                cm.mass_flow_fuel,
            ),
        )
        out: dict[str, MeasurementWithUncertainty] = dict(avg)

        pc = avg[f"avg_{cm.chamber_pressure}"]
        thrust = avg[f"avg_{cm.thrust}"]
        mf_ox = avg[f"avg_{cm.mass_flow_ox}"]
        mf_fuel = avg[f"avg_{cm.mass_flow_fuel}"]

        a_throat = self._resolve_geometry_value(ctx, "throat_area_mm2")
        a_throat_u = float(
            ctx.geometry_uncertainties.get("throat_area_mm2", 0.0)
        )

        out["mf_total"] = self._propagate(
            "mf_total", "g/s",
            _mf_total_from_inputs,
            inputs={"mf_ox_g_s": mf_ox.value, "mf_fuel_g_s": mf_fuel.value},
            uncertainties={
                "mf_ox_g_s": mf_ox.uncertainty,
                "mf_fuel_g_s": mf_fuel.uncertainty,
            },
        )
        out["of_ratio"] = self._propagate(
            "of_ratio", "",
            _of_ratio_from_inputs,
            inputs={"mf_ox_g_s": mf_ox.value, "mf_fuel_g_s": mf_fuel.value},
            uncertainties={
                "mf_ox_g_s": mf_ox.uncertainty,
                "mf_fuel_g_s": mf_fuel.uncertainty,
            },
        )
        out["c_star"] = self._propagate(
            "c_star", "m/s",
            _c_star_from_inputs,
            inputs={
                "pc_bar": pc.value,
                "mf_ox_g_s": mf_ox.value,
                "mf_fuel_g_s": mf_fuel.value,
                "throat_area_mm2": a_throat,
            },
            uncertainties={
                "pc_bar": pc.uncertainty,
                "mf_ox_g_s": mf_ox.uncertainty,
                "mf_fuel_g_s": mf_fuel.uncertainty,
                "throat_area_mm2": a_throat_u,
            },
        )
        out["isp"] = self._propagate(
            "isp", "s",
            _isp_from_inputs,
            inputs={
                "thrust_n": thrust.value,
                "mf_ox_g_s": mf_ox.value,
                "mf_fuel_g_s": mf_fuel.value,
            },
            uncertainties={
                "thrust_n": thrust.uncertainty,
                "mf_ox_g_s": mf_ox.uncertainty,
                "mf_fuel_g_s": mf_fuel.uncertainty,
            },
        )
        return out

    @staticmethod
    def _propagate(name, unit, fn, *, inputs, uncertainties):
        """Propagate one derived metric.

        A degenerate input (e.g. zero fuel flow making OF undefined) emits
        the metric as NaN ± NaN with a warning to the structured log,
        rather than aborting the whole hot-fire analysis. The other
        metrics — pc, thrust, mf_total, c*, isp — typically remain
        computable and persisting them is more valuable than losing the
        whole run to one degenerate scalar.
        """
        try:
            value, u = propagate_analytical(fn, inputs, uncertainties)
        except AnalysisError as e:
            _log.warning(
                "hot_fire: '%s' yielded a non-finite result; emitting NaN. inputs=%s err=%s",
                name, dict(inputs), e,
            )
            value, u = float("nan"), float("nan")
        return MeasurementWithUncertainty(
            name=name,
            value=value,
            uncertainty=u,
            unit=unit,
            provenance=Provenance.SENSOR,
        )

    def _steady_means(
        self, ctx: AnalysisContext, columns: Sequence[str]
    ) -> dict[str, MeasurementWithUncertainty]:
        ts = ctx.timestamp_column
        out: dict[str, MeasurementWithUncertainty] = {}
        for col in columns:
            if col not in ctx.steady_df.columns:
                raise ConfigError(
                    f"hot_fire: required channel '{col}' is not in steady_df "
                    f"(have: {sorted(ctx.steady_df.columns)})"
                )
            if col == ts:
                continue
            arr = ctx.steady_df[col].to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                raise AnalysisError(
                    f"hot_fire: channel '{col}' has no finite samples in the steady window"
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

    @staticmethod
    def _resolve_geometry_value(ctx: AnalysisContext, key: str) -> float:
        if key not in ctx.geometry:
            raise ConfigError(
                f"hot_fire: required geometry parameter '{key}' missing "
                "(check metadata.geometry)"
            )
        try:
            return float(ctx.geometry[key])
        except (TypeError, ValueError) as e:
            raise ConfigError(
                f"hot_fire: geometry['{key}'] is not numeric: {e}"
            ) from e
