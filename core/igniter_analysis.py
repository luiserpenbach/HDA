"""
Igniter hot-fire post-test analysis — N₂O / Ethanol (IGN-CP01 class).

Physics ported from igniter_test_utils.py (Hopper torch igniter):
  - N₂O mass flow: NHNE / Dyer two-phase model (SPI + HEM bracketing)
  - Ethanol: SPI (subcooled liquid)
  - CEA: rocketcea iterative Pc solve and c* efficiency
  - Cd back-calculation from measured N₂O flow

No Streamlit or Qt imports — safe for unit tests and plugins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import CoolProp.CoolProp as CP

    COOLPROP_AVAILABLE = True
except ImportError:
    CP = None  # type: ignore
    COOLPROP_AVAILABLE = False

try:
    from rocketcea.cea_obj_w_units import CEA_Obj

    ROCKETCEA_AVAILABLE = True
except ImportError:
    CEA_Obj = None  # type: ignore
    ROCKETCEA_AVAILABLE = False

N2O_FLUID = "HEOS::N2O"
ETHANOL_FLUID = "HEOS::Ethanol"
NOZZLE_SIZES_MM = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
G0 = 9.80665

_cea_instance: Optional[Any] = None


def coolprop_available() -> bool:
    return COOLPROP_AVAILABLE


def rocketcea_available() -> bool:
    return ROCKETCEA_AVAILABLE


def missing_dependencies() -> List[str]:
    missing: List[str] = []
    if not COOLPROP_AVAILABLE:
        missing.append("CoolProp")
    if not ROCKETCEA_AVAILABLE:
        missing.append("rocketcea")
    return missing


def get_cea() -> Any:
    """Lazy singleton CEA object for N2O / C2H5OH."""
    global _cea_instance
    if not ROCKETCEA_AVAILABLE:
        raise ImportError("rocketcea is required for CEA calculations (pip install rocketcea)")
    if _cea_instance is None:
        _cea_instance = CEA_Obj(
            oxName="N2O",
            fuelName="C2H5OH",
            pressure_units="Bar",
            cstar_units="m/s",
            temperature_units="K",
            isp_units="sec",
            density_units="kg/m^3",
        )
    return _cea_instance


@dataclass
class IgniterHardware:
    """Fixed hardware and model defaults for torch igniter tests."""

    throat_diameter_mm: float = 4.4
    cd_n2o: float = 0.77
    cd_eth: float = 0.77
    t_n2o_c: float = 20.0
    t_eth_c: float = 20.0
    eta_cstar_target: float = 0.85
    d_n2o_orifice_mm: float = 0.6
    d_eth_orifice_mm: float = 0.5
    p_amb_bar: float = 1.01325

    @property
    def throat_area_m2(self) -> float:
        return math.pi / 4.0 * (self.throat_diameter_mm * 1e-3) ** 2

    @property
    def t_n2o_k(self) -> float:
        return self.t_n2o_c + 273.15

    @property
    def t_eth_k(self) -> float:
        return self.t_eth_c + 273.15

    @classmethod
    def from_metadata(cls, metadata: Optional[Dict[str, Any]]) -> "IgniterHardware":
        if not metadata:
            return cls()
        hw = metadata.get("igniter_hardware") or metadata.get("hardware") or {}
        geom = metadata.get("geometry") or {}
        throat_mm = hw.get("throat_diameter_mm") or geom.get("throat_diameter_mm")
        kwargs: Dict[str, Any] = {}
        if throat_mm is not None:
            kwargs["throat_diameter_mm"] = float(throat_mm)
        for key in (
            "cd_n2o",
            "cd_eth",
            "t_n2o_c",
            "t_eth_c",
            "eta_cstar_target",
            "d_n2o_orifice_mm",
            "d_eth_orifice_mm",
            "p_amb_bar",
        ):
            if key in hw:
                kwargs[key] = float(hw[key])
        return cls(**kwargs)


@dataclass
class IgniterTestInputs:
    """Measured or averaged values for post-test analysis."""

    pc_bar: float
    mdot_n2o_g_s: float
    mdot_eth_g_s: Optional[float] = None
    p_eth_upstream_bar: Optional[float] = None
    d_eth_orifice_mm: Optional[float] = None
    cd_eth_override: Optional[float] = None
    p_n2o_upstream_bar: Optional[float] = None
    d_n2o_orifice_mm: Optional[float] = None
    oxidizer_name: str = "N2O"
    fuel_name: str = "Ethanol"
    input_source: str = "manual"


@dataclass
class IgniterAnalysisResult:
    """Complete igniter post-test analysis output."""

    pc_bar: float
    mdot_n2o_g_s: float
    mdot_eth_g_s: float
    mdot_eth_source: str
    mdot_total_g_s: float
    of_ratio: float
    cstar_actual_m_s: float
    cstar_theoretical_m_s: Optional[float] = None
    eta_cstar: Optional[float] = None
    pc_predicted_bar: Optional[float] = None
    tc_theo_k: Optional[float] = None
    cd_n2o_back: Optional[float] = None
    cd_n2o_delta: Optional[float] = None
    oxidizer_name: str = "N2O"
    fuel_name: str = "Ethanol"
    throat_diameter_mm: float = 4.4
    input_source: str = "manual"
    warnings: List[str] = field(default_factory=list)
    n2o_flow_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_report_rows(self) -> List[Dict[str, str]]:
        rows = [
            ("Oxidizer", self.oxidizer_name),
            ("Fuel", self.fuel_name),
            ("Pc measured [bar]", f"{self.pc_bar:.2f}"),
            ("ṁ N₂O [g/s]", f"{self.mdot_n2o_g_s:.3f}"),
            ("ṁ EtOH [g/s]", f"{self.mdot_eth_g_s:.3f} ({self.mdot_eth_source})"),
            ("ṁ total [g/s]", f"{self.mdot_total_g_s:.3f}"),
            ("O/F", f"{self.of_ratio:.3f}"),
            ("c* actual [m/s]", f"{self.cstar_actual_m_s:.2f}"),
            (
                "c* theoretical [m/s]",
                f"{self.cstar_theoretical_m_s:.2f}" if self.cstar_theoretical_m_s else "—",
            ),
            (
                "η_c*",
                f"{self.eta_cstar:.4f} ({self.eta_cstar * 100:.1f} %)"
                if self.eta_cstar is not None
                else "—",
            ),
            (
                "Cd N₂O (back-calc)",
                f"{self.cd_n2o_back:.4f}" if self.cd_n2o_back is not None else "—",
            ),
            (
                "Pc predicted [bar]",
                f"{self.pc_predicted_bar:.2f}" if self.pc_predicted_bar else "—",
            ),
            (
                "T_c theoretical [K]",
                f"{self.tc_theo_k:.0f}" if self.tc_theo_k else "—",
            ),
        ]
        return [{"Parameter": name, "Value": val} for name, val in rows]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def n2o_nhne(
    d_mm: float,
    cd: float,
    t_k: float,
    p_up_bar: float,
    p_dn_bar: float,
) -> Dict[str, Any]:
    """
    N₂O mass flow via NHNE (Dyer 2007).

    Returns SPI / HEM / NHNE mass flows plus diagnostics.
    """
    if not COOLPROP_AVAILABLE:
        raise ImportError("CoolProp is required for N₂O NHNE flow (pip install CoolProp)")

    fluid = N2O_FLUID
    p0 = p_up_bar * 1e5
    p1 = p_dn_bar * 1e5
    area = math.pi / 4.0 * (d_mm * 1e-3) ** 2

    h0 = CP.PropsSI("H", "T", t_k, "P", p0, fluid)
    s0 = CP.PropsSI("S", "T", t_k, "P", p0, fluid)

    phase_0 = int(CP.PropsSI("Phase", "T", t_k, "P", p0, fluid))
    if phase_0 in (0, 3):
        rho_l = CP.PropsSI("D", "T", t_k, "P", p0, fluid)
    else:
        rho_l = CP.PropsSI("D", "Q", 0, "P", p0, fluid)

    try:
        psat = CP.PropsSI("P", "T", t_k, "Q", 0, fluid)
    except Exception:
        psat = np.inf

    def g_hem(pt: float) -> float:
        try:
            s_l = CP.PropsSI("S", "Q", 0, "P", pt, fluid)
            s_v = CP.PropsSI("S", "Q", 1, "P", pt, fluid)
            h_l = CP.PropsSI("H", "Q", 0, "P", pt, fluid)
            h_v = CP.PropsSI("H", "Q", 1, "P", pt, fluid)
            r_l = CP.PropsSI("D", "Q", 0, "P", pt, fluid)
            r_v = CP.PropsSI("D", "Q", 1, "P", pt, fluid)
            x = (s0 - s_l) / (s_v - s_l)
            if x < 0.0:
                ht = CP.PropsSI("H", "P", pt, "S", s0, fluid)
                rt = CP.PropsSI("D", "P", pt, "S", s0, fluid)
            elif x > 1.0:
                ht = CP.PropsSI("H", "P", pt, "S", s0, fluid)
                rt = CP.PropsSI("D", "P", pt, "S", s0, fluid)
            else:
                ht = h_l + x * (h_v - h_l)
                rt = 1.0 / ((1.0 - x) / r_l + x / r_v)
            dh = h0 - ht
            return (rt * math.sqrt(2.0 * dh)) if dh > 0 else 0.0
        except Exception:
            return 0.0

    ps = np.linspace(0.999 * p0, 0.005 * p0, 300)
    gs = np.array([g_hem(p) for p in ps])
    idx = int(np.nanargmax(gs))
    pcrit = float(ps[idx])
    gcrit = float(gs[idx])

    from scipy.optimize import minimize_scalar

    lo = float(ps[min(idx + 3, 299)])
    hi = float(ps[max(idx - 3, 0)])
    try:
        opt = minimize_scalar(
            lambda p: -g_hem(p),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 100.0},
        )
        if -opt.fun > gcrit:
            pcrit, gcrit = opt.x, -opt.fun
    except Exception:
        pass

    choked = p1 <= pcrit
    g_eff = gcrit if choked else g_hem(p1)
    dp = max(p0 - p1, 0.0)

    m_spi = cd * area * math.sqrt(2.0 * rho_l * dp) if dp > 0 else 0.0
    m_hem = cd * area * g_eff
    kappa = float("nan")
    m_nhne = m_spi

    if np.isfinite(psat) and dp > 0:
        karg = (psat - p1) / (p0 - p1)
        kappa = math.sqrt(max(karg, 0.0))
        wspi = 1.0 / (1.0 + kappa)
        m_nhne = wspi * m_spi + (1.0 - wspi) * m_hem

    return {
        "mdot_spi": m_spi,
        "mdot_hem": m_hem,
        "mdot_nhne": m_nhne,
        "kappa": kappa,
        "Psat_bar": psat / 1e5 if np.isfinite(psat) else None,
        "Pcrit_bar": pcrit / 1e5,
        "Gcrit": gcrit,
        "choked": choked,
        "regime": "CHOKED" if choked else "SUBCRITICAL",
    }


def eth_spi(
    d_mm: float,
    cd: float,
    t_k: float,
    p_up_bar: float,
    p_dn_bar: float,
) -> float:
    """Ethanol SPI mass flow — subcooled liquid Bernoulli."""
    if not COOLPROP_AVAILABLE:
        raise ImportError("CoolProp is required for ethanol SPI flow (pip install CoolProp)")

    rho = CP.PropsSI("D", "T", t_k, "P", p_up_bar * 1e5, ETHANOL_FLUID)
    dp = max((p_up_bar - p_dn_bar) * 1e5, 0.0)
    area = math.pi / 4.0 * (d_mm * 1e-3) ** 2
    return cd * area * math.sqrt(2.0 * rho * dp) if dp > 0 else 0.0


def solve_pc(
    of_ratio: float,
    mdot_kg_s: float,
    eta: float,
    throat_area_m2: float,
    pc_init: float = 30.0,
) -> float:
    """Iterative chamber pressure: Pc = mdot * c*(Pc, OF) * eta / At."""
    cea = get_cea()
    pc = pc_init
    for _ in range(50):
        cstar = cea.get_Cstar(Pc=pc, MR=of_ratio)
        pc_new = mdot_kg_s * cstar * eta / throat_area_m2 / 1e5
        if abs(pc_new - pc) < 1e-4:
            return pc_new
        pc = 0.55 * pc + 0.45 * pc_new
    return pc


def cea_point(
    of_ratio: float,
    mdot_kg_s: float,
    eta: float,
    throat_area_m2: float,
) -> Dict[str, float]:
    """Full CEA evaluation at one operating point."""
    cea = get_cea()
    pc = solve_pc(of_ratio, mdot_kg_s, eta, throat_area_m2)
    tc = cea.get_Tcomb(Pc=pc, MR=of_ratio)
    cs = cea.get_Cstar(Pc=pc, MR=of_ratio)
    _, gamma = cea.get_Chamber_MolWt_gamma(Pc=pc, MR=of_ratio)
    return {
        "pc": pc,
        "Tc": tc,
        "cstar": cs,
        "gamma": gamma,
        "mdot_ox": mdot_kg_s * of_ratio / (1 + of_ratio),
        "mdot_fuel": mdot_kg_s / (1 + of_ratio),
    }


def resolve_time_column(df: pd.DataFrame) -> Optional[str]:
    for name in ("time_s", "time_ms", "timestamp", "Time", "TIME", "t"):
        if name in df.columns:
            return name
    return None


def steady_window_slice(
    df: pd.DataFrame,
    steady_window_s: Tuple[float, float],
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return rows inside the steady-state window (seconds)."""
    if df is None or df.empty:
        return df
    col = time_col or resolve_time_column(df)
    if not col:
        return df.iloc[0:0]

    t = df[col].astype(float)
    if col == "time_ms" or (col == "timestamp" and t.max() > 100.0):
        t = t / 1000.0
    start, end = steady_window_s
    mask = (t >= start) & (t <= end)
    return df.loc[mask]


def column_means(
    df: pd.DataFrame,
    columns: Dict[str, str],
) -> Dict[str, float]:
    """Average numeric columns; keys are role names, values are sensor column names."""
    out: Dict[str, float] = {}
    for role, col in columns.items():
        if col and col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                out[role] = float(series.mean())
    return out


def inputs_from_steady_state(
    df: pd.DataFrame,
    sensor_roles: Dict[str, str],
    steady_window_s: Tuple[float, float],
    hardware: IgniterHardware,
    time_col: Optional[str] = None,
) -> IgniterTestInputs:
    """
    Build igniter inputs from steady-window averages.

    Expected sensor_roles keys (any subset):
      chamber_pressure, mass_flow_ox, mass_flow_fuel,
      upstream_pressure (fuel upstream for ΔP estimate)
    """
    steady_df = steady_window_slice(df, steady_window_s, time_col)
    avgs = column_means(steady_df, sensor_roles)

    pc = avgs.get("chamber_pressure")
    mdot_ox = avgs.get("mass_flow_ox")
    if pc is None or mdot_ox is None:
        missing = []
        if pc is None:
            missing.append("chamber_pressure")
        if mdot_ox is None:
            missing.append("mass_flow_ox")
        raise ValueError(
            f"Steady-state averages missing required roles: {', '.join(missing)}"
        )

    mdot_fuel = avgs.get("mass_flow_fuel")
    p_up = avgs.get("upstream_pressure")

    return IgniterTestInputs(
        pc_bar=pc,
        mdot_n2o_g_s=mdot_ox,
        mdot_eth_g_s=mdot_fuel,
        p_eth_upstream_bar=p_up,
        d_eth_orifice_mm=hardware.d_eth_orifice_mm,
        cd_eth_override=hardware.cd_eth,
        p_n2o_upstream_bar=None,
        d_n2o_orifice_mm=hardware.d_n2o_orifice_mm,
        input_source="steady_state",
    )


def analyze_igniter_post_test(
    inputs: IgniterTestInputs,
    hardware: IgniterHardware,
) -> IgniterAnalysisResult:
    """
    Post-test igniter analysis: O/F, c* efficiency, Cd back-calculation.

    Ethanol mass flow uses measured FU-FM if provided; otherwise SPI estimate
    from fuel upstream pressure and chamber pressure.
    """
    warnings: List[str] = []
    if missing := missing_dependencies():
        warnings.append(f"Optional packages missing: {', '.join(missing)}")

    pc = inputs.pc_bar
    mdot_n2o_kg = inputs.mdot_n2o_g_s * 1e-3

    mdot_eth_kg: float
    eth_source: str
    if inputs.mdot_eth_g_s is not None and inputs.mdot_eth_g_s > 0:
        mdot_eth_kg = inputs.mdot_eth_g_s * 1e-3
        eth_source = "measured"
    else:
        if not COOLPROP_AVAILABLE:
            raise ImportError(
                "CoolProp required to estimate ethanol flow from injector ΔP"
            )
        p_eth_up = inputs.p_eth_upstream_bar
        if p_eth_up is None:
            raise ValueError(
                "Fuel mass flow not measured — provide upstream_pressure in steady data "
                "or enter ethanol upstream P manually."
            )
        d_eth = inputs.d_eth_orifice_mm or hardware.d_eth_orifice_mm
        cd_eth = inputs.cd_eth_override if inputs.cd_eth_override is not None else hardware.cd_eth
        p_eth_dn = max(pc, 0.5)
        mdot_eth_kg = eth_spi(d_eth, cd_eth, hardware.t_eth_k, p_eth_up, p_eth_dn)
        eth_source = "estimated_spi"

    mdot_total_kg = mdot_n2o_kg + mdot_eth_kg
    of_ratio = mdot_n2o_kg / mdot_eth_kg if mdot_eth_kg > 0 else float("nan")
    cstar_actual = pc * 1e5 * hardware.throat_area_m2 / mdot_total_kg

    cstar_theo: Optional[float] = None
    eta_cstar: Optional[float] = None
    pc_predicted: Optional[float] = None
    tc_theo: Optional[float] = None

    if ROCKETCEA_AVAILABLE and math.isfinite(of_ratio):
        try:
            cea = get_cea()
            cstar_theo = float(cea.get_Cstar(Pc=pc, MR=of_ratio))
            eta_cstar = cstar_actual / cstar_theo if cstar_theo > 0 else None
            pc_predicted = solve_pc(
                of_ratio,
                mdot_total_kg,
                hardware.eta_cstar_target,
                hardware.throat_area_m2,
            )
            tc_theo = float(cea.get_Tcomb(Pc=pc, MR=of_ratio))
        except Exception as exc:
            warnings.append(f"CEA calculation failed: {exc}")
    elif not ROCKETCEA_AVAILABLE:
        warnings.append("Install rocketcea for theoretical c* and η_c*")

    cd_n2o_back: Optional[float] = None
    cd_delta: Optional[float] = None
    n2o_diag: Dict[str, Any] = {}

    p_n2o_up = inputs.p_n2o_upstream_bar
    d_n2o = inputs.d_n2o_orifice_mm or hardware.d_n2o_orifice_mm
    if p_n2o_up is not None and COOLPROP_AVAILABLE:
        try:
            r_n2o = n2o_nhne(d_n2o, 1.0, hardware.t_n2o_k, p_n2o_up, pc)
            n2o_diag = r_n2o
            if r_n2o["mdot_nhne"] > 0:
                cd_n2o_back = mdot_n2o_kg / r_n2o["mdot_nhne"]
                cd_delta = cd_n2o_back - hardware.cd_n2o
        except Exception as exc:
            warnings.append(f"N₂O Cd back-calculation failed: {exc}")

    return IgniterAnalysisResult(
        pc_bar=pc,
        mdot_n2o_g_s=inputs.mdot_n2o_g_s,
        mdot_eth_g_s=mdot_eth_kg * 1e3,
        mdot_eth_source=eth_source,
        mdot_total_g_s=mdot_total_kg * 1e3,
        of_ratio=of_ratio,
        cstar_actual_m_s=cstar_actual,
        cstar_theoretical_m_s=cstar_theo,
        eta_cstar=eta_cstar,
        pc_predicted_bar=pc_predicted,
        tc_theo_k=tc_theo,
        cd_n2o_back=cd_n2o_back,
        cd_n2o_delta=cd_delta,
        oxidizer_name=inputs.oxidizer_name,
        fuel_name=inputs.fuel_name,
        throat_diameter_mm=hardware.throat_diameter_mm,
        input_source=inputs.input_source,
        warnings=warnings,
        n2o_flow_diagnostics=n2o_diag,
    )
