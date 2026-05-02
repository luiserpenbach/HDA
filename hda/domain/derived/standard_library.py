"""Standard derived-formula library.

Pure numerical functions registered into a ``FormulaLibrary`` and called
by the evaluator. Each function:

    - accepts numpy arrays or python scalars (numpy broadcasts);
    - uses SI internally (Pa, kg/m^3, m^2, kg/s, m/s, N, K) but accepts the
      common engineering units that show up in HDA configs and converts at
      the boundary (bar, mm^2, g/s);
    - is registered with a version string. When a formula's math changes,
      bump the version. The combined library version contributes to the
      run's processing version so traceability survives.

Library scope here is intentionally small: the helpers most of the
Streamlit cold-flow / hot-fire workflows need. Add more in follow-up
commits as plugins demand them.
"""

from __future__ import annotations

import numpy as np

from hda.domain.derived.spec import FormulaLibrary

_LIBRARY_VERSION = "1.0.0"

_BAR_TO_PA = 1.0e5
_MM2_TO_M2 = 1.0e-6
_KG_TO_G = 1.0e3
_G0_M_S2 = 9.80665  # standard gravity, ISO 80000-3


def _safe_divide(num, den):
    """Element-wise division; returns NaN where ``den`` is zero."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0.0, np.nan, num / np.where(den == 0.0, 1.0, den))
    if out.ndim == 0:
        return float(out)
    return out


def ratio(num, den):
    """``num / den``, NaN where ``den == 0``."""
    return _safe_divide(num, den)


def subtract(a, b):
    """``a - b``. Useful for pressure drops."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = a_arr - b_arr
    return float(out) if out.ndim == 0 else out


def cd_orifice_incompressible(
    p_up_bar,
    p_down_bar,
    density_kg_m3,
    area_mm2,
    cd,
):
    """Incompressible-flow orifice mass flow estimator [g/s].

    .. math::
        \\dot m = C_d \\cdot A \\cdot \\sqrt{2 \\rho \\Delta p}

    All inputs may be arrays (broadcasted). Returns g/s. Yields NaN where
    ``p_up < p_down`` (back-flow, outside the model's validity).
    """
    p_up = np.asarray(p_up_bar, dtype=float) * _BAR_TO_PA
    p_down = np.asarray(p_down_bar, dtype=float) * _BAR_TO_PA
    rho = np.asarray(density_kg_m3, dtype=float)
    a = np.asarray(area_mm2, dtype=float) * _MM2_TO_M2
    cd_arr = np.asarray(cd, dtype=float)
    dp = p_up - p_down
    valid = dp >= 0.0
    safe_dp = np.where(valid, dp, 0.0)
    mdot_kg_s = cd_arr * a * np.sqrt(2.0 * rho * safe_dp)
    mdot_kg_s = np.where(valid, mdot_kg_s, np.nan)
    out = mdot_kg_s * _KG_TO_G
    return float(out) if out.ndim == 0 else out


def c_star(pc_bar, mf_total_g_s, throat_area_mm2):
    """Characteristic velocity c* [m/s].

    .. math:: c^* = \\frac{p_c \\cdot A_t}{\\dot m}
    """
    pc_pa = np.asarray(pc_bar, dtype=float) * _BAR_TO_PA
    mf_kg_s = np.asarray(mf_total_g_s, dtype=float) / _KG_TO_G
    at_m2 = np.asarray(throat_area_mm2, dtype=float) * _MM2_TO_M2
    out = _safe_divide(pc_pa * at_m2, mf_kg_s)
    return out


def isp(thrust_n, mf_total_g_s):
    """Specific impulse [s].

    .. math:: I_{sp} = \\frac{F}{\\dot m \\cdot g_0}
    """
    f_n = np.asarray(thrust_n, dtype=float)
    mf_kg_s = np.asarray(mf_total_g_s, dtype=float) / _KG_TO_G
    return _safe_divide(f_n, mf_kg_s * _G0_M_S2)


def of_ratio(mf_ox_g_s, mf_fuel_g_s):
    """Oxidizer/fuel mass-flow ratio [-]."""
    return _safe_divide(mf_ox_g_s, mf_fuel_g_s)


def peak(series):
    """Maximum value in a 1D series. Channel→scalar reducer."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("peak() expects a 1-D series")
    if arr.size == 0:
        raise ValueError("peak() on empty series")
    return float(np.nanmax(arr))


def integrate_trapezoid(series, dt_s):
    """Trapezoidal integral of a uniformly-sampled series. Channel→scalar."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("integrate_trapezoid() expects a 1-D series")
    if arr.size < 2:
        raise ValueError("integrate_trapezoid() needs >= 2 samples")
    dt = float(dt_s)
    if dt <= 0:
        raise ValueError(f"dt_s must be > 0, got {dt}")
    return float(np.trapezoid(arr, dx=dt))


def standard_library() -> FormulaLibrary:
    """Return a fresh FormulaLibrary preloaded with the standard functions."""
    lib = FormulaLibrary()
    lib.register("ratio", ratio, version=_LIBRARY_VERSION)
    lib.register("subtract", subtract, version=_LIBRARY_VERSION)
    lib.register(
        "cd_orifice_incompressible",
        cd_orifice_incompressible,
        version=_LIBRARY_VERSION,
    )
    lib.register("c_star", c_star, version=_LIBRARY_VERSION)
    lib.register("isp", isp, version=_LIBRARY_VERSION)
    lib.register("of_ratio", of_ratio, version=_LIBRARY_VERSION)
    lib.register("peak", peak, version=_LIBRARY_VERSION)
    lib.register("integrate_trapezoid", integrate_trapezoid, version=_LIBRARY_VERSION)
    return lib
