"""Uncertainty propagation.

Two methods, both available on every derived measurement:

    Analytical (numerical Jacobian)
        sigma_y^2 = sum_i (df/dx_i)^2 * sigma_x_i^2
        Partials are computed by central finite differences with an
        adaptive step h = max(|x|, 1) * 1e-6. Fast (2 evaluations per
        input), exact for linear functions, accurate to ~ppm for smooth
        ones.

    Monte Carlo
        Sample each input from N(x, sigma) n_samples times, evaluate the
        function on the joint sample, return mean + std. Robust to
        nonlinearity and pathological derivatives, slower. The default
        n_samples=10_000 keeps stochastic noise in the output uncertainty
        below ~1% for typical inputs.

Both signatures are identical so callers swap methods by name. Inputs are
specified by name → mean and name → standard uncertainty; constants the
function needs but whose uncertainty is not propagated go in ``fixed``.

Sensor uncertainty model:
    Three calibration shapes used across rocket-test instrumentation:
    absolute, relative-to-reading, and percent-of-full-scale. The
    ``standard_uncertainty(reading)`` method always returns a magnitude
    (positive). The legacy app's signed-vs-abs() bug is moot here because
    standard uncertainty is by definition non-negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

import numpy as np

from hda.domain.errors import AnalysisError, ConfigError


class UncertaintyKind(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    PERCENT_FS = "percent_fs"


@dataclass(frozen=True, slots=True)
class SensorUncertainty:
    kind: UncertaintyKind
    value: float
    full_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ConfigError(
                f"SensorUncertainty.value must be >= 0, got {self.value}"
            )
        if self.kind is UncertaintyKind.PERCENT_FS and self.full_scale is None:
            raise ConfigError(
                "SensorUncertainty.PERCENT_FS requires a full_scale"
            )

    def standard_uncertainty(self, reading: float) -> float:
        if self.kind is UncertaintyKind.ABSOLUTE:
            return self.value
        if self.kind is UncertaintyKind.RELATIVE:
            return abs(reading) * self.value
        if self.kind is UncertaintyKind.PERCENT_FS:
            assert self.full_scale is not None
            return abs(self.full_scale) * self.value
        raise ConfigError(f"Unknown uncertainty kind {self.kind}")


def _step_for(x: float) -> float:
    return max(abs(x), 1.0) * 1e-6


def propagate_analytical(
    fn: Callable[..., float],
    inputs: Mapping[str, float],
    uncertainties: Mapping[str, float],
    fixed: Optional[Mapping[str, Any]] = None,
) -> tuple[float, float]:
    """Return (value, std uncertainty) by central-difference Jacobian.

    Inputs whose uncertainty is 0 contribute nothing — their partials are
    skipped, so the function call count is 1 + 2*nonzero_inputs.
    """
    fixed = dict(fixed or {})
    inputs = dict(inputs)
    try:
        value = float(fn(**inputs, **fixed))
    except Exception as e:
        raise AnalysisError(f"propagate_analytical: function failed at nominal point: {e}") from e
    if not math.isfinite(value):
        raise AnalysisError(
            f"propagate_analytical: function returned non-finite value {value} at nominal point"
        )

    var = 0.0
    for name, x in inputs.items():
        sigma = float(uncertainties.get(name, 0.0))
        if sigma == 0.0:
            continue
        h = _step_for(x)
        plus = dict(inputs)
        plus[name] = x + h
        minus = dict(inputs)
        minus[name] = x - h
        try:
            f_plus = float(fn(**plus, **fixed))
            f_minus = float(fn(**minus, **fixed))
        except Exception as e:
            raise AnalysisError(
                f"propagate_analytical: function failed in finite-difference "
                f"step for input '{name}': {e}"
            ) from e
        if not (math.isfinite(f_plus) and math.isfinite(f_minus)):
            raise AnalysisError(
                f"propagate_analytical: non-finite finite-difference for '{name}' "
                f"(h={h}, f+={f_plus}, f-={f_minus})"
            )
        partial = (f_plus - f_minus) / (2.0 * h)
        var += (partial * sigma) ** 2
    return value, math.sqrt(var)


def propagate_monte_carlo(
    fn: Callable[..., float],
    inputs: Mapping[str, float],
    uncertainties: Mapping[str, float],
    fixed: Optional[Mapping[str, Any]] = None,
    n_samples: int = 10_000,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Return (mean, std) by joint-Gaussian Monte Carlo over inputs.

    Tries a single vectorized call first (most numpy-based formulas
    accept arrays via broadcasting). Falls back to per-sample evaluation
    if the function is scalar-only.
    """
    if n_samples < 100:
        raise ConfigError(f"n_samples must be >= 100, got {n_samples}")
    fixed = dict(fixed or {})
    rng = np.random.default_rng(seed)
    samples: dict[str, np.ndarray] = {}
    for name, x in inputs.items():
        sigma = float(uncertainties.get(name, 0.0))
        if sigma > 0.0:
            samples[name] = rng.normal(loc=x, scale=sigma, size=n_samples)
        else:
            samples[name] = np.full(n_samples, x, dtype=float)

    out: np.ndarray
    try:
        candidate = fn(**samples, **fixed)
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape == (n_samples,):
            out = candidate
        else:
            raise TypeError("not vectorized")
    except Exception:
        out = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            try:
                out[i] = float(
                    fn(**{k: float(v[i]) for k, v in samples.items()}, **fixed)
                )
            except Exception:
                out[i] = np.nan

    finite = out[np.isfinite(out)]
    if finite.size == 0:
        raise AnalysisError(
            "propagate_monte_carlo: every sample produced a non-finite result"
        )
    if finite.size < n_samples * 0.5:
        raise AnalysisError(
            f"propagate_monte_carlo: {n_samples - finite.size}/{n_samples} samples "
            "produced non-finite output; the formula may be ill-conditioned at "
            "these inputs"
        )
    return float(np.mean(finite)), float(np.std(finite, ddof=1))
