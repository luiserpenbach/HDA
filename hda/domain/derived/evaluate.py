"""Evaluator for derived channels (per-sample) and derived measurements (scalar).

A derived channel is a numpy array of the same shape as its input sensor
channels, computed during preprocessing. Derived channels participate in
QC, steady-state detection, plotting, and analysis identically to raw
channels.

A derived measurement is a per-test scalar, computed during analysis from
either raw scalars (e.g., the steady-state mean of a sensor channel) or
other derived measurements. It is persisted as a ``MeasurementWithUncertainty``
with ``provenance=DERIVED``.

Topological evaluation:
    Each spec declares ``inputs`` whose values are source names. Source names
    may refer to sensors, scalars in the context, or other derived specs.
    The evaluator orders specs so dependencies are evaluated first and
    raises ``ConfigError`` on cycles or unresolved sources.

Uncertainty propagation lives in a follow-up commit; this commit produces
values only and emits ``MeasurementWithUncertainty`` with ``uncertainty=0.0``
for derived measurements with ``UncertaintyMethod.NONE``. Other methods
raise ``NotImplementedError`` so we never silently fabricate a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from hda.domain.derived.spec import (
    DerivedChannelSpec,
    DerivedMeasurementSpec,
    FormulaLibrary,
    UncertaintyMethod,
)
from hda.domain.errors import AnalysisError, ConfigError
from hda.domain.types import MeasurementWithUncertainty, Provenance
from hda.domain.uncertainty import propagate_analytical, propagate_monte_carlo


@dataclass(frozen=True, slots=True)
class DerivedContext:
    """Inputs to the derived evaluator.

    Two scalar layers, distinguished by whether the value carries an
    uncertainty:

        ``sensor_measurements``: name -> MeasurementWithUncertainty.
            For inputs that have a known uncertainty (sensor steady-state
            means produced by a plugin, geometry parameters with calibration
            uncertainty). Used for scalar uncertainty propagation.

        ``sensor_scalars`` / ``geometry`` / ``metadata``: name -> float / Any.
            For inputs treated as exact constants. Resolved when an input
            is not present in ``sensor_measurements``; uncertainty defaults
            to 0 for those.

    Channel inputs (per-sample arrays) live in ``sensor_channels`` and are
    used only by ``evaluate_channels``.
    """

    sensor_channels: Mapping[str, np.ndarray] = field(default_factory=dict)
    sensor_scalars: Mapping[str, float] = field(default_factory=dict)
    sensor_measurements: Mapping[str, MeasurementWithUncertainty] = field(
        default_factory=dict
    )
    geometry: Mapping[str, float] = field(default_factory=dict)
    geometry_uncertainties: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def all_scalar_names(self) -> set[str]:
        return (
            set(self.sensor_scalars)
            | set(self.sensor_measurements)
            | set(self.geometry)
            | set(self.metadata)
        )

    def all_channel_names(self) -> set[str]:
        return set(self.sensor_channels)

    def lookup_scalar(self, name: str) -> Any:
        if name in self.sensor_measurements:
            return self.sensor_measurements[name].value
        if name in self.sensor_scalars:
            return self.sensor_scalars[name]
        if name in self.geometry:
            return self.geometry[name]
        if name in self.metadata:
            return self.metadata[name]
        raise KeyError(name)

    def lookup_uncertainty(self, name: str) -> float:
        """Return the std uncertainty associated with a scalar input.

        Defaults to 0 for inputs sourced from ``sensor_scalars``,
        ``metadata``, or ``geometry`` that have no entry in
        ``geometry_uncertainties``.
        """
        if name in self.sensor_measurements:
            return float(self.sensor_measurements[name].uncertainty)
        if name in self.geometry_uncertainties:
            return float(self.geometry_uncertainties[name])
        return 0.0


def _topo_order(
    specs: Sequence[DerivedChannelSpec | DerivedMeasurementSpec],
    available: Iterable[str],
) -> List[DerivedChannelSpec | DerivedMeasurementSpec]:
    """Kahn's algorithm. Raises ConfigError on cycles or unresolved sources."""
    by_name = {s.name: s for s in specs}
    if len(by_name) != len(specs):
        seen: set[str] = set()
        for s in specs:
            if s.name in seen:
                raise ConfigError(f"Duplicate derived spec name: {s.name}")
            seen.add(s.name)

    available_set = set(available)
    derived_names = set(by_name.keys())

    deps: Dict[str, set[str]] = {}
    for s in specs:
        srcs = set(s.source_names())
        unresolved = srcs - available_set - derived_names
        if unresolved:
            raise ConfigError(
                f"Derived spec '{s.name}' references unknown sources: "
                f"{sorted(unresolved)}"
            )
        deps[s.name] = srcs & derived_names

    ordered: List[DerivedChannelSpec | DerivedMeasurementSpec] = []
    ready = [name for name, d in deps.items() if not d]
    pending = {name for name, d in deps.items() if d}

    while ready:
        name = ready.pop(0)
        ordered.append(by_name[name])
        for other in list(pending):
            if name in deps[other]:
                deps[other].discard(name)
                if not deps[other]:
                    ready.append(other)
                    pending.remove(other)

    if pending:
        raise ConfigError(
            f"Cyclic dependency among derived specs: {sorted(pending)}"
        )
    return ordered


def evaluate_channels(
    specs: Sequence[DerivedChannelSpec],
    ctx: DerivedContext,
    library: FormulaLibrary,
) -> Dict[str, np.ndarray]:
    """Evaluate derived channels in dependency order.

    Returns a mapping name -> np.ndarray. Each output array is broadcast to
    the same length as the longest input array.
    """
    available = ctx.all_channel_names() | ctx.all_scalar_names()
    ordered = _topo_order(list(specs), available)
    out: Dict[str, np.ndarray] = {}

    for spec in ordered:
        assert isinstance(spec, DerivedChannelSpec)
        kwargs: Dict[str, Any] = {}
        for kwarg, src in spec.inputs.items():
            if src in ctx.sensor_channels:
                kwargs[kwarg] = ctx.sensor_channels[src]
            elif src in out:
                kwargs[kwarg] = out[src]
            else:
                try:
                    kwargs[kwarg] = ctx.lookup_scalar(src)
                except KeyError:
                    raise ConfigError(
                        f"Derived channel '{spec.name}': cannot resolve "
                        f"input '{src}' (kwarg '{kwarg}')"
                    )
        kwargs.update(spec.params)
        fn = library.get(spec.formula)
        try:
            value = fn(**kwargs)
        except Exception as e:
            raise AnalysisError(
                f"Derived channel '{spec.name}' failed in formula "
                f"'{spec.formula}': {e}"
            ) from e
        out[spec.name] = np.asarray(value, dtype=float)
    return out


def evaluate_measurements(
    specs: Sequence[DerivedMeasurementSpec],
    ctx: DerivedContext,
    library: FormulaLibrary,
    monte_carlo_samples: int = 10_000,
    monte_carlo_seed: int | None = None,
) -> Dict[str, MeasurementWithUncertainty]:
    """Evaluate scalar derived measurements in dependency order.

    Each spec's ``uncertainty_method`` controls propagation:

        NONE          -> uncertainty = 0.0
        ANALYTICAL    -> central-difference Jacobian
        MONTE_CARLO   -> joint-Gaussian sampling
        PROPAGATE     -> alias for ANALYTICAL

    Inputs sourced from ``sensor_measurements`` carry their own uncertainty
    forward; inputs from ``sensor_scalars`` / ``metadata`` / ``geometry``
    contribute zero unless an entry is provided in
    ``geometry_uncertainties``. Earlier-evaluated derived measurements
    feed into later specs as ``MeasurementWithUncertainty`` so chained
    derivations propagate correctly.

    Reduction of time-series channels to steady-state scalars (e.g. mean
    of ``mf_fuel`` over the steady window) is the caller's responsibility
    and lives in the analysis service — keeping the reduction policy
    out of the evaluator means it stays explicit and reviewable.
    """
    available = ctx.all_scalar_names()
    ordered = _topo_order(list(specs), available)
    derived_scalars: Dict[str, MeasurementWithUncertainty] = {}
    out: Dict[str, MeasurementWithUncertainty] = {}

    for spec in ordered:
        assert isinstance(spec, DerivedMeasurementSpec)
        input_kwargs: Dict[str, Any] = {}
        input_uncs: Dict[str, float] = {}
        for kwarg, src in spec.inputs.items():
            if src in derived_scalars:
                m = derived_scalars[src]
                input_kwargs[kwarg] = m.value
                input_uncs[kwarg] = m.uncertainty
            else:
                try:
                    input_kwargs[kwarg] = ctx.lookup_scalar(src)
                except KeyError:
                    raise ConfigError(
                        f"Derived measurement '{spec.name}': cannot resolve "
                        f"input '{src}' (kwarg '{kwarg}')"
                    )
                input_uncs[kwarg] = ctx.lookup_uncertainty(src)
        fn = library.get(spec.formula)
        fixed_params = dict(spec.params)
        method = spec.uncertainty_method

        if method is UncertaintyMethod.NONE:
            try:
                value = float(fn(**input_kwargs, **fixed_params))
            except Exception as e:
                raise AnalysisError(
                    f"Derived measurement '{spec.name}' failed in formula "
                    f"'{spec.formula}': {e}"
                ) from e
            uncertainty = 0.0
        elif method in (UncertaintyMethod.ANALYTICAL, UncertaintyMethod.PROPAGATE):
            try:
                value, uncertainty = propagate_analytical(
                    fn, input_kwargs, input_uncs, fixed=fixed_params
                )
            except AnalysisError as e:
                raise AnalysisError(
                    f"Derived measurement '{spec.name}' analytical "
                    f"propagation failed: {e}"
                ) from e
        elif method is UncertaintyMethod.MONTE_CARLO:
            try:
                value, uncertainty = propagate_monte_carlo(
                    fn,
                    input_kwargs,
                    input_uncs,
                    fixed=fixed_params,
                    n_samples=monte_carlo_samples,
                    seed=monte_carlo_seed,
                )
            except AnalysisError as e:
                raise AnalysisError(
                    f"Derived measurement '{spec.name}' Monte Carlo "
                    f"propagation failed: {e}"
                ) from e
        else:
            raise ConfigError(
                f"Unknown UncertaintyMethod {method} for derived measurement "
                f"'{spec.name}'"
            )

        m = MeasurementWithUncertainty(
            name=spec.name,
            value=value,
            uncertainty=uncertainty,
            unit=spec.unit,
            provenance=Provenance.DERIVED,
        )
        derived_scalars[spec.name] = m
        out[spec.name] = m
    return out
