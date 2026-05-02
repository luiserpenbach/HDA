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


@dataclass(frozen=True, slots=True)
class DerivedContext:
    """Inputs to the derived evaluator.

    Channel inputs and scalar inputs live in separate maps so a name like
    ``mf_fuel`` can unambiguously resolve to either a per-sample array (when
    evaluating channels) or a steady-state scalar (when evaluating
    measurements).
    """

    sensor_channels: Mapping[str, np.ndarray] = field(default_factory=dict)
    sensor_scalars: Mapping[str, float] = field(default_factory=dict)
    geometry: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def all_scalar_names(self) -> set[str]:
        return set(self.sensor_scalars) | set(self.geometry) | set(self.metadata)

    def all_channel_names(self) -> set[str]:
        return set(self.sensor_channels)

    def lookup_scalar(self, name: str) -> Any:
        if name in self.sensor_scalars:
            return self.sensor_scalars[name]
        if name in self.geometry:
            return self.geometry[name]
        if name in self.metadata:
            return self.metadata[name]
        raise KeyError(name)


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
) -> Dict[str, MeasurementWithUncertainty]:
    """Evaluate scalar derived measurements in dependency order.

    The caller is responsible for placing channel-scope inputs (e.g. the
    steady-state mean of ``mf_fuel``) into ``ctx.sensor_scalars`` before
    invoking the evaluator. Channels are not auto-reduced here so the
    reduction policy (mean / median / robust) stays explicit and visible.
    """
    available = ctx.all_scalar_names()
    ordered = _topo_order(list(specs), available)
    scalars: Dict[str, float] = {}
    out: Dict[str, MeasurementWithUncertainty] = {}

    for spec in ordered:
        assert isinstance(spec, DerivedMeasurementSpec)
        kwargs: Dict[str, Any] = {}
        for kwarg, src in spec.inputs.items():
            if src in scalars:
                kwargs[kwarg] = scalars[src]
            else:
                try:
                    kwargs[kwarg] = ctx.lookup_scalar(src)
                except KeyError:
                    raise ConfigError(
                        f"Derived measurement '{spec.name}': cannot resolve "
                        f"input '{src}' (kwarg '{kwarg}')"
                    )
        kwargs.update(spec.params)
        fn = library.get(spec.formula)
        try:
            value = float(fn(**kwargs))
        except Exception as e:
            raise AnalysisError(
                f"Derived measurement '{spec.name}' failed in formula "
                f"'{spec.formula}': {e}"
            ) from e

        if spec.uncertainty_method is UncertaintyMethod.NONE:
            uncertainty = 0.0
        else:
            raise NotImplementedError(
                f"Uncertainty method {spec.uncertainty_method.value} for "
                f"derived measurement '{spec.name}' is not implemented yet. "
                "Use UncertaintyMethod.NONE until propagation lands."
            )

        scalars[spec.name] = value
        out[spec.name] = MeasurementWithUncertainty(
            name=spec.name,
            value=value,
            uncertainty=uncertainty,
            unit=spec.unit,
            provenance=Provenance.DERIVED,
        )
    return out
