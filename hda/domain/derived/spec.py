"""Specifications for calculated channels and measurements.

A derived channel is a time-series computed during preprocessing from raw
sensor channels (e.g., fuel mass flow estimated from upstream pressure and a
discharge coefficient). It then participates in QC, steady-state detection,
plotting, and analysis identically to a raw channel.

A derived measurement is a scalar produced during analysis (e.g., O/F ratio
from two mass flows, one of which may itself be derived).

This module declares the data shapes and the formula registry. The evaluator
that consumes them lives in a later commit (``hda.domain.derived.evaluate``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Mapping, Sequence

from hda.domain.errors import ConfigError


class UncertaintyMethod(str, Enum):
    ANALYTICAL = "analytical"
    MONTE_CARLO = "monte_carlo"
    PROPAGATE = "propagate"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class DerivedChannelSpec:
    """Spec for a per-sample derived time-series channel."""

    name: str
    unit: str
    formula: str
    inputs: Sequence[str]
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.PROPAGATE
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("DerivedChannelSpec.name must be non-empty")
        if not self.formula:
            raise ConfigError(
                f"DerivedChannelSpec '{self.name}' requires a formula"
            )
        if self.name in self.inputs:
            raise ConfigError(
                f"DerivedChannelSpec '{self.name}' cannot reference itself in inputs"
            )


@dataclass(frozen=True, slots=True)
class DerivedMeasurementSpec:
    """Spec for a per-test scalar derived measurement."""

    name: str
    unit: str
    formula: str
    inputs: Sequence[str]
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.ANALYTICAL
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("DerivedMeasurementSpec.name must be non-empty")
        if not self.formula:
            raise ConfigError(
                f"DerivedMeasurementSpec '{self.name}' requires a formula"
            )


FormulaFn = Callable[..., float]


class FormulaLibrary:
    """Registry of named formula functions referenced by derived specs.

    Specs reference functions by name (e.g. ``cd_orifice``); the evaluator
    looks them up here. Keeping this layer indirect lets the formula library
    be versioned and shipped in templates without serializing live callables.
    """

    def __init__(self) -> None:
        self._fns: Dict[str, FormulaFn] = {}
        self._versions: Dict[str, str] = {}

    def register(self, name: str, fn: FormulaFn, version: str = "1.0.0") -> None:
        if name in self._fns:
            raise ConfigError(f"Formula '{name}' already registered")
        self._fns[name] = fn
        self._versions[name] = version

    def get(self, name: str) -> FormulaFn:
        if name not in self._fns:
            raise ConfigError(f"Unknown formula '{name}'")
        return self._fns[name]

    def version(self, name: str) -> str:
        if name not in self._versions:
            raise ConfigError(f"Unknown formula '{name}'")
        return self._versions[name]

    def names(self) -> Sequence[str]:
        return tuple(self._fns.keys())
