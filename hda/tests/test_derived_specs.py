"""Derived-channel/measurement spec invariants and the formula registry."""

from __future__ import annotations

import pytest

from hda.domain.derived import (
    DerivedChannelSpec,
    DerivedMeasurementSpec,
    FormulaLibrary,
    UncertaintyMethod,
)
from hda.domain.errors import ConfigError


def test_channel_spec_requires_name_and_formula():
    with pytest.raises(ConfigError):
        DerivedChannelSpec(name="", unit="g/s", formula="cd_orifice", inputs={})
    with pytest.raises(ConfigError):
        DerivedChannelSpec(name="mf_fuel", unit="g/s", formula="", inputs={})


def test_channel_spec_rejects_self_reference():
    with pytest.raises(ConfigError):
        DerivedChannelSpec(
            name="mf_fuel",
            unit="g/s",
            formula="cd_orifice",
            inputs={"p_up": "PT-up", "echo": "mf_fuel"},
        )


def test_channel_spec_source_names():
    spec = DerivedChannelSpec(
        name="mf_fuel",
        unit="g/s",
        formula="cd_orifice",
        inputs={"p_up": "PT-fuel-up", "p_down": "PT-fuel-down"},
    )
    assert set(spec.source_names()) == {"PT-fuel-up", "PT-fuel-down"}


def test_measurement_spec_basic_construction():
    spec = DerivedMeasurementSpec(
        name="of_ratio",
        unit="",
        formula="ratio",
        inputs={"num": "mf_ox", "den": "mf_fuel"},
        uncertainty_method=UncertaintyMethod.ANALYTICAL,
    )
    assert spec.name == "of_ratio"
    assert spec.uncertainty_method is UncertaintyMethod.ANALYTICAL
    assert set(spec.source_names()) == {"mf_ox", "mf_fuel"}


def test_measurement_spec_rejects_self_reference():
    with pytest.raises(ConfigError):
        DerivedMeasurementSpec(
            name="of",
            unit="",
            formula="ratio",
            inputs={"num": "mf_ox", "den": "of"},
        )


def test_formula_library_register_and_get():
    lib = FormulaLibrary()
    lib.register("cd_orifice", lambda **kw: 0.65, version="1.2.0")
    assert lib.get("cd_orifice")() == 0.65
    assert lib.version("cd_orifice") == "1.2.0"
    assert "cd_orifice" in lib.names()


def test_formula_library_rejects_duplicate_register():
    lib = FormulaLibrary()
    lib.register("ratio", lambda a, b: a / b)
    with pytest.raises(ConfigError):
        lib.register("ratio", lambda a, b: a / b)


def test_formula_library_unknown_lookup_raises():
    lib = FormulaLibrary()
    with pytest.raises(ConfigError):
        lib.get("does_not_exist")
    with pytest.raises(ConfigError):
        lib.version("does_not_exist")
