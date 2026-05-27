"""Tests for Single Test Analysis handoff helpers."""

from hda.test_type_utils import normalize_test_type


def test_normalize_test_type_cold_flow_aliases():
    assert normalize_test_type("CF") == "cold_flow"
    assert normalize_test_type("cold_flow") == "cold_flow"
    assert normalize_test_type("cold flow") == "cold_flow"


def test_normalize_test_type_hot_fire_aliases():
    assert normalize_test_type("HF") == "hot_fire"
    assert normalize_test_type("hot_fire") == "hot_fire"
    assert normalize_test_type("hot fire") == "hot_fire"


def test_normalize_test_type_unknown_defaults_cold_flow():
    assert normalize_test_type("") == "cold_flow"
    assert normalize_test_type("unknown") == "cold_flow"
