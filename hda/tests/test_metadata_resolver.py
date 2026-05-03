"""Three-layer metadata resolution + sidecar loader + canonical hashing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hda.domain.errors import IngestError
from hda.domain.metadata import (
    FieldType,
    MetadataField,
    MetadataLayer,
    MetadataSchema,
    canonical_metadata_hash,
    load_sidecar,
    resolve_metadata,
)


def _schema() -> MetadataSchema:
    return MetadataSchema(
        fields=(
            MetadataField("part_number", FieldType.STRING, required=True),
            MetadataField("serial_number", FieldType.STRING, required=True),
            MetadataField("fuel_additive", FieldType.STRING),
            MetadataField("additive_pct", FieldType.FLOAT),
            MetadataField("operator", FieldType.STRING),
        )
    )


def test_sidecar_wins_over_campaign_and_operator():
    res = resolve_metadata(
        _schema(),
        sidecar={"part_number": "PN-FROM-SIDECAR"},
        campaign_defaults={"part_number": "PN-FROM-CAMPAIGN"},
        operator={"part_number": "PN-FROM-OPERATOR"},
    )
    assert res.values["part_number"] == "PN-FROM-SIDECAR"
    assert res.sources["part_number"] is MetadataLayer.SIDECAR


def test_campaign_fills_when_sidecar_omits():
    res = resolve_metadata(
        _schema(),
        sidecar={"part_number": "PN-1"},
        campaign_defaults={"operator": "alice"},
        operator=None,
    )
    assert res.values["operator"] == "alice"
    assert res.sources["operator"] is MetadataLayer.CAMPAIGN


def test_operator_fills_only_remaining():
    res = resolve_metadata(
        _schema(),
        sidecar={"part_number": "PN-1"},
        campaign_defaults={"operator": "alice"},
        operator={"operator": "bob", "serial_number": "SN-1"},
    )
    assert res.values["operator"] == "alice"
    assert res.values["serial_number"] == "SN-1"
    assert res.sources["serial_number"] is MetadataLayer.OPERATOR


def test_missing_required_reported_not_raised():
    res = resolve_metadata(_schema(), sidecar={"part_number": "PN-1"})
    assert not res.complete
    assert "serial_number" in res.missing_required


def test_unknown_field_surfaces_as_error():
    res = resolve_metadata(_schema(), operator={"junk": 1})
    assert any(e.field_name == "junk" for e in res.errors)


def test_load_sidecar_round_trip(tmp_path: Path):
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps({"part_number": "PN-1", "additive_pct": 5.0}))
    data = load_sidecar(p)
    assert data == {"part_number": "PN-1", "additive_pct": 5.0}


def test_load_sidecar_missing(tmp_path: Path):
    with pytest.raises(IngestError):
        load_sidecar(tmp_path / "nope.json")


def test_load_sidecar_not_object(tmp_path: Path):
    p = tmp_path / "metadata.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(IngestError):
        load_sidecar(p)


def test_load_sidecar_invalid_json(tmp_path: Path):
    p = tmp_path / "metadata.json"
    p.write_text("{not json")
    with pytest.raises(IngestError):
        load_sidecar(p)


def test_canonical_hash_is_order_independent():
    a = canonical_metadata_hash({"x": 1, "y": 2})
    b = canonical_metadata_hash({"y": 2, "x": 1})
    assert a == b


def test_canonical_hash_is_stable_across_calls():
    payload = {"part": "PN-1", "additive": "TEAL", "pct": 5.0}
    assert canonical_metadata_hash(payload) == canonical_metadata_hash(payload)


def test_canonical_hash_distinguishes_int_from_float():
    assert canonical_metadata_hash({"x": 1}) != canonical_metadata_hash({"x": 1.0})


def test_canonical_hash_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        canonical_metadata_hash({"x": math.nan})
    with pytest.raises(ValueError):
        canonical_metadata_hash({"x": math.inf})


def test_canonical_hash_rejects_non_native_types():
    class Custom:
        pass

    with pytest.raises(ValueError):
        canonical_metadata_hash({"x": Custom()})


def test_canonical_hash_handles_nested_structures():
    h = canonical_metadata_hash(
        {"geometry": {"area": 12.5, "cd": 0.65}, "tags": ["acceptance", "Q1"]}
    )
    assert isinstance(h, str)
    assert len(h) == 64
