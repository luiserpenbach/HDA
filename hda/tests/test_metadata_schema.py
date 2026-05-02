"""Plugin metadata schema validation."""

from __future__ import annotations

import pytest

from hda.domain.metadata import (
    FieldType,
    MetadataField,
    MetadataSchema,
)


def test_field_rejects_empty_name():
    with pytest.raises(ValueError):
        MetadataField(name="", type=FieldType.STRING)


def test_choice_field_requires_choices():
    with pytest.raises(ValueError):
        MetadataField(name="fluid", type=FieldType.CHOICE)


def test_int_coerces_string():
    f = MetadataField(name="cycles", type=FieldType.INT)
    assert f.coerce("42") == 42


def test_int_rejects_bool():
    f = MetadataField(name="cycles", type=FieldType.INT)
    with pytest.raises(ValueError):
        f.coerce(True)


def test_float_coerces_int():
    f = MetadataField(name="pct", type=FieldType.FLOAT)
    assert f.coerce(5) == 5.0


def test_bool_parses_strings():
    f = MetadataField(name="x", type=FieldType.BOOL)
    assert f.coerce("yes") is True
    assert f.coerce("FALSE") is False
    with pytest.raises(ValueError):
        f.coerce("maybe")


def test_choice_rejects_out_of_set():
    f = MetadataField(name="fluid", type=FieldType.CHOICE, choices=("N2", "GHe"))
    assert f.coerce("N2") == "N2"
    with pytest.raises(ValueError):
        f.coerce("LOX")


def test_schema_rejects_duplicate_field_names():
    with pytest.raises(ValueError):
        MetadataSchema(
            fields=(
                MetadataField("a", FieldType.STRING),
                MetadataField("a", FieldType.INT),
            )
        )


def test_schema_validate_required_missing():
    schema = MetadataSchema(
        fields=(
            MetadataField("part_number", FieldType.STRING, required=True),
            MetadataField("operator", FieldType.STRING, required=True),
        )
    )
    res = schema.validate({"part_number": "PN-1"})
    assert not res.ok
    assert res.missing_required == ("operator",)


def test_schema_validate_unknown_field_is_error():
    schema = MetadataSchema(fields=(MetadataField("x", FieldType.STRING),))
    res = schema.validate({"x": "ok", "junk": 1})
    assert any(e.field_name == "junk" for e in res.errors)


def test_schema_default_applied_when_field_omitted():
    schema = MetadataSchema(
        fields=(MetadataField("cd", FieldType.FLOAT, default=0.65),)
    )
    res = schema.validate({})
    assert res.ok
    assert res.values["cd"] == pytest.approx(0.65)


def test_schema_coerces_then_validates():
    schema = MetadataSchema(
        fields=(MetadataField("cycles", FieldType.INT, required=True),)
    )
    res = schema.validate({"cycles": "12"})
    assert res.ok
    assert res.values["cycles"] == 12


def test_schema_merge_combines_fields():
    a = MetadataSchema(fields=(MetadataField("x", FieldType.STRING),))
    b = MetadataSchema(fields=(MetadataField("y", FieldType.INT),))
    m = a.merge(b)
    assert {f.name for f in m.fields} == {"x", "y"}


def test_schema_merge_rejects_duplicate():
    a = MetadataSchema(fields=(MetadataField("x", FieldType.STRING),))
    b = MetadataSchema(fields=(MetadataField("x", FieldType.INT),))
    with pytest.raises(ValueError):
        a.merge(b)
