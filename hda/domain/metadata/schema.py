"""Plugin-declared metadata schema.

Each plugin declares the metadata fields it expects (e.g., a hot-fire plugin
declares ``fuel_additive`` and ``additive_pct``). The schema validates an
incoming mapping, coerces types, and reports missing required fields. The
same schema drives form generation in the UI and filter dimensions in the
analytics layer.

Validation never silently coerces unknown types or drops malformed values;
every problem becomes a typed ``ValidationError`` so the UI can surface it
field-by-field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class FieldType(str, Enum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"


@dataclass(frozen=True, slots=True)
class ValidationError:
    field_name: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    values: Mapping[str, Any]
    errors: Sequence[ValidationError]
    missing_required: Sequence[str]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.missing_required


@dataclass(frozen=True, slots=True)
class MetadataField:
    name: str
    type: FieldType
    required: bool = False
    unit: str = ""
    help: str = ""
    choices: Sequence[str] = ()
    default: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MetadataField.name must be non-empty")
        if self.type is FieldType.CHOICE and not self.choices:
            raise ValueError(
                f"Field '{self.name}' is CHOICE but declares no choices"
            )

    def coerce(self, raw: Any) -> Any:
        """Convert a raw value (e.g., from JSON or a UI widget) to the field type.

        Returns the coerced value on success. Raises ValueError otherwise; the
        caller wraps it as a ValidationError.
        """
        if raw is None:
            return None
        t = self.type
        if t is FieldType.STRING:
            return str(raw)
        if t is FieldType.INT:
            if isinstance(raw, bool):
                raise ValueError(f"'{self.name}' expects int, got bool")
            return int(raw)
        if t is FieldType.FLOAT:
            if isinstance(raw, bool):
                raise ValueError(f"'{self.name}' expects float, got bool")
            return float(raw)
        if t is FieldType.BOOL:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                low = raw.strip().lower()
                if low in {"true", "1", "yes", "y"}:
                    return True
                if low in {"false", "0", "no", "n"}:
                    return False
                raise ValueError(f"'{self.name}': cannot parse bool from '{raw}'")
            raise ValueError(f"'{self.name}' expects bool")
        if t is FieldType.CHOICE:
            s = str(raw)
            if s not in self.choices:
                raise ValueError(
                    f"'{self.name}': '{s}' not in choices {list(self.choices)}"
                )
            return s
        raise ValueError(f"Unknown field type {t}")


@dataclass(frozen=True, slots=True)
class MetadataSchema:
    fields: Sequence[MetadataField] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        dups = {n for n in names if names.count(n) > 1}
        if dups:
            raise ValueError(f"Duplicate field names: {sorted(dups)}")

    def field_by_name(self, name: str) -> MetadataField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def required_names(self) -> frozenset[str]:
        return frozenset(f.name for f in self.fields if f.required)

    def validate(self, values: Mapping[str, Any]) -> ValidationResult:
        coerced: dict[str, Any] = {}
        errors: list[ValidationError] = []
        for f in self.fields:
            if f.name in values:
                try:
                    v = f.coerce(values[f.name])
                except ValueError as e:
                    errors.append(ValidationError(f.name, str(e)))
                    continue
                if v is None and f.default is not None:
                    coerced[f.name] = f.default
                elif v is not None:
                    coerced[f.name] = v
            elif f.default is not None:
                coerced[f.name] = f.default
        for unknown in set(values.keys()) - {f.name for f in self.fields}:
            errors.append(
                ValidationError(unknown, f"Unknown metadata field '{unknown}'")
            )
        missing = sorted(self.required_names() - set(coerced.keys()))
        return ValidationResult(
            values=coerced, errors=tuple(errors), missing_required=tuple(missing)
        )

    def merge(self, other: "MetadataSchema") -> "MetadataSchema":
        """Combine two schemas. Duplicate names raise ValueError."""
        return MetadataSchema(fields=tuple(self.fields) + tuple(other.fields))
