"""Three-layer metadata resolution.

Sources, in priority order:

    1. SIDECAR     ``metadata.json`` next to the data file (DAQ writes it).
                   Authoritative ground truth — wins over everything.
    2. CAMPAIGN    Campaign-template defaults (test stand, fluids, etc.).
                   Fills fields the sidecar omitted.
    3. OPERATOR    Operator dialog / drag-drop form. Fills what is still
                   missing; the operator never silently overrides the sidecar.

The resolved metadata is the union, with first-set-wins semantics. Each
field's source is recorded so the UI / report can show "this came from the
sidecar" vs "operator-supplied". The resolver also reports which required
fields are still missing — when non-empty, the TestRun enters
``AWAITING_METADATA`` rather than failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hda.domain.errors import IngestError
from hda.domain.metadata.schema import (
    MetadataSchema,
    ValidationError,
    ValidationResult,
)


class MetadataLayer(str, Enum):
    SIDECAR = "sidecar"
    CAMPAIGN = "campaign"
    OPERATOR = "operator"


@dataclass(frozen=True, slots=True)
class ResolvedMetadata:
    values: Mapping[str, Any]
    sources: Mapping[str, MetadataLayer]
    missing_required: Sequence[str]
    errors: Sequence[ValidationError]

    @property
    def complete(self) -> bool:
        return not self.missing_required and not self.errors


def resolve_metadata(
    schema: MetadataSchema,
    sidecar: Optional[Mapping[str, Any]] = None,
    campaign_defaults: Optional[Mapping[str, Any]] = None,
    operator: Optional[Mapping[str, Any]] = None,
) -> ResolvedMetadata:
    """Resolve a metadata mapping from the three layers.

    First-set-wins: sidecar > campaign > operator. The validator runs once at
    the end against the unioned dict, so type errors and unknown fields are
    reported with their resolved value's source attributed.
    """
    sources: dict[str, MetadataLayer] = {}
    merged: dict[str, Any] = {}

    for layer, payload in (
        (MetadataLayer.SIDECAR, sidecar),
        (MetadataLayer.CAMPAIGN, campaign_defaults),
        (MetadataLayer.OPERATOR, operator),
    ):
        if not payload:
            continue
        for k, v in payload.items():
            if k in merged:
                continue
            merged[k] = v
            sources[k] = layer

    result: ValidationResult = schema.validate(merged)
    final_sources = {k: sources[k] for k in result.values.keys() if k in sources}
    return ResolvedMetadata(
        values=result.values,
        sources=final_sources,
        missing_required=result.missing_required,
        errors=result.errors,
    )


def load_sidecar(path: Path) -> Mapping[str, Any]:
    """Load a ``metadata.json`` sidecar file.

    Raises:
        IngestError: file missing, unreadable, or not a JSON object.
    """
    if not path.exists():
        raise IngestError(f"Sidecar not found: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise IngestError(f"Cannot read sidecar {path}: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise IngestError(f"Sidecar {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise IngestError(f"Sidecar {path} must be a JSON object, got {type(data).__name__}")
    return data
