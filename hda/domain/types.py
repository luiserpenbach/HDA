"""Domain types — frozen dataclasses, no I/O.

Every TestRun snapshot is immutable; transitions produce a new instance via
``dataclasses.replace``. This eliminates the class of "Test A's result saved
against Test B's metadata" bugs that plagued the Streamlit app's mutable
session_state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hda.domain.state import TestState


class Provenance(str, Enum):
    SENSOR = "sensor"
    DERIVED = "derived"


class QCStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class MeasurementWithUncertainty:
    name: str
    value: float
    uncertainty: float
    unit: str
    provenance: Provenance = Provenance.SENSOR

    def __post_init__(self) -> None:
        if self.uncertainty < 0:
            raise ValueError(
                f"Uncertainty must be non-negative for {self.name}, got {self.uncertainty}"
            )

    @property
    def rel_uncertainty_pct(self) -> Optional[float]:
        if self.value == 0:
            return None
        return abs(100.0 * self.uncertainty / self.value)


@dataclass(frozen=True, slots=True)
class Hardware:
    part_number: str
    serial_number: str

    def __post_init__(self) -> None:
        if not self.part_number or not self.serial_number:
            raise ValueError("Hardware requires non-empty part_number and serial_number")


@dataclass(frozen=True, slots=True)
class TestMetadata:
    """Test article + run metadata.

    The typed fields cover what every test must declare. ``extra`` carries
    plugin-declared fields (e.g. ``fuel_additive``, ``additive_pct``) and
    participates in the metadata hash so plugin-relevant state is part of
    traceability.
    """

    hardware: Hardware
    fluid: str
    operator: str
    test_id: str
    geometry: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def merged_extra(self, other: Mapping[str, Any]) -> "TestMetadata":
        from dataclasses import replace
        return replace(self, extra={**self.extra, **other})


@dataclass(frozen=True, slots=True)
class Campaign:
    id: str
    name: str
    test_type: str
    created_at: datetime
    archived: bool = False


@dataclass(frozen=True, slots=True)
class SteadyWindow:
    start_s: float
    end_s: float
    method: str
    confidence: float

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"Steady window end ({self.end_s}) must exceed start ({self.start_s})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class QCFinding:
    check_name: str
    status: QCStatus
    message: str
    blocking: bool = False


@dataclass(frozen=True, slots=True)
class QCReport:
    findings: Sequence[QCFinding]

    @property
    def passed(self) -> bool:
        return not any(f.blocking and f.status == QCStatus.FAIL for f in self.findings)

    @property
    def blocking_failures(self) -> Sequence[QCFinding]:
        return tuple(
            f for f in self.findings if f.blocking and f.status == QCStatus.FAIL
        )


@dataclass(frozen=True, slots=True)
class TraceabilityRecord:
    file_hash: str
    config_hash: str
    metadata_hash: str
    processing_version: str
    plugin_name: str
    plugin_version: str
    analyst: str
    analyzed_at: datetime


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    measurements: Mapping[str, MeasurementWithUncertainty]
    qc_report: QCReport
    traceability: TraceabilityRecord
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        for name, m in self.measurements.items():
            if name != m.name:
                raise ValueError(
                    f"Measurement key '{name}' does not match measurement.name '{m.name}'"
                )


@dataclass(frozen=True, slots=True)
class TestRun:
    """Immutable snapshot of a test through the analysis pipeline.

    Transitions produce a new TestRun via ``dataclasses.replace`` and validate
    the state change through ``hda.domain.state.transition``.
    """

    id: str
    campaign_id: str
    file_path: Path
    file_hash: str
    state: TestState
    metadata: Optional[TestMetadata] = None
    steady_window: Optional[SteadyWindow] = None
    qc_report: Optional[QCReport] = None
    result: Optional[AnalysisResult] = None
    error_message: Optional[str] = None
    discovered_at: Optional[datetime] = None
