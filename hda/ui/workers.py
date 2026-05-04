"""Background workers for ingest + analysis.

Long ops run on QThreadPool so the GUI never blocks. Workers communicate
back to the UI thread via Qt signals. The pure-Python ``run_pipeline``
function does the actual work and is independently testable without Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from PySide6.QtCore import QObject, QRunnable, Signal

from hda.domain.errors import HDAError
from hda.domain.metadata import canonical_metadata_hash
from hda.domain.state import TestState
from hda.domain.types import Hardware, TestMetadata
from hda.services.ingest import IngestRequest, IngestSource
from hda.ui.logging_setup import get_logger
from hda.ui.workspace import Workspace


_log = get_logger("workers")


@dataclass(slots=True)
class PipelineResult:
    test_run_id: str
    final_state: TestState
    duplicate_of: Optional[str] = None
    missing_metadata: tuple[str, ...] = ()
    error: Optional[str] = None


def run_pipeline(
    workspace: Workspace,
    file_path: Path,
    campaign_id: str,
    operator: Optional[str] = None,
    sidecar_metadata: Optional[Mapping[str, Any]] = None,
    source: IngestSource = IngestSource.FILE_DIALOG,
    analyst: str = "operator",
) -> PipelineResult:
    """Drive a file all the way to PERSISTED (or to the appropriate
    intermediate state). Pure function; the Qt worker just calls this.
    """
    if workspace.ingest_service is None or workspace.analysis_service is None:
        raise HDAError("Workspace services were not configured")

    _log.info("ingest start: %s -> campaign=%s", file_path, campaign_id)
    ingest_outcome = workspace.ingest_service.process(
        IngestRequest(
            file_path=file_path,
            campaign_id=campaign_id,
            source=source,
            sidecar_metadata=sidecar_metadata,
            operator=operator,
        )
    )
    _log.info(
        "ingest done: id=%s state=%s duplicate=%s",
        ingest_outcome.test_run_id,
        ingest_outcome.state.value,
        bool(ingest_outcome.duplicate_of),
    )
    if (
        ingest_outcome.duplicate_of is not None
        or ingest_outcome.preprocessed is None
    ):
        return PipelineResult(
            test_run_id=ingest_outcome.test_run_id,
            final_state=ingest_outcome.state,
            duplicate_of=ingest_outcome.duplicate_of,
            missing_metadata=ingest_outcome.missing_metadata,
        )

    metadata = _build_metadata(ingest_outcome, sidecar_metadata, operator)
    metadata_hash = canonical_metadata_hash(_metadata_for_hash(metadata))

    _log.info("analysis start: id=%s", ingest_outcome.test_run_id)
    analysis_outcome = workspace.analysis_service.submit(
        test_run_id=ingest_outcome.test_run_id,
        campaign_id=campaign_id,
        preprocessed=ingest_outcome.preprocessed,
        metadata=metadata,
        config_hash="",
        metadata_hash=metadata_hash,
        analyst=analyst,
    )
    _log.info(
        "analysis done: id=%s state=%s qc_passed=%s",
        ingest_outcome.test_run_id,
        analysis_outcome.final_state.value,
        analysis_outcome.qc_passed,
    )
    return PipelineResult(
        test_run_id=ingest_outcome.test_run_id,
        final_state=analysis_outcome.final_state,
    )


def _build_metadata(ingest_outcome, sidecar, operator) -> TestMetadata:
    """Best-effort TestMetadata construction from the resolved values."""
    src = dict(sidecar or {})
    if operator and "operator" not in src:
        src["operator"] = operator
    return TestMetadata(
        hardware=Hardware(
            part_number=str(src.get("part_number", "PN-UNKNOWN")),
            serial_number=str(src.get("serial_number", "SN-UNKNOWN")),
        ),
        fluid=str(src.get("fluid", "")),
        operator=str(src.get("operator", "")),
        test_id=str(src.get("test_id", ingest_outcome.test_run_id[:8])),
        notes=str(src.get("notes", "")),
    )


def _metadata_for_hash(md: TestMetadata) -> dict[str, Any]:
    return {
        "part_number": md.hardware.part_number,
        "serial_number": md.hardware.serial_number,
        "fluid": md.fluid,
        "operator": md.operator,
        "test_id": md.test_id,
        "notes": md.notes,
    }


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class IngestAndAnalyzeWorker(QRunnable):
    """QRunnable wrapper around ``run_pipeline``."""

    def __init__(
        self,
        workspace: Workspace,
        file_path: Path,
        campaign_id: str,
        operator: Optional[str] = None,
        sidecar_metadata: Optional[Mapping[str, Any]] = None,
        source: IngestSource = IngestSource.FILE_DIALOG,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.file_path = Path(file_path)
        self.campaign_id = campaign_id
        self.operator = operator
        self.sidecar_metadata = sidecar_metadata
        self.source = source
        self.signals = WorkerSignals()

    def run(self) -> None:  # noqa: D401 - QRunnable contract
        try:
            result = run_pipeline(
                workspace=self.workspace,
                file_path=self.file_path,
                campaign_id=self.campaign_id,
                operator=self.operator,
                sidecar_metadata=self.sidecar_metadata,
                source=self.source,
            )
            self.signals.finished.emit(result)
        except Exception as e:  # surface as a UI message, do not crash
            _log.exception("pipeline worker failed")
            self.signals.failed.emit(f"{type(e).__name__}: {e}")
