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
from hda.domain.state import TestState
from hda.domain.types import Hardware, SteadyWindow, TestMetadata
from hda.services.ingest import IngestRequest, IngestSource
from hda.services.preprocessed_cache import CachedPreprocessed
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
    if ingest_outcome.duplicate_of is not None:
        return PipelineResult(
            test_run_id=ingest_outcome.test_run_id,
            final_state=ingest_outcome.state,
            duplicate_of=ingest_outcome.duplicate_of,
            missing_metadata=ingest_outcome.missing_metadata,
        )

    # Cache the preprocessed data even when metadata is incomplete so the
    # detail panel can preview the time-series immediately. Metadata can
    # be filled in later via complete_metadata_and_analyze.
    metadata_for_cache = (
        ingest_outcome.metadata
        if ingest_outcome.metadata is not None
        else _placeholder_metadata(ingest_outcome.test_run_id)
    )
    if ingest_outcome.preprocessed is not None:
        workspace.preprocessed_cache.put(
            CachedPreprocessed(
                test_run_id=ingest_outcome.test_run_id,
                data=ingest_outcome.preprocessed,
                metadata=metadata_for_cache,
                config_hash="",
                metadata_hash=ingest_outcome.metadata_hash,
            )
        )

    if ingest_outcome.state is TestState.AWAITING_METADATA:
        _log.info(
            "ingest awaiting metadata: id=%s missing=%s",
            ingest_outcome.test_run_id,
            list(ingest_outcome.missing_metadata),
        )
        return PipelineResult(
            test_run_id=ingest_outcome.test_run_id,
            final_state=ingest_outcome.state,
            missing_metadata=ingest_outcome.missing_metadata,
        )

    assert ingest_outcome.metadata is not None
    _log.info("analysis start: id=%s", ingest_outcome.test_run_id)
    analysis_outcome = workspace.analysis_service.submit(
        test_run_id=ingest_outcome.test_run_id,
        campaign_id=campaign_id,
        preprocessed=ingest_outcome.preprocessed,
        metadata=ingest_outcome.metadata,
        config_hash="",
        metadata_hash=ingest_outcome.metadata_hash,
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


def _placeholder_metadata(run_id: str) -> TestMetadata:
    """A do-not-persist metadata used only as a CachedPreprocessed
    handle while the test is in AWAITING_METADATA. The real metadata
    is assembled by complete_metadata_and_analyze and stored in the DB."""
    return TestMetadata(
        hardware=Hardware(
            part_number="__awaiting__", serial_number=run_id[:8]
        ),
        fluid="",
        operator="",
        test_id=run_id[:8],
    )


def complete_metadata_and_analyze(
    workspace: Workspace,
    test_run_id: str,
    operator_metadata: Mapping[str, Any],
    analyst: str = "operator",
) -> PipelineResult:
    """Resolve the missing metadata, persist it, and run analysis.

    The detail panel calls this when the operator submits the
    "Complete metadata" form. Errors propagate as HDAError /
    ConfigError so the dialog can show them inline.
    """
    if workspace.ingest_service is None or workspace.analysis_service is None:
        raise HDAError("Workspace services were not configured")

    cached = workspace.preprocessed_cache.get(test_run_id)
    if cached is None:
        raise HDAError(
            f"Preprocessed data for {test_run_id} is not in the cache. "
            "Re-ingest the source file to enable analysis."
        )

    outcome = workspace.ingest_service.complete_metadata(
        test_run_id, operator_metadata
    )
    # Refresh the cache entry so the metadata it carries reflects what
    # was just persisted (later reanalysis reads metadata from cache).
    workspace.preprocessed_cache.put(
        CachedPreprocessed(
            test_run_id=test_run_id,
            data=cached.data,
            metadata=outcome.metadata,
            config_hash=cached.config_hash,
            metadata_hash=outcome.metadata_hash,
        )
    )

    campaign_id = _lookup_campaign_id(workspace, test_run_id)
    _log.info("analysis start (post-complete-metadata): id=%s", test_run_id)
    analysis_outcome = workspace.analysis_service.submit(
        test_run_id=test_run_id,
        campaign_id=campaign_id,
        preprocessed=cached.data,
        metadata=outcome.metadata,
        config_hash="",
        metadata_hash=outcome.metadata_hash,
        analyst=analyst,
    )
    return PipelineResult(
        test_run_id=test_run_id,
        final_state=analysis_outcome.final_state,
    )


def reanalyze_with_window(
    workspace: Workspace,
    test_run_id: str,
    manual_window: SteadyWindow,
    analyst: str = "operator-reanalyze",
) -> PipelineResult:
    """Re-run analysis on a cached preprocessed dataset with a manual window.

    The dashboard's drag-handle preview commits via this entry point.
    Raises HDAError if the test's preprocessed data is not in the cache
    (e.g. because the cache was evicted) so the UI can surface
    "re-ingest the source file to enable reanalysis".
    """
    if workspace.analysis_service is None:
        raise HDAError("Workspace services were not configured")
    cached = workspace.preprocessed_cache.get(test_run_id)
    if cached is None:
        raise HDAError(
            f"Preprocessed data for {test_run_id} is not in the cache. "
            "Re-ingest the source file to enable reanalysis."
        )
    campaign_id = _lookup_campaign_id(workspace, test_run_id)
    _log.info(
        "reanalyze: id=%s window=[%.3f,%.3f]",
        test_run_id,
        manual_window.start_s,
        manual_window.end_s,
    )
    outcome = workspace.analysis_service.reanalyze(
        test_run_id=test_run_id,
        campaign_id=campaign_id,
        preprocessed=cached.data,
        metadata=cached.metadata,
        manual_window=manual_window,
        config_hash=cached.config_hash,
        metadata_hash=cached.metadata_hash,
        analyst=analyst,
    )
    return PipelineResult(
        test_run_id=test_run_id,
        final_state=outcome.final_state,
    )


def _lookup_campaign_id(workspace: Workspace, test_run_id: str) -> str:
    conn = workspace.db.connect()
    row = conn.execute(
        "SELECT campaign_id FROM test_runs WHERE id = ?", (test_run_id,)
    ).fetchone()
    if row is None:
        raise HDAError(f"TestRun {test_run_id} not found in database")
    return str(row["campaign_id"])


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
        except Exception as e:
            _log.exception("pipeline worker failed")
            self.signals.failed.emit(f"{type(e).__name__}: {e}")


class ReanalyzeWorker(QRunnable):
    """QRunnable wrapper around ``reanalyze_with_window``."""

    def __init__(
        self,
        workspace: Workspace,
        test_run_id: str,
        manual_window: SteadyWindow,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.test_run_id = test_run_id
        self.manual_window = manual_window
        self.signals = WorkerSignals()

    def run(self) -> None:  # noqa: D401 - QRunnable contract
        try:
            result = reanalyze_with_window(
                self.workspace, self.test_run_id, self.manual_window
            )
            self.signals.finished.emit(result)
        except Exception as e:
            _log.exception("reanalyze worker failed")
            self.signals.failed.emit(f"{type(e).__name__}: {e}")


class CompleteMetadataAndAnalyzeWorker(QRunnable):
    """QRunnable wrapper around ``complete_metadata_and_analyze``."""

    def __init__(
        self,
        workspace: Workspace,
        test_run_id: str,
        operator_metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.test_run_id = test_run_id
        self.operator_metadata = dict(operator_metadata)
        self.signals = WorkerSignals()

    def run(self) -> None:  # noqa: D401 - QRunnable contract
        try:
            result = complete_metadata_and_analyze(
                self.workspace,
                self.test_run_id,
                self.operator_metadata,
            )
            self.signals.finished.emit(result)
        except Exception as e:
            _log.exception("complete-metadata worker failed")
            self.signals.failed.emit(f"{type(e).__name__}: {e}")
