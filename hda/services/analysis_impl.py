"""Concrete AnalysisService.

Drives a TestRun from PREPROCESSED through:

    STEADY_DETECTED → QC_RUN →
        ANALYZED  (qc passed, confidence ≥ auto threshold) → PERSISTED
        NEEDS_REVIEW (qc passed, low confidence)
        QC_FAILED  (qc failed)

Every state transition goes through ``TestRunRepository.update_state``,
which validates against the domain DAG before touching the row, so the
orchestrator cannot accidentally skip a phase.

The orchestrator is sync today; the Qt layer wraps ``submit`` in a
``QThreadPool`` worker so the GUI never blocks. Progress callbacks fire
after each transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from hda.domain.errors import AnalysisError, ConfigError
from hda.domain.plugins import AnalysisContext, AnalysisPlugin, PluginRegistry
from hda.domain.qc import QCConfig, run_qc
from hda.domain.state import TestState
from hda.domain.steady_state import detect_cv, detect_simple
from hda.domain.types import (
    AnalysisResult,
    QCReport,
    SteadyWindow,
    TestMetadata,
    TraceabilityRecord,
)
from hda.persistence.db import Database
from hda.persistence.repositories import (
    MeasurementsRepository,
    QCFindingsRepository,
    TestRunRepository,
)
from hda.services.preprocessing import PreprocessedData


@dataclass(frozen=True, slots=True)
class AnalysisProfile:
    """Per-campaign analysis configuration.

    The profile is locked to a campaign and selected once when the campaign
    is created — it pins the plugin, QC thresholds, steady-state policy,
    and which sensor calibration uncertainties to use, so per-test
    operator decisions reduce to "approve or reject the auto result".
    """

    plugin_name: str
    qc_config: QCConfig = field(default_factory=QCConfig)
    steady_state_signal: Optional[str] = None
    steady_state_cv_threshold: float = 0.02
    steady_state_window_s: float = 1.0
    steady_state_min_duration_s: float = 2.0
    auto_confirm_confidence: float = 0.7
    sensor_uncertainties: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    test_run_id: str
    final_state: TestState
    steady_window: Optional[SteadyWindow]
    result: Optional[AnalysisResult]
    qc_passed: bool


StateChangeCallback = Callable[[str, TestState], None]


class AnalysisServiceImpl:
    def __init__(
        self,
        db: Database,
        plugins: PluginRegistry,
        profiles: Mapping[str, AnalysisProfile],
        processing_version: str = "3.0.0-dev",
    ) -> None:
        self._db = db
        self._plugins = plugins
        self._profiles = dict(profiles)
        self._processing_version = processing_version
        self._test_runs = TestRunRepository(db)
        self._measurements = MeasurementsRepository(db)
        self._qc_findings = QCFindingsRepository(db)

    def submit(
        self,
        test_run_id: str,
        campaign_id: str,
        preprocessed: PreprocessedData,
        metadata: TestMetadata,
        config_hash: str,
        metadata_hash: str,
        analyst: str,
        steady_window_override: Optional[SteadyWindow] = None,
        on_state_change: Optional[StateChangeCallback] = None,
    ) -> AnalysisOutcome:
        profile = self._profiles.get(campaign_id)
        if profile is None:
            raise ConfigError(
                f"No AnalysisProfile registered for campaign '{campaign_id}'"
            )
        plugin = self._plugins.get(profile.plugin_name)

        try:
            window = self._detect_or_override(
                preprocessed, profile, steady_window_override
            )
            self._transition(
                test_run_id, TestState.STEADY_DETECTED, on_state_change
            )

            steady_df = self._slice_steady(preprocessed, window)
            qc = self._run_qc_suite(preprocessed, steady_df, profile.qc_config)
            self._qc_findings.write_all(test_run_id, qc.findings, qc.passed)

            if not qc.passed:
                self._transition(
                    test_run_id, TestState.QC_RUN, on_state_change
                )
                self._transition(
                    test_run_id, TestState.QC_FAILED, on_state_change
                )
                return AnalysisOutcome(
                    test_run_id=test_run_id,
                    final_state=TestState.QC_FAILED,
                    steady_window=window,
                    result=None,
                    qc_passed=False,
                )

            self._transition(test_run_id, TestState.QC_RUN, on_state_change)

            self._validate_required_channels(plugin, steady_df)
            ctx = AnalysisContext(
                df=preprocessed.df,
                steady_df=steady_df,
                steady_window=window,
                metadata=metadata,
                sensor_uncertainties=profile.sensor_uncertainties,
                geometry=metadata.geometry,
            )
            measurements = dict(plugin.compute(ctx))
            traceability = TraceabilityRecord(
                file_hash=self._file_hash_for(test_run_id),
                config_hash=config_hash,
                metadata_hash=metadata_hash,
                processing_version=self._processing_version,
                plugin_name=plugin.name,
                plugin_version=plugin.version,
                analyst=analyst,
                analyzed_at=datetime.utcnow(),
            )
            result = AnalysisResult(
                measurements=measurements,
                qc_report=qc,
                traceability=traceability,
                confidence=window.confidence,
            )

            if window.confidence < profile.auto_confirm_confidence:
                self._transition(
                    test_run_id, TestState.NEEDS_REVIEW, on_state_change
                )
                return AnalysisOutcome(
                    test_run_id=test_run_id,
                    final_state=TestState.NEEDS_REVIEW,
                    steady_window=window,
                    result=result,
                    qc_passed=True,
                )

            self._transition(test_run_id, TestState.ANALYZED, on_state_change)
            self._persist(
                test_run_id=test_run_id,
                window=window,
                result=result,
                plugin=plugin,
                config_hash=config_hash,
            )
            self._transition(
                test_run_id, TestState.PERSISTED, on_state_change
            )
            return AnalysisOutcome(
                test_run_id=test_run_id,
                final_state=TestState.PERSISTED,
                steady_window=window,
                result=result,
                qc_passed=True,
            )

        except (AnalysisError, ConfigError) as e:
            self._test_runs.update_state(
                test_run_id, TestState.ERROR, error_message=str(e)
            )
            if on_state_change is not None:
                on_state_change(test_run_id, TestState.ERROR)
            raise

    def confirm_review(
        self,
        test_run_id: str,
        approve: bool,
        on_state_change: Optional[StateChangeCallback] = None,
    ) -> TestState:
        """Operator decision on a NEEDS_REVIEW run."""
        target = TestState.ANALYZED if approve else TestState.QC_FAILED
        self._transition(test_run_id, target, on_state_change)
        return target

    def _detect_or_override(
        self,
        pp: PreprocessedData,
        profile: AnalysisProfile,
        override: Optional[SteadyWindow],
    ) -> SteadyWindow:
        if override is not None:
            return override
        ts = "timestamp"
        time_s = pp.df[ts].to_numpy(dtype=float)
        signal_col = profile.steady_state_signal or self._auto_signal(pp, ts)
        if signal_col not in pp.df.columns:
            raise AnalysisError(
                f"Steady-state signal '{signal_col}' not in preprocessed data"
            )
        signal = pp.df[signal_col].to_numpy(dtype=float)
        cv_window = detect_cv(
            signal,
            time_s,
            cv_threshold=profile.steady_state_cv_threshold,
            window_s=profile.steady_state_window_s,
            min_duration_s=profile.steady_state_min_duration_s,
        )
        if cv_window is not None:
            return cv_window
        return detect_simple(time_s, fraction=0.5)

    @staticmethod
    def _auto_signal(pp: PreprocessedData, ts: str) -> str:
        for col in pp.df.columns:
            if col == ts:
                continue
            return col
        raise AnalysisError("No non-timestamp channels available")

    @staticmethod
    def _slice_steady(pp: PreprocessedData, window: SteadyWindow):
        df = pp.df
        ts = "timestamp"
        mask = (df[ts] >= window.start_s) & (df[ts] <= window.end_s)
        out = df.loc[mask].reset_index(drop=True)
        if out.empty:
            raise AnalysisError(
                f"Steady-state window [{window.start_s}, {window.end_s}] "
                "contains no samples"
            )
        return out

    @staticmethod
    def _run_qc_suite(pp: PreprocessedData, steady_df, qc_config: QCConfig) -> QCReport:
        ts = "timestamp"
        time_s = pp.df[ts].to_numpy(dtype=float)
        channels = {
            col: pp.df[col].to_numpy(dtype=float)
            for col in pp.df.columns
            if col != ts
        }
        return run_qc(time_s, channels, qc_config)

    @staticmethod
    def _validate_required_channels(plugin: AnalysisPlugin, steady_df) -> None:
        required = set(plugin.required_channels())
        missing = required - set(steady_df.columns)
        if missing:
            raise ConfigError(
                f"Plugin '{plugin.name}' requires channels not present in "
                f"steady_df: {sorted(missing)}"
            )

    def _file_hash_for(self, test_run_id: str) -> str:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT file_hash FROM test_runs WHERE id = ?", (test_run_id,)
        ).fetchone()
        if row is None:
            raise AnalysisError(f"TestRun {test_run_id} not found")
        return row["file_hash"]

    def _persist(
        self,
        test_run_id: str,
        window: SteadyWindow,
        result: AnalysisResult,
        plugin: AnalysisPlugin,
        config_hash: str,
    ) -> None:
        conn = self._db.connect()
        from hda.persistence.db import transaction

        with transaction(self._db, write=True) as conn:
            conn.execute(
                """
                UPDATE test_runs
                   SET steady_start_s = ?,
                       steady_end_s = ?,
                       steady_method = ?,
                       steady_confidence = ?,
                       confidence = ?,
                       processing_version = ?,
                       plugin_name = ?,
                       plugin_version = ?,
                       config_hash = ?
                 WHERE id = ?
                """,
                (
                    float(window.start_s),
                    float(window.end_s),
                    window.method,
                    float(window.confidence),
                    float(result.confidence),
                    result.traceability.processing_version,
                    plugin.name,
                    plugin.version,
                    config_hash,
                    test_run_id,
                ),
            )
        self._measurements.write_all(test_run_id, result.measurements)

    def _transition(
        self,
        test_run_id: str,
        target: TestState,
        on_state_change: Optional[StateChangeCallback],
    ) -> None:
        self._test_runs.update_state(test_run_id, target)
        if on_state_change is not None:
            on_state_change(test_run_id, target)
