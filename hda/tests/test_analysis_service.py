"""End-to-end AnalysisServiceImpl: PREPROCESSED → PERSISTED through every branch."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hda.domain.errors import ConfigError
from hda.domain.metadata import FieldType, MetadataField, MetadataSchema
from hda.domain.plugin_modules import BasicMeansPlugin
from hda.domain.plugins import PluginRegistry
from hda.domain.qc import QCConfig, SensorRange
from hda.domain.state import TestState
from hda.domain.types import (
    Campaign,
    Hardware,
    SteadyWindow,
    TestMetadata,
)
from hda.persistence import Database, apply_migrations
from hda.persistence.repositories import (
    CampaignRepository,
    MeasurementsRepository,
    QCFindingsRepository,
    TestRunRepository,
)
from hda.services import (
    AnalysisProfile,
    AnalysisServiceImpl,
    IngestPipeline,
    IngestRequest,
    IngestServiceImpl,
    IngestSource,
    NaNPolicy,
    PreprocessingConfig,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "hda.db")
    apply_migrations(d)
    return d


@pytest.fixture
def campaign_id(db: Database) -> str:
    cid = "INJ-CF-C1"
    CampaignRepository(db).create(
        Campaign(id=cid, name="Demo", test_type="cold_flow",
                 created_at=datetime(2026, 1, 1))
    )
    return cid


def _write_steady_csv(path: Path, n_steady: int = 800):
    n_ramp = 100
    n = 2 * n_ramp + n_steady
    t_ms = np.arange(n) * 10.0  # 100 Hz, ms
    pt_up = np.concatenate([
        np.linspace(0.0, 10.0, n_ramp),
        np.full(n_steady, 10.0) + 0.0005 * np.random.default_rng(0).standard_normal(n_steady),
        np.linspace(10.0, 0.0, n_ramp),
    ])
    pt_down = np.concatenate([
        np.linspace(0.0, 5.0, n_ramp),
        np.full(n_steady, 5.0) + 0.0005 * np.random.default_rng(1).standard_normal(n_steady),
        np.linspace(5.0, 0.0, n_ramp),
    ])
    pd.DataFrame({"timestamp": t_ms, "PT-up": pt_up, "PT-down": pt_down}).to_csv(
        path, index=False
    )


def _schema():
    return MetadataSchema(fields=(
        MetadataField("part_number", FieldType.STRING, required=True),
        MetadataField("serial_number", FieldType.STRING, required=True),
        MetadataField("operator", FieldType.STRING, required=True),
        MetadataField("test_id", FieldType.STRING),
        MetadataField("fluid", FieldType.STRING),
    ))


def _ingest_pipeline():
    return IngestPipeline(
        metadata_schema=_schema(),
        preprocessing=PreprocessingConfig(
            timestamp_column="timestamp",
            resample_freq_hz=None,
            nan_policy=NaNPolicy.LEAVE,
        ),
    )


def _profile(plugin_name: str = "basic_means", auto_thresh: float = 0.5):
    return AnalysisProfile(
        plugin_name=plugin_name,
        qc_config=QCConfig(expected_sample_rate_hz=100.0),
        steady_state_signal="PT-up",
        steady_state_cv_threshold=0.005,
        steady_state_window_s=0.2,
        steady_state_min_duration_s=2.0,
        auto_confirm_confidence=auto_thresh,
        sensor_uncertainties={"PT-up": 0.05, "PT-down": 0.05},
    )


def _ingest_one(db: Database, tmp_path: Path, campaign_id: str):
    csv = tmp_path / "test.csv"
    _write_steady_csv(csv)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "part_number": "PN-1", "serial_number": "SN-1",
        "operator": "alice", "test_id": "T-001",
    }))
    svc = IngestServiceImpl(db, pipelines={campaign_id: _ingest_pipeline()})
    return svc.process(IngestRequest(
        file_path=csv, campaign_id=campaign_id, source=IngestSource.FILE_DIALOG,
    ))


def _build_metadata():
    return TestMetadata(
        hardware=Hardware(part_number="PN-1", serial_number="SN-1"),
        fluid="N2", operator="alice", test_id="T-001",
    )


def test_happy_path_preprocessed_to_persisted(db: Database, tmp_path: Path, campaign_id: str):
    outcome_ingest = _ingest_one(db, tmp_path, campaign_id)
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    svc = AnalysisServiceImpl(
        db, plugins, profiles={campaign_id: _profile(auto_thresh=0.0)}
    )
    states_seen = []
    outcome = svc.submit(
        test_run_id=outcome_ingest.test_run_id,
        campaign_id=campaign_id,
        preprocessed=outcome_ingest.preprocessed,
        metadata=_build_metadata(),
        config_hash="cfg" * 16 + "...",
        metadata_hash="md" * 32,
        analyst="alice",
        on_state_change=lambda _id, s: states_seen.append(s),
    )
    assert outcome.final_state is TestState.PERSISTED
    assert outcome.qc_passed is True
    assert outcome.result is not None
    assert "avg_PT-up" in outcome.result.measurements

    # Persisted into DB
    runs = TestRunRepository(db)
    assert runs.get_state(outcome_ingest.test_run_id) is TestState.PERSISTED
    saved = MeasurementsRepository(db).get_for_run(outcome_ingest.test_run_id)
    assert {m.name for m in saved} >= {"avg_PT-up", "avg_PT-down"}
    qc = QCFindingsRepository(db).get_for_run(outcome_ingest.test_run_id)
    assert qc, "QC findings should be persisted"
    assert TestState.PERSISTED in states_seen


def test_low_confidence_routes_to_needs_review(db: Database, tmp_path: Path, campaign_id: str):
    outcome_ingest = _ingest_one(db, tmp_path, campaign_id)
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    # auto_confirm_confidence above achievable -> route to NEEDS_REVIEW
    svc = AnalysisServiceImpl(
        db, plugins, profiles={campaign_id: _profile(auto_thresh=0.99)}
    )
    outcome = svc.submit(
        test_run_id=outcome_ingest.test_run_id,
        campaign_id=campaign_id,
        preprocessed=outcome_ingest.preprocessed,
        metadata=_build_metadata(),
        config_hash="cfg",
        metadata_hash="md",
        analyst="alice",
    )
    assert outcome.final_state is TestState.NEEDS_REVIEW
    assert outcome.qc_passed is True


def test_review_approval_advances_state(db: Database, tmp_path: Path, campaign_id: str):
    outcome_ingest = _ingest_one(db, tmp_path, campaign_id)
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    svc = AnalysisServiceImpl(
        db, plugins, profiles={campaign_id: _profile(auto_thresh=0.99)}
    )
    svc.submit(
        test_run_id=outcome_ingest.test_run_id,
        campaign_id=campaign_id,
        preprocessed=outcome_ingest.preprocessed,
        metadata=_build_metadata(),
        config_hash="cfg",
        metadata_hash="md",
        analyst="alice",
    )
    final = svc.confirm_review(outcome_ingest.test_run_id, approve=True)
    assert final is TestState.ANALYZED


def test_qc_failure_routes_to_qc_failed(db: Database, tmp_path: Path, campaign_id: str):
    # Build a CSV whose PT-down is constant (flatline) -> QC FAIL
    csv = tmp_path / "test.csv"
    n = 1000
    t_ms = np.arange(n) * 10.0
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "timestamp": t_ms,
        "PT-up": 10.0 + 0.001 * rng.standard_normal(n),
        "PT-down": np.full(n, 5.0),  # exactly constant -> flatline
    }).to_csv(csv, index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "part_number": "PN-1", "serial_number": "SN-1", "operator": "alice",
    }))
    svc_in = IngestServiceImpl(db, pipelines={campaign_id: _ingest_pipeline()})
    ing = svc_in.process(IngestRequest(
        file_path=csv, campaign_id=campaign_id, source=IngestSource.FILE_DIALOG,
    ))
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    svc = AnalysisServiceImpl(
        db, plugins, profiles={campaign_id: _profile(auto_thresh=0.0)}
    )
    outcome = svc.submit(
        test_run_id=ing.test_run_id,
        campaign_id=campaign_id,
        preprocessed=ing.preprocessed,
        metadata=_build_metadata(),
        config_hash="cfg",
        metadata_hash="md",
        analyst="alice",
    )
    assert outcome.final_state is TestState.QC_FAILED
    assert outcome.qc_passed is False
    qc = QCFindingsRepository(db).get_for_run(ing.test_run_id)
    assert any("flatline" in f.check_name for f in qc)


def test_unknown_campaign_raises(db: Database):
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    svc = AnalysisServiceImpl(db, plugins, profiles={})
    with pytest.raises(ConfigError, match="No AnalysisProfile"):
        svc.submit(
            test_run_id="x", campaign_id="ghost", preprocessed=None,  # type: ignore[arg-type]
            metadata=_build_metadata(),
            config_hash="", metadata_hash="", analyst="",
        )


def test_derived_measurement_chained_after_plugin(
    db: Database, tmp_path: Path, campaign_id: str
):
    """Plugin emits avg_PT-up and avg_PT-down with uncertainty. A derived
    measurement subtracts them to produce dp_mean — uncertainty must
    propagate end-to-end and the persisted measurement carries
    Provenance.DERIVED.
    """
    from hda.domain.derived import DerivedMeasurementSpec, UncertaintyMethod
    from hda.domain.types import Provenance

    ing = _ingest_one(db, tmp_path, campaign_id)
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    profile = AnalysisProfile(
        plugin_name="basic_means",
        qc_config=QCConfig(expected_sample_rate_hz=100.0),
        steady_state_signal="PT-up",
        steady_state_cv_threshold=0.005,
        steady_state_window_s=0.2,
        steady_state_min_duration_s=2.0,
        auto_confirm_confidence=0.0,
        sensor_uncertainties={"PT-up": 0.05, "PT-down": 0.05},
        derived_measurements=(
            DerivedMeasurementSpec(
                name="dp_mean",
                unit="bar",
                formula="subtract",
                inputs={"a": "avg_PT-up", "b": "avg_PT-down"},
                uncertainty_method=UncertaintyMethod.ANALYTICAL,
            ),
        ),
    )
    svc = AnalysisServiceImpl(db, plugins, profiles={campaign_id: profile})
    outcome = svc.submit(
        test_run_id=ing.test_run_id,
        campaign_id=campaign_id,
        preprocessed=ing.preprocessed,
        metadata=_build_metadata(),
        config_hash="cfg",
        metadata_hash="md",
        analyst="alice",
    )
    assert outcome.final_state is TestState.PERSISTED
    assert "dp_mean" in outcome.result.measurements
    dp = outcome.result.measurements["dp_mean"]
    assert dp.provenance is Provenance.DERIVED
    assert dp.value == pytest.approx(5.0, abs=0.05)
    assert dp.uncertainty > 0.0

    # Persisted into the measurements table with provenance preserved.
    saved = MeasurementsRepository(db).get_for_run(ing.test_run_id)
    saved_by_name = {m.name: m for m in saved}
    assert saved_by_name["dp_mean"].provenance is Provenance.DERIVED


def test_derived_measurement_name_collision_with_plugin_raises(
    db: Database, tmp_path: Path, campaign_id: str
):
    from hda.domain.derived import DerivedMeasurementSpec, UncertaintyMethod

    ing = _ingest_one(db, tmp_path, campaign_id)
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    # Deliberately pick a name the plugin will produce.
    # Spec uses a different formula on different inputs but names itself
    # "avg_PT-up" — the same name the plugin produces. Self-reference is
    # blocked at spec construction; the runtime collision check handles
    # this distinct case.
    colliding_spec = DerivedMeasurementSpec(
        name="avg_PT-up",
        unit="",
        formula="ratio",
        inputs={"num": "avg_PT-down", "den": "avg_PT-down"},
        uncertainty_method=UncertaintyMethod.NONE,
    )
    profile = AnalysisProfile(
        plugin_name="basic_means",
        qc_config=QCConfig(expected_sample_rate_hz=100.0),
        steady_state_signal="PT-up",
        steady_state_cv_threshold=0.005,
        steady_state_window_s=0.2,
        steady_state_min_duration_s=2.0,
        auto_confirm_confidence=0.0,
        sensor_uncertainties={"PT-up": 0.05, "PT-down": 0.05},
        derived_measurements=(colliding_spec,),
    )
    svc = AnalysisServiceImpl(db, plugins, profiles={campaign_id: profile})
    with pytest.raises(ConfigError, match="collide"):
        svc.submit(
            test_run_id=ing.test_run_id,
            campaign_id=campaign_id,
            preprocessed=ing.preprocessed,
            metadata=_build_metadata(),
            config_hash="cfg",
            metadata_hash="md",
            analyst="alice",
        )


def test_steady_window_override(db: Database, tmp_path: Path, campaign_id: str):
    ing = _ingest_one(db, tmp_path, campaign_id)
    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())
    svc = AnalysisServiceImpl(
        db, plugins, profiles={campaign_id: _profile(auto_thresh=0.0)}
    )
    override = SteadyWindow(start_s=2.0, end_s=8.0, method="manual", confidence=1.0)
    outcome = svc.submit(
        test_run_id=ing.test_run_id, campaign_id=campaign_id,
        preprocessed=ing.preprocessed,
        metadata=_build_metadata(),
        config_hash="cfg", metadata_hash="md", analyst="alice",
        steady_window_override=override,
    )
    assert outcome.steady_window.method == "manual"
    assert outcome.final_state is TestState.PERSISTED
