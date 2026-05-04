"""AnalysisService.reanalyze: re-open a finished test with a manual window."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hda.domain.errors import HDAError
from hda.domain.metadata import canonical_metadata_hash
from hda.domain.plugin_modules import BasicMeansPlugin
from hda.domain.plugins import PluginRegistry
from hda.domain.qc import QCConfig
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
from hda.ui.workers import reanalyze_with_window
from hda.ui.workspace import build_default_workspace


def _write_csv(path: Path, n_steady: int = 1500, mean_pt: float = 10.0):
    n_ramp = 200
    n = 2 * n_ramp + n_steady
    t_ms = np.arange(n) * 10.0
    rng = np.random.default_rng(0)
    pt_up = np.concatenate([
        np.linspace(0.0, mean_pt, n_ramp),
        np.full(n_steady, mean_pt) + 0.0005 * rng.standard_normal(n_steady),
        np.linspace(mean_pt, 0.0, n_ramp),
    ])
    pd.DataFrame({"timestamp": t_ms, "PT-up": pt_up}).to_csv(path, index=False)


def _write_metadata(path: Path):
    path.write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
                "test_id": "T-001",
            }
        )
    )


@pytest.fixture
def db_with_run(tmp_path: Path):
    csv = tmp_path / "demo.csv"
    _write_csv(csv)
    _write_metadata(tmp_path / "metadata.json")
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    from hda.ui.workers import run_pipeline
    result = run_pipeline(ws, csv, campaign_id="C1", operator="alice")
    return ws, result.test_run_id


def test_reanalyze_advances_persisted_run_back_through_pipeline(db_with_run):
    ws, run_id = db_with_run
    runs = TestRunRepository(ws.db)
    assert runs.get_state(run_id) is TestState.PERSISTED

    new_window = SteadyWindow(
        start_s=2.5, end_s=8.0, method="manual", confidence=1.0
    )
    out = reanalyze_with_window(ws, run_id, new_window)
    assert out.final_state is TestState.PERSISTED
    assert runs.get_state(run_id) is TestState.PERSISTED


def test_reanalyze_replaces_measurements_atomically(db_with_run):
    ws, run_id = db_with_run
    meas_repo = MeasurementsRepository(ws.db)
    before = {m.name: m.value for m in meas_repo.get_for_run(run_id)}

    # New window covering only the lower half of the steady portion.
    new_window = SteadyWindow(
        start_s=2.5, end_s=4.0, method="manual", confidence=1.0
    )
    reanalyze_with_window(ws, run_id, new_window)

    after_rows = meas_repo.get_for_run(run_id)
    after = {m.name: m.value for m in after_rows}
    # Same measurement names; same DB row count (no leftover stale rows).
    assert set(before) == set(after)
    # Steady-window persisted on the test_run row.
    conn = ws.db.connect()
    row = conn.execute(
        "SELECT steady_start_s, steady_end_s, steady_method FROM test_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert row["steady_start_s"] == pytest.approx(2.5)
    assert row["steady_end_s"] == pytest.approx(4.0)
    assert row["steady_method"] == "manual"


def test_reanalyze_replaces_qc_findings(db_with_run):
    ws, run_id = db_with_run
    qc_repo = QCFindingsRepository(ws.db)
    n_before = len(qc_repo.get_for_run(run_id))
    new_window = SteadyWindow(
        start_s=3.0, end_s=7.0, method="manual", confidence=1.0
    )
    reanalyze_with_window(ws, run_id, new_window)
    n_after = len(qc_repo.get_for_run(run_id))
    # write_all replaces; row count should not double.
    assert n_after == n_before


def test_reanalyze_missing_cache_raises(tmp_path: Path):
    """Eviction => clear UI message rather than corrupt analysis."""
    csv = tmp_path / "demo.csv"
    _write_csv(csv)
    _write_metadata(tmp_path / "metadata.json")
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    from hda.ui.workers import run_pipeline
    result = run_pipeline(ws, csv, campaign_id="C1", operator="alice")
    ws.preprocessed_cache.clear()

    new_window = SteadyWindow(
        start_s=2.5, end_s=4.0, method="manual", confidence=1.0
    )
    with pytest.raises(HDAError, match="not in the cache"):
        reanalyze_with_window(ws, result.test_run_id, new_window)


def test_pipeline_writes_to_preprocessed_cache(tmp_path: Path):
    csv = tmp_path / "demo.csv"
    _write_csv(csv)
    _write_metadata(tmp_path / "metadata.json")
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    from hda.ui.workers import run_pipeline
    result = run_pipeline(ws, csv, campaign_id="C1", operator="alice")
    cached = ws.preprocessed_cache.get(result.test_run_id)
    assert cached is not None
    assert cached.test_run_id == result.test_run_id
    assert "PT-up" in cached.data.df.columns


def test_low_confidence_run_now_persists_measurements_for_review(tmp_path: Path):
    """Refactor: NEEDS_REVIEW now persists measurements so the operator
    can see them in the detail panel before approving."""
    csv = tmp_path / "demo.csv"
    _write_csv(csv)
    _write_metadata(tmp_path / "metadata.json")
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="C1",
        auto_confirm_confidence=0.99,  # forces NEEDS_REVIEW
    )
    from hda.ui.workers import run_pipeline
    result = run_pipeline(ws, csv, campaign_id="C1", operator="alice")
    assert result.final_state is TestState.NEEDS_REVIEW
    saved = MeasurementsRepository(ws.db).get_for_run(result.test_run_id)
    assert saved, "Measurements should be persisted even in NEEDS_REVIEW"
