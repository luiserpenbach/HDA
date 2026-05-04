"""End-to-end UI pipeline runner: file -> ingested -> analyzed -> persisted.

Tests ``run_pipeline`` directly (the same function the QRunnable wraps),
so we don't need a QApplication. This is the closest thing to "drop a CSV
in the UI and check what happens" without a display server.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hda.domain.state import TestState
from hda.persistence.repositories import (
    MeasurementsRepository,
    QCFindingsRepository,
    TestRunRepository,
)
from hda.ui.workers import run_pipeline
from hda.ui.workspace import build_default_workspace


def _write_steady_csv(path: Path, n: int = 1000):
    rng = np.random.default_rng(0)
    t_ms = np.arange(n) * 10.0
    p_up = np.full(n, 10.0) + 0.001 * rng.standard_normal(n)
    p_down = np.full(n, 5.0) + 0.001 * rng.standard_normal(n)
    pd.DataFrame({"timestamp": t_ms, "PT-up": p_up, "PT-down": p_down}).to_csv(
        path, index=False
    )


def test_run_pipeline_drives_test_to_persisted(tmp_path: Path):
    csv = tmp_path / "demo.csv"
    _write_steady_csv(csv)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
            }
        )
    )
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db",
        campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    result = run_pipeline(ws, csv, campaign_id="C1", operator="alice")
    assert result.final_state is TestState.PERSISTED
    assert result.error is None

    runs = TestRunRepository(ws.db)
    assert runs.get_state(result.test_run_id) is TestState.PERSISTED
    measurements = MeasurementsRepository(ws.db).get_for_run(result.test_run_id)
    assert any(m.name.startswith("avg_") for m in measurements)
    qc = QCFindingsRepository(ws.db).get_for_run(result.test_run_id)
    assert qc, "QC findings must persist"


def test_run_pipeline_idempotent_on_duplicate_file(tmp_path: Path):
    csv = tmp_path / "demo.csv"
    _write_steady_csv(csv)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
            }
        )
    )
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db",
        campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    a = run_pipeline(ws, csv, campaign_id="C1")
    b = run_pipeline(ws, csv, campaign_id="C1")
    assert a.test_run_id == b.test_run_id
    assert b.duplicate_of == a.test_run_id


def test_run_pipeline_reports_missing_metadata(tmp_path: Path):
    csv = tmp_path / "demo.csv"
    _write_steady_csv(csv)
    # No sidecar metadata.json and no operator -> required fields missing.
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="C1"
    )
    result = run_pipeline(ws, csv, campaign_id="C1")
    assert result.final_state is TestState.AWAITING_METADATA
    assert "part_number" in result.missing_metadata
