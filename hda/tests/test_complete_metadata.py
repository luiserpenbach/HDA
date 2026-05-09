"""IngestServiceImpl.complete_metadata + the AWAITING_METADATA → analysis flow.

Pins the user-visible behavior we now rely on:
  - Ingest of a CSV with no sidecar lands in AWAITING_METADATA but with
    the preprocessed data cached (so the detail panel can preview).
  - complete_metadata fills the missing fields and transitions to
    PREPROCESSED; the hardware row appears.
  - complete_metadata_and_analyze finishes the run and writes
    measurements / qc_findings.
  - Submitting an incomplete operator dict still raises ConfigError.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest

from hda.domain.errors import ConfigError, HDAError
from hda.domain.state import TestState
from hda.persistence.repositories import (
    HardwareRepository,
    MeasurementsRepository,
    TestRunRepository,
)
from hda.services.ingest import IngestRequest, IngestSource
from hda.ui.workers import (
    complete_metadata_and_analyze,
    run_pipeline,
)
from hda.ui.workspace import build_default_workspace


def _csv(path: Path, n: int = 1000):
    rng = np.random.default_rng(0)
    t_ms = np.arange(n) * 10.0
    p_up = np.full(n, 10.0) + 0.001 * rng.standard_normal(n)
    p_down = np.full(n, 5.0) + 0.001 * rng.standard_normal(n)
    pd.DataFrame({"timestamp": t_ms, "PT-up": p_up, "PT-down": p_down}).to_csv(
        path, index=False
    )


@pytest.fixture
def ws_with_awaiting(tmp_path: Path):
    """A workspace that contains a CSV ingested without metadata."""
    csv = tmp_path / "demo.csv"
    _csv(csv)
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    result = run_pipeline(ws, csv, campaign_id="C1")
    assert result.final_state is TestState.AWAITING_METADATA
    return ws, result.test_run_id


def test_awaiting_run_has_preprocessed_in_cache(ws_with_awaiting):
    ws, run_id = ws_with_awaiting
    cached = ws.preprocessed_cache.get(run_id)
    assert cached is not None
    assert "PT-up" in cached.data.df.columns


def test_complete_metadata_fills_required_and_advances_state(ws_with_awaiting):
    ws, run_id = ws_with_awaiting
    outcome = ws.ingest_service.complete_metadata(
        run_id,
        {
            "part_number": "PN-1",
            "serial_number": "SN-1",
            "operator": "alice",
        },
    )
    runs = TestRunRepository(ws.db)
    assert runs.get_state(run_id) is TestState.PREPROCESSED
    # Hardware row is now present.
    hw = HardwareRepository(ws.db).find_by_part("PN-1")
    assert any(h.serial_number == "SN-1" for h in hw)
    # Metadata stored on the run row.
    conn = ws.db.connect()
    row = conn.execute(
        "SELECT operator, metadata_json FROM test_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["operator"] == "alice"
    stored = json.loads(row["metadata_json"])
    assert stored["serial_number"] == "SN-1"


def test_complete_metadata_rejects_still_incomplete(ws_with_awaiting):
    ws, run_id = ws_with_awaiting
    with pytest.raises(ConfigError, match="still missing"):
        ws.ingest_service.complete_metadata(
            run_id, {"part_number": "PN-1"}  # serial + operator missing
        )


def test_complete_metadata_rejects_invalid_state(tmp_path: Path):
    """A test that's already PERSISTED cannot have its metadata
    replayed via complete_metadata; that path is reanalysis."""
    csv = tmp_path / "demo.csv"
    _csv(csv)
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
        db_path=tmp_path / "hda.db", campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    result = run_pipeline(ws, csv, campaign_id="C1")
    assert result.final_state is TestState.PERSISTED
    with pytest.raises(ConfigError, match="AWAITING_METADATA"):
        ws.ingest_service.complete_metadata(
            result.test_run_id, {"operator": "bob"}
        )


def test_complete_metadata_and_analyze_drives_to_persisted(ws_with_awaiting):
    ws, run_id = ws_with_awaiting
    result = complete_metadata_and_analyze(
        ws,
        run_id,
        {
            "part_number": "PN-1",
            "serial_number": "SN-1",
            "operator": "alice",
        },
    )
    assert result.final_state is TestState.PERSISTED
    saved = MeasurementsRepository(ws.db).get_for_run(run_id)
    assert any(m.name.startswith("avg_") for m in saved)


def test_complete_metadata_and_analyze_without_cache_raises(tmp_path: Path):
    csv = tmp_path / "demo.csv"
    _csv(csv)
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="C1",
        auto_confirm_confidence=0.0,
    )
    result = run_pipeline(ws, csv, campaign_id="C1")
    ws.preprocessed_cache.clear()
    with pytest.raises(HDAError, match="not in the cache"):
        complete_metadata_and_analyze(
            ws,
            result.test_run_id,
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
            },
        )


def test_existing_metadata_for_run_returns_partial(ws_with_awaiting):
    ws, run_id = ws_with_awaiting
    # Even though no fields were supplied, the resolver may have written
    # campaign defaults — return whatever's stored.
    existing = ws.ingest_service.existing_metadata_for_run(run_id)
    assert isinstance(existing, dict)


def test_metadata_schema_for_run(ws_with_awaiting):
    ws, run_id = ws_with_awaiting
    schema = ws.ingest_service.metadata_schema_for_run(run_id)
    names = {f.name for f in schema.fields}
    assert {"part_number", "serial_number", "operator"}.issubset(names)
