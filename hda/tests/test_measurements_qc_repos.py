"""Measurements + QC findings repositories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from hda.domain.state import TestState
from hda.domain.types import (
    Campaign,
    Hardware,
    MeasurementWithUncertainty,
    Provenance,
    QCFinding,
    QCStatus,
    TestRun,
)
from hda.persistence import Database, apply_migrations
from hda.persistence.repositories import (
    CampaignRepository,
    HardwareRepository,
    MeasurementsRepository,
    QCFindingsRepository,
    TestRunRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "hda.db")
    apply_migrations(d)
    return d


def _seed(db: Database, run_id: str = "run-1", campaign_id: str = "C1"):
    CampaignRepository(db).create(
        Campaign(id=campaign_id, name=campaign_id, test_type="cold_flow",
                 created_at=datetime(2026, 1, 1))
    )
    runs = TestRunRepository(db)
    run = TestRun(
        id=run_id,
        campaign_id=campaign_id,
        file_path=Path("/data/x.csv"),
        file_hash="a" * 64,
        state=TestState.PREPROCESSED,
    )
    runs.insert_initial(run, hardware_id=None, metadata_values={}, metadata_hash="")
    return run_id


def test_measurements_write_then_read_round_trip(db: Database):
    run_id = _seed(db)
    repo = MeasurementsRepository(db)
    repo.write_all(run_id, {
        "avg_PT-up": MeasurementWithUncertainty("avg_PT-up", 10.0, 0.1, "bar"),
        "avg_mf": MeasurementWithUncertainty(
            "avg_mf", 50.0, 0.5, "g/s", provenance=Provenance.DERIVED
        ),
    })
    out = {m.name: m for m in repo.get_for_run(run_id)}
    assert out["avg_PT-up"].value == 10.0
    assert out["avg_mf"].provenance is Provenance.DERIVED


def test_measurements_write_all_replaces_previous(db: Database):
    run_id = _seed(db)
    repo = MeasurementsRepository(db)
    repo.write_all(run_id, {
        "x": MeasurementWithUncertainty("x", 1.0, 0.1, ""),
    })
    repo.write_all(run_id, {
        "y": MeasurementWithUncertainty("y", 2.0, 0.2, ""),
    })
    names = {m.name for m in repo.get_for_run(run_id)}
    assert names == {"y"}


def test_qc_findings_round_trip(db: Database):
    run_id = _seed(db)
    repo = QCFindingsRepository(db)
    findings = [
        QCFinding("ts_monotonic", QCStatus.PASS, "", blocking=True),
        QCFinding("flatline:PT-01", QCStatus.FAIL, "stuck", blocking=True),
    ]
    repo.write_all(run_id, findings, qc_passed=False)
    out = repo.get_for_run(run_id)
    assert len(out) == 2
    assert any(f.status is QCStatus.FAIL for f in out)
    conn = db.connect()
    row = conn.execute(
        "SELECT qc_passed FROM test_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["qc_passed"] == 0


def test_hardware_history_cross_campaign(db: Database):
    # Two campaigns, same part, two tests each, distinct measurements
    CampaignRepository(db).create(
        Campaign(id="C1", name="C1", test_type="cold_flow",
                 created_at=datetime(2026, 1, 1))
    )
    CampaignRepository(db).create(
        Campaign(id="C2", name="C2", test_type="cold_flow",
                 created_at=datetime(2026, 2, 1))
    )
    hw = HardwareRepository(db)
    hw_id = hw.get_or_create(Hardware(part_number="PN-X", serial_number="SN-1"))
    runs_repo = TestRunRepository(db)
    meas_repo = MeasurementsRepository(db)
    for i, c in enumerate(["C1", "C1", "C2", "C2"]):
        rid = f"run-{i}"
        run = TestRun(
            id=rid, campaign_id=c, file_path=Path(f"/data/{i}.csv"),
            file_hash=str(i) * 64, state=TestState.PREPROCESSED,
            discovered_at=datetime(2026, 1, 1 + i),
        )
        runs_repo.insert_initial(
            run, hardware_id=hw_id, metadata_values={}, metadata_hash=""
        )
        meas_repo.write_all(rid, {
            "avg_cd": MeasurementWithUncertainty("avg_cd", 0.6 + 0.01 * i, 0.01, ""),
        })
    df = meas_repo.hardware_history(part_number="PN-X", measurement_name="avg_cd")
    assert len(df) == 4
    assert set(df["campaign_id"]) == {"C1", "C2"}
