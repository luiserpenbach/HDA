"""Hardware + TestRun repository invariants."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from hda.domain.errors import DBError
from hda.domain.state import TestState
from hda.domain.types import Campaign, Hardware, TestRun
from hda.persistence import Database, apply_migrations
from hda.persistence.repositories import (
    CampaignRepository,
    HardwareRepository,
    TestRunRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "hda.db")
    apply_migrations(d)
    return d


@pytest.fixture
def campaign(db: Database) -> Campaign:
    repo = CampaignRepository(db)
    c = Campaign(
        id="C1", name="C1", test_type="cold_flow", created_at=datetime(2026, 1, 1)
    )
    repo.create(c)
    return c


def test_hardware_get_or_create_inserts_once(db: Database):
    repo = HardwareRepository(db)
    a = Hardware(part_number="PN-1", serial_number="SN-1")
    id1 = repo.get_or_create(a)
    id2 = repo.get_or_create(a)
    assert id1 == id2
    assert repo.get(id1) == a


def test_hardware_distinct_serials_get_distinct_ids(db: Database):
    repo = HardwareRepository(db)
    id1 = repo.get_or_create(Hardware(part_number="PN-1", serial_number="SN-1"))
    id2 = repo.get_or_create(Hardware(part_number="PN-1", serial_number="SN-2"))
    assert id1 != id2


def test_hardware_find_by_part_returns_all_serials(db: Database):
    repo = HardwareRepository(db)
    repo.get_or_create(Hardware(part_number="PN-X", serial_number="SN-1"))
    repo.get_or_create(Hardware(part_number="PN-X", serial_number="SN-2"))
    repo.get_or_create(Hardware(part_number="PN-Y", serial_number="SN-1"))
    found = repo.find_by_part("PN-X")
    assert {h.serial_number for h in found} == {"SN-1", "SN-2"}


def test_test_run_insert_initial_round_trip(db: Database, campaign: Campaign):
    runs = TestRunRepository(db)
    hw_repo = HardwareRepository(db)
    hw_id = hw_repo.get_or_create(Hardware(part_number="PN-1", serial_number="SN-1"))
    run = TestRun(
        id="run-1",
        campaign_id=campaign.id,
        file_path=Path("/data/test.csv"),
        file_hash="a" * 64,
        state=TestState.PREPROCESSED,
    )
    runs.insert_initial(
        run, hardware_id=hw_id, metadata_values={"x": 1}, metadata_hash="b" * 64
    )
    assert runs.get_state("run-1") is TestState.PREPROCESSED
    assert runs.find_by_file_hash("a" * 64) == ["run-1"]


def test_test_run_insert_terminal_state_rejected(db: Database, campaign: Campaign):
    runs = TestRunRepository(db)
    run = TestRun(
        id="run-2",
        campaign_id=campaign.id,
        file_path=Path("/data/test.csv"),
        file_hash="c" * 64,
        state=TestState.PERSISTED,
    )
    with pytest.raises(DBError, match="terminal state"):
        runs.insert_initial(run, hardware_id=None, metadata_values={}, metadata_hash="")


def test_test_run_update_state_validates_transition(db: Database, campaign: Campaign):
    runs = TestRunRepository(db)
    run = TestRun(
        id="run-3",
        campaign_id=campaign.id,
        file_path=Path("/data/test.csv"),
        file_hash="d" * 64,
        state=TestState.PREPROCESSED,
    )
    runs.insert_initial(run, hardware_id=None, metadata_values={}, metadata_hash="")
    runs.update_state("run-3", TestState.STEADY_DETECTED)
    assert runs.get_state("run-3") is TestState.STEADY_DETECTED

    from hda.domain.errors import IllegalTransition

    with pytest.raises(IllegalTransition):
        runs.update_state("run-3", TestState.PERSISTED)


def test_test_run_update_state_unknown_id_raises(db: Database):
    runs = TestRunRepository(db)
    with pytest.raises(DBError, match="does not exist"):
        runs.update_state("does-not-exist", TestState.STEADY_DETECTED)


def test_test_run_persisted_at_set_only_on_persisted(db: Database, campaign: Campaign):
    runs = TestRunRepository(db)
    run = TestRun(
        id="run-4",
        campaign_id=campaign.id,
        file_path=Path("/data/test.csv"),
        file_hash="e" * 64,
        state=TestState.ANALYZED,
    )
    runs.insert_initial(run, hardware_id=None, metadata_values={}, metadata_hash="")
    runs.update_state("run-4", TestState.PERSISTED)
    conn = db.connect()
    row = conn.execute(
        "SELECT persisted_at FROM test_runs WHERE id = ?", ("run-4",)
    ).fetchone()
    assert row["persisted_at"] is not None


def test_list_for_campaign_orders_by_recent(db: Database, campaign: Campaign):
    runs = TestRunRepository(db)
    for i, h in enumerate(("a", "b", "c")):
        run = TestRun(
            id=f"run-{i}",
            campaign_id=campaign.id,
            file_path=Path(f"/data/{i}.csv"),
            file_hash=h * 64,
            state=TestState.PREPROCESSED,
            discovered_at=datetime(2026, 1, 1 + i),
        )
        runs.insert_initial(
            run, hardware_id=None, metadata_values={}, metadata_hash=""
        )
    listing = runs.list_for_campaign(campaign.id)
    ids = [r["id"] for r in listing]
    assert ids == ["run-2", "run-1", "run-0"]
