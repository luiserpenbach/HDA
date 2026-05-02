"""Persistence: WAL setup, transactional migrations, repository round-trips."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import pytest

from hda.domain.errors import DBError
from hda.domain.types import Campaign
from hda.persistence import Database, apply_migrations, current_version
from hda.persistence.db import transaction
from hda.persistence.repositories import CampaignRepository
from hda.persistence.schema import SCHEMA_VERSION


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "hda.db")


def test_wal_mode_enabled(db: Database):
    conn = db.connect()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_foreign_keys_enabled(db: Database):
    conn = db.connect()
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_busy_timeout_set(db: Database):
    conn = db.connect()
    bt = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert bt >= 5000


def test_apply_migrations_reaches_schema_version(db: Database):
    assert current_version(db) == 0
    final = apply_migrations(db)
    assert final == SCHEMA_VERSION
    assert current_version(db) == SCHEMA_VERSION


def test_apply_migrations_is_idempotent(db: Database):
    apply_migrations(db)
    final = apply_migrations(db)
    assert final == SCHEMA_VERSION


def test_apply_migrations_creates_expected_tables(db: Database):
    apply_migrations(db)
    conn = db.connect()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    names = {r["name"] for r in rows}
    expected = {
        "_meta",
        "campaigns",
        "hardware",
        "test_runs",
        "measurements",
        "qc_findings",
        "derived_specs",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


def test_indexes_exist_for_cross_campaign_analytics(db: Database):
    apply_migrations(db)
    conn = db.connect()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    ).fetchall()
    names = {r["name"] for r in rows}
    for required in (
        "idx_hardware_part",
        "idx_hardware_serial",
        "idx_test_runs_campaign",
        "idx_test_runs_hardware",
        "idx_measurements_run_name",
    ):
        assert required in names, f"Missing index {required}"


def test_transaction_rolls_back_on_exception(db: Database):
    apply_migrations(db)
    repo = CampaignRepository(db)
    repo.create(
        Campaign(
            id="C1",
            name="C1",
            test_type="cold_flow",
            created_at=datetime(2026, 1, 1),
        )
    )
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(db, write=True) as conn:
            conn.execute(
                "INSERT INTO campaigns(id, name, test_type, created_at, archived) "
                "VALUES ('C2','C2','cold_flow','2026-01-01',0)"
            )
            conn.execute(
                "INSERT INTO campaigns(id, name, test_type, created_at, archived) "
                "VALUES ('C2','dup','cold_flow','2026-01-01',0)"
            )
    assert repo.get("C2") is None, "Failed transaction must roll back"


def test_campaign_repository_round_trip(db: Database):
    apply_migrations(db)
    repo = CampaignRepository(db)
    c = Campaign(
        id="INJ-CF-C1",
        name="INJ Cold Flow",
        test_type="cold_flow",
        created_at=datetime(2026, 5, 2, 9, 0, 0),
    )
    repo.create(c)
    got = repo.get("INJ-CF-C1")
    assert got == c


def test_campaign_list_excludes_archived_by_default(db: Database):
    apply_migrations(db)
    repo = CampaignRepository(db)
    repo.create(
        Campaign(id="A", name="A", test_type="cold_flow", created_at=datetime.utcnow())
    )
    repo.create(
        Campaign(
            id="B",
            name="B",
            test_type="cold_flow",
            created_at=datetime.utcnow(),
            archived=True,
        )
    )
    assert {c.id for c in repo.list()} == {"A"}
    assert {c.id for c in repo.list(include_archived=True)} == {"A", "B"}


def test_archive_campaign(db: Database):
    apply_migrations(db)
    repo = CampaignRepository(db)
    repo.create(
        Campaign(id="X", name="X", test_type="cold_flow", created_at=datetime.utcnow())
    )
    repo.archive("X")
    assert repo.list() == []
    assert repo.get("X").archived is True


def test_refuse_to_open_newer_schema(tmp_path: Path):
    db = Database(tmp_path / "future.db")
    apply_migrations(db)
    conn = db.connect()
    with transaction(db, write=True) as c:
        c.execute(
            "INSERT INTO _meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION + 99),),
        )
    with pytest.raises(DBError):
        apply_migrations(db)


def test_concurrent_writes_serialize_via_lock(db: Database):
    apply_migrations(db)
    repo = CampaignRepository(db)
    errors: list[Exception] = []

    def worker(i: int):
        try:
            repo.create(
                Campaign(
                    id=f"C{i}",
                    name=f"C{i}",
                    test_type="cold_flow",
                    created_at=datetime.utcnow(),
                )
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(repo.list()) == 8
