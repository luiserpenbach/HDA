"""Cross-campaign analytics repository queries."""

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
    TestRun,
)
from hda.persistence import Database, apply_migrations
from hda.persistence.repositories import (
    CampaignRepository,
    HardwareRepository,
    MeasurementsRepository,
    TestRunRepository,
)


@pytest.fixture
def populated_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "hda.db")
    apply_migrations(db)
    campaigns = CampaignRepository(db)
    hardware = HardwareRepository(db)
    runs = TestRunRepository(db)
    meas = MeasurementsRepository(db)

    for cid in ("C1", "C2"):
        campaigns.create(
            Campaign(id=cid, name=cid, test_type="cold_flow",
                     created_at=datetime(2026, 1, 1))
        )

    # Two parts, two serials each, distributed across both campaigns.
    spec = [
        ("PN-A", "SN-1", "C1", 0.65),
        ("PN-A", "SN-1", "C1", 0.66),
        ("PN-A", "SN-1", "C2", 0.64),
        ("PN-A", "SN-2", "C1", 0.70),
        ("PN-B", "SN-1", "C2", 0.55),
    ]
    for i, (pn, sn, cid, cd) in enumerate(spec):
        hw_id = hardware.get_or_create(Hardware(part_number=pn, serial_number=sn))
        rid = f"run-{i}"
        runs.insert_initial(
            TestRun(
                id=rid,
                campaign_id=cid,
                file_path=Path(f"/data/{i}.csv"),
                file_hash=str(i) * 64,
                state=TestState.PREPROCESSED,
                discovered_at=datetime(2026, 1, 1 + i),
            ),
            hardware_id=hw_id,
            metadata_values={},
            metadata_hash="",
        )
        meas.write_all(
            rid,
            {
                "avg_cd": MeasurementWithUncertainty(
                    name="avg_cd",
                    value=cd,
                    uncertainty=0.01,
                    unit="",
                ),
                "avg_pt": MeasurementWithUncertainty(
                    name="avg_pt",
                    value=10.0 + i,
                    uncertainty=0.05,
                    unit="bar",
                ),
            },
        )
    return db


def test_list_parts_with_measurements(populated_db: Database):
    repo = MeasurementsRepository(populated_db)
    assert repo.list_parts_with_measurements() == ["PN-A", "PN-B"]


def test_list_serials_for_part(populated_db: Database):
    repo = MeasurementsRepository(populated_db)
    assert repo.list_serials_for_part("PN-A") == ["SN-1", "SN-2"]
    assert repo.list_serials_for_part("PN-B") == ["SN-1"]


def test_list_measurement_names_for_part(populated_db: Database):
    repo = MeasurementsRepository(populated_db)
    names = repo.list_measurement_names_for_part("PN-A")
    assert set(names) == {"avg_cd", "avg_pt"}
    assert names == sorted(names)


def test_list_measurement_names_filtered_by_serial(populated_db: Database):
    repo = MeasurementsRepository(populated_db)
    only_sn1 = repo.list_measurement_names_for_part("PN-A", serial_number="SN-1")
    assert set(only_sn1) == {"avg_cd", "avg_pt"}


def test_hardware_history_crosses_campaigns(populated_db: Database):
    repo = MeasurementsRepository(populated_db)
    df = repo.hardware_history(part_number="PN-A", measurement_name="avg_cd")
    assert len(df) == 4
    assert set(df["campaign_id"]) == {"C1", "C2"}
    assert set(df["serial_number"]) == {"SN-1", "SN-2"}
    # Time-ordered by persisted_at|discovered_at
    seq = list(df["discovered_at"])
    assert seq == sorted(seq)


def test_hardware_history_filtered_by_serial(populated_db: Database):
    repo = MeasurementsRepository(populated_db)
    df = repo.hardware_history(
        part_number="PN-A", measurement_name="avg_cd", serial_number="SN-1"
    )
    assert len(df) == 3
    assert set(df["serial_number"]) == {"SN-1"}


def test_empty_db_returns_empty_lists(tmp_path: Path):
    db = Database(tmp_path / "hda.db")
    apply_migrations(db)
    repo = MeasurementsRepository(db)
    assert repo.list_parts_with_measurements() == []
    assert repo.list_serials_for_part("PN-X") == []
    assert repo.list_measurement_names_for_part("PN-X") == []
