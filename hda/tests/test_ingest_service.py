"""End-to-end ingest: file → preprocess → metadata → DB."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hda.domain.derived import DerivedChannelSpec, UncertaintyMethod
from hda.domain.errors import ConfigError, IngestError
from hda.domain.metadata import FieldType, MetadataField, MetadataSchema
from hda.domain.state import TestState
from hda.domain.types import Campaign
from hda.persistence import Database, apply_migrations
from hda.persistence.repositories import (
    CampaignRepository,
    HardwareRepository,
    TestRunRepository,
)
from hda.services import (
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


def _seed_campaign(db: Database, campaign_id: str = "INJ-CF-C1") -> str:
    CampaignRepository(db).create(
        Campaign(
            id=campaign_id,
            name="Demo",
            test_type="cold_flow",
            created_at=datetime(2026, 1, 1),
        )
    )
    return campaign_id


def _schema() -> MetadataSchema:
    return MetadataSchema(
        fields=(
            MetadataField("part_number", FieldType.STRING, required=True),
            MetadataField("serial_number", FieldType.STRING, required=True),
            MetadataField("operator", FieldType.STRING, required=True),
            MetadataField("fluid", FieldType.STRING),
            MetadataField("test_id", FieldType.STRING),
            MetadataField("fuel_additive", FieldType.STRING),
        )
    )


def _write_csv(path: Path, n: int = 100) -> None:
    t = np.linspace(0.0, (n - 1) * 10.0, n)
    df = pd.DataFrame(
        {
            "timestamp": t,
            "PT-up": np.linspace(10.0, 12.0, n),
            "PT-down": np.linspace(5.0, 6.0, n),
        }
    )
    df.to_csv(path, index=False)


def _pipeline(derived=()) -> IngestPipeline:
    return IngestPipeline(
        metadata_schema=_schema(),
        preprocessing=PreprocessingConfig(
            timestamp_column="timestamp",
            resample_freq_hz=None,
            nan_policy=NaNPolicy.LEAVE,
            derived_channels=derived,
        ),
        campaign_metadata_defaults={"fluid": "N2"},
    )


def _request(campaign_id: str, csv_path: Path, sidecar=None, operator=None) -> IngestRequest:
    return IngestRequest(
        file_path=csv_path,
        campaign_id=campaign_id,
        source=IngestSource.FILE_DIALOG,
        sidecar_metadata=sidecar,
        operator=operator,
    )


def test_ingest_complete_metadata_lands_in_preprocessed(db: Database, tmp_path: Path):
    campaign = _seed_campaign(db)
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
                "test_id": "T-001",
            }
        )
    )
    svc = IngestServiceImpl(db, pipelines={campaign: _pipeline()})
    outcome = svc.process(_request(campaign, csv))
    assert outcome.state is TestState.PREPROCESSED
    assert outcome.preprocessed is not None
    assert outcome.preprocessed.n_samples == 100
    assert outcome.missing_metadata == ()


def test_ingest_missing_required_metadata_lands_in_awaiting(
    db: Database, tmp_path: Path
):
    campaign = _seed_campaign(db)
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    svc = IngestServiceImpl(db, pipelines={campaign: _pipeline()})
    outcome = svc.process(
        _request(campaign, csv, sidecar={"part_number": "PN-1"})
    )
    assert outcome.state is TestState.AWAITING_METADATA
    assert "serial_number" in outcome.missing_metadata
    assert "operator" in outcome.missing_metadata
    assert outcome.preprocessed is None


def test_ingest_persists_test_run_and_hardware(db: Database, tmp_path: Path):
    campaign = _seed_campaign(db)
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
            }
        )
    )
    svc = IngestServiceImpl(db, pipelines={campaign: _pipeline()})
    test_run_id = svc.enqueue(_request(campaign, csv))

    runs = TestRunRepository(db)
    assert runs.get_state(test_run_id) is TestState.PREPROCESSED

    hw = HardwareRepository(db)
    found = hw.find_by_part("PN-1")
    assert len(found) == 1 and found[0].serial_number == "SN-1"


def test_ingest_idempotent_on_duplicate_file(db: Database, tmp_path: Path):
    campaign = _seed_campaign(db)
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
            }
        )
    )
    svc = IngestServiceImpl(db, pipelines={campaign: _pipeline()})
    first = svc.process(_request(campaign, csv))
    second = svc.process(_request(campaign, csv))
    assert first.test_run_id == second.test_run_id
    assert second.duplicate_of == first.test_run_id


def test_ingest_unknown_campaign_raises(db: Database, tmp_path: Path):
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    svc = IngestServiceImpl(db, pipelines={})
    with pytest.raises(ConfigError, match="No ingest pipeline"):
        svc.process(_request("nope", csv))


def test_ingest_campaign_not_in_db_raises(db: Database, tmp_path: Path):
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    svc = IngestServiceImpl(
        db, pipelines={"ghost": _pipeline()}
    )
    with pytest.raises(ConfigError, match="does not exist"):
        svc.process(_request("ghost", csv))


def test_ingest_missing_file_raises(db: Database, tmp_path: Path):
    campaign = _seed_campaign(db)
    svc = IngestServiceImpl(db, pipelines={campaign: _pipeline()})
    with pytest.raises(IngestError):
        svc.process(_request(campaign, tmp_path / "nope.csv"))


def test_ingest_sidecar_overrides_campaign_default(db: Database, tmp_path: Path):
    campaign = _seed_campaign(db)
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
                "fluid": "GHe",  # overrides campaign default of N2
            }
        )
    )
    svc = IngestServiceImpl(db, pipelines={campaign: _pipeline()})
    outcome = svc.process(_request(campaign, csv))

    runs = TestRunRepository(db)
    conn = db.connect()
    row = conn.execute(
        "SELECT fluid FROM test_runs WHERE id = ?", (outcome.test_run_id,)
    ).fetchone()
    assert row["fluid"] == "GHe"


def test_ingest_evaluates_derived_channels_during_preprocessing(
    db: Database, tmp_path: Path
):
    campaign = _seed_campaign(db)
    csv = tmp_path / "test.csv"
    _write_csv(csv)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "part_number": "PN-1",
                "serial_number": "SN-1",
                "operator": "alice",
            }
        )
    )
    dp = DerivedChannelSpec(
        name="dp_bar",
        unit="bar",
        formula="subtract",
        inputs={"a": "PT-up", "b": "PT-down"},
        uncertainty_method=UncertaintyMethod.NONE,
    )
    svc = IngestServiceImpl(
        db, pipelines={campaign: _pipeline(derived=(dp,))}
    )
    outcome = svc.process(_request(campaign, csv))
    assert "dp_bar" in outcome.preprocessed.df.columns
    assert outcome.preprocessed.derived_channel_names == ("dp_bar",)
