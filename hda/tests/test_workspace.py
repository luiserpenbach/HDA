"""Workspace factory: builds a usable ingest+analysis substrate."""

from __future__ import annotations

from pathlib import Path

from hda.persistence import apply_migrations
from hda.persistence.repositories import CampaignRepository
from hda.ui.workspace import Workspace, build_default_workspace


def test_build_default_workspace_creates_campaign(tmp_path: Path):
    ws = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="DEMO-1"
    )
    assert isinstance(ws, Workspace)
    assert ws.ingest_service is not None
    assert ws.analysis_service is not None
    campaign = CampaignRepository(ws.db).get("DEMO-1")
    assert campaign is not None
    assert campaign.test_type == "cold_flow"


def test_build_default_workspace_idempotent_on_existing_campaign(tmp_path: Path):
    ws1 = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="X"
    )
    ws2 = build_default_workspace(
        db_path=tmp_path / "hda.db", campaign_id="X"
    )
    assert (
        CampaignRepository(ws1.db).get("X")
        == CampaignRepository(ws2.db).get("X")
    )


def test_workspace_registers_basic_means(tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    assert "basic_means" in ws.plugins.names()


def test_workspace_db_is_migrated(tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    conn = ws.db.connect()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert {"campaigns", "test_runs", "measurements", "qc_findings"}.issubset(names)
