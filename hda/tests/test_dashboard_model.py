"""Dashboard model: Qt-aware. Uses the offscreen QPA platform so the test
suite can run without a display server. Skips entirely when PySide6's
QtWidgets cannot load (e.g. CI containers missing libEGL)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication
except (ImportError, OSError) as e:
    pytest.skip(
        f"PySide6 QtWidgets unavailable in this environment: {e}",
        allow_module_level=True,
    )

from hda.persistence.repositories import (  # noqa: E402
    CampaignRepository,
    TestRunRepository,
)
from hda.domain.state import TestState  # noqa: E402
from hda.domain.types import Campaign, TestRun  # noqa: E402
from hda.ui.dashboard import DashboardData, DashboardModel  # noqa: E402
from hda.ui.workspace import build_default_workspace  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QCoreApplication.instance() or QApplication([])
    yield instance


def _seed(ws, n: int = 3):
    runs = TestRunRepository(ws.db)
    for i in range(n):
        run = TestRun(
            id=f"run-{i}",
            campaign_id="DEMO-C1",
            file_path=Path(f"/data/{i}.csv"),
            file_hash=str(i) * 64,
            state=TestState.PREPROCESSED,
            discovered_at=datetime(2026, 1, 1 + i),
        )
        runs.insert_initial(
            run, hardware_id=None, metadata_values={}, metadata_hash=""
        )


def test_dashboard_data_lists_seeded_runs(tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    _seed(ws, n=3)
    data = DashboardData(ws.db)
    data.set_campaign("DEMO-C1")
    assert len(data.rows) == 3
    # list_for_campaign orders by recency, so run-2 first
    assert data.row(0).test_run_id == "run-2"


def test_dashboard_model_basic_layout(app, tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    _seed(ws, n=2)
    data = DashboardData(ws.db)
    data.set_campaign("DEMO-C1")
    model = DashboardModel(data)
    assert model.rowCount() == 2
    assert model.columnCount() == len(DashboardData.COLUMNS)
    assert model.headerData(0, Qt.Horizontal, Qt.DisplayRole) == "Test ID"
    assert model.test_run_id_at(0) == "run-1"
    assert model.test_run_id_at(99) is None


def test_dashboard_model_set_campaign_resets(app, tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db", campaign_id="A")
    _seed_one(ws, "A")
    CampaignRepository(ws.db).create(
        Campaign(id="B", name="B", test_type="cold_flow",
                 created_at=datetime(2026, 1, 1))
    )

    data = DashboardData(ws.db)
    model = DashboardModel(data)
    model.set_campaign("A")
    assert model.rowCount() == 1
    model.set_campaign("B")
    assert model.rowCount() == 0


def _seed_one(ws, campaign_id: str):
    TestRunRepository(ws.db).insert_initial(
        TestRun(
            id="x",
            campaign_id=campaign_id,
            file_path=Path("/x.csv"),
            file_hash="a" * 64,
            state=TestState.PREPROCESSED,
        ),
        hardware_id=None,
        metadata_values={},
        metadata_hash="",
    )
