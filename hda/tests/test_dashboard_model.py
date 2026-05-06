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
    from PySide6.QtGui import QBrush
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
from hda.ui.dashboard import (  # noqa: E402
    STATE_COLORS,
    TEST_RUN_ID_ROLE,
    DashboardData,
    DashboardModel,
    state_colors,
)
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


def test_test_run_id_role_returns_full_uuid(app, tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    _seed(ws, n=2)
    data = DashboardData(ws.db)
    data.set_campaign("DEMO-C1")
    model = DashboardModel(data)
    idx = model.index(0, 0)
    assert model.data(idx, TEST_RUN_ID_ROLE) == "run-1"
    # Same regardless of column the index points at.
    assert model.data(model.index(0, 4), TEST_RUN_ID_ROLE) == "run-1"


def test_state_column_has_color_pair_per_state(app, tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    _seed(ws, n=1)
    data = DashboardData(ws.db)
    data.set_campaign("DEMO-C1")
    model = DashboardModel(data)
    state_idx = model.index(0, 1)
    bg = model.data(state_idx, Qt.BackgroundRole)
    fg = model.data(state_idx, Qt.ForegroundRole)
    assert isinstance(bg, QBrush)
    assert isinstance(fg, QBrush)


def test_state_colors_table_covers_every_TestState(app):
    # Every TestState value defined in the domain has a UI color.
    for s in TestState:
        assert s.value in STATE_COLORS, f"missing color for state {s.value!r}"


def test_state_colors_helper_falls_back_for_unknown(app):
    fg, bg = state_colors("never_seen_state")
    assert fg is not None and bg is not None  # fallback present


def test_test_id_column_carries_full_uuid_as_tooltip(app, tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    _seed(ws, n=1)
    data = DashboardData(ws.db)
    data.set_campaign("DEMO-C1")
    model = DashboardModel(data)
    idx = model.index(0, 0)
    assert model.data(idx, Qt.ToolTipRole) == "run-0"


def test_date_columns_right_aligned(app, tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    _seed(ws, n=1)
    data = DashboardData(ws.db)
    data.set_campaign("DEMO-C1")
    model = DashboardModel(data)
    align_discovered = model.data(model.index(0, 3), Qt.TextAlignmentRole)
    assert align_discovered & Qt.AlignRight
    align_persisted = model.data(model.index(0, 4), Qt.TextAlignmentRole)
    assert align_persisted & Qt.AlignRight


def test_display_id_uses_test_id_label_when_present(app, tmp_path: Path):
    ws = build_default_workspace(db_path=tmp_path / "hda.db")
    runs = TestRunRepository(ws.db)
    from hda.domain.types import TestMetadata, Hardware

    runs.insert_initial(
        TestRun(
            id="abcd1234ef" * 6 + "abcd",  # long uuid-ish
            campaign_id="DEMO-C1",
            file_path=Path("/x.csv"),
            file_hash="z" * 64,
            state=TestState.PREPROCESSED,
            metadata=TestMetadata(
                hardware=Hardware(part_number="PN", serial_number="SN"),
                fluid="N2",
                operator="alice",
                test_id="HF-2026-001",
            ),
        ),
        hardware_id=None,
        metadata_values={"test_id": "HF-2026-001"},
        metadata_hash="",
    )
    data = DashboardData(ws.db)
    data.set_campaign("DEMO-C1")
    model = DashboardModel(data)
    assert model.data(model.index(0, 0), Qt.DisplayRole) == "HF-2026-001"
