"""Dashboard test list.

A QAbstractTableModel that surfaces the persisted test runs of a
campaign, plus a small ``DashboardData`` shim that is independent of Qt
so the data layer can be unit-tested without instantiating QApplication.

State colors are exposed via ``Qt.BackgroundRole`` so the operator can
scan a 10-tests/day list at a glance and drill into the yellows / reds.
The mapping is in ``STATE_COLORS`` (single source of truth, also used by
the detail panel's header tint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor

from hda.persistence import Database
from hda.persistence.repositories import TestRunRepository


# (foreground, background) per state. Subdued so the table reads cleanly;
# the foreground is used to tint the detail-panel header.
STATE_COLORS: dict[str, tuple[QColor, QColor]] = {
    "discovered":         (QColor("#52525b"), QColor("#f4f4f5")),
    "ingesting":          (QColor("#0c4a6e"), QColor("#e0f2fe")),
    "awaiting_metadata":  (QColor("#92400e"), QColor("#fef3c7")),
    "preprocessed":       (QColor("#0c4a6e"), QColor("#e0f2fe")),
    "steady_detected":    (QColor("#0c4a6e"), QColor("#e0f2fe")),
    "qc_run":             (QColor("#0c4a6e"), QColor("#e0f2fe")),
    "needs_review":       (QColor("#92400e"), QColor("#fef3c7")),
    "analyzed":           (QColor("#14532d"), QColor("#dcfce7")),
    "persisted":          (QColor("#14532d"), QColor("#dcfce7")),
    "qc_failed":          (QColor("#7f1d1d"), QColor("#fee2e2")),
    "error":              (QColor("#7f1d1d"), QColor("#fee2e2")),
}


def state_colors(state: str) -> tuple[QColor, QColor]:
    return STATE_COLORS.get(
        state, (QColor("#27272a"), QColor("#fafafa"))
    )


@dataclass(frozen=True, slots=True)
class TestRow:
    test_run_id: str
    test_id_label: str
    state: str
    operator: str
    discovered_at: str
    persisted_at: str

    @property
    def display_id(self) -> str:
        return self.test_id_label or self.test_run_id[:8]


class DashboardData:
    """Pure-Python data source. Owns the DB read; the Qt model wraps it."""

    COLUMNS: tuple[str, ...] = (
        "Test ID",
        "State",
        "Operator",
        "Discovered",
        "Persisted",
    )

    def __init__(self, db: Database) -> None:
        self._db = db
        self._rows: List[TestRow] = []
        self._campaign_id: Optional[str] = None

    def set_campaign(self, campaign_id: str) -> None:
        self._campaign_id = campaign_id
        self.refresh()

    def refresh(self) -> None:
        if self._campaign_id is None:
            self._rows = []
            return
        runs = TestRunRepository(self._db).list_for_campaign(self._campaign_id)
        self._rows = [
            TestRow(
                test_run_id=r["id"],
                test_id_label=r.get("test_id_label") or "",
                state=r["state"],
                operator=r.get("operator") or "",
                discovered_at=(r.get("discovered_at") or "")[:19],
                persisted_at=(r.get("persisted_at") or "")[:19],
            )
            for r in runs
        ]

    @property
    def rows(self) -> List[TestRow]:
        return list(self._rows)

    def row(self, index: int) -> TestRow:
        return self._rows[index]


# Custom data role for "give me this row's test_run_id" — survives sort.
TEST_RUN_ID_ROLE = Qt.UserRole + 1


class DashboardModel(QAbstractTableModel):
    def __init__(self, data: DashboardData) -> None:
        super().__init__()
        self._data = data

    def reload(self) -> None:
        self.beginResetModel()
        self._data.refresh()
        self.endResetModel()

    def set_campaign(self, campaign_id: str) -> None:
        self.beginResetModel()
        self._data.set_campaign(campaign_id)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._data.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(DashboardData.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.DisplayRole:
            return DashboardData.COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row = self._data.row(index.row())
        col = index.column()

        if role == TEST_RUN_ID_ROLE:
            return row.test_run_id

        if role == Qt.DisplayRole:
            if col == 0:
                return row.display_id
            if col == 1:
                return row.state
            if col == 2:
                return row.operator
            if col == 3:
                return row.discovered_at
            if col == 4:
                return row.persisted_at
            return None

        if role == Qt.ToolTipRole:
            if col == 0:
                return row.test_run_id  # full UUID on hover
            return None

        if role == Qt.BackgroundRole and col == 1:
            _, bg = state_colors(row.state)
            return QBrush(bg)
        if role == Qt.ForegroundRole and col == 1:
            fg, _ = state_colors(row.state)
            return QBrush(fg)

        if role == Qt.TextAlignmentRole:
            if col in (3, 4):
                return int(Qt.AlignVCenter | Qt.AlignRight)
        return None

    def test_run_id_at(self, row: int) -> Optional[str]:
        if 0 <= row < len(self._data.rows):
            return self._data.rows[row].test_run_id
        return None
