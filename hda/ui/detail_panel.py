"""Test-detail panel.

For a selected test_run_id, renders measurements (value ± uncertainty)
and QC findings. Pulls from the repositories on demand; refreshes when
the dashboard signals a state change for the matching id.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hda.persistence import Database
from hda.persistence.repositories import (
    MeasurementsRepository,
    QCFindingsRepository,
    TestRunRepository,
)


class DetailPanel(QWidget):
    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._test_run_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._header = QLabel("No test selected")
        self._header.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(self._header)

        self._meas_box = QGroupBox("Measurements")
        meas_layout = QVBoxLayout(self._meas_box)
        self._meas_table = QTableWidget(0, 4)
        self._meas_table.setHorizontalHeaderLabels(
            ["Name", "Value", "Uncertainty", "Unit"]
        )
        self._meas_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._meas_table.setEditTriggers(QTableWidget.NoEditTriggers)
        meas_layout.addWidget(self._meas_table)
        layout.addWidget(self._meas_box, stretch=2)

        self._qc_box = QGroupBox("QC findings")
        qc_layout = QVBoxLayout(self._qc_box)
        self._qc_table = QTableWidget(0, 4)
        self._qc_table.setHorizontalHeaderLabels(
            ["Check", "Status", "Blocking", "Message"]
        )
        self._qc_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._qc_table.setEditTriggers(QTableWidget.NoEditTriggers)
        qc_layout.addWidget(self._qc_table)
        layout.addWidget(self._qc_box, stretch=1)

    def show_test_run(self, test_run_id: str | None) -> None:
        self._test_run_id = test_run_id
        if test_run_id is None:
            self._header.setText("No test selected")
            self._meas_table.setRowCount(0)
            self._qc_table.setRowCount(0)
            return

        run_state = TestRunRepository(self._db).get_state(test_run_id)
        state_str = run_state.value if run_state is not None else "?"
        self._header.setText(f"Test {test_run_id[:8]} — state: {state_str}")

        measurements = MeasurementsRepository(self._db).get_for_run(test_run_id)
        self._meas_table.setRowCount(len(measurements))
        for r, m in enumerate(measurements):
            self._meas_table.setItem(r, 0, _item(m.name))
            self._meas_table.setItem(r, 1, _item(f"{m.value:.6g}"))
            self._meas_table.setItem(r, 2, _item(f"{m.uncertainty:.6g}"))
            self._meas_table.setItem(r, 3, _item(m.unit))

        findings = QCFindingsRepository(self._db).get_for_run(test_run_id)
        self._qc_table.setRowCount(len(findings))
        for r, f in enumerate(findings):
            self._qc_table.setItem(r, 0, _item(f.check_name))
            self._qc_table.setItem(r, 1, _item(f.status.value))
            self._qc_table.setItem(r, 2, _item("yes" if f.blocking else ""))
            self._qc_table.setItem(r, 3, _item(f.message))


def _item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    return item
