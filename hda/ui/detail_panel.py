"""Test-detail panel.

For a selected test_run_id renders three sections:
  1. Header line (id + state).
  2. Interactive steady-state preview (when preprocessed data is in the
     cache) with live window stats and an Apply-window button.
  3. Measurements + QC findings tables, refreshed on demand.

The steady-state preview is wired through ``window_committed`` to a
ReanalyzeWorker; on success the panel reloads measurements + QC.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hda.domain.types import SteadyWindow
from hda.persistence import Database
from hda.persistence.repositories import (
    MeasurementsRepository,
    QCFindingsRepository,
    TestRunRepository,
)
from hda.ui.logging_setup import get_logger
from hda.ui.steady_state_preview import PYQTGRAPH_AVAILABLE
from hda.ui.workers import PipelineResult, ReanalyzeWorker
from hda.ui.workspace import Workspace

if PYQTGRAPH_AVAILABLE:
    from hda.ui.steady_state_preview import SteadyStatePreview


_log = get_logger("detail_panel")


class DetailPanel(QWidget):
    """Test-detail view. Owns the steady-state preview and the
    measurements / QC tables for the selected run."""

    reanalyzed = Signal(str)

    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._workspace = workspace
        self._db: Database = workspace.db
        self._test_run_id: Optional[str] = None
        self._pool = QThreadPool.globalInstance()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._header = QLabel("No test selected")
        self._header.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(self._header)

        self._preview: Optional[SteadyStatePreview] = None
        if PYQTGRAPH_AVAILABLE:
            self._preview_box = QGroupBox("Steady-state window")
            preview_layout = QVBoxLayout(self._preview_box)
            self._preview = SteadyStatePreview()
            self._preview.window_committed.connect(self._on_window_committed)
            preview_layout.addWidget(self._preview)
            self._preview_box.setVisible(False)
            layout.addWidget(self._preview_box, stretch=3)
        else:
            self._preview_box = None  # type: ignore[assignment]

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
            if self._preview is not None:
                self._preview.clear()
                self._preview_box.setVisible(False)
            return

        run_state = TestRunRepository(self._db).get_state(test_run_id)
        state_str = run_state.value if run_state is not None else "?"
        self._header.setText(f"Test {test_run_id[:8]} — state: {state_str}")

        self._populate_preview(test_run_id)
        self._populate_measurements(test_run_id)
        self._populate_qc(test_run_id)

    def _populate_preview(self, test_run_id: str) -> None:
        if self._preview is None:
            return
        cached = self._workspace.preprocessed_cache.get(test_run_id)
        steady_row = self._lookup_steady_window(test_run_id)
        if cached is None or steady_row is None:
            self._preview.clear()
            self._preview_box.setVisible(False)
            return
        self._preview.show_data(
            df=cached.data.df,
            initial_window=steady_row,
            timestamp_column="timestamp",
        )
        self._preview_box.setVisible(True)

    def _lookup_steady_window(self, test_run_id: str) -> Optional[SteadyWindow]:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT steady_start_s, steady_end_s, steady_method, steady_confidence "
            "FROM test_runs WHERE id = ?",
            (test_run_id,),
        ).fetchone()
        if row is None:
            return None
        s, e = row["steady_start_s"], row["steady_end_s"]
        if s is None or e is None:
            return None
        try:
            return SteadyWindow(
                start_s=float(s),
                end_s=float(e),
                method=row["steady_method"] or "stored",
                confidence=float(row["steady_confidence"] or 0.0),
            )
        except ValueError:
            return None

    def _populate_measurements(self, test_run_id: str) -> None:
        measurements = MeasurementsRepository(self._db).get_for_run(test_run_id)
        self._meas_table.setRowCount(len(measurements))
        for r, m in enumerate(measurements):
            self._meas_table.setItem(r, 0, _item(m.name))
            self._meas_table.setItem(r, 1, _item(f"{m.value:.6g}"))
            self._meas_table.setItem(r, 2, _item(f"{m.uncertainty:.6g}"))
            self._meas_table.setItem(r, 3, _item(m.unit))

    def _populate_qc(self, test_run_id: str) -> None:
        findings = QCFindingsRepository(self._db).get_for_run(test_run_id)
        self._qc_table.setRowCount(len(findings))
        for r, f in enumerate(findings):
            self._qc_table.setItem(r, 0, _item(f.check_name))
            self._qc_table.setItem(r, 1, _item(f.status.value))
            self._qc_table.setItem(r, 2, _item("yes" if f.blocking else ""))
            self._qc_table.setItem(r, 3, _item(f.message))

    def _on_window_committed(self, window: SteadyWindow) -> None:
        if self._test_run_id is None:
            return
        _log.info(
            "operator commit window: id=%s [%.3f, %.3f]",
            self._test_run_id,
            window.start_s,
            window.end_s,
        )
        worker = ReanalyzeWorker(self._workspace, self._test_run_id, window)
        worker.signals.finished.connect(self._on_reanalyze_finished)
        worker.signals.failed.connect(self._on_reanalyze_failed)
        self._pool.start(worker)

    def _on_reanalyze_finished(self, result: PipelineResult) -> None:
        self.show_test_run(result.test_run_id)
        self.reanalyzed.emit(result.test_run_id)

    def _on_reanalyze_failed(self, message: str) -> None:
        _log.error("reanalyze failed: %s", message)
        QMessageBox.critical(self, "Reanalysis failed", message)


def _item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    return item
