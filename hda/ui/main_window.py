"""Main window: dashboard + detail panel + add-test action.

Wiring:
    Add Test button   ->  file dialog  ->  IngestAndAnalyzeWorker
    Worker.finished   ->  reload dashboard, select the new run
    Dashboard click   ->  DetailPanel.show_test_run
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hda.ui.dashboard import DashboardData, DashboardModel
from hda.ui.detail_panel import DetailPanel
from hda.ui.logging_setup import get_logger
from hda.ui.workers import IngestAndAnalyzeWorker, PipelineResult
from hda.ui.workspace import Workspace


_log = get_logger("main_window")


class MainWindow(QMainWindow):
    def __init__(
        self, workspace: Workspace, default_campaign_id: str
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._campaign_id = default_campaign_id
        self._pool = QThreadPool.globalInstance()

        self.setWindowTitle("Hopper Data Studio v3")
        self.resize(1200, 720)

        self._dash_data = DashboardData(workspace.db)
        self._dash_data.set_campaign(default_campaign_id)
        self._dash_model = DashboardModel(self._dash_data)

        self._table = QTableView()
        self._table.setModel(self._dash_model)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self._detail = DetailPanel(workspace)
        self._detail.reanalyzed.connect(self._on_reanalyzed)

        splitter = QSplitter()
        splitter.addWidget(self._table)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 700])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._campaign_bar(), 0)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        toolbar = QToolBar("Actions")
        self.addToolBar(toolbar)
        add_action = QAction("Add Test…", self)
        add_action.setShortcut("Ctrl+O")
        add_action.triggered.connect(self._on_add_test)
        toolbar.addAction(add_action)
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._dash_model.reload)
        toolbar.addAction(refresh_action)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            f"Campaign: {self._campaign_id} • DB: {workspace.db.path}"
        )

    def _campaign_bar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 6, 8, 6)
        label = QLabel(f"Active campaign: <b>{self._campaign_id}</b>")
        h.addWidget(label)
        h.addStretch(1)
        add_btn = QPushButton("+ Add Test")
        add_btn.clicked.connect(self._on_add_test)
        h.addWidget(add_btn)
        return bar

    def _on_add_test(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Add test data file",
            str(Path.home()),
            "Test data (*.csv);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        self.statusBar().showMessage(f"Ingesting {path.name}…")
        worker = IngestAndAnalyzeWorker(
            workspace=self._workspace,
            file_path=path,
            campaign_id=self._campaign_id,
        )
        worker.signals.finished.connect(self._on_pipeline_finished)
        worker.signals.failed.connect(self._on_pipeline_failed)
        self._pool.start(worker)

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
        self._dash_model.reload()
        self._select_run(result.test_run_id)
        self.statusBar().showMessage(
            f"{result.test_run_id[:8]} → {result.final_state.value}"
        )

    def _on_pipeline_failed(self, message: str) -> None:
        _log.error("pipeline failed: %s", message)
        self.statusBar().showMessage("Pipeline failed.")
        QMessageBox.critical(self, "Ingest / analysis failed", message)

    def _select_run(self, test_run_id: str) -> None:
        for r in range(self._dash_model.rowCount()):
            if self._dash_model.test_run_id_at(r) == test_run_id:
                self._table.selectRow(r)
                break

    def _on_selection_changed(self, *_args) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._detail.show_test_run(None)
            return
        rid = self._dash_model.test_run_id_at(rows[0].row())
        self._detail.show_test_run(rid)

    def _on_reanalyzed(self, test_run_id: str) -> None:
        self._dash_model.reload()
        self._select_run(test_run_id)
        self.statusBar().showMessage(f"Reanalyzed {test_run_id[:8]}.")
