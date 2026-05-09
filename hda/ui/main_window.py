"""Main window: dashboard + detail panel + add-test action.

Wiring:
    Add Test button   ->  file dialog (multi-select)  ->  IngestAndAnalyzeWorker
    Drag-and-drop     ->  same enqueue path
    Worker.finished   ->  reload dashboard, select the new run
    Dashboard click   ->  DetailPanel.show_test_run

UX touches:
  - Drag-and-drop CSVs onto the window (so the operator can drop the
    whole test-day's batch in one motion).
  - Multi-file selection in the dialog (Ctrl+O).
  - Sortable dashboard columns; a UserRole exposes the test_run_id so
    the selection survives a column sort.
  - Window title = "Hopper Data Studio v3 — <campaign> — <db filename>".
  - Status bar: transient action message on the left, persistent
    campaign + DB widget on the right (never overwritten).
  - "In-flight: N" badge while ingest workers are running.
  - Empty-state overlay on the table when there are zero tests.
  - Toolbar non-floatable, no context menu.
  - Window geometry + last-used directory persisted via QSettings.
  - "Open log folder" toolbar action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PySide6.QtCore import QSettings, QSortFilterProxyModel, QSize, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedLayout,
    QStatusBar,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hda.domain.state import TestState
from hda.ui.analytics_window import PYQTGRAPH_AVAILABLE as _ANALYTICS_OK
from hda.ui.dashboard import (
    TEST_RUN_ID_ROLE,
    DashboardData,
    DashboardModel,
)
from hda.ui.detail_panel import DetailPanel
from hda.ui.logging_setup import get_logger
from hda.ui.metadata_dialog import MetadataCompletionDialog
from hda.ui.workers import (
    CompleteMetadataAndAnalyzeWorker,
    IngestAndAnalyzeWorker,
    PipelineResult,
)
from hda.ui.workspace import Workspace

if _ANALYTICS_OK:
    from hda.ui.analytics_window import AnalyticsWindow


_log = get_logger("main_window")


class MainWindow(QMainWindow):
    def __init__(
        self, workspace: Workspace, default_campaign_id: str
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._campaign_id = default_campaign_id
        self._settings = QSettings("HopperPropulsion", "HDA")
        self._in_flight: int = 0

        self._update_window_title()
        self.resize(1280, 760)
        self.setAcceptDrops(True)

        self._dash_data = DashboardData(workspace.db)
        self._dash_data.set_campaign(default_campaign_id)
        self._dash_model = DashboardModel(self._dash_data)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._dash_model)
        self._proxy.setSortRole(Qt.DisplayRole)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(3, Qt.DescendingOrder)  # newest discovered first
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(26)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setHighlightSections(False)
        self._table.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._size_columns()

        self._empty_label = QLabel(
            "No tests yet.\n\n"
            "Drop CSVs here or use Add Test… (Ctrl+O) to begin."
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "color:#71717a; font-size:14px; padding:32px;"
        )
        self._dash_model.modelReset.connect(self._refresh_empty_state)
        self._dash_model.rowsInserted.connect(self._refresh_empty_state)
        self._dash_model.rowsRemoved.connect(self._refresh_empty_state)

        table_container = QWidget()
        table_stack = QStackedLayout(table_container)
        table_stack.setStackingMode(QStackedLayout.StackAll)
        table_stack.setContentsMargins(0, 0, 0, 0)
        table_stack.addWidget(self._table)
        table_stack.addWidget(self._empty_label)
        self._table_stack = table_stack

        self._detail = DetailPanel(workspace)
        self._detail.reanalyzed.connect(self._on_reanalyzed)
        self._detail.busy_changed.connect(self._on_detail_busy_changed)
        self._detail.complete_metadata_requested.connect(
            self._prompt_complete_metadata
        )

        splitter = QSplitter()
        splitter.addWidget(table_container)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 760])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._build_toolbar()
        self._build_status_bar()
        self._restore_geometry()
        self._refresh_empty_state()

    # ---------------------------------------------------------------- helpers

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Actions", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        add_action = QAction("Add Test…", self)
        add_action.setShortcut("Ctrl+O")
        add_action.setStatusTip("Add one or more CSV files to the current campaign")
        add_action.triggered.connect(self._on_add_test)
        toolbar.addAction(add_action)

        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.setStatusTip("Reload the dashboard from the database")
        refresh_action.triggered.connect(self._dash_model.reload)
        toolbar.addAction(refresh_action)

        toolbar.addSeparator()

        analytics_action = QAction("Hardware Analytics…", self)
        analytics_action.setShortcut("Ctrl+H")
        analytics_action.setStatusTip(
            "Cross-campaign view of one part / measurement"
        )
        analytics_action.triggered.connect(self._open_analytics)
        toolbar.addAction(analytics_action)

        toolbar.addSeparator()

        log_action = QAction("Open log folder", self)
        log_action.setStatusTip("Reveal the log directory in the file manager")
        log_action.triggered.connect(self._open_log_folder)
        toolbar.addAction(log_action)

        self._analytics_window: Optional[QWidget] = None

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        self._activity_label = QLabel("Ready.")
        self._activity_label.setStyleSheet("color:#52525b;")
        bar.addWidget(self._activity_label, 1)

        self._inflight_label = QLabel("")
        self._inflight_label.setStyleSheet(
            "color:#92400e; font-weight:600; padding-right:8px;"
        )
        bar.addPermanentWidget(self._inflight_label)

        self._workspace_label = QLabel(
            f"Campaign: <b>{self._campaign_id}</b>  •  "
            f"DB: {self._workspace.db.path}"
        )
        self._workspace_label.setStyleSheet("color:#27272a;")
        bar.addPermanentWidget(self._workspace_label)

    def _size_columns(self) -> None:
        # Test ID + State narrow; dates wider; Operator middle.
        widths = [110, 110, 130, 170, 170]
        for i, w in enumerate(widths):
            self._table.setColumnWidth(i, w)

    def _update_window_title(self) -> None:
        db_name = Path(str(self._workspace.db.path)).name
        self.setWindowTitle(
            f"Hopper Data Studio v3 — {self._campaign_id} — {db_name}"
        )

    def _restore_geometry(self) -> None:
        geo = self._settings.value("main/geometry")
        if geo is not None:
            try:
                self.restoreGeometry(geo)
            except Exception:
                pass

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._settings.setValue("main/geometry", self.saveGeometry())
        super().closeEvent(event)

    def _refresh_empty_state(self, *_args) -> None:
        empty = self._proxy.rowCount() == 0
        self._empty_label.setVisible(empty)

    # ---------------------------------------------------------------- ingest

    def _on_add_test(self) -> None:
        last_dir = self._settings.value(
            "ingest/last_dir", str(Path.home()), type=str
        )
        paths_str, _ = QFileDialog.getOpenFileNames(
            self,
            "Add test data file(s)",
            last_dir,
            "Test data (*.csv);;All files (*)",
        )
        if not paths_str:
            return
        first_dir = str(Path(paths_str[0]).parent)
        self._settings.setValue("ingest/last_dir", first_dir)
        self._enqueue_paths([Path(p) for p in paths_str])

    def _enqueue_paths(self, paths: Iterable[Path]) -> None:
        paths = list(paths)
        if not paths:
            return
        for path in paths:
            self._spawn_worker(path)
        self._activity_label.setText(
            f"Ingesting {len(paths)} file{'s' if len(paths) > 1 else ''}…"
        )

    def _spawn_worker(self, path: Path) -> None:
        worker = IngestAndAnalyzeWorker(
            workspace=self._workspace,
            file_path=path,
            campaign_id=self._campaign_id,
        )
        worker.signals.finished.connect(self._on_pipeline_finished)
        worker.signals.failed.connect(self._on_pipeline_failed)
        self._in_flight += 1
        self._update_inflight_badge()
        from PySide6.QtCore import QThreadPool

        QThreadPool.globalInstance().start(worker)

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        self._dash_model.reload()
        self._select_run(result.test_run_id)
        self._update_inflight_badge()
        suffix = " (duplicate)" if result.duplicate_of else ""
        self._activity_label.setText(
            f"{result.test_run_id[:8]} → {result.final_state.value}{suffix}"
        )
        if (
            result.final_state is TestState.AWAITING_METADATA
            and not result.duplicate_of
            and self._in_flight == 0
        ):
            # Only auto-prompt when the queue is drained so we don't pop
            # 5 dialogs after a 5-file drag-drop. The detail panel still
            # exposes a "Complete metadata" button for the rest.
            self._prompt_complete_metadata(result.test_run_id)

    def _on_pipeline_failed(self, message: str) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        self._update_inflight_badge()
        _log.error("pipeline failed: %s", message)
        self._activity_label.setText("Pipeline failed.")
        QMessageBox.critical(self, "Ingest / analysis failed", message)

    def _update_inflight_badge(self) -> None:
        self._inflight_label.setText(
            f"In flight: {self._in_flight}" if self._in_flight else ""
        )

    # ---------------------------------------------------------------- detail

    def _select_run(self, test_run_id: str) -> None:
        for r in range(self._proxy.rowCount()):
            idx = self._proxy.index(r, 0)
            if self._proxy.data(idx, TEST_RUN_ID_ROLE) == test_run_id:
                self._table.selectRow(r)
                self._table.scrollTo(idx)
                return

    def _on_selection_changed(self, *_args) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._detail.show_test_run(None)
            return
        rid = self._proxy.data(rows[0], TEST_RUN_ID_ROLE)
        self._detail.show_test_run(rid)

    def _on_reanalyzed(self, test_run_id: str) -> None:
        self._dash_model.reload()
        self._select_run(test_run_id)
        self._activity_label.setText(f"Reanalyzed {test_run_id[:8]}.")

    def prompt_complete_metadata(self, test_run_id: str) -> None:
        """Public entry point so the detail panel can ask us to open the
        dialog when the user clicks 'Complete metadata'."""
        self._prompt_complete_metadata(test_run_id)

    def _prompt_complete_metadata(self, test_run_id: str) -> None:
        ingest_svc = self._workspace.ingest_service
        if ingest_svc is None:
            return
        try:
            schema = ingest_svc.metadata_schema_for_run(test_run_id)
            existing = ingest_svc.existing_metadata_for_run(test_run_id)
        except Exception as e:
            QMessageBox.critical(
                self, "Cannot complete metadata", f"{type(e).__name__}: {e}"
            )
            return
        # Re-validate to know which required fields are still missing.
        missing = tuple(schema.validate(dict(existing)).missing_required)
        dialog = MetadataCompletionDialog(
            schema=schema,
            existing=existing,
            missing_required=missing,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        operator_md = dialog.values()
        worker = CompleteMetadataAndAnalyzeWorker(
            self._workspace, test_run_id, operator_md
        )
        worker.signals.finished.connect(self._on_complete_meta_finished)
        worker.signals.failed.connect(self._on_pipeline_failed)
        self._in_flight += 1
        self._update_inflight_badge()
        from PySide6.QtCore import QThreadPool

        QThreadPool.globalInstance().start(worker)
        self._activity_label.setText(
            f"Completing metadata for {test_run_id[:8]}…"
        )

    def _on_complete_meta_finished(self, result: PipelineResult) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        self._dash_model.reload()
        self._select_run(result.test_run_id)
        self._update_inflight_badge()
        self._activity_label.setText(
            f"{result.test_run_id[:8]} → {result.final_state.value}"
        )

    def _on_detail_busy_changed(self, busy: bool) -> None:
        if busy:
            self._activity_label.setText("Reanalyzing…")

    # ---------------------------------------------------------------- actions

    def _open_analytics(self) -> None:
        if not _ANALYTICS_OK:
            QMessageBox.information(
                self,
                "Analytics unavailable",
                "Hardware analytics requires pyqtgraph and Qt widgets.",
            )
            return
        if self._analytics_window is None:
            self._analytics_window = AnalyticsWindow(self._workspace)
        self._analytics_window.show()
        self._analytics_window.raise_()
        self._analytics_window.activateWindow()

    def _open_log_folder(self) -> None:
        log_dir = self._workspace.log_dir
        if log_dir is None or not Path(str(log_dir)).exists():
            QMessageBox.information(
                self,
                "Log folder unavailable",
                "No log directory is configured for this workspace.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

    # ---------------------------------------------------------------- DnD

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if not event.mimeData().hasUrls():
            return
        paths: list[Path] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            p = Path(url.toLocalFile())
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                paths.extend(sorted(p.glob("*.csv")))
        paths = [p for p in paths if p.suffix.lower() == ".csv"]
        if not paths:
            QMessageBox.information(
                self, "Nothing to ingest",
                "Drop one or more .csv files (or a folder containing some).",
            )
            return
        event.acceptProposedAction()
        self._enqueue_paths(paths)
