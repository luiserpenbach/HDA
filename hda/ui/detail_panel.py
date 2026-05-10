"""Test-detail panel.

For a selected test_run_id renders three sections:
  1. Header line — id + state, tinted with the same color used by the
     dashboard's state column so the operator's eye lands on the same
     hue across views.
  2. Interactive steady-state preview (when preprocessed data is in the
     cache) with live window stats and an Apply-window button.
  3. Measurements + QC findings tables. Group titles include counts.

The steady-state preview is wired through ``window_committed`` to a
ReanalyzeWorker; on success the panel reloads measurements + QC and
emits ``reanalyzed(test_run_id)`` so the dashboard refreshes. While a
reanalysis is in flight the Apply button is disabled — preventing
double-clicks — and the parent gets a ``busy_changed(True/False)``
signal so the status bar reflects "Reanalyzing…".
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, QThreadPool, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hda.domain.state import TestState
from hda.domain.steady_state import detect_cv, detect_simple
from hda.domain.types import SteadyWindow
from hda.persistence import Database
from hda.persistence.repositories import (
    MeasurementsRepository,
    QCFindingsRepository,
    TestRunRepository,
)
from hda.ui.dashboard import state_colors
from hda.ui.logging_setup import get_logger
from hda.ui.steady_state_preview import PYQTGRAPH_AVAILABLE
from hda.ui.workers import PipelineResult, ReanalyzeWorker
from hda.ui.workspace import Workspace

if PYQTGRAPH_AVAILABLE:
    from hda.ui.steady_state_preview import SteadyStatePreview


_log = get_logger("detail_panel")


_QC_STATUS_BG: dict[str, QColor] = {
    "pass": QColor("#dcfce7"),
    "warn": QColor("#fef3c7"),
    "fail": QColor("#fee2e2"),
}
_QC_STATUS_FG: dict[str, QColor] = {
    "pass": QColor("#14532d"),
    "warn": QColor("#92400e"),
    "fail": QColor("#7f1d1d"),
}


class DetailPanel(QWidget):
    """Test-detail view. Owns the steady-state preview and the
    measurements / QC tables for the selected run."""

    reanalyzed = Signal(str)
    busy_changed = Signal(bool)
    complete_metadata_requested = Signal(str)  # test_run_id

    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._workspace = workspace
        self._db: Database = workspace.db
        self._test_run_id: Optional[str] = None
        self._busy: bool = False
        self._pool = QThreadPool.globalInstance()
        self._settings = QSettings("HopperPropulsion", "HDA")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self._header = QLabel("No test selected")
        self._header.setStyleSheet(
            "font-weight:600; font-size:14px; padding:6px 10px;"
            " border-radius:4px; background:#f4f4f5; color:#27272a;"
        )
        outer.addWidget(self._header)

        # AWAITING_METADATA banner — visible only in that state.
        self._banner = QWidget()
        self._banner.setVisible(False)
        banner_layout = QHBoxLayout(self._banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        self._banner_label = QLabel()
        self._banner_label.setWordWrap(True)
        self._banner_btn = QPushButton("Complete metadata…")
        self._banner_btn.setStyleSheet("font-weight:600;")
        self._banner_btn.clicked.connect(self._emit_complete_metadata)
        banner_layout.addWidget(self._banner_label, stretch=1)
        banner_layout.addWidget(self._banner_btn, stretch=0)
        self._banner.setStyleSheet(
            "background:#fef3c7; color:#92400e; border:1px solid #fcd34d;"
            " border-radius:4px;"
        )
        outer.addWidget(self._banner)

        # Three sections live in a draggable vertical splitter so the
        # operator can grow the plot when they want to inspect a long
        # transient or grow the tables when they want to scan many
        # measurements. Each section has a minimum height to keep it
        # usable; the splitter remembers its geometry across launches.
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setHandleWidth(6)
        self._splitter.setChildrenCollapsible(False)

        self._preview: Optional[SteadyStatePreview] = None
        if PYQTGRAPH_AVAILABLE:
            self._preview_box = QGroupBox("Steady-state window")
            preview_layout = QVBoxLayout(self._preview_box)
            preview_layout.setContentsMargins(8, 16, 8, 8)
            self._preview = SteadyStatePreview()
            self._preview.window_committed.connect(self._on_window_committed)
            preview_layout.addWidget(self._preview)
            self._preview_box.setMinimumHeight(280)
            self._preview_box.setVisible(False)
            self._splitter.addWidget(self._preview_box)
        else:
            self._preview_box = None  # type: ignore[assignment]

        self._meas_box = QGroupBox("Measurements")
        meas_layout = QVBoxLayout(self._meas_box)
        meas_layout.setContentsMargins(8, 16, 8, 8)
        self._meas_table = QTableWidget(0, 4)
        self._meas_table.setHorizontalHeaderLabels(
            ["Name", "Value", "Uncertainty", "Unit"]
        )
        meas_header = self._meas_table.horizontalHeader()
        meas_header.setSectionResizeMode(0, QHeaderView.Stretch)
        meas_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        meas_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        meas_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._meas_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._meas_table.setAlternatingRowColors(True)
        self._meas_table.verticalHeader().setVisible(False)
        meas_layout.addWidget(self._meas_table)
        self._meas_box.setMinimumHeight(140)
        self._splitter.addWidget(self._meas_box)

        self._qc_box = QGroupBox("QC findings")
        qc_layout = QVBoxLayout(self._qc_box)
        qc_layout.setContentsMargins(8, 16, 8, 8)
        self._qc_table = QTableWidget(0, 4)
        self._qc_table.setHorizontalHeaderLabels(
            ["Check", "Status", "Blocking", "Message"]
        )
        qc_header = self._qc_table.horizontalHeader()
        qc_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        qc_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        qc_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        qc_header.setSectionResizeMode(3, QHeaderView.Stretch)
        self._qc_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._qc_table.setAlternatingRowColors(True)
        self._qc_table.verticalHeader().setVisible(False)
        qc_layout.addWidget(self._qc_table)
        self._qc_box.setMinimumHeight(120)
        self._splitter.addWidget(self._qc_box)

        # Stretch factors set the *preferred* growth per section when the
        # window is resized. The plot grows fastest, then measurements,
        # then QC.
        if PYQTGRAPH_AVAILABLE:
            self._splitter.setStretchFactor(0, 5)
            self._splitter.setStretchFactor(1, 3)
            self._splitter.setStretchFactor(2, 2)
            self._splitter.setSizes([460, 240, 160])
        else:
            self._splitter.setStretchFactor(0, 3)
            self._splitter.setStretchFactor(1, 2)
            self._splitter.setSizes([320, 200])
        self._restore_splitter_sizes()
        self._splitter.splitterMoved.connect(self._save_splitter_sizes)

        outer.addWidget(self._splitter, 1)

    # ----------------------------------------------------- splitter state

    def _restore_splitter_sizes(self) -> None:
        sizes = self._settings.value("detail/splitter_sizes")
        if sizes is None:
            return
        try:
            ints = [int(s) for s in sizes]
        except (TypeError, ValueError):
            return
        if len(ints) == self._splitter.count():
            self._splitter.setSizes(ints)

    def _save_splitter_sizes(self, *_args) -> None:
        self._settings.setValue(
            "detail/splitter_sizes", list(self._splitter.sizes())
        )

    # ---------------------------------------------------------- public API

    def show_test_run(self, test_run_id: str | None) -> None:
        self._test_run_id = test_run_id
        if test_run_id is None:
            self._set_header("No test selected", "discovered")
            self._meas_table.setRowCount(0)
            self._qc_table.setRowCount(0)
            self._meas_box.setTitle("Measurements")
            self._qc_box.setTitle("QC findings")
            self._banner.setVisible(False)
            if self._preview is not None:
                self._preview.clear()
                self._preview_box.setVisible(False)
            return

        run_state = TestRunRepository(self._db).get_state(test_run_id)
        state_str = run_state.value if run_state is not None else "?"
        self._set_header(
            f"Test {test_run_id[:8]} — state: {state_str}", state_str
        )

        self._refresh_banner(test_run_id, run_state)
        self._populate_preview(test_run_id, run_state)
        self._populate_measurements(test_run_id)
        self._populate_qc(test_run_id)

    # ------------------------------------------------------------- private

    def _set_header(self, text: str, state: str) -> None:
        fg, bg = state_colors(state)
        self._header.setText(text)
        self._header.setStyleSheet(
            f"font-weight:600; font-size:14px; padding:6px 10px;"
            f" border-radius:4px; background:{bg.name()}; color:{fg.name()};"
        )

    def _populate_preview(
        self, test_run_id: str, run_state: Optional[TestState]
    ) -> None:
        if self._preview is None:
            return
        cached = self._workspace.preprocessed_cache.get(test_run_id)
        if cached is None:
            self._preview.clear()
            self._preview_box.setVisible(False)
            return

        steady_row = self._lookup_steady_window(test_run_id)
        if steady_row is None:
            steady_row = _auto_detect_window(cached.data.df)
        if steady_row is None:
            self._preview.clear()
            self._preview_box.setVisible(False)
            return

        title = (
            "Steady-state window (preview — apply to analyze)"
            if run_state is TestState.AWAITING_METADATA
            else "Steady-state window"
        )
        self._preview_box.setTitle(title)
        self._preview.show_data(
            df=cached.data.df,
            initial_window=steady_row,
            timestamp_column="timestamp",
        )
        # In AWAITING_METADATA the operator can browse + drag, but
        # Apply is meaningless until metadata is filled in.
        if run_state is TestState.AWAITING_METADATA:
            self._preview.set_apply_enabled(False)
            self._preview.set_apply_tooltip(
                "Complete metadata first — Apply will run analysis once required fields are set."
            )
        else:
            self._preview.set_apply_tooltip(None)
            self._preview.set_apply_enabled(True)
        self._preview.set_busy(self._busy)
        self._preview_box.setVisible(True)

    def _refresh_banner(
        self, test_run_id: str, run_state: Optional[TestState]
    ) -> None:
        if run_state is not TestState.AWAITING_METADATA:
            self._banner.setVisible(False)
            return
        ingest_svc = self._workspace.ingest_service
        missing: tuple[str, ...] = ()
        if ingest_svc is not None:
            try:
                schema = ingest_svc.metadata_schema_for_run(test_run_id)
                existing = ingest_svc.existing_metadata_for_run(test_run_id)
                missing = tuple(schema.validate(dict(existing)).missing_required)
            except Exception:
                missing = ()
        if missing:
            self._banner_label.setText(
                "<b>Awaiting metadata.</b> Required fields still missing: "
                + ", ".join(missing)
                + ". Click <b>Complete metadata</b> to fill them in and analyze."
            )
        else:
            self._banner_label.setText(
                "<b>Awaiting metadata.</b> Click <b>Complete metadata</b> "
                "to review and submit."
            )
        self._banner.setVisible(True)

    def _emit_complete_metadata(self) -> None:
        if self._test_run_id is None:
            return
        self.complete_metadata_requested.emit(self._test_run_id)

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
        self._meas_box.setTitle(f"Measurements ({len(measurements)})")
        self._meas_table.setRowCount(len(measurements))
        for r, m in enumerate(measurements):
            self._meas_table.setItem(r, 0, _name_item(m.name))
            self._meas_table.setItem(r, 1, _numeric_item(f"{m.value:.6g}"))
            self._meas_table.setItem(r, 2, _numeric_item(f"{m.uncertainty:.4g}"))
            self._meas_table.setItem(r, 3, _name_item(m.unit))

    def _populate_qc(self, test_run_id: str) -> None:
        findings = QCFindingsRepository(self._db).get_for_run(test_run_id)
        self._qc_box.setTitle(f"QC findings ({len(findings)})")
        self._qc_table.setRowCount(len(findings))
        for r, f in enumerate(findings):
            check = _name_item(f.check_name)
            status = _name_item(f.status.value)
            bg = _QC_STATUS_BG.get(f.status.value)
            fg = _QC_STATUS_FG.get(f.status.value)
            if bg is not None:
                status.setBackground(QBrush(bg))
            if fg is not None:
                status.setForeground(QBrush(fg))
            blocking = _name_item("yes" if f.blocking else "")
            blocking.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self._qc_table.setItem(r, 0, check)
            self._qc_table.setItem(r, 1, status)
            self._qc_table.setItem(r, 2, blocking)
            self._qc_table.setItem(r, 3, _name_item(f.message))

    # ----------------------------------------------------------- reanalyze

    def _on_window_committed(self, window: SteadyWindow) -> None:
        if self._test_run_id is None or self._busy:
            return
        _log.info(
            "operator commit window: id=%s [%.3f, %.3f]",
            self._test_run_id, window.start_s, window.end_s,
        )
        self._set_busy(True)
        worker = ReanalyzeWorker(self._workspace, self._test_run_id, window)
        worker.signals.finished.connect(self._on_reanalyze_finished)
        worker.signals.failed.connect(self._on_reanalyze_failed)
        self._pool.start(worker)

    def _on_reanalyze_finished(self, result: PipelineResult) -> None:
        self._set_busy(False)
        self.show_test_run(result.test_run_id)
        self.reanalyzed.emit(result.test_run_id)

    def _on_reanalyze_failed(self, message: str) -> None:
        self._set_busy(False)
        _log.error("reanalyze failed: %s", message)
        QMessageBox.critical(self, "Reanalysis failed", message)

    def _set_busy(self, busy: bool) -> None:
        if busy == self._busy:
            return
        self._busy = busy
        if self._preview is not None:
            self._preview.set_busy(busy)
        self.busy_changed.emit(busy)


def _mono_font() -> QFont:
    f = QFont("Menlo")
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(10)
    return f


def _name_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    return item


def _numeric_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
    item.setFont(_mono_font())
    return item


def _auto_detect_window(df) -> Optional[SteadyWindow]:
    """Pick a sensible initial window for the AWAITING_METADATA preview:
    try CV-based detection on the first non-timestamp channel, fall back
    to the centered 50% if CV finds nothing."""
    import numpy as np

    cols = [c for c in df.columns if c != "timestamp"]
    if not cols:
        return None
    t = df["timestamp"].to_numpy(dtype=float)
    if t.size < 4:
        return None
    sig = df[cols[0]].to_numpy(dtype=float)
    try:
        cv = detect_cv(sig, t)
        if cv is not None:
            return cv
        return detect_simple(t, fraction=0.5)
    except Exception:
        return None
