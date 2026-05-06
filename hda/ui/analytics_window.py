"""Hardware analytics: a measurement across every campaign of a part.

A separate window opened from the main toolbar (Ctrl+H). Filter by part /
serial / measurement name; the plot, table, and summary all repaint
together. Built directly on ``MeasurementsRepository.hardware_history``,
which is a single indexed join on the v3 single-DB schema — no
multi-database UNION the legacy app needed.

Skips at module load when pyqtgraph or QtWidgets cannot load (e.g. in
libEGL-less containers); the tests skip the same way.
"""

from __future__ import annotations

from typing import Optional

try:
    import pyqtgraph as pg
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QStackedLayout,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    PYQTGRAPH_AVAILABLE = True
except (ImportError, OSError):
    PYQTGRAPH_AVAILABLE = False


from hda.persistence.repositories import MeasurementsRepository
from hda.ui.analytics_data import (
    HistoryPoint,
    HistorySummary,
    format_summary,
    history_to_points,
    summarize,
)
from hda.ui.logging_setup import get_logger
from hda.ui.workspace import Workspace


_log = get_logger("analytics")


_ANY = "(any)"


if PYQTGRAPH_AVAILABLE:

    class AnalyticsWindow(QMainWindow):
        """Cross-campaign hardware analytics window.

        Layout (top to bottom):
            Filter bar (part / serial / measurement / refresh)
            Plot (value ± uncertainty over time)
            Table (test_run_id, campaign, serial, persisted_at, value ± u)
            Status bar (summary line)
        """

        def __init__(self, workspace: Workspace, parent: QWidget | None = None):
            super().__init__(parent)
            self._workspace = workspace
            self._repo = MeasurementsRepository(workspace.db)
            self._points: list[HistoryPoint] = []
            self._unit: str = ""

            self.setWindowTitle("Hardware Analytics — cross-campaign")
            self.resize(1200, 760)

            self._part_combo = QComboBox()
            self._serial_combo = QComboBox()
            self._measurement_combo = QComboBox()
            self._refresh_btn = QPushButton("Refresh")
            self._refresh_btn.setStyleSheet("font-weight: 600;")

            self._part_combo.currentTextChanged.connect(self._on_part_changed)
            self._serial_combo.currentTextChanged.connect(
                self._on_serial_changed
            )
            self._refresh_btn.clicked.connect(self._reload_data)
            self._measurement_combo.currentTextChanged.connect(self._reload_data)

            filter_bar = self._build_filter_bar()

            self._plot = pg.PlotWidget(
                axisItems={"bottom": pg.DateAxisItem(orientation="bottom")}
            )
            self._plot.setLabel("bottom", "persisted")
            self._plot.showGrid(x=True, y=True, alpha=0.2)

            self._hover_label = pg.TextItem(
                anchor=(0, 1),
                fill=pg.mkBrush(255, 255, 255, 220),
                color=(24, 24, 27),
                border=pg.mkPen("#a1a1aa"),
            )
            self._hover_label.setZValue(50)
            self._plot.addItem(self._hover_label)
            self._hover_label.hide()

            self._scatter = pg.ScatterPlotItem(
                size=10,
                brush=pg.mkBrush("#18181b"),
                pen=pg.mkPen("#18181b"),
                hoverable=True,
                hoverBrush=pg.mkBrush("#0c4a6e"),
                hoverSize=14,
            )
            self._scatter.sigHovered.connect(self._on_point_hovered)
            self._plot.addItem(self._scatter)

            self._error_bars = pg.ErrorBarItem(
                pen=pg.mkPen("#71717a", width=1), beam=0.0
            )
            self._plot.addItem(self._error_bars)

            self._empty_overlay = QLabel(
                "Pick a part, serial, and measurement to see history."
            )
            self._empty_overlay.setAlignment(Qt.AlignCenter)
            self._empty_overlay.setStyleSheet(
                "color:#71717a; font-size:13px; padding:24px;"
            )

            plot_container = QWidget()
            plot_stack = QStackedLayout(plot_container)
            plot_stack.setStackingMode(QStackedLayout.StackAll)
            plot_stack.setContentsMargins(0, 0, 0, 0)
            plot_stack.addWidget(self._plot)
            plot_stack.addWidget(self._empty_overlay)
            self._plot_stack = plot_stack

            self._table = QTableWidget(0, 6)
            self._table.setHorizontalHeaderLabels(
                ["test_run_id", "campaign", "serial", "persisted_at", "value", "u"]
            )
            t_header = self._table.horizontalHeader()
            t_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            t_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            t_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            t_header.setSectionResizeMode(3, QHeaderView.Stretch)
            t_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            t_header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            self._table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._table.setSelectionBehavior(QTableWidget.SelectRows)
            self._table.setAlternatingRowColors(True)
            self._table.verticalHeader().setVisible(False)

            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(plot_container)
            splitter.addWidget(self._table)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)

            central = QWidget()
            v = QVBoxLayout(central)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(filter_bar)
            v.addWidget(splitter, 1)
            self.setCentralWidget(central)

            self.setStatusBar(QStatusBar())
            self._set_summary(HistorySummary.empty())

            self._populate_parts()

        def _build_filter_bar(self) -> QWidget:
            bar = QWidget()
            h = QHBoxLayout(bar)
            h.setContentsMargins(8, 6, 8, 6)
            h.addWidget(QLabel("Part:"))
            h.addWidget(self._part_combo, stretch=1)
            h.addWidget(QLabel("Serial:"))
            h.addWidget(self._serial_combo, stretch=1)
            h.addWidget(QLabel("Measurement:"))
            h.addWidget(self._measurement_combo, stretch=1)
            h.addWidget(self._refresh_btn)
            return bar

        # --- population ----------------------------------------------------

        def _populate_parts(self) -> None:
            self._part_combo.blockSignals(True)
            self._part_combo.clear()
            parts = self._repo.list_parts_with_measurements()
            if not parts:
                self._part_combo.addItem("(no measurements yet)")
                self._part_combo.setEnabled(False)
                self._part_combo.blockSignals(False)
                return
            self._part_combo.addItems(parts)
            self._part_combo.setEnabled(True)
            self._part_combo.blockSignals(False)
            self._on_part_changed(self._part_combo.currentText())

        def _on_part_changed(self, part: str) -> None:
            if not part or part == "(no measurements yet)":
                return
            self._serial_combo.blockSignals(True)
            self._serial_combo.clear()
            self._serial_combo.addItem(_ANY)
            for s in self._repo.list_serials_for_part(part):
                self._serial_combo.addItem(s)
            self._serial_combo.blockSignals(False)
            self._on_serial_changed(self._serial_combo.currentText())

        def _on_serial_changed(self, _serial: str) -> None:
            part = self._part_combo.currentText()
            serial = self._current_serial()
            self._measurement_combo.blockSignals(True)
            self._measurement_combo.clear()
            for m in self._repo.list_measurement_names_for_part(part, serial):
                self._measurement_combo.addItem(m)
            self._measurement_combo.blockSignals(False)
            self._reload_data()

        def _current_serial(self) -> Optional[str]:
            s = self._serial_combo.currentText()
            return None if s in ("", _ANY) else s

        # --- data + plot ---------------------------------------------------

        def _reload_data(self) -> None:
            part = self._part_combo.currentText()
            measurement = self._measurement_combo.currentText()
            if not part or not measurement or part == "(no measurements yet)":
                self._points = []
                self._unit = ""
                self._render([])
                return
            try:
                df = self._repo.hardware_history(
                    part_number=part,
                    measurement_name=measurement,
                    serial_number=self._current_serial(),
                )
            except Exception as e:
                _log.exception("analytics query failed")
                QMessageBox.critical(self, "Query failed", str(e))
                return
            self._points = history_to_points(df)
            self._unit = (
                df["unit"].iloc[0] if not df.empty and "unit" in df.columns else ""
            )
            self._render(self._points)

        def _render(self, points: list[HistoryPoint]) -> None:
            self._hover_label.hide()
            if points:
                xs = _npx([p.timestamp_unix for p in points])
                ys = _npx([p.value for p in points])
                us = _npx([p.uncertainty for p in points])
                self._error_bars.setData(x=xs, y=ys, height=2 * us)
                self._scatter.setData(
                    x=xs,
                    y=ys,
                    data=points,  # so sigHovered can recover the HistoryPoint
                )
                self._empty_overlay.hide()
            else:
                self._error_bars.setData(x=[], y=[], height=[])
                self._scatter.setData(x=[], y=[])
                self._empty_overlay.setText(
                    "No measurements match the current filter."
                    if self._part_combo.isEnabled()
                    else "No measurements in the database yet."
                )
                self._empty_overlay.show()

            measurement = self._measurement_combo.currentText()
            self._plot.setLabel(
                "left",
                f"{measurement} ({self._unit})" if self._unit else measurement,
            )
            self._plot.getViewBox().enableAutoRange(enable=True)

            self._table.setRowCount(len(points))
            for r, p in enumerate(points):
                tid = _name_item(p.test_run_id[:8])
                tid.setToolTip(p.test_run_id)
                self._table.setItem(r, 0, tid)
                self._table.setItem(r, 1, _name_item(p.campaign_id))
                self._table.setItem(r, 2, _name_item(p.serial_number))
                self._table.setItem(r, 3, _name_item(p.timestamp_iso[:19]))
                self._table.setItem(r, 4, _numeric_item(f"{p.value:.6g}"))
                self._table.setItem(r, 5, _numeric_item(f"{p.uncertainty:.4g}"))

            self._set_summary(summarize(points))

        def _on_point_hovered(self, _scatter, hovered_points, _ev=None):
            if not hovered_points:
                self._hover_label.hide()
                return
            spot = hovered_points[0]
            point: HistoryPoint = spot.data()
            text = (
                f"{point.test_run_id[:8]}  ·  {point.campaign_id}\n"
                f"{point.serial_number}  ·  {point.timestamp_iso[:19]}\n"
                f"value: {point.value:.6g}  ±  {point.uncertainty:.4g}"
            )
            self._hover_label.setText(text)
            self._hover_label.setPos(spot.pos())
            self._hover_label.show()

        def _set_summary(self, summary: HistorySummary) -> None:
            self.statusBar().showMessage(format_summary(summary, self._unit))


def _name_item(text: str):
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    return item


def _numeric_item(text: str):
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
    f = QFont("Menlo")
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(10)
    item.setFont(f)
    return item


def _npx(xs):
    """Tiny numpy import dance kept inside the function so this module
    still imports when pyqtgraph isn't present."""
    import numpy as np

    return np.asarray(xs, dtype=float)
