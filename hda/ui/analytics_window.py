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
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
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

            self._table = QTableWidget(0, 6)
            self._table.setHorizontalHeaderLabels(
                ["test_run_id", "campaign", "serial", "persisted_at", "value", "u"]
            )
            self._table.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch
            )
            self._table.setEditTriggers(QTableWidget.NoEditTriggers)
            self._table.setSelectionBehavior(QTableWidget.SelectRows)

            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(self._plot)
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
            self._plot.clear()
            if points:
                xs = [p.timestamp_unix for p in points]
                ys = [p.value for p in points]
                us = [p.uncertainty for p in points]
                self._plot.addItem(
                    pg.ErrorBarItem(
                        x=_npx(xs),
                        y=_npx(ys),
                        height=_npx([2 * u for u in us]),
                        beam=0.0,
                        pen=pg.mkPen("#71717a", width=1),
                    )
                )
                self._plot.plot(
                    xs, ys,
                    pen=None,
                    symbol="o",
                    symbolSize=8,
                    symbolBrush=pg.mkBrush("#18181b"),
                    symbolPen=pg.mkPen("#18181b"),
                )
            measurement = self._measurement_combo.currentText()
            self._plot.setLabel(
                "left",
                f"{measurement} ({self._unit})" if self._unit else measurement,
            )

            self._table.setRowCount(len(points))
            for r, p in enumerate(points):
                self._table.setItem(r, 0, _item(p.test_run_id[:8]))
                self._table.setItem(r, 1, _item(p.campaign_id))
                self._table.setItem(r, 2, _item(p.serial_number))
                self._table.setItem(r, 3, _item(p.timestamp_iso[:19]))
                self._table.setItem(r, 4, _item(f"{p.value:.6g}"))
                self._table.setItem(r, 5, _item(f"{p.uncertainty:.4g}"))

            self._set_summary(summarize(points))

        def _set_summary(self, summary: HistorySummary) -> None:
            self.statusBar().showMessage(format_summary(summary, self._unit))


def _item(text: str):
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    return item


def _npx(xs):
    """Tiny numpy import dance kept inside the function so this module
    still imports when pyqtgraph isn't present."""
    import numpy as np

    return np.asarray(xs, dtype=float)
