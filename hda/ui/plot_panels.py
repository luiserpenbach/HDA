"""Dockable pyqtgraph plot panels for manual sensor analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.exporters import SVGExporter

    _PG_OK = True
except Exception:
    _PG_OK = False
    SVGExporter = None  # type: ignore

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hda.ui.style import PLOT_BG, PLOT_FG, SZ_SM, TEXT_PRIMARY

TRACE_COLORS = [
    "#3b82f6", "#ef4444", "#16a34a", "#d97706",
    "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
    "#f97316", "#6366f1",
]

TRIM_LINE_COLOR = "#ef4444"
STEADY_REGION_BRUSH = (59, 130, 246, 25)
STEADY_REGION_PEN = "#3b82f6"


@dataclass
class PlotRenderContext:
    """Data required to draw time-series on every plot panel."""

    df: Any = None
    time_seconds: np.ndarray = field(default_factory=lambda: np.array([]))
    available_channels: List[str] = field(default_factory=list)
    trim_range: Optional[Tuple[float, float]] = None
    show_trim_lines: bool = False
    trim_line_positions: Optional[Tuple[float, float]] = None
    steady_range: Optional[Tuple[float, float]] = None
    highlight_trim: bool = False


if _PG_OK:

    class SecondsAxis(pg.AxisItem):
        def tickStrings(self, values, scale, spacing):  # noqa: N802
            return [f"{v:.3f}" for v in values]


def _plot_channel_series(
    plot: "pg.PlotWidget",
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    name: str,
    trim_range: Optional[Tuple[float, float]],
    highlight_trim: bool,
) -> None:
    """Plot a channel, dimming data outside the trim keep-window when requested."""
    if not _PG_OK:
        return
    pen_full = pg.mkPen(color=color, width=1.8, alpha=255)
    pen_dim = pg.mkPen(color=color, width=1.2, alpha=70)

    if not highlight_trim or trim_range is None or len(time_s) == 0:
        plot.plot(time_s, values, pen=pen_full, name=name)
        return

    lo, hi = trim_range
    if lo > hi:
        lo, hi = hi, lo
    inside = (time_s >= lo) & (time_s <= hi)
    if inside.any():
        plot.plot(time_s[inside], values[inside], pen=pen_full, name=name)
    outside = ~inside
    if outside.any():
        plot.plot(time_s[outside], values[outside], pen=pen_dim)


class PlotPanelWidget(QWidget):
    """One sensor plot with its own channel picker and SVG export."""

    channels_changed = Signal()
    trim_lines_changed = Signal(float, float)
    steady_region_changed = Signal(float, float)

    def __init__(
        self,
        title: str,
        *,
        available_channels: Optional[List[str]] = None,
        default_channels: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self._available: List[str] = list(available_channels or [])
        self._selected: Set[str] = set(default_channels or self._available[:3])
        self._overlay_items: List[Any] = []
        self._trim_start_line: Any = None
        self._trim_end_line: Any = None
        self._steady_region: Any = None
        self._overlay_sync = False

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {SZ_SM}; font-weight: 600; background: transparent;"
        )
        header.addWidget(title_lbl, 1)

        self._sensor_btn = QToolButton()
        self._sensor_btn.setText("Sensors")
        self._sensor_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._sensor_menu = QMenu(self)
        self._sensor_btn.setMenu(self._sensor_menu)
        header.addWidget(self._sensor_btn)

        export_btn = QPushButton("SVG")
        export_btn.setToolTip("Export this panel to SVG")
        export_btn.setProperty("secondary", True)
        export_btn.clicked.connect(self._export_svg)
        header.addWidget(export_btn)
        root.addLayout(header)

        if _PG_OK:
            self.plot = pg.PlotWidget(axisItems={"bottom": SecondsAxis(orientation="bottom")})
            self.plot.setBackground(PLOT_BG)
            self.plot.showGrid(x=True, y=True, alpha=0.35)
            self.plot.getAxis("bottom").setPen(PLOT_FG)
            self.plot.getAxis("left").setPen(PLOT_FG)
            self.plot.getAxis("bottom").setTextPen(PLOT_FG)
            self.plot.getAxis("left").setTextPen(PLOT_FG)
            self.plot.setLabel("bottom", "Time (s)")
            self.plot.addLegend(offset=(8, 8))
            self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            root.addWidget(self.plot, 1)
        else:
            self.plot = None
            root.addWidget(QLabel("pyqtgraph not installed"), 1)

        self._rebuild_sensor_menu()

    def set_available_channels(self, channels: List[str]) -> None:
        self._available = list(channels)
        self._selected = {c for c in self._selected if c in self._available}
        if not self._selected and self._available:
            self._selected = set(self._available[: min(3, len(self._available))])
        self._rebuild_sensor_menu()

    def selected_channels(self) -> List[str]:
        return [c for c in self._available if c in self._selected]

    def _rebuild_sensor_menu(self) -> None:
        self._sensor_menu.clear()
        if not self._available:
            empty = self._sensor_menu.addAction("(no channels)")
            empty.setEnabled(False)
            return
        for ch in self._available:
            act = self._sensor_menu.addAction(ch)
            act.setCheckable(True)
            act.setChecked(ch in self._selected)
            act.toggled.connect(lambda checked, c=ch: self._on_sensor_toggled(c, checked))

    def _on_sensor_toggled(self, channel: str, checked: bool) -> None:
        if checked:
            self._selected.add(channel)
        else:
            self._selected.discard(channel)
        self.channels_changed.emit()

    def clear_overlays(self) -> None:
        if not _PG_OK or self.plot is None:
            return
        for item in self._overlay_items:
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self._overlay_items.clear()
        self._trim_start_line = None
        self._trim_end_line = None
        self._steady_region = None

    def apply_overlays(
        self,
        *,
        show_trim: bool,
        trim_positions: Optional[Tuple[float, float]],
        steady_range: Optional[Tuple[float, float]],
    ) -> None:
        self.clear_overlays()
        if not _PG_OK or self.plot is None:
            return

        if show_trim and trim_positions:
            lo, hi = trim_positions
            if lo > hi:
                lo, hi = hi, lo
            pen = pg.mkPen(TRIM_LINE_COLOR, width=2)
            self._trim_start_line = pg.InfiniteLine(pos=lo, angle=90, movable=True, pen=pen)
            self._trim_end_line = pg.InfiniteLine(pos=hi, angle=90, movable=True, pen=pen)
            self._trim_start_line.sigPositionChanged.connect(self._on_trim_line_moved)
            self._trim_end_line.sigPositionChanged.connect(self._on_trim_line_moved)
            self.plot.addItem(self._trim_start_line)
            self.plot.addItem(self._trim_end_line)
            self._overlay_items.extend([self._trim_start_line, self._trim_end_line])

        if steady_range:
            lo, hi = steady_range
            if lo > hi:
                lo, hi = hi, lo
            self._steady_region = pg.LinearRegionItem(
                [lo, hi],
                movable=True,
                brush=pg.mkBrush(*STEADY_REGION_BRUSH),
                pen=pg.mkPen(color=STEADY_REGION_PEN, width=1),
            )
            self._steady_region.sigRegionChanged.connect(self._on_steady_region_moved)
            self.plot.addItem(self._steady_region)
            self._overlay_items.append(self._steady_region)

    def set_trim_positions(self, start_s: float, end_s: float) -> None:
        if not _PG_OK or self._trim_start_line is None or self._trim_end_line is None:
            return
        lo, hi = (start_s, end_s) if start_s <= end_s else (end_s, start_s)
        self._overlay_sync = True
        try:
            self._trim_start_line.setValue(lo)
            self._trim_end_line.setValue(hi)
        finally:
            self._overlay_sync = False

    def set_steady_region(self, start_s: float, end_s: float) -> None:
        if not _PG_OK or self._steady_region is None:
            return
        lo, hi = (start_s, end_s) if start_s <= end_s else (end_s, start_s)
        self._overlay_sync = True
        try:
            self._steady_region.setRegion([lo, hi])
        finally:
            self._overlay_sync = False

    def _on_trim_line_moved(self, *_args) -> None:
        if self._overlay_sync or self._trim_start_line is None or self._trim_end_line is None:
            return
        lo = float(self._trim_start_line.value())
        hi = float(self._trim_end_line.value())
        self.trim_lines_changed.emit(min(lo, hi), max(lo, hi))

    def _on_steady_region_moved(self) -> None:
        if self._overlay_sync or self._steady_region is None:
            return
        lo, hi = self._steady_region.getRegion()
        self.steady_region_changed.emit(float(lo), float(hi))

    def render(self, ctx: PlotRenderContext) -> None:
        if not _PG_OK or self.plot is None or ctx.df is None:
            return

        self.plot.clear()
        self.plot.addLegend(offset=(8, 8))

        t = ctx.time_seconds
        if len(t) == 0:
            return

        trim_range = ctx.trim_range if ctx.highlight_trim else None
        color_idx = 0
        for col in self.selected_channels():
            if col not in ctx.df.columns:
                continue
            color = TRACE_COLORS[color_idx % len(TRACE_COLORS)]
            color_idx += 1
            _plot_channel_series(
                self.plot,
                t,
                ctx.df[col].values,
                color=color,
                name=col,
                trim_range=trim_range,
                highlight_trim=ctx.highlight_trim,
            )

    def _export_svg(self) -> None:
        if not _PG_OK or self.plot is None or SVGExporter is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export plot as SVG",
            f"{self.title.replace(' ', '_').lower()}.svg",
            "SVG files (*.svg)",
        )
        if not path:
            return
        exporter = SVGExporter(self.plot.plotItem)
        exporter.export(path)


class ClosablePlotDock(QDockWidget):
    """Dock widget that notifies when the user closes it."""

    dock_closed = Signal(object)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.dock_closed.emit(self)
        super().closeEvent(event)


class PlotDockWorkspace(QMainWindow):
    """Dockable multi-panel plot workspace."""

    trim_lines_changed = Signal(float, float)
    steady_region_changed = Signal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )

        self._panels: List[PlotPanelWidget] = []
        self._docks: List[QDockWidget] = []
        self._panel_counter = 0
        self._ctx = PlotRenderContext()
        self._overlay_updating = False

        placeholder = QFrame()
        placeholder.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(placeholder)

    def add_panel(
        self,
        *,
        available_channels: Optional[List[str]] = None,
        default_channels: Optional[List[str]] = None,
    ) -> PlotPanelWidget:
        self._panel_counter += 1
        title = f"Plot {self._panel_counter}"
        panel = PlotPanelWidget(
            title,
            available_channels=available_channels or self._ctx.available_channels,
            default_channels=default_channels,
            parent=self,
        )
        panel.channels_changed.connect(self.render_traces)
        panel.trim_lines_changed.connect(self._on_panel_trim_changed)
        panel.steady_region_changed.connect(self._on_panel_steady_changed)

        dock = ClosablePlotDock(title, self)
        dock.setObjectName(f"plot_dock_{self._panel_counter}")
        dock.setWidget(panel)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        dock.dock_closed.connect(self._on_dock_closed)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._panels.append(panel)
        self._docks.append(dock)
        panel.render(self._ctx)
        panel.apply_overlays(
            show_trim=bool(self._ctx.show_trim_lines),
            trim_positions=self._ctx.trim_line_positions,
            steady_range=self._ctx.steady_range,
        )
        return panel

    def _on_panel_trim_changed(self, lo: float, hi: float) -> None:
        if self._overlay_updating:
            return
        self._overlay_updating = True
        try:
            for panel in self._panels:
                panel.set_trim_positions(lo, hi)
        finally:
            self._overlay_updating = False
        self.trim_lines_changed.emit(lo, hi)

    def _on_panel_steady_changed(self, lo: float, hi: float) -> None:
        if self._overlay_updating:
            return
        self._overlay_updating = True
        try:
            for panel in self._panels:
                panel.set_steady_region(lo, hi)
        finally:
            self._overlay_updating = False
        self.steady_region_changed.emit(lo, hi)

    def _on_dock_closed(self, dock: QDockWidget) -> None:
        panel = dock.widget()
        if isinstance(panel, PlotPanelWidget):
            try:
                self._panels.remove(panel)
            except ValueError:
                pass
        try:
            self._docks.remove(dock)
        except ValueError:
            pass
        if not self._panels:
            self.add_panel()

    def update_context(self, **kwargs) -> None:
        for key, val in kwargs.items():
            if hasattr(self._ctx, key):
                setattr(self._ctx, key, val)
        channels = kwargs.get("available_channels")
        if channels is not None:
            for panel in self._panels:
                panel.set_available_channels(channels)

    def render_traces(self) -> None:
        """Redraw data traces only — leave overlays untouched."""
        for panel in self._panels:
            panel.render(self._ctx)

    def refresh(self) -> None:
        trim_pos = self._ctx.trim_line_positions if self._ctx.show_trim_lines else None
        for panel in self._panels:
            panel.render(self._ctx)
            panel.apply_overlays(
                show_trim=bool(self._ctx.show_trim_lines),
                trim_positions=trim_pos,
                steady_range=self._ctx.steady_range,
            )

    def sync_overlay_positions(self) -> None:
        """Move existing overlays to match context without clearing plots."""
        trim_pos = self._ctx.trim_line_positions if self._ctx.show_trim_lines else None
        self._overlay_updating = True
        try:
            for panel in self._panels:
                if trim_pos is not None:
                    panel.set_trim_positions(trim_pos[0], trim_pos[1])
                if self._ctx.steady_range is not None:
                    panel.set_steady_region(self._ctx.steady_range[0], self._ctx.steady_range[1])
        finally:
            self._overlay_updating = False

    def set_trim_line_positions(self, start_s: float, end_s: float) -> None:
        lo, hi = (start_s, end_s) if start_s <= end_s else (end_s, start_s)
        self._ctx.trim_line_positions = (lo, hi)
        self._ctx.show_trim_lines = True
        self.sync_overlay_positions()

    def set_steady_region(self, start_s: float, end_s: float) -> None:
        lo, hi = (start_s, end_s) if start_s <= end_s else (end_s, start_s)
        self._ctx.steady_range = (lo, hi)
        self.sync_overlay_positions()
