"""Interactive steady-state preview.

A pyqtgraph PlotWidget for the chosen channel of a preprocessed test, with
two draggable vertical lines marking the steady-state window. While the
operator drags either handle, ``window_stats`` is recomputed for every
visible channel and a small stats panel updates live (mean / std / n /
CV%). On commit, the parent widget kicks off a ReanalyzeWorker.

This is the killer-feature widget: in the legacy app, tweaking the
window required a full re-run (~30 s) per iteration and the operator
worked blind on the time-series plot. Here the readout is microsecond-
fast on every drag pixel.

Skips at module load when pyqtgraph is unavailable; the rest of the
detail panel keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

try:
    import pyqtgraph as pg
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    PYQTGRAPH_AVAILABLE = True
except (ImportError, OSError):
    PYQTGRAPH_AVAILABLE = False

from hda.domain.steady_state import ChannelStats, window_stats
from hda.domain.types import SteadyWindow


@dataclass(frozen=True, slots=True)
class _PreviewState:
    df: pd.DataFrame
    timestamp_column: str
    plotted_channel: str
    initial_window: SteadyWindow


if PYQTGRAPH_AVAILABLE:

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOption("antialias", True)

    class SteadyStatePreview(QWidget):
        """Interactive steady-state preview with live window stats.

        Signals:
            window_committed(SteadyWindow): emitted when the operator clicks
                "Apply window". Parent wires this to a ReanalyzeWorker.
        """

        window_committed = Signal(object)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._df: Optional[pd.DataFrame] = None
            self._ts_col = "timestamp"
            self._initial: Optional[SteadyWindow] = None
            self._busy: bool = False
            self._apply_allowed: bool = True

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            controls = QHBoxLayout()
            controls.setContentsMargins(8, 4, 8, 0)
            controls.addWidget(QLabel("Channel:"))
            self._channel_combo = QComboBox()
            self._channel_combo.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Fixed
            )
            self._channel_combo.currentTextChanged.connect(self._on_channel_changed)
            controls.addWidget(self._channel_combo, stretch=1)
            self._reset_btn = QPushButton("Reset window")
            self._reset_btn.setToolTip("Restore the currently-persisted steady-state window.")
            self._reset_btn.clicked.connect(self._reset_window)
            controls.addWidget(self._reset_btn)
            self._apply_btn = QPushButton("Apply window")
            self._apply_btn.setStyleSheet("font-weight: 600;")
            self._apply_btn.setToolTip(
                "Re-run QC + analysis on the highlighted window. (Ctrl+Enter)"
            )
            self._apply_btn.setShortcut("Ctrl+Return")
            self._apply_btn.clicked.connect(self._on_apply)
            controls.addWidget(self._apply_btn)
            layout.addLayout(controls)

            self._plot = pg.PlotWidget()
            self._plot.setLabel("bottom", "time", units="s")
            self._plot.showGrid(x=True, y=True, alpha=0.2)
            self._plot.getViewBox().enableAutoRange(axis="y", enable=True)
            self._curve = self._plot.plot(pen=pg.mkPen("#18181b", width=1))
            self._region = pg.LinearRegionItem(
                brush=pg.mkBrush(24, 24, 27, 35),
                pen=pg.mkPen("#18181b", width=1, style=Qt.DashLine),
            )
            self._region.sigRegionChanged.connect(self._on_region_changed)
            self._plot.addItem(self._region)
            layout.addWidget(self._plot, stretch=1)

            self._stats_label = QLabel("Drag the shaded region to adjust the window.")
            self._stats_label.setFrameStyle(QFrame.StyledPanel)
            mono = QFont("Menlo")
            mono.setStyleHint(QFont.Monospace)
            mono.setPointSize(10)
            self._stats_label.setFont(mono)
            self._stats_label.setStyleSheet(
                "padding:6px 10px; background:#f4f4f5; color:#27272a;"
                " border:1px solid #e4e4e7; border-radius:3px;"
            )
            self._stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._stats_label.setWordWrap(False)
            layout.addWidget(self._stats_label)

            self._set_enabled(False)

        # ---- public API ------------------------------------------------

        def show_data(
            self,
            df: pd.DataFrame,
            initial_window: SteadyWindow,
            timestamp_column: str = "timestamp",
            preferred_channel: Optional[str] = None,
        ) -> None:
            """Populate the widget with a dataset and an initial window."""
            self._df = df
            self._ts_col = timestamp_column
            self._initial = initial_window

            channels = [
                c
                for c in df.columns
                if c != timestamp_column and pd.api.types.is_numeric_dtype(df[c])
            ]
            self._channel_combo.blockSignals(True)
            self._channel_combo.clear()
            self._channel_combo.addItems(channels)
            if preferred_channel and preferred_channel in channels:
                self._channel_combo.setCurrentText(preferred_channel)
            elif channels:
                self._channel_combo.setCurrentIndex(0)
            self._channel_combo.blockSignals(False)

            self._render_curve()

            t = df[timestamp_column].to_numpy(dtype=float)
            t_min = float(t[0]) if t.size else 0.0
            t_max = float(t[-1]) if t.size else 1.0
            self._region.setBounds((t_min, t_max))
            self._region.setRegion(
                (initial_window.start_s, initial_window.end_s)
            )

            self._set_enabled(True)
            self._refresh_stats()

        def clear(self) -> None:
            self._df = None
            self._initial = None
            self._channel_combo.clear()
            self._curve.setData([], [])
            self._stats_label.setText("No test selected.")
            self._set_enabled(False)

        def set_busy(self, busy: bool) -> None:
            """Disable the Apply / Reset / channel controls while a
            reanalysis worker is running so a slow click cannot stack
            multiple jobs."""
            self._busy = busy
            has_data = self._df is not None
            self._apply_btn.setEnabled(
                has_data and not busy and self._apply_allowed
            )
            self._reset_btn.setEnabled(has_data and not busy)
            self._channel_combo.setEnabled(has_data and not busy)
            self._region.setMovable(has_data and not busy)
            self._apply_btn.setText("Reanalyzing…" if busy else "Apply window")

        def set_apply_enabled(self, allowed: bool) -> None:
            """External lock on Apply (e.g. AWAITING_METADATA). Drag and
            channel-switch stay live so the operator can still inspect."""
            self._apply_allowed = allowed
            self._apply_btn.setEnabled(
                self._df is not None and not self._busy and allowed
            )

        def set_apply_tooltip(self, text: Optional[str]) -> None:
            self._apply_btn.setToolTip(
                text
                if text is not None
                else "Re-run QC + analysis on the highlighted window. (Ctrl+Enter)"
            )

        def current_window(self) -> SteadyWindow:
            start, end = self._region.getRegion()
            method = "manual"
            if self._initial is not None and (
                abs(start - self._initial.start_s) < 1e-9
                and abs(end - self._initial.end_s) < 1e-9
            ):
                method = self._initial.method
            return SteadyWindow(
                start_s=float(start),
                end_s=float(end),
                method=method,
                confidence=1.0 if method == "manual" else self._initial.confidence,
            )

        # ---- handlers --------------------------------------------------

        def _set_enabled(self, enabled: bool) -> None:
            allow = enabled and not self._busy
            self._channel_combo.setEnabled(allow)
            self._reset_btn.setEnabled(allow)
            self._apply_btn.setEnabled(allow and self._apply_allowed)
            self._region.setMovable(allow)

        def _render_curve(self) -> None:
            if self._df is None or self._channel_combo.count() == 0:
                return
            channel = self._channel_combo.currentText()
            if channel not in self._df.columns:
                return
            t = self._df[self._ts_col].to_numpy(dtype=float)
            y = self._df[channel].to_numpy(dtype=float)
            self._curve.setData(t, y)
            self._plot.setLabel("left", channel)

        def _on_channel_changed(self, _channel: str) -> None:
            self._render_curve()
            self._refresh_stats()

        def _on_region_changed(self) -> None:
            self._refresh_stats()

        def _reset_window(self) -> None:
            if self._initial is None:
                return
            self._region.setRegion(
                (self._initial.start_s, self._initial.end_s)
            )

        def _on_apply(self) -> None:
            window = self.current_window()
            self.window_committed.emit(window)

        def _refresh_stats(self) -> None:
            if self._df is None:
                return
            start, end = self._region.getRegion()
            try:
                stats = window_stats(self._df, float(start), float(end), self._ts_col)
            except Exception as e:
                self._stats_label.setText(f"Stats error: {e}")
                return
            self._stats_label.setText(_format_stats(start, end, stats))


def _format_stats(start: float, end: float, stats: Mapping[str, ChannelStats]) -> str:
    duration = end - start
    n_max = max((s.n for s in stats.values()), default=0)
    lines = [
        f"window: [{start:.3f}, {end:.3f}] s   "
        f"duration: {duration:.3f} s   n: {n_max}",
    ]
    for name, s in stats.items():
        if s.n == 0:
            lines.append(f"  {name:<14}  no samples")
            continue
        cv_str = "—" if not np.isfinite(s.cv) else f"{s.cv * 100:5.2f}%"
        lines.append(
            f"  {name:<14}  mean={s.mean:11.4g}  std={s.std:9.3g}  cv={cv_str}"
        )
    return "\n".join(lines)
