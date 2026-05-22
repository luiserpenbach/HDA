"""Single Test Analysis page for the HDA Qt desktop UI.

Pipeline (left panel drives everything):
  1. Load CSV  →  detect time column, list numeric channels
  2. Pick config  →  badge shows test type
  3. Map sensor roles  →  tell analysis which column is upstream_pressure, mass_flow, etc.
  4. Choose plot channels  →  live time-series via pyqtgraph
  5. Detect / set steady-state window  →  draggable LinearRegionItem on the plot
  6. Run analysis  →  AnalysisResult with QC, measurements, traceability
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    _PG_OK = True
except Exception:
    _PG_OK = False

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    QThreadPool,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hda.ui.pages.base import BasePage, InfoBanner, MetricCard
from hda.ui.style import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BORDER,
    CONTENT_BG,
    CONTENT_SECONDARY_BG,
    FONT_FAMILY,
    SZ_BASE,
    SZ_SM,
    SZ_XS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# ---------------------------------------------------------------------------
# Colours for plot traces (cycles through these)
# ---------------------------------------------------------------------------
_TRACE_COLORS = [
    "#3b82f6", "#ef4444", "#16a34a", "#d97706",
    "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
    "#f97316", "#6366f1",
]

# Cold-flow sensor role labels (name → display label)
_CF_ROLES: List[Tuple[str, str, bool]] = [
    ("upstream_pressure",   "Upstream pressure",     True),
    ("mass_flow",           "Mass flow",              True),
    ("downstream_pressure", "Downstream pressure",   False),
    ("temperature",         "Temperature",            False),
]
_HF_ROLES: List[Tuple[str, str, bool]] = [
    ("chamber_pressure",    "Chamber pressure",      True),
    ("oxidizer_flow",       "Oxidizer mass flow",    True),
    ("fuel_flow",           "Fuel mass flow",        True),
    ("thrust",              "Thrust",                False),
    ("upstream_pressure",   "Upstream pressure",     False),
]


# ===========================================================================
# Background workers
# ===========================================================================

class _Sigs(QObject):
    loaded   = Signal(object, str, list)  # (df, time_col, numeric_cols)
    detected = Signal(float, float, str)  # (start_s, end_s, method)
    finished = Signal(object)             # AnalysisResult
    failed   = Signal(str)


class _LoadWorker(QRunnable):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._path = path
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            import pandas as pd
            df = pd.read_csv(self._path)
            time_col = self._detect_time_col(df)
            num_cols = [
                c for c in df.select_dtypes(include=["number"]).columns
                if c not in {"time_s", "time_ms", "timestamp", "Time", "TIME", "t"}
            ]
            self.signals.loaded.emit(df, time_col or "", num_cols)
        except Exception as exc:
            self.signals.failed.emit(str(exc))

    @staticmethod
    def _detect_time_col(df) -> Optional[str]:
        for name in ["time_s", "time_ms", "timestamp", "Time", "TIME", "t"]:
            if name in df.columns:
                return name
        return None


class _DetectWorker(QRunnable):
    def __init__(self, df, config: dict) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._df = df
        self._cfg = config
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            from core.steady_state_detection import detect_steady_state_auto
            start, end, method = detect_steady_state_auto(self._df, self._cfg)
            if start is None or end is None:
                self.signals.failed.emit("No steady state found — adjust detection parameters.")
            else:
                self.signals.detected.emit(float(start), float(end), method)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _AnalysisWorker(QRunnable):
    def __init__(
        self,
        df,
        config: dict,
        steady_s: Tuple[float, float],
        test_type: str,
        test_id: str,
        file_path: str,
        resample_hz: Optional[float],
        time_col: str,
        time_unit: str,
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._df = df
        self._cfg = config
        self._steady = steady_s
        self._test_type = test_type
        self._test_id = test_id
        self._file_path = file_path or None
        self._resample_hz = resample_hz
        self._time_col = time_col
        self._time_unit = time_unit
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            import pandas as pd
            import numpy as np
            from core.integrated_analysis import (
                analyze_cold_flow_test,
                analyze_hot_fire_test,
            )

            df = self._preprocess(self._df.copy())

            fn = (
                analyze_cold_flow_test
                if self._test_type == "cold_flow"
                else analyze_hot_fire_test
            )
            result = fn(
                df=df,
                config=self._cfg,
                steady_window=self._steady,
                test_id=self._test_id,
                file_path=self._file_path,
                skip_qc=False,
            )
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))

    def _preprocess(self, df):
        import pandas as pd
        import numpy as np

        tc = self._time_col
        if tc and tc in df.columns:
            df = df.sort_values(tc).reset_index(drop=True)
            df = df.drop_duplicates(subset=[tc], keep="first")

            unit = self._time_unit
            if unit == "ms":
                df["time_s"] = df[tc] / 1000.0
            elif unit == "μs":
                df["time_s"] = df[tc] / 1_000_000.0
            else:
                df["time_s"] = df[tc].astype(float)

            df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
            df["time_ms"] = df["time_s"] * 1000.0

        if self._resample_hz and self._resample_hz > 0 and "time_s" in df.columns:
            t = df["time_s"].values
            new_t = np.arange(t[0], t[-1], 1.0 / self._resample_hz)
            out = {"time_s": new_t, "time_ms": new_t * 1000.0}
            for col in df.select_dtypes(include=["number"]).columns:
                if col in {"time_s", "time_ms"}:
                    continue
                out[col] = np.interp(new_t, t, df[col].values)
            df = pd.DataFrame(out)

        return df


# ===========================================================================
# Left-panel section header
# ===========================================================================

def _section(title: str) -> QLabel:
    lbl = QLabel(title.upper())
    lbl.setStyleSheet(
        f"color: {TEXT_MUTED}; font-size: {SZ_XS}; font-weight: 700; "
        f"letter-spacing: 0.08em; background: transparent; "
        f"padding-top: 10px; padding-bottom: 2px;"
    )
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {BORDER}; border: none;")
    return line


def _form_row(label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {SZ_SM}; background: transparent;")
    lbl.setFixedWidth(140)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


# ===========================================================================
# Results widget
# ===========================================================================

class _ResultsWidget(QWidget):
    """Shown below the plot after a successful analysis run."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # QC row
        self._qc_banner = InfoBanner(parent=self)
        layout.addWidget(self._qc_banner)

        # Metric cards grid
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        layout.addWidget(self._grid_widget)

        # Traceability block
        self._trace_label = QLabel()
        self._trace_label.setWordWrap(True)
        self._trace_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_XS}; background: {CONTENT_SECONDARY_BG}; "
            f"border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 10px;"
        )
        layout.addWidget(self._trace_label)
        layout.addStretch()

    def populate(self, result) -> None:
        # QC banner
        if result.passed_qc:
            self._qc_banner.show_message("QC passed — all checks within limits.", "success")
        else:
            fails = getattr(result.qc_report, "blocking_failures", [])
            names = ", ".join(getattr(f, "name", str(f)) for f in fails)
            self._qc_banner.show_message(f"QC failed: {names}", "error")

        # Clear old metric cards
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Render metric cards (up to 8)
        items = list(result.measurements.items())[:8]
        cols = 4
        for i, (name, meas) in enumerate(items):
            row, col = divmod(i, cols)
            val = getattr(meas, "value", meas)
            unc = getattr(meas, "uncertainty", None)
            unit = getattr(meas, "unit", "")
            val_str = f"{val:.4g} {unit}".strip()
            if unc is not None:
                val_str += f" ±{unc:.3g}"
            card = MetricCard(name, val_str)
            self._grid.addWidget(card, row, col)

        # Traceability
        tr = result.traceability or {}
        raw_hash = tr.get("raw_data_hash", "—")[:16] if isinstance(tr, dict) else "—"
        cfg_hash  = tr.get("config_hash",  "—")[:16] if isinstance(tr, dict) else "—"
        analyst   = tr.get("analyst_username", "—") if isinstance(tr, dict) else "—"
        ts        = tr.get("analysis_timestamp", "—") if isinstance(tr, dict) else "—"
        proc_ver  = tr.get("processing_version", "—") if isinstance(tr, dict) else "—"
        self._trace_label.setText(
            f"data: {raw_hash}…  cfg: {cfg_hash}…  "
            f"analyst: {analyst}  ts: {ts}  proc: {proc_ver}"
        )


# ===========================================================================
# Main page
# ===========================================================================

class SingleTestAnalysisPage(BasePage):
    """Full single-test analysis pipeline in a native Qt UI."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "Single Test Analysis",
            "Load a CSV, pick a config, map sensor roles, define the steady-state window, then run.",
            parent=parent,
        )
        self._settings = QSettings("HopperPropulsion", "HDA")

        # State
        self._df = None                  # raw DataFrame after load
        self._time_col: str = ""
        self._numeric_cols: List[str] = []
        self._plot_items: Dict[str, Any] = {}   # col → PlotDataItem
        self._region: Any = None                # LinearRegionItem
        self._region_updating = False           # recursion guard
        self._config_id: str = ""
        self._file_path: str = ""

        # ── Main splitter ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")
        self.content_layout.addWidget(splitter, 1)

        # Left panel (scrollable controls)
        splitter.addWidget(self._build_left_panel())

        # Right panel (plot + results)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 0, 0, 0)
        right_lay.setSpacing(8)

        self._banner = InfoBanner(parent=right)
        right_lay.addWidget(self._banner)

        if _PG_OK:
            self._plot = pg.PlotWidget()
            self._plot.setBackground("w")
            self._plot.showGrid(x=True, y=True, alpha=0.25)
            self._plot.setLabel("bottom", "Time", units="s")
            self._plot.addLegend(offset=(10, 10))
            self._plot.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            right_lay.addWidget(self._plot, 3)
            self._add_region(0.0, 1.0)
        else:
            placeholder = QLabel("pyqtgraph not available — install it to see the plot.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(f"color: {TEXT_MUTED};")
            right_lay.addWidget(placeholder, 3)
            self._plot = None

        # Results (hidden until analysis runs)
        self._results = _ResultsWidget(right)
        self._results.hide()
        right_lay.addWidget(self._results, 2)

        splitter.addWidget(right)
        splitter.setSizes([340, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Keyboard shortcut
        sc = QShortcut(QKeySequence("F5"), self)
        sc.activated.connect(self._run_analysis)

    # ------------------------------------------------------------------ panels

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedWidth(340)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {CONTENT_BG}; border: none; border-right: 1px solid {BORDER}; }}"
        )

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 4, 12, 16)
        lay.setSpacing(2)

        # ── Data ────────────────────────────────────────────────────────────
        lay.addWidget(_section("Data"))
        lay.addWidget(_divider())

        self._browse_btn = QPushButton("Browse CSV…")
        self._browse_btn.clicked.connect(self._browse_csv)
        lay.addWidget(self._browse_btn)

        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_XS}; background: transparent;"
        )
        lay.addWidget(self._file_label)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {SZ_XS}; background: transparent;"
        )
        lay.addWidget(self._info_label)

        # ── Configuration ───────────────────────────────────────────────────
        lay.addWidget(_section("Configuration"))
        lay.addWidget(_divider())

        self._config_combo = QComboBox()
        self._config_combo.currentIndexChanged.connect(self._on_config_changed)
        lay.addWidget(self._config_combo)

        self._config_type_lbl = QLabel("")
        self._config_type_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_XS}; background: transparent;"
        )
        lay.addWidget(self._config_type_lbl)

        # ── Preprocessing ───────────────────────────────────────────────────
        lay.addWidget(_section("Preprocessing"))
        lay.addWidget(_divider())

        self._time_col_combo = QComboBox()
        lay.addLayout(_form_row("Time col", self._time_col_combo))

        self._time_unit_combo = QComboBox()
        self._time_unit_combo.addItems(["ms", "s", "μs"])
        lay.addLayout(_form_row("Time unit", self._time_unit_combo))

        resample_row = QHBoxLayout()
        resample_row.setContentsMargins(0, 0, 0, 0)
        self._resample_chk = QCheckBox("Resample to")
        self._resample_chk.setStyleSheet(f"font-size: {SZ_SM}; background: transparent;")
        self._resample_hz = QDoubleSpinBox()
        self._resample_hz.setRange(1, 10000)
        self._resample_hz.setValue(100)
        self._resample_hz.setSuffix(" Hz")
        self._resample_hz.setDecimals(0)
        self._resample_hz.setEnabled(False)
        self._resample_chk.toggled.connect(self._resample_hz.setEnabled)
        resample_row.addWidget(self._resample_chk)
        resample_row.addWidget(self._resample_hz, 1)
        lay.addLayout(resample_row)

        # ── Plot channels ───────────────────────────────────────────────────
        lay.addWidget(_section("Plot channels"))
        lay.addWidget(_divider())

        self._channel_list = QListWidget()
        self._channel_list.setMaximumHeight(110)
        self._channel_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {BORDER}; font-size: {SZ_SM}; }}"
            f"QListWidget::item {{ padding: 2px 4px; }}"
        )
        self._channel_list.itemChanged.connect(self._on_channel_toggled)
        lay.addWidget(self._channel_list)

        # ── Sensor roles ─────────────────────────────────────────────────────
        lay.addWidget(_section("Sensor roles"))
        lay.addWidget(_divider())

        self._role_combos: Dict[str, QComboBox] = {}
        self._role_box = QWidget()
        self._role_layout = QVBoxLayout(self._role_box)
        self._role_layout.setContentsMargins(0, 0, 0, 0)
        self._role_layout.setSpacing(3)
        lay.addWidget(self._role_box)
        self._rebuild_role_combos("cold_flow")

        # ── Steady state ────────────────────────────────────────────────────
        lay.addWidget(_section("Steady state"))
        lay.addWidget(_divider())

        detect_row = QHBoxLayout()
        detect_row.setContentsMargins(0, 0, 0, 0)
        self._detect_method = QComboBox()
        self._detect_method.addItems(["cv", "ml", "derivative", "simple"])
        detect_row.addWidget(self._detect_method, 1)
        self._detect_btn = QPushButton("Auto-detect")
        self._detect_btn.setFixedWidth(90)
        self._detect_btn.clicked.connect(self._auto_detect)
        detect_row.addWidget(self._detect_btn)
        lay.addLayout(detect_row)

        self._ss_start = QDoubleSpinBox()
        self._ss_start.setRange(0, 100000)
        self._ss_start.setDecimals(3)
        self._ss_start.setSuffix(" s")
        self._ss_start.setSingleStep(0.1)
        self._ss_start.valueChanged.connect(self._on_spinbox_changed)
        lay.addLayout(_form_row("Start", self._ss_start))

        self._ss_end = QDoubleSpinBox()
        self._ss_end.setRange(0, 100000)
        self._ss_end.setDecimals(3)
        self._ss_end.setSuffix(" s")
        self._ss_end.setSingleStep(0.1)
        self._ss_end.setValue(1.0)
        self._ss_end.valueChanged.connect(self._on_spinbox_changed)
        lay.addLayout(_form_row("End", self._ss_end))

        # ── Run ─────────────────────────────────────────────────────────────
        lay.addWidget(_section("Analysis"))
        lay.addWidget(_divider())

        self._test_id_edit = QLineEdit()
        self._test_id_edit.setPlaceholderText("e.g. INJ-CF-042")
        lay.addLayout(_form_row("Test ID", self._test_id_edit))

        self._operator_edit = QLineEdit()
        self._operator_edit.setPlaceholderText("your name")
        lay.addLayout(_form_row("Operator", self._operator_edit))

        self._skip_qc_chk = QCheckBox("Skip QC (not recommended)")
        self._skip_qc_chk.setStyleSheet(f"font-size: {SZ_SM}; background: transparent;")
        lay.addWidget(self._skip_qc_chk)

        self._run_btn = QPushButton("Run Analysis  (F5)")
        self._run_btn.clicked.connect(self._run_analysis)
        lay.addWidget(self._run_btn)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ------------------------------------------------------------------ region

    def _add_region(self, start: float, end: float) -> None:
        if not _PG_OK or self._plot is None:
            return
        if self._region is not None:
            self._plot.removeItem(self._region)
        self._region = pg.LinearRegionItem(
            [start, end],
            movable=True,
            brush=pg.mkBrush(59, 130, 246, 25),
            pen=pg.mkPen(color="#3b82f6", width=1),
        )
        self._region.sigRegionChanged.connect(self._on_region_changed)
        self._plot.addItem(self._region)

    @Slot()
    def _on_region_changed(self) -> None:
        if self._region_updating:
            return
        self._region_updating = True
        try:
            lo, hi = self._region.getRegion()
            self._ss_start.setValue(lo)
            self._ss_end.setValue(hi)
        finally:
            self._region_updating = False

    @Slot()
    def _on_spinbox_changed(self) -> None:
        if self._region_updating or self._region is None:
            return
        self._region_updating = True
        try:
            self._region.setRegion([self._ss_start.value(), self._ss_end.value()])
        finally:
            self._region_updating = False

    # ------------------------------------------------------------------ plot

    def _rebuild_plot(self) -> None:
        if not _PG_OK or self._plot is None or self._df is None:
            return

        # Remove old items (keep region and legend)
        for item in list(self._plot_items.values()):
            self._plot.removeItem(item)
        self._plot_items.clear()

        time_col = self._time_col_combo.currentText()
        if not time_col or time_col not in self._df.columns:
            return

        t = self._df[time_col].values

        color_idx = 0
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            col = item.text()
            if col not in self._df.columns:
                continue
            color = _TRACE_COLORS[color_idx % len(_TRACE_COLORS)]
            color_idx += 1
            pen = pg.mkPen(color=color, width=1.5)
            pi = self._plot.plot(t, self._df[col].values, pen=pen, name=col)
            self._plot_items[col] = pi

        # Ensure region stays on top
        if self._region is not None:
            self._plot.removeItem(self._region)
            self._plot.addItem(self._region)

    # ------------------------------------------------------------------ slots: data

    @Slot()
    def _browse_csv(self) -> None:
        last_dir = self._settings.value("sta/last_csv_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open test CSV", last_dir, "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        self._settings.setValue("sta/last_csv_dir", str(Path(path).parent))
        self._file_path = path
        self._file_label.setText(Path(path).name)
        self._info_label.setText("Loading…")
        self._banner.show_message("Loading CSV…", "info")

        worker = _LoadWorker(path)
        worker.signals.loaded.connect(self._on_csv_loaded)
        worker.signals.failed.connect(lambda e: self._banner.show_message(f"Load failed: {e}", "error"))
        QThreadPool.globalInstance().start(worker)

    @Slot(object, str, list)
    def _on_csv_loaded(self, df, time_col: str, numeric_cols: List[str]) -> None:
        self._df = df
        self._time_col = time_col
        self._numeric_cols = numeric_cols

        rows, cols = df.shape
        t_min = t_max = 0.0
        if time_col and time_col in df.columns:
            t_min = float(df[time_col].min())
            t_max = float(df[time_col].max())
        self._info_label.setText(f"{rows:,} rows × {cols} cols  |  {t_max - t_min:.1f} time units")

        # Populate time-col combo
        self._time_col_combo.blockSignals(True)
        self._time_col_combo.clear()
        time_candidates = [
            c for c in df.columns
            if c in {"time_s", "time_ms", "timestamp", "Time", "TIME", "t"}
        ]
        other_candidates = [c for c in df.columns if c not in time_candidates]
        for c in time_candidates + other_candidates:
            self._time_col_combo.addItem(c)
        if time_col:
            idx = self._time_col_combo.findText(time_col)
            if idx >= 0:
                self._time_col_combo.setCurrentIndex(idx)
        self._time_col_combo.blockSignals(False)

        # Populate channel checklist (first 8 checked)
        self._channel_list.blockSignals(True)
        self._channel_list.clear()
        for i, col in enumerate(numeric_cols):
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < 8 else Qt.Unchecked)
            self._channel_list.addItem(item)
        self._channel_list.blockSignals(False)

        # Populate sensor role combos with empty + column names
        blank_plus_cols = [""] + numeric_cols
        for combo in self._role_combos.values():
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(blank_plus_cols)
            combo.blockSignals(False)

        # Set steady-state region to middle 50%
        if time_col and time_col in df.columns:
            dur = t_max - t_min
            lo = t_min + dur * 0.25
            hi = t_min + dur * 0.75
            self._ss_start.setValue(lo)
            self._ss_end.setValue(hi)
            if self._region is not None:
                self._add_region(lo, hi)

        self._rebuild_plot()
        self._banner.show_message(
            f"Loaded {Path(self._file_path).name} — {rows:,} rows, {len(numeric_cols)} channels.", "success"
        )

    @Slot()
    def _on_channel_toggled(self, _item: QListWidgetItem) -> None:
        self._rebuild_plot()

    # ------------------------------------------------------------------ slots: config

    @Slot()
    def _on_config_changed(self, index: int) -> None:
        self._config_id = self._config_combo.itemData(index) or ""
        cfg = self._load_config_obj()
        test_type = getattr(cfg, "test_type", "cold_flow") if cfg else "cold_flow"
        self._rebuild_role_combos(test_type)

        type_text = test_type.replace("_", " ").title()
        color = ACCENT_BLUE if test_type == "cold_flow" else ACCENT_AMBER
        self._config_type_lbl.setText(f"Type: {type_text}")
        self._config_type_lbl.setStyleSheet(
            f"color: {color}; font-size: {SZ_XS}; font-weight: 600; background: transparent;"
        )

    def _load_config_obj(self):
        if not self._config_id:
            return None
        try:
            from pathlib import Path as P
            from core.saved_configs import SavedConfigManager
            mgr = SavedConfigManager(str(P(__file__).resolve().parents[3] / "saved_configs"))
            return mgr.get_template(self._config_id)
        except Exception:
            return None

    def _build_analysis_config(self) -> Optional[dict]:
        cfg_obj = self._load_config_obj()
        if cfg_obj is None:
            return None
        config = cfg_obj.to_config()
        # Inject sensor roles from the combo selections
        roles = {
            role: self._role_combos[role].currentText()
            for role in self._role_combos
            if self._role_combos[role].currentText()
        }
        if roles:
            config["sensor_roles"] = roles
        return config

    def _rebuild_role_combos(self, test_type: str) -> None:
        # Clear existing
        while self._role_layout.count():
            item = self._role_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._role_combos.clear()

        roles = _HF_ROLES if test_type == "hot_fire" else _CF_ROLES
        current_cols = [""] + self._numeric_cols

        for key, label, required in roles:
            combo = QComboBox()
            combo.addItems(current_cols)
            suffix = "" if required else " (opt)"
            self._role_layout.addLayout(_form_row(label + suffix, combo))
            self._role_combos[key] = combo

    # ------------------------------------------------------------------ slots: steady state

    @Slot()
    def _auto_detect(self) -> None:
        if self._df is None:
            self._banner.show_message("Load a CSV first.", "warning")
            return
        config = self._build_analysis_config() or {}
        method = self._detect_method.currentText()
        config["preferred_method"] = method
        self._banner.show_message("Detecting steady state…", "info")
        worker = _DetectWorker(self._df, config)
        worker.signals.detected.connect(self._on_detected)
        worker.signals.failed.connect(lambda e: self._banner.show_message(e, "warning"))
        QThreadPool.globalInstance().start(worker)

    @Slot(float, float, str)
    def _on_detected(self, start: float, end: float, method: str) -> None:
        self._ss_start.setValue(start)
        self._ss_end.setValue(end)
        self._add_region(start, end)
        self._banner.show_message(
            f"Steady state: {start:.3f}–{end:.3f} s  ({end - start:.2f} s, method: {method})",
            "success",
        )

    # ------------------------------------------------------------------ slots: run

    @Slot()
    def _run_analysis(self) -> None:
        if self._df is None:
            self._banner.show_message("Load a CSV first.", "warning")
            return

        config = self._build_analysis_config()
        if config is None:
            self._banner.show_message("Select a configuration.", "warning")
            return

        test_type = config.get("test_type", "cold_flow")
        test_id = self._test_id_edit.text().strip() or "HDA-TEST-001"
        steady = (self._ss_start.value(), self._ss_end.value())

        if steady[1] <= steady[0]:
            self._banner.show_message("Steady-state end must be after start.", "warning")
            return

        resample_hz: Optional[float] = (
            self._resample_hz.value() if self._resample_chk.isChecked() else None
        )
        time_col = self._time_col_combo.currentText()
        time_unit = self._time_unit_combo.currentText()

        self._banner.show_message("Running analysis…", "info")
        self._run_btn.setEnabled(False)
        self._results.hide()

        worker = _AnalysisWorker(
            df=self._df,
            config=config,
            steady_s=steady,
            test_type=test_type,
            test_id=test_id,
            file_path=self._file_path,
            resample_hz=resample_hz,
            time_col=time_col,
            time_unit=time_unit,
        )
        worker.signals.finished.connect(self._on_analysis_done)
        worker.signals.failed.connect(self._on_analysis_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_analysis_done(self, result) -> None:
        self._run_btn.setEnabled(True)
        n = len(result.measurements)
        status = "passed" if result.passed_qc else "FAILED QC"
        self._banner.show_message(
            f"Analysis complete — {n} metrics, QC {status}.",
            "success" if result.passed_qc else "error",
        )
        self._results.populate(result)
        self._results.show()

    @Slot(str)
    def _on_analysis_failed(self, error: str) -> None:
        self._run_btn.setEnabled(True)
        self._banner.show_message(f"Analysis failed: {error}", "error")

    # ------------------------------------------------------------------ lifecycle

    def on_context_changed(self) -> None:
        self._reload_config_list()

    def _reload_config_list(self) -> None:
        try:
            from pathlib import Path as P
            from core.saved_configs import SavedConfigManager
            mgr = SavedConfigManager(str(P(__file__).resolve().parents[3] / "saved_configs"))
            templates = mgr.list_templates(include_builtin=True)
        except Exception:
            templates = []

        prev_id = self._config_id
        self._config_combo.blockSignals(True)
        self._config_combo.clear()
        for t in templates:
            tid = t.get("id", t.get("name", ""))
            label = t.get("name", tid)
            self._config_combo.addItem(label, userData=tid)
        self._config_combo.blockSignals(False)

        # Restore previous selection
        if prev_id:
            for i in range(self._config_combo.count()):
                if self._config_combo.itemData(i) == prev_id:
                    self._config_combo.setCurrentIndex(i)
                    break
        if self._config_combo.count() > 0:
            self._on_config_changed(self._config_combo.currentIndex())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._config_combo.count() == 0:
            self._reload_config_list()
