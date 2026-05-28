"""Single Test Analysis page for the HDA Qt desktop UI.

Workflow:
  1. Load CSV (or handoff from Test Explorer)
  2. Preprocess — time axis, resample, trim, channel mapping
  3. Inspect data — dockable pyqtgraph panels, trim lines, steady region
  4. Run analysis — core.integrated_analysis with full P0 integrity
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from hda.ui.style import configure_pyqtgraph

    configure_pyqtgraph()
except Exception:
    pass

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
from PySide6.QtGui import QKeySequence, QShortcut
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hda.preprocessing import (
    detect_time_unit,
    preview_time_seconds,
    run_preprocessing_pipeline,
)
from hda.test_type_utils import normalize_test_type
from hda.ui.pages.base import BasePage, InfoBanner, MetricCard
from hda.ui.plot_panels import PlotDockWorkspace, PlotRenderContext
from hda.ui.style import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BORDER,
    CONTENT_SECONDARY_BG,
    SZ_BASE,
    SZ_SM,
    SZ_XS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# Time unit options: (label, internal key)
_TIME_UNIT_OPTIONS: List[Tuple[str, str]] = [
    ("Unix timestamp (ms)", "unix_ms"),
    ("Unix timestamp (s)", "unix_s"),
    ("Relative ms", "ms"),
    ("Relative s", "s"),
    ("Relative μs", "us"),
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

STA_PANEL_MIN = 260
STA_PANEL_MAX = 520
STA_PANEL_DEFAULT = 320


from hda.plot_utils import default_steady_window
# ===========================================================================
# Background workers
# ===========================================================================

class _Sigs(QObject):
    loaded       = Signal(object, str, list)    # (df, time_col, numeric_cols)
    preprocessed = Signal(object, object, object)  # (df, stats, df_before_trim)
    detected     = Signal(float, float, str)    # (start_s, end_s, method)
    finished     = Signal(object)               # AnalysisResult
    failed       = Signal(str)


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


class _PreprocessWorker(QRunnable):
    def __init__(self, df, options: dict) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._df = df
        self._options = options
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            df_out, stats, df_before = run_preprocessing_pipeline(self._df, **self._options)
            self.signals.preprocessed.emit(df_out, stats, df_before)
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
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._df = df
        self._cfg = config
        self._steady = steady_s
        self._test_type = test_type
        self._test_id = test_id
        self._file_path = file_path or None
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            from core.integrated_analysis import (
                analyze_cold_flow_test,
                analyze_hot_fire_test,
            )

            fn = (
                analyze_cold_flow_test
                if self._test_type == "cold_flow"
                else analyze_hot_fire_test
            )
            result = fn(
                df=self._df,
                config=self._cfg,
                steady_window=self._steady,
                test_id=self._test_id,
                file_path=self._file_path,
                skip_qc=False,
            )
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


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


def _form_row(
    label: str,
    widget: QWidget,
    *,
    optional: bool = False,
    required: bool = False,
) -> QWidget:
    """Label above control in a widget row (safe to add/remove as one unit)."""
    row = QWidget()
    col = QVBoxLayout(row)
    col.setContentsMargins(0, 6, 0, 0)
    col.setSpacing(5)
    lbl = QLabel()
    lbl.setObjectName("FormFieldLabel")
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
    if optional:
        lbl.setText(f"{label} (optional)")
    elif required:
        lbl.setText(f"{label} *")
    else:
        lbl.setText(label)
    col.addWidget(lbl)
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    col.addWidget(widget)
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
            "Load and preprocess data, then detect steady state and run analysis.",
            parent=parent,
        )
        self._settings = QSettings("HopperPropulsion", "HDA")

        # State
        self._df_raw = None
        self._df_processed = None
        self._df_plot_source = None
        self._trim_highlight: Optional[Tuple[float, float]] = None
        self._preprocess_stats: dict = {}
        self._time_col: str = ""
        self._numeric_cols: List[str] = []
        self._region_updating = False
        self._trim_updating = False
        self._config_id: str = ""
        self._file_path: str = ""
        self._test_folder_path: str = ""
        self._pending_config_id: str = ""
        self._plot_area_hosts: List[QWidget] = []

        # ── Tabbed workflow (mirrors Streamlit STA tabs) ─────────────────────
        self._tabs = QTabWidget()
        self.content_layout.addWidget(self._tabs, 1)

        self._plot_area = self._build_plot_area()

        pre_tab, pre_host = self._create_split_tab(self._build_preprocess_panel())
        self._tabs.addTab(pre_tab, "Preprocessing")
        self._plot_area_hosts.append(pre_host)

        steady_tab, steady_host = self._create_split_tab(self._build_steady_panel())
        self._tabs.addTab(steady_tab, "Steady State")
        self._plot_area_hosts.append(steady_host)
        self._add_plot_panel()

        for title in ("Analyze", "Results", "Export"):
            self._tabs.addTab(self._placeholder_tab(title), title)

        self._tabs.currentChanged.connect(self._on_workflow_tab_changed)
        self._on_workflow_tab_changed(0)

        # Keyboard shortcuts
        sc_run = QShortcut(QKeySequence("F5"), self)
        sc_run.activated.connect(self._run_analysis)
        sc_open = QShortcut(QKeySequence("Ctrl+O"), self)
        sc_open.activated.connect(self._browse_csv)

    def _placeholder_tab(self, title: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch()
        lbl = QLabel(f"{title} — coming soon.")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_BASE}; background: transparent;")
        lay.addWidget(lbl)
        lay.addStretch()
        return w

    def _create_split_tab(self, left_panel: QWidget) -> Tuple[QWidget, QWidget]:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_panel)
        host = QWidget()
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.setSpacing(0)
        splitter.addWidget(host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)
        return tab, host

    def _build_plot_area(self) -> QWidget:
        area = QWidget()
        lay = QVBoxLayout(area)
        lay.setContentsMargins(16, 0, 0, 0)
        lay.setSpacing(8)

        self._banner = InfoBanner(parent=area)
        lay.addWidget(self._banner)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {SZ_SM}; background: transparent;"
        )
        lay.addWidget(self._status_lbl)

        plot_toolbar = QHBoxLayout()
        plot_toolbar.setSpacing(8)
        self._add_plot_btn = QPushButton("Add plot panel")
        self._add_plot_btn.setProperty("secondary", True)
        self._add_plot_btn.clicked.connect(self._add_plot_panel)
        plot_toolbar.addWidget(self._add_plot_btn)
        plot_toolbar.addStretch()
        lay.addLayout(plot_toolbar)

        self._plot_workspace = PlotDockWorkspace(area)
        self._plot_workspace.setMinimumHeight(280)
        self._plot_workspace.trim_lines_changed.connect(self._on_trim_lines_dragged)
        self._plot_workspace.steady_region_changed.connect(self._on_steady_region_dragged)
        lay.addWidget(self._plot_workspace, 3)

        self._results = _ResultsWidget(area)
        self._results.hide()
        lay.addWidget(self._results, 2)
        return area

    def _on_workflow_tab_changed(self, index: int) -> None:
        if index >= len(self._plot_area_hosts):
            self._plot_area.hide()
            return
        host = self._plot_area_hosts[index]
        self._plot_area.setParent(None)
        host.layout().addWidget(self._plot_area)
        self._plot_area.show()

    # ------------------------------------------------------------------ panels

    def _build_scroll_panel(self) -> Tuple[QWidget, QVBoxLayout]:
        container = QWidget()
        container.setMinimumWidth(STA_PANEL_MIN)
        container.setMaximumWidth(STA_PANEL_MAX)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; "
            f"border-right: 1px solid {BORDER}; }}"
        )
        inner = QWidget()
        inner.setMinimumWidth(STA_PANEL_MIN - 24)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 4, 12, 16)
        lay.setSpacing(6)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return container, lay

    def _build_preprocess_panel(self) -> QWidget:
        container, lay = self._build_scroll_panel()
        lay.addWidget(_section("Data"))
        lay.addWidget(_divider())

        self._browse_btn = QPushButton("Browse CSV…")
        self._browse_btn.setToolTip("Open a CSV file (Ctrl+O)")
        self._browse_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self._config_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        lay.addWidget(_form_row("Time col", self._time_col_combo))

        self._time_unit_combo = QComboBox()
        for label, key in _TIME_UNIT_OPTIONS:
            self._time_unit_combo.addItem(label, key)
        lay.addWidget(_form_row("Time unit", self._time_unit_combo))

        self._shift_zero_chk = QCheckBox("Shift time to t = 0")
        self._shift_zero_chk.setChecked(True)
        self._shift_zero_chk.setStyleSheet(f"font-size: {SZ_SM}; background: transparent;")
        self._shift_zero_chk.toggled.connect(self._on_plot_settings_changed)
        lay.addWidget(self._shift_zero_chk)

        self._mapping_chk = QCheckBox("Apply channel mapping from config")
        self._mapping_chk.setChecked(True)
        self._mapping_chk.setStyleSheet(f"font-size: {SZ_SM}; background: transparent;")
        lay.addWidget(self._mapping_chk)

        self._nan_method_combo = QComboBox()
        self._nan_method_combo.addItems(
            ["interpolate+ffill", "interpolate", "ffill", "drop", "none"]
        )
        lay.addWidget(_form_row("Gap filling", self._nan_method_combo))

        self._resample_chk = QCheckBox("Enable resampling")
        self._resample_chk.setStyleSheet(f"font-size: {SZ_SM}; background: transparent;")
        self._resample_chk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._resample_hz = QDoubleSpinBox()
        self._resample_hz.setRange(1, 10000)
        self._resample_hz.setValue(100)
        self._resample_hz.setSuffix(" Hz")
        self._resample_hz.setDecimals(0)
        self._resample_hz.setEnabled(False)
        self._resample_chk.toggled.connect(self._resample_hz.setEnabled)
        lay.addWidget(self._resample_chk)
        lay.addWidget(_form_row("Rate", self._resample_hz))

        self._trim_chk = QCheckBox("Trim time window")
        self._trim_chk.setStyleSheet(f"font-size: {SZ_SM}; background: transparent;")
        self._trim_chk.toggled.connect(self._on_trim_toggled)
        lay.addWidget(self._trim_chk)
        self._trim_start = QDoubleSpinBox()
        self._trim_start.setRange(0, 1_000_000)
        self._trim_start.setDecimals(3)
        self._trim_start.setSuffix(" s")
        self._trim_start.setEnabled(False)
        self._trim_end = QDoubleSpinBox()
        self._trim_end.setRange(0, 1_000_000)
        self._trim_end.setDecimals(3)
        self._trim_end.setSuffix(" s")
        self._trim_end.setEnabled(False)
        self._trim_chk.toggled.connect(self._trim_start.setEnabled)
        self._trim_chk.toggled.connect(self._trim_end.setEnabled)
        self._trim_start.valueChanged.connect(self._on_trim_spinbox_changed)
        self._trim_end.valueChanged.connect(self._on_trim_spinbox_changed)
        lay.addWidget(_form_row("Trim start", self._trim_start))
        lay.addWidget(_form_row("Trim end", self._trim_end))

        self._preprocess_btn = QPushButton("Run preprocessing")
        self._preprocess_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._preprocess_btn.clicked.connect(self._run_preprocessing)
        lay.addWidget(self._preprocess_btn)

        self._save_processed_btn = QPushButton("Save processed CSV…")
        self._save_processed_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._save_processed_btn.setEnabled(False)
        self._save_processed_btn.clicked.connect(self._save_processed_csv)
        lay.addWidget(self._save_processed_btn)

        lay.addWidget(_section("Plot channels"))
        lay.addWidget(_divider())

        self._channel_list = QListWidget()
        self._channel_list.setMaximumHeight(110)
        self._channel_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {BORDER}; font-size: {SZ_SM}; }}"
            f"QListWidget::item {{ padding: 2px 4px; }}"
        )
        self._channel_list.itemChanged.connect(self._on_channel_toggled)
        lay.addWidget(_form_row("Plot template channels", self._channel_list))
        hint = QLabel("Checked channels are used as defaults when adding a new plot panel.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_XS}; background: transparent;"
        )
        lay.addWidget(hint)

        self._time_col_combo.currentIndexChanged.connect(self._on_plot_settings_changed)
        self._time_unit_combo.currentIndexChanged.connect(self._on_plot_settings_changed)

        lay.addStretch()
        return container

    def _build_steady_panel(self) -> QWidget:
        container, lay = self._build_scroll_panel()

        lay.addWidget(_section("Sensor roles"))
        lay.addWidget(_divider())

        self._role_combos: Dict[str, QComboBox] = {}
        self._role_box = QWidget()
        self._role_layout = QVBoxLayout(self._role_box)
        self._role_layout.setContentsMargins(0, 0, 0, 0)
        self._role_layout.setSpacing(10)
        lay.addWidget(self._role_box)
        self._rebuild_role_combos("cold_flow")

        lay.addWidget(_section("Steady state"))
        lay.addWidget(_divider())

        self._detect_method = QComboBox()
        self._detect_method.addItems(["cv", "ml", "derivative", "simple"])
        lay.addWidget(_form_row("Detection method", self._detect_method))

        self._detect_btn = QPushButton("Auto-detect")
        self._detect_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._detect_btn.clicked.connect(self._auto_detect)
        lay.addWidget(self._detect_btn)

        self._ss_start = QDoubleSpinBox()
        self._ss_start.setRange(0, 100000)
        self._ss_start.setDecimals(3)
        self._ss_start.setSuffix(" s")
        self._ss_start.setSingleStep(0.1)
        self._ss_start.valueChanged.connect(self._on_spinbox_changed)
        lay.addWidget(_form_row("Start", self._ss_start))

        self._ss_end = QDoubleSpinBox()
        self._ss_end.setRange(0, 100000)
        self._ss_end.setDecimals(3)
        self._ss_end.setSuffix(" s")
        self._ss_end.setSingleStep(0.1)
        self._ss_end.setValue(1.0)
        self._ss_end.valueChanged.connect(self._on_spinbox_changed)
        lay.addWidget(_form_row("End", self._ss_end))

        lay.addWidget(_section("Analysis"))
        lay.addWidget(_divider())

        self._test_id_edit = QLineEdit()
        self._test_id_edit.setPlaceholderText("e.g. INJ-CF-042")
        lay.addWidget(_form_row("Test ID", self._test_id_edit))

        self._operator_edit = QLineEdit()
        self._operator_edit.setPlaceholderText("your name")
        lay.addWidget(_form_row("Operator", self._operator_edit))

        self._skip_qc_chk = QCheckBox("Skip QC (not recommended)")
        self._skip_qc_chk.setStyleSheet(f"font-size: {SZ_SM}; background: transparent;")
        lay.addWidget(self._skip_qc_chk)

        self._run_btn = QPushButton("Run Analysis  (F5)")
        self._run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._run_btn.clicked.connect(self._run_analysis)
        lay.addWidget(self._run_btn)

        lay.addStretch()
        return container

    # ------------------------------------------------------------------ plot workspace

    def _default_channels_from_template(self) -> List[str]:
        channel_list = getattr(self, "_channel_list", None)
        if channel_list is None:
            return self._numeric_cols[:3]
        cols: List[str] = []
        for i in range(channel_list.count()):
            item = channel_list.item(i)
            if item.checkState() == Qt.Checked:
                cols.append(item.text())
        return cols or self._numeric_cols[:3]

    def _add_plot_panel(self) -> None:
        self._plot_workspace.add_panel(
            available_channels=self._numeric_cols,
            default_channels=self._default_channels_from_template(),
        )
        self._rebuild_plot()

    def _plot_dataframe(self):
        if self._df_plot_source is not None:
            return self._df_plot_source
        if self._df_processed is not None:
            return self._df_processed
        return self._df_raw

    def _build_plot_context(self) -> PlotRenderContext:
        df = self._plot_dataframe()
        t = np.array([])
        if df is not None:
            if "time_s" in df.columns:
                t = df["time_s"].values.astype(float)
            else:
                t = preview_time_seconds(
                    df,
                    self._time_col_combo.currentText(),
                    self._selected_time_unit(),
                    shift_to_zero=self._shift_zero_chk.isChecked(),
                )

        trim_lo = self._trim_start.value()
        trim_hi = self._trim_end.value()
        show_trim = self._trim_chk.isChecked()
        highlight = self._trim_highlight is not None and self._df_plot_source is not None
        trim_range = self._trim_highlight if highlight else None
        if highlight and trim_range is None:
            trim_range = (min(trim_lo, trim_hi), max(trim_lo, trim_hi))

        preview_highlight = (
            show_trim
            and not highlight
            and df is not None
            and len(t) > 0
        )
        if preview_highlight:
            trim_range = (min(trim_lo, trim_hi), max(trim_lo, trim_hi))

        return PlotRenderContext(
            df=df,
            time_seconds=t,
            available_channels=self._numeric_cols,
            trim_range=trim_range,
            show_trim_lines=show_trim,
            trim_line_positions=(min(trim_lo, trim_hi), max(trim_lo, trim_hi)) if show_trim else None,
            steady_range=(self._ss_start.value(), self._ss_end.value()),
            highlight_trim=highlight or preview_highlight,
        )

    @Slot()
    def _on_trim_toggled(self, enabled: bool) -> None:
        self._rebuild_plot()

    @Slot()
    def _on_trim_spinbox_changed(self, *_args) -> None:
        if self._trim_updating or not self._trim_chk.isChecked():
            return
        self._rebuild_plot()

    @Slot(float, float)
    def _on_trim_lines_dragged(self, lo: float, hi: float) -> None:
        if self._trim_updating:
            return
        lo, hi = min(lo, hi), max(lo, hi)
        self._trim_updating = True
        try:
            self._trim_start.blockSignals(True)
            self._trim_end.blockSignals(True)
            self._trim_start.setValue(lo)
            self._trim_end.setValue(hi)
            self._trim_start.blockSignals(False)
            self._trim_end.blockSignals(False)
        finally:
            self._trim_updating = False
        ctx = self._build_plot_context()
        self._plot_workspace.update_context(
            trim_line_positions=ctx.trim_line_positions,
            trim_range=ctx.trim_range,
            highlight_trim=ctx.highlight_trim,
        )
        self._plot_workspace.render_traces()

    @Slot(float, float)
    def _on_steady_region_dragged(self, lo: float, hi: float) -> None:
        if self._region_updating:
            return
        lo, hi = min(lo, hi), max(lo, hi)
        self._region_updating = True
        try:
            self._ss_start.blockSignals(True)
            self._ss_end.blockSignals(True)
            self._ss_start.setValue(lo)
            self._ss_end.setValue(hi)
            self._ss_start.blockSignals(False)
            self._ss_end.blockSignals(False)
        finally:
            self._region_updating = False
        self._plot_workspace.update_context(steady_range=(lo, hi))

    @Slot()
    def _on_spinbox_changed(self, *_args) -> None:
        if self._region_updating:
            return
        self._plot_workspace.set_steady_region(
            self._ss_start.value(), self._ss_end.value()
        )

    def _set_steady_window_seconds(self, start_s: float, end_s: float) -> None:
        self._ss_start.blockSignals(True)
        self._ss_end.blockSignals(True)
        self._ss_start.setValue(start_s)
        self._ss_end.setValue(end_s)
        self._ss_start.blockSignals(False)
        self._ss_end.blockSignals(False)
        self._plot_workspace.set_steady_region(start_s, end_s)

    @Slot()
    def _on_plot_settings_changed(self, *_args) -> None:
        self._rebuild_plot()

    def _rebuild_plot(self) -> None:
        ctx = self._build_plot_context()
        self._plot_workspace.update_context(
            df=ctx.df,
            time_seconds=ctx.time_seconds,
            available_channels=ctx.available_channels,
            trim_range=ctx.trim_range,
            show_trim_lines=ctx.show_trim_lines,
            trim_line_positions=ctx.trim_line_positions,
            steady_range=ctx.steady_range,
            highlight_trim=ctx.highlight_trim,
        )
        self._plot_workspace.refresh()

    def _selected_time_unit(self) -> str:
        data = self._time_unit_combo.currentData()
        if data:
            return str(data)
        return "unix_ms"

    def _set_time_unit(self, unit_key: str) -> None:
        idx = self._time_unit_combo.findData(unit_key)
        if idx >= 0:
            self._time_unit_combo.setCurrentIndex(idx)

    def _active_df(self):
        return self._df_processed if self._df_processed is not None else self._df_raw

    def _update_data_status_label(self) -> None:
        df = self._active_df()
        if df is None:
            self._status_lbl.setText("")
            return
        rows = len(df)
        status = "Preprocessed" if self._df_processed is not None else "Raw"
        if self._trim_highlight and self._df_plot_source is not None:
            status = "Preprocessed (trim preview)"
        dur = "—"
        plot_df = self._plot_dataframe()
        if plot_df is None:
            plot_df = df
        if "time_s" in plot_df.columns and len(plot_df) > 0:
            t = plot_df["time_s"].values.astype(float)
            dur = f"{float(t[-1] - t[0]):.3f} s"
        else:
            t = preview_time_seconds(
                plot_df,
                self._time_col_combo.currentText(),
                self._selected_time_unit(),
                shift_to_zero=self._shift_zero_chk.isChecked(),
            )
            if len(t) > 0:
                dur = f"{float(t[-1] - t[0]):.3f} s"
        parts = [f"{rows:,} rows", f"Duration: {dur}", status]
        if self._preprocess_stats:
            rs = self._preprocess_stats.get("resample", {})
            if rs.get("resampled_rows"):
                parts.append(f"Resampled: {rs['resampled_rows']:,} pts")
            if self._preprocess_stats.get("trim"):
                parts.append("Trimmed")
        self._status_lbl.setText("  ·  ".join(parts))

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
        self._load_csv_path(path)

    def _load_csv_path(self, path: str) -> None:
        """Start background load of a CSV file path."""
        self._test_folder_path = ""
        self._file_path = path
        self._file_label.setText(Path(path).name)
        self._info_label.setText("Loading…")
        self._banner.show_message("Loading CSV…", "info")
        self.status_message.emit(f"Loading {Path(path).name}…")

        worker = _LoadWorker(path)
        worker.signals.loaded.connect(self._on_csv_loaded)
        worker.signals.failed.connect(self._on_csv_load_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_csv_load_failed(self, error: str) -> None:
        self._banner.show_message(f"Load failed: {error}", "error")
        self.status_message.emit(f"Load failed: {error}")

    @Slot(object, str, list)
    def _on_csv_loaded(self, df, time_col: str, numeric_cols: List[str]) -> None:
        self._df_raw = df
        self._df_processed = None
        self._df_plot_source = None
        self._trim_highlight = None
        self._preprocess_stats = {}
        self._save_processed_btn.setEnabled(False)
        self._time_col = time_col
        self._numeric_cols = numeric_cols

        rows, cols = df.shape
        self._info_label.setText(f"{rows:,} rows × {cols} cols  ·  raw data")

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

        self._refresh_channel_list(numeric_cols)
        self._refresh_role_combos(numeric_cols)

        if time_col and time_col in df.columns:
            detected = detect_time_unit(df[time_col])
            self._set_time_unit(detected)
            t = preview_time_seconds(
                df,
                time_col,
                detected,
                shift_to_zero=self._shift_zero_chk.isChecked(),
            )
            if len(t) > 0:
                t_min = float(np.min(t))
                t_max = float(np.max(t))
                self._trim_start.setValue(t_min)
                self._trim_end.setValue(max(t_max, t_min + 0.001))
                lo, hi = default_steady_window(t_min, t_max)
                self._set_steady_window_seconds(lo, hi)

        self._update_data_status_label()
        self._rebuild_plot()

        if self._test_folder_path:
            self._apply_test_folder_metadata(self._test_folder_path)

        self._banner.show_message(
            f"Loaded {Path(self._file_path).name} — {rows:,} rows, {len(numeric_cols)} channels.",
            "success",
        )
        self.status_message.emit(
            f"Loaded {Path(self._file_path).name} — {rows:,} rows, {len(numeric_cols)} channels."
        )

    def _refresh_channel_list(self, numeric_cols: List[str]) -> None:
        self._numeric_cols = numeric_cols
        self._channel_list.blockSignals(True)
        self._channel_list.clear()
        for i, col in enumerate(numeric_cols):
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < 8 else Qt.Unchecked)
            self._channel_list.addItem(item)
        self._channel_list.blockSignals(False)

    def _refresh_role_combos(self, numeric_cols: List[str]) -> None:
        blank_plus_cols = [""] + numeric_cols
        for combo in self._role_combos.values():
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(blank_plus_cols)
            combo.blockSignals(False)

    @Slot()
    def _run_preprocessing(self) -> None:
        if self._df_raw is None:
            self._banner.show_message("Load a CSV first.", "warning")
            return
        time_col = self._time_col_combo.currentText()
        if not time_col:
            self._banner.show_message("Select a time column.", "warning")
            return

        config = self._build_analysis_config()
        options = {
            "time_col": time_col,
            "time_unit": self._selected_time_unit(),
            "shift_to_zero": self._shift_zero_chk.isChecked(),
            "nan_method": self._nan_method_combo.currentText(),
            "resample_hz": self._resample_hz.value() if self._resample_chk.isChecked() else None,
            "config": config,
            "apply_mapping": self._mapping_chk.isChecked() and config is not None,
        }
        if self._trim_chk.isChecked():
            options["trim_start_s"] = self._trim_start.value()
            options["trim_end_s"] = self._trim_end.value()

        self._preprocess_btn.setEnabled(False)
        self._banner.show_message("Preprocessing…", "info")
        self.status_message.emit("Preprocessing…")
        worker = _PreprocessWorker(self._df_raw, options)
        worker.signals.preprocessed.connect(self._on_preprocessed)
        worker.signals.failed.connect(self._on_preprocess_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_preprocess_failed(self, error: str) -> None:
        self._preprocess_btn.setEnabled(True)
        self._banner.show_message(f"Preprocessing failed: {error}", "error")
        self.status_message.emit(f"Preprocessing failed: {error}")

    @Slot(object, object, object)
    def _on_preprocessed(self, df, stats: dict, df_before_trim=None) -> None:
        self._preprocess_btn.setEnabled(True)
        self._df_processed = df
        self._preprocess_stats = stats or {}

        trim_stats = (stats or {}).get("trim") or {}
        if df_before_trim is not None and trim_stats:
            self._df_plot_source = df_before_trim
            self._trim_highlight = (
                float(trim_stats.get("start_s", self._trim_start.value())),
                float(trim_stats.get("end_s", self._trim_end.value())),
            )
        else:
            self._df_plot_source = df
            self._trim_highlight = None

        time_derived = {"time_s", "time_ms"}
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c not in time_derived
        ]
        self._refresh_channel_list(numeric_cols)
        self._refresh_role_combos(numeric_cols)

        dur = float(stats.get("duration_s", 0.0) or 0.0)
        if dur <= 0 and "time_s" in df.columns and len(df) > 0:
            dur = float(df["time_s"].max() - df["time_s"].min())

        plot_dur = dur
        if self._df_plot_source is not None and "time_s" in self._df_plot_source.columns:
            src_t = self._df_plot_source["time_s"]
            if len(src_t) > 0:
                plot_dur = float(src_t.max() - src_t.min())

        if trim_stats:
            self._trim_start.setValue(float(trim_stats.get("start_s", 0.0)))
            self._trim_end.setValue(float(trim_stats.get("end_s", plot_dur)))
        else:
            self._trim_start.setValue(0.0)
            self._trim_end.setValue(max(plot_dur, 0.001))
        if "time_s" in df.columns and len(df) > 0:
            t_min = float(df["time_s"].min())
            t_max = float(df["time_s"].max())
            lo, hi = default_steady_window(t_min, t_max)
            self._set_steady_window_seconds(lo, hi)

        self._save_processed_btn.setEnabled(True)
        self._update_data_status_label()
        self._rebuild_plot()

        rows = stats.get("final_rows", len(df))
        msg = f"Preprocessing complete — {rows:,} rows ready for analysis."
        self._banner.show_message(msg, "success")
        self.status_message.emit(msg)

    @Slot()
    def _save_processed_csv(self) -> None:
        if self._df_processed is None:
            self._banner.show_message("Run preprocessing first.", "warning")
            return

        default_dir = str(Path(self._file_path).parent) if self._file_path else ""
        default_name = "processed_data.csv"
        if self._test_folder_path:
            proc_dir = Path(self._test_folder_path) / "processed"
            proc_dir.mkdir(parents=True, exist_ok=True)
            default_dir = str(proc_dir)
            default_name = "data.csv"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save processed CSV",
            str(Path(default_dir) / default_name),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            out = self._df_processed.copy()
            if "time_s" in out.columns:
                out["time_s"] = out["time_s"].round(3)
            if "time_ms" in out.columns:
                out["time_ms"] = (out["time_s"] * 1000.0).round(1)
            out.to_csv(path, index=False, float_format="%.3f")
            msg = f"Saved processed data to {Path(path).name}"
            self._banner.show_message(msg, "success")
            self.status_message.emit(msg)
        except Exception as exc:
            self._banner.show_message(f"Save failed: {exc}", "error")
            self.status_message.emit(f"Save failed: {exc}")

    @Slot()
    def _on_channel_toggled(self, _item: QListWidgetItem) -> None:
        # Template channels only apply when adding new plot panels.
        pass

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

        cfg_dict = cfg.to_config() if cfg else {}
        rate = (cfg_dict.get("settings") or {}).get("sample_rate_hz", 100)
        self._resample_hz.setValue(float(rate))
        channel_config = cfg_dict.get("channel_config") or cfg_dict.get("columns") or {}
        has_mapping = bool(channel_config)
        self._mapping_chk.setEnabled(has_mapping)
        if not has_mapping:
            self._mapping_chk.setChecked(False)

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
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if required:
                self._role_layout.addWidget(_form_row(label, combo, required=True))
            else:
                self._role_layout.addWidget(_form_row(label, combo, optional=True))
            self._role_combos[key] = combo

    # ------------------------------------------------------------------ slots: steady state

    @Slot()
    def _auto_detect(self) -> None:
        df = self._df_processed
        if df is None:
            self._banner.show_message("Run preprocessing on the Preprocessing tab first.", "warning")
            return
        config = self._build_analysis_config() or {}
        method = self._detect_method.currentText()
        config["preferred_method"] = method
        self._banner.show_message("Detecting steady state…", "info")
        self.status_message.emit("Detecting steady state…")
        worker = _DetectWorker(df, config)
        worker.signals.detected.connect(self._on_detected)
        worker.signals.failed.connect(self._on_detect_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_detect_failed(self, error: str) -> None:
        self._banner.show_message(error, "warning")
        self.status_message.emit(error)

    @Slot(float, float, str)
    def _on_detected(self, start: float, end: float, method: str) -> None:
        self._set_steady_window_seconds(start, end)
        msg = f"Steady state: {start:.3f}–{end:.3f} s  ({end - start:.2f} s, method: {method})"
        self._banner.show_message(msg, "success")
        self.status_message.emit(msg)

    # ------------------------------------------------------------------ slots: run

    @Slot()
    def _run_analysis(self) -> None:
        if self._df_processed is None:
            self._banner.show_message(
                "Run preprocessing on the Preprocessing tab before analysis.",
                "warning",
            )
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

        self._banner.show_message("Running analysis…", "info")
        self.status_message.emit(f"Running analysis for {test_id}…")
        self._run_btn.setEnabled(False)
        self._results.hide()

        worker = _AnalysisWorker(
            df=self._df_processed,
            config=config,
            steady_s=steady,
            test_type=test_type,
            test_id=test_id,
            file_path=self._file_path,
        )
        worker.signals.finished.connect(self._on_analysis_done)
        worker.signals.failed.connect(self._on_analysis_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_analysis_done(self, result) -> None:
        self._run_btn.setEnabled(True)
        n = len(result.measurements)
        status = "passed" if result.passed_qc else "FAILED QC"
        msg = f"Analysis complete — {n} metrics, QC {status}."
        self._banner.show_message(
            msg,
            "success" if result.passed_qc else "error",
        )
        self.status_message.emit(msg)
        self._results.populate(result)
        self._results.show()

    @Slot(str)
    def _on_analysis_failed(self, error: str) -> None:
        self._run_btn.setEnabled(True)
        self._banner.show_message(f"Analysis failed: {error}", "error")
        self.status_message.emit(f"Analysis failed: {error}")

    # ------------------------------------------------------------------ public API

    def load_test_from_path(self, folder_path: str) -> None:
        """Load a test folder from Test Explorer (CSV + metadata prefill)."""
        from core.test_metadata import find_raw_data_file, load_test_from_folder

        folder = Path(folder_path)
        if not folder.is_dir():
            self._banner.show_message(f"Test folder not found: {folder_path}", "error")
            self.status_message.emit(f"Test folder not found: {folder_path}")
            return

        test_data = load_test_from_folder(folder)
        csv_path = test_data.get("raw_data_file")
        if not csv_path:
            csv_path_obj = find_raw_data_file(folder)
            csv_path = str(csv_path_obj) if csv_path_obj else None

        if not csv_path:
            self._banner.show_message(
                f"No CSV found in {folder.name} — check raw_data/ or upload manually.",
                "warning",
            )
            self.status_message.emit(f"No CSV in {folder.name}")
            metadata = test_data.get("metadata") or {}
            test_id = metadata.get("test_id") or folder.name
            self._test_id_edit.setText(test_id)
            if metadata.get("operator"):
                self._operator_edit.setText(str(metadata["operator"]))
            self._test_folder_path = str(folder)
            self._select_config_for_metadata(metadata)
            return

        self._test_folder_path = str(folder)
        metadata = test_data.get("metadata") or {}
        test_id = metadata.get("test_id") or folder.name
        self._test_id_edit.setText(test_id)
        if metadata.get("operator"):
            self._operator_edit.setText(str(metadata["operator"]))
        self._select_config_for_metadata(metadata)
        self._load_csv_path(csv_path)

    def set_active_config(self, config_id: str) -> None:
        """Select a saved configuration (from Configurations page handoff)."""
        if not config_id:
            return
        self._pending_config_id = config_id
        if self._config_combo.count() == 0:
            self._reload_config_list()
        self._apply_config_selection(config_id)

    # ------------------------------------------------------------------ lifecycle

    def on_context_changed(self) -> None:
        self._reload_config_list()
        if self._pending_config_id:
            self._apply_config_selection(self._pending_config_id)

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
        if self._pending_config_id:
            self._apply_config_selection(self._pending_config_id)
        elif self._config_combo.count() > 0:
            self._on_config_changed(self._config_combo.currentIndex())

    def _apply_config_selection(self, config_id: str) -> None:
        for i in range(self._config_combo.count()):
            if self._config_combo.itemData(i) == config_id:
                self._config_combo.setCurrentIndex(i)
                self._on_config_changed(i)
                return

    def _select_config_for_metadata(self, metadata: dict) -> None:
        """Pick a saved config matching test metadata test_type."""
        test_type = normalize_test_type(str(metadata.get("test_type", "")))
        if self._config_combo.count() == 0:
            self._reload_config_list()

        preferred = metadata.get("active_config") or metadata.get("config_name")
        if preferred:
            self._apply_config_selection(str(preferred))

        if self._config_id:
            cfg = self._load_config_obj()
            if cfg and normalize_test_type(cfg.test_type) == test_type:
                return

        for i in range(self._config_combo.count()):
            tid = self._config_combo.itemData(i)
            cfg = self._load_config_obj_by_id(tid) if tid else None
            if cfg and normalize_test_type(cfg.test_type) == test_type:
                self._config_combo.setCurrentIndex(i)
                return

    def _load_config_obj_by_id(self, config_id: str):
        if not config_id:
            return None
        try:
            from core.saved_configs import SavedConfigManager
            mgr = SavedConfigManager(str(Path(__file__).resolve().parents[3] / "saved_configs"))
            return mgr.get_template(config_id)
        except Exception:
            return None

    def _apply_test_folder_metadata(self, folder_path: str) -> None:
        """After CSV load, auto-map sensor roles from the active config."""
        config = self._build_analysis_config()
        if not config:
            return
        roles = config.get("sensor_roles") or config.get("columns") or {}
        col_set = set(self._numeric_cols)
        for role, sensor in roles.items():
            if role not in self._role_combos:
                continue
            if sensor in col_set:
                idx = self._role_combos[role].findText(sensor)
                if idx >= 0:
                    self._role_combos[role].setCurrentIndex(idx)

        channel_config = config.get("channel_config") or {}
        for ch_id, sensor in channel_config.items():
            if ch_id in col_set:
                for role, combo in self._role_combos.items():
                    if combo.currentText():
                        continue
                    if sensor in col_set:
                        idx = combo.findText(sensor)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                            break

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._config_combo.count() == 0:
            self._reload_config_list()
        if self._pending_config_id:
            self._apply_config_selection(self._pending_config_id)
