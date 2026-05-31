"""Campaign Analysis page — SPC, summary, and reports via core.campaign_manager_v2 + core.spc."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pyqtgraph as pg
    from hda.ui.style import PLOT_BG, PLOT_FG, configure_pyqtgraph

    configure_pyqtgraph()
    _PG_OK = True
except Exception:
    _PG_OK = False

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.campaign_manager_v2 import (
    create_campaign,
    get_available_campaigns,
    get_campaign_data,
    get_campaign_info,
)
from core.export import export_campaign_csv, export_campaign_excel
from core.reporting import generate_campaign_report
from core.spc import (
    SPCAnalysis,
    create_imr_chart,
    create_xbar_r_chart,
    format_spc_summary,
)

from hda.campaign_helpers import (
    campaign_overview_stats,
    campaign_type_from_info,
    filter_campaign_df,
    metric_columns,
    primary_metric_for_type,
    summary_display_columns,
)
from hda.ui.pages.base import BasePage, InfoBanner, MetricCard
from hda.ui.style import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BORDER,
    CONTENT_SECONDARY_BG,
    PLOT_BG,
    PLOT_FG,
    SZ_SM,
    SZ_XS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def _secondary_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setProperty("secondary", "true")
    return btn


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class _Sigs(QObject):
    campaigns_ready = Signal(list)
    loaded = Signal(object, object)   # info dict, DataFrame
    spc_ready = Signal(object)       # SPCAnalysis or (SPCAnalysis, SPCAnalysis)
    file_saved = Signal(str)
    failed = Signal(str)


class _CampaignListWorker(QRunnable):
    def __init__(self) -> None:
        super().__init__()
        self.signals = _Sigs()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.campaigns_ready.emit(get_available_campaigns())
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _CampaignLoadWorker(QRunnable):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._name = name
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            info = get_campaign_info(self._name) or {}
            df = get_campaign_data(self._name)
            self.signals.loaded.emit(info, df)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _SPCWorker(QRunnable):
    def __init__(
        self,
        df,
        parameter: str,
        chart_type: str,
        usl: Optional[float],
        lsl: Optional[float],
        target: Optional[float],
        subgroup_size: int = 5,
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._df = df
        self._parameter = parameter
        self._chart_type = chart_type
        self._usl = usl
        self._lsl = lsl
        self._target = target
        self._subgroup_size = subgroup_size
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            test_id_col = "test_id" if "test_id" in self._df.columns else self._df.columns[0]
            if self._chart_type == "X-bar/R":
                xbar, _r = create_xbar_r_chart(
                    self._df,
                    parameter=self._parameter,
                    subgroup_size=self._subgroup_size,
                    test_id_col=test_id_col,
                    usl=self._usl,
                    lsl=self._lsl,
                    target=self._target,
                )
                self.signals.spc_ready.emit(xbar)
            else:
                analysis = create_imr_chart(
                    self._df,
                    parameter=self._parameter,
                    test_id_col=test_id_col,
                    usl=self._usl,
                    lsl=self._lsl,
                    target=self._target,
                )
                self.signals.spc_ready.emit(analysis)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _ReportWorker(QRunnable):
    def __init__(
        self,
        campaign_name: str,
        df,
        parameters: List[str],
        output_path: str,
        mode: str,
        info: Dict[str, Any],
        spc_analysis: Optional[SPCAnalysis] = None,
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._campaign_name = campaign_name
        self._df = df
        self._parameters = parameters
        self._output_path = output_path
        self._mode = mode
        self._info = info
        self._spc = spc_analysis
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            if self._mode == "html":
                spc_map = {}
                if self._spc is not None and self._spc.parameter_name:
                    spc_map[self._spc.parameter_name] = self._spc
                html = generate_campaign_report(
                    self._campaign_name,
                    self._df,
                    self._parameters,
                    spc_analyses=spc_map or None,
                )
                Path(self._output_path).write_text(html, encoding="utf-8")
            elif self._mode == "excel":
                spc_map = {}
                if self._spc is not None:
                    spc_map[self._spc.parameter_name] = self._spc
                export_campaign_excel(
                    self._df,
                    self._output_path,
                    campaign_info=self._info,
                    spc_summary=spc_map or None,
                )
            else:
                export_campaign_csv(self._df, self._output_path, campaign_info=self._info)
            self.signals.file_saved.emit(self._output_path)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# SPC chart widget
# ---------------------------------------------------------------------------

class _SPCChartWidget(QWidget):
    """pyqtgraph control chart with CL / UCL / LCL lines."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if _PG_OK:
            self._plot = pg.PlotWidget()
            self._plot.setBackground(PLOT_BG)
            self._plot.showGrid(x=True, y=True, alpha=0.35)
            self._plot.getAxis("bottom").setPen(PLOT_FG)
            self._plot.getAxis("left").setPen(PLOT_FG)
            self._plot.getAxis("bottom").setTextPen(PLOT_FG)
            self._plot.getAxis("left").setTextPen(PLOT_FG)
            self._plot.setLabel("bottom", "Test index")
            lay.addWidget(self._plot, 1)
        else:
            self._plot = None
            lbl = QLabel("Install pyqtgraph to view control charts.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {TEXT_MUTED};")
            lay.addWidget(lbl, 1)

    def render(self, analysis: SPCAnalysis) -> None:
        if not _PG_OK or self._plot is None:
            return
        self._plot.clear()
        xs = np.array([p.index for p in analysis.points], dtype=float)
        ys = np.array([p.value for p in analysis.points], dtype=float)

        in_x, in_y, out_x, out_y = [], [], [], []
        for p in analysis.points:
            if p.in_control:
                in_x.append(p.index)
                in_y.append(p.value)
            else:
                out_x.append(p.index)
                out_y.append(p.value)

        if in_x:
            self._plot.plot(
                in_x, in_y, pen=pg.mkPen(ACCENT_BLUE, width=1),
                symbol="o", symbolBrush=ACCENT_BLUE, symbolSize=7,
                name="In control",
            )
        if out_x:
            self._plot.plot(
                out_x, out_y, pen=None,
                symbol="x", symbolBrush=ACCENT_RED, symbolPen=ACCENT_RED, symbolSize=10,
                name="Violation",
            )

        lim = analysis.limits
        for val, label, color in (
            (lim.center_line, "CL", "#16a34a"),
            (lim.ucl, "UCL", ACCENT_RED),
            (lim.lcl, "LCL", ACCENT_RED),
        ):
            line = pg.InfiniteLine(
                pos=val, angle=0,
                pen=pg.mkPen(color=color, width=1, style=Qt.DashLine),
                label=label,
            )
            self._plot.addItem(line)

        if analysis.capability:
            cap = analysis.capability
            for val, label in ((cap.usl, "USL"), (cap.lsl, "LSL"), (cap.target, "Target")):
                if val is not None:
                    self._plot.addItem(pg.InfiniteLine(
                        pos=val, angle=0,
                        pen=pg.mkPen(color="#d97706", width=1, style=Qt.DotLine),
                        label=label,
                    ))

        title = f"{analysis.parameter_name} — {analysis.chart_type.value}"
        self._plot.setTitle(title)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

class CampaignAnalysisPage(BasePage):
    """Campaign-level SPC, summary table, and report export."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "Campaign Analysis",
            "SPC control charts, capability indices, and trend analysis across a campaign",
            parent=parent,
        )

        self._campaigns: List[Dict[str, Any]] = []
        self._campaign_name: str = ""
        self._info: Dict[str, Any] = {}
        self._df = None
        self._filtered_df = None
        self._last_spc: Optional[SPCAnalysis] = None
        self._pending_select_campaign: str = ""
        self._opened_from_sta_name: str = ""

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        toolbar.addWidget(QLabel("Campaign"))
        self._campaign_combo = QComboBox()
        self._campaign_combo.setMinimumWidth(220)
        self._campaign_combo.currentIndexChanged.connect(self._on_campaign_selected)
        toolbar.addWidget(self._campaign_combo, 1)

        refresh_btn = _secondary_btn("Refresh")
        refresh_btn.setToolTip("Reload campaign list (F5)")
        refresh_btn.clicked.connect(self.refresh_campaigns)
        toolbar.addWidget(refresh_btn)

        new_btn = _secondary_btn("+ New")
        new_btn.clicked.connect(self._create_campaign)
        toolbar.addWidget(new_btn)

        self.content_layout.addLayout(toolbar)

        self._banner = InfoBanner(parent=self)
        self.content_layout.addWidget(self._banner)

        # ── Metric cards ────────────────────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self._m_tests = MetricCard("Total Tests", "—")
        self._m_qc = MetricCard("QC Passed", "—")
        self._m_type = MetricCard("Type", "—")
        self._m_schema = MetricCard("Schema", "—")
        for card in (self._m_tests, self._m_qc, self._m_type, self._m_schema):
            metrics_row.addWidget(card)
        self.content_layout.addLayout(metrics_row)

        # ── Filters ───────────────────────────────────────────────────────
        filter_box = QGroupBox("Filters")
        filter_lay = QHBoxLayout(filter_box)
        self._part_combo = QComboBox()
        self._part_combo.setMinimumWidth(140)
        self._part_combo.currentIndexChanged.connect(self._apply_filters)
        self._serial_combo = QComboBox()
        self._serial_combo.setMinimumWidth(140)
        self._serial_combo.currentIndexChanged.connect(self._apply_filters)
        filter_lay.addWidget(QLabel("Part"))
        filter_lay.addWidget(self._part_combo)
        filter_lay.addWidget(QLabel("Serial"))
        filter_lay.addWidget(self._serial_combo)
        filter_lay.addStretch()
        self.content_layout.addWidget(filter_box)

        # ── Tabs ──────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self.content_layout.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_summary_tab(), "Summary")
        self._tabs.addTab(self._build_spc_tab(), "SPC Analysis")
        self._tabs.addTab(self._build_reports_tab(), "Reports")

        sc = QShortcut(QKeySequence("F5"), self)
        sc.activated.connect(self.refresh_campaigns)

        self.refresh_campaigns()

    # ------------------------------------------------------------------ tabs

    def _build_summary_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        self._summary_table = QTableWidget(0, 0)
        self._summary_table.setAlternatingRowColors(True)
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self._summary_table, 1)
        return w

    def _build_spc_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        controls = QWidget()
        controls.setFixedWidth(280)
        c_lay = QVBoxLayout(controls)
        c_lay.setContentsMargins(0, 8, 8, 0)
        c_lay.setSpacing(8)

        form = QFormLayout()
        self._chart_type = QComboBox()
        self._chart_type.addItems(["I-MR", "X-bar/R"])
        form.addRow("Chart", self._chart_type)

        self._param_combo = QComboBox()
        form.addRow("Parameter", self._param_combo)

        self._subgroup_spin = QComboBox()
        self._subgroup_spin.addItems([str(n) for n in range(2, 11)])
        self._subgroup_spin.setCurrentText("5")
        form.addRow("Subgroup (X-bar/R)", self._subgroup_spin)

        c_lay.addLayout(form)

        self._use_specs = QCheckBox("Use specification limits")
        c_lay.addWidget(self._use_specs)

        spec_form = QFormLayout()
        self._lsl_spin = QDoubleSpinBox()
        self._lsl_spin.setDecimals(6)
        self._lsl_spin.setRange(-1e9, 1e9)
        self._usl_spin = QDoubleSpinBox()
        self._usl_spin.setDecimals(6)
        self._usl_spin.setRange(-1e9, 1e9)
        self._target_spin = QDoubleSpinBox()
        self._target_spin.setDecimals(6)
        self._target_spin.setRange(-1e9, 1e9)
        for spin in (self._lsl_spin, self._usl_spin, self._target_spin):
            spin.setEnabled(False)
        self._use_specs.toggled.connect(self._toggle_specs)
        spec_form.addRow("LSL", self._lsl_spin)
        spec_form.addRow("USL", self._usl_spin)
        spec_form.addRow("Target", self._target_spin)
        c_lay.addLayout(spec_form)

        run_btn = QPushButton("Run SPC")
        run_btn.clicked.connect(self._run_spc)
        c_lay.addWidget(run_btn)

        self._spc_status = QLabel("")
        self._spc_status.setWordWrap(True)
        self._spc_status.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {SZ_SM}; background: transparent;"
        )
        c_lay.addWidget(self._spc_status)
        c_lay.addStretch()
        splitter.addWidget(controls)

        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 8, 0, 0)
        self._spc_chart = _SPCChartWidget()
        r_lay.addWidget(self._spc_chart, 2)

        self._spc_summary = QPlainTextEdit()
        self._spc_summary.setReadOnly(True)
        self._spc_summary.setMaximumHeight(160)
        self._spc_summary.setStyleSheet(
            f"font-size: {SZ_XS}; background: {CONTENT_SECONDARY_BG}; border: 1px solid {BORDER};"
        )
        r_lay.addWidget(self._spc_summary)

        self._violations_table = QTableWidget(0, 3)
        self._violations_table.setHorizontalHeaderLabels(["Test ID", "Value", "Violations"])
        self._violations_table.horizontalHeader().setStretchLastSection(True)
        self._violations_table.setMaximumHeight(140)
        r_lay.addWidget(self._violations_table)
        splitter.addWidget(right)
        splitter.setSizes([280, 900])
        return splitter

    def _build_reports_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 16, 0, 0)
        lay.setSpacing(12)

        info = QLabel(
            "Export campaign results with traceability metadata. "
            "HTML reports include summary statistics; Excel adds SPC sheet when available."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        lay.addWidget(info)

        btn_row = QHBoxLayout()
        html_btn = QPushButton("Generate HTML Report…")
        html_btn.clicked.connect(lambda: self._export_report("html"))
        excel_btn = _secondary_btn("Export Excel…")
        excel_btn.clicked.connect(lambda: self._export_report("excel"))
        csv_btn = _secondary_btn("Export CSV…")
        csv_btn.clicked.connect(lambda: self._export_report("csv"))
        btn_row.addWidget(html_btn)
        btn_row.addWidget(excel_btn)
        btn_row.addWidget(csv_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        lay.addStretch()
        return w

    # ------------------------------------------------------------------ data

    def refresh_campaigns(self) -> None:
        self.status_message.emit("Loading campaigns…")
        self._banner.show_message("Loading campaigns…", "info")
        worker = _CampaignListWorker()
        worker.signals.campaigns_ready.connect(self._on_campaigns_loaded)
        worker.signals.failed.connect(self._on_error)
        QThreadPool.globalInstance().start(worker)

    def mark_opened_from_sta(self, campaign_name: str) -> None:
        """Set context note for campaign opened from STA save handoff."""
        self._opened_from_sta_name = (campaign_name or "").strip()

    def select_campaign(self, campaign_name: str, refresh: bool = True) -> None:
        """Select a campaign in the combo (optionally refreshing list first)."""
        name = (campaign_name or "").strip()
        if not name:
            return
        self._pending_select_campaign = name
        if refresh:
            self.refresh_campaigns()
            return
        for i in range(self._campaign_combo.count()):
            if self._campaign_combo.itemData(i) == name:
                self._campaign_combo.setCurrentIndex(i)
                self._on_campaign_selected(i)
                self._pending_select_campaign = ""
                return

    @Slot(list)
    def _on_campaigns_loaded(self, campaigns: List[Dict[str, Any]]) -> None:
        self._campaigns = campaigns
        prev = self._campaign_name
        pending = self._pending_select_campaign
        self._campaign_combo.blockSignals(True)
        self._campaign_combo.clear()
        if not campaigns:
            self._campaign_combo.blockSignals(False)
            self._banner.show_message(
                "No campaigns found in campaigns/ — create one with + New.", "warning"
            )
            self.status_message.emit("No campaigns found.")
            self._clear_data()
            return

        for c in campaigns:
            name = c["name"]
            count = c.get("test_count", 0)
            self._campaign_combo.addItem(f"{name}  ({count} tests)", userData=name)

        self._campaign_combo.blockSignals(False)

        if pending:
            for i in range(self._campaign_combo.count()):
                if self._campaign_combo.itemData(i) == pending:
                    self._campaign_combo.setCurrentIndex(i)
                    self._pending_select_campaign = ""
                    return

        if prev:
            for i in range(self._campaign_combo.count()):
                if self._campaign_combo.itemData(i) == prev:
                    self._campaign_combo.setCurrentIndex(i)
                    return
        self._campaign_combo.setCurrentIndex(0)

    @Slot()
    def _on_campaign_selected(self, index: int) -> None:
        if index < 0:
            return
        name = self._campaign_combo.itemData(index)
        if not name:
            return
        self._load_campaign(name)

    def _load_campaign(self, name: str) -> None:
        self._campaign_name = name
        self.status_message.emit(f"Loading campaign {name}…")
        self._banner.show_message(f"Loading {name}…", "info")
        worker = _CampaignLoadWorker(name)
        worker.signals.loaded.connect(self._on_campaign_loaded)
        worker.signals.failed.connect(self._on_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object)
    def _on_campaign_loaded(self, info: Dict[str, Any], df) -> None:
        self._info = info or {}
        self._df = df
        self._populate_filters()
        self._apply_filters()
        stats = campaign_overview_stats(self._filtered_df, self._info)
        self._m_tests.set_value(stats["tests"])
        self._m_qc.set_value(stats["qc_passed"])
        self._m_type.set_value(stats["type"])
        self._m_schema.set_value(stats["schema"])

        n = len(self._filtered_df) if self._filtered_df is not None else 0
        msg = f"Loaded {self._campaign_name} — {n} test(s)."
        if self._opened_from_sta_name and self._opened_from_sta_name == self._campaign_name:
            msg = f"{msg} Opened from STA save."
            self._opened_from_sta_name = ""
        self._banner.show_message(msg, "success" if n else "warning")
        self.status_message.emit(msg)

        self._populate_param_combo()
        self._last_spc = None
        self._spc_summary.clear()
        self._violations_table.setRowCount(0)
        if _PG_OK and self._spc_chart._plot is not None:
            self._spc_chart._plot.clear()

    def _clear_data(self) -> None:
        self._df = None
        self._filtered_df = None
        self._summary_table.setRowCount(0)
        self._summary_table.setColumnCount(0)
        for card in (self._m_tests, self._m_qc, self._m_type, self._m_schema):
            card.set_value("—")

    def _populate_filters(self) -> None:
        self._part_combo.blockSignals(True)
        self._serial_combo.blockSignals(True)
        self._part_combo.clear()
        self._serial_combo.clear()
        self._part_combo.addItem("All parts", userData=None)
        self._serial_combo.addItem("All serials", userData=None)

        if self._df is not None and len(self._df):
            if "part" in self._df.columns:
                for p in sorted(self._df["part"].dropna().unique()):
                    self._part_combo.addItem(str(p), userData=str(p))
            if "serial_num" in self._df.columns:
                for s in sorted(self._df["serial_num"].dropna().unique()):
                    self._serial_combo.addItem(str(s), userData=str(s))

        self._part_combo.blockSignals(False)
        self._serial_combo.blockSignals(False)

    @Slot()
    def _apply_filters(self) -> None:
        if self._df is None:
            self._filtered_df = None
            self._refresh_summary_table()
            return

        part = self._part_combo.currentData()
        serial = self._serial_combo.currentData()
        parts = [part] if part else None
        serials = [serial] if serial else None
        self._filtered_df = filter_campaign_df(self._df, parts, serials)
        self._refresh_summary_table()
        stats = campaign_overview_stats(self._filtered_df, self._info)
        self._m_tests.set_value(stats["tests"])
        self._m_qc.set_value(stats["qc_passed"])
        self._populate_param_combo()

    def _refresh_summary_table(self) -> None:
        df = self._filtered_df
        if df is None or df.empty:
            self._summary_table.setRowCount(0)
            self._summary_table.setColumnCount(0)
            return

        cols = summary_display_columns(df)
        if not cols:
            cols = list(df.columns[: min(12, len(df.columns))])

        self._summary_table.setColumnCount(len(cols))
        self._summary_table.setHorizontalHeaderLabels(cols)
        self._summary_table.setRowCount(len(df))

        for row in range(len(df)):
            for col_idx, col_name in enumerate(cols):
                val = df.iloc[row][col_name]
                text = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)
                item = QTableWidgetItem(text)
                if col_name == "qc_passed":
                    passed = bool(val) if val is not None else False
                    item.setForeground(
                        QBrush(QColor(ACCENT_GREEN if passed else ACCENT_RED))
                    )
                self._summary_table.setItem(row, col_idx, item)

    def _populate_param_combo(self) -> None:
        self._param_combo.blockSignals(True)
        self._param_combo.clear()
        df = self._filtered_df
        if df is not None and len(df):
            cols = metric_columns(df)
            ctype = campaign_type_from_info(self._info)
            primary = primary_metric_for_type(ctype)
            for c in cols:
                self._param_combo.addItem(c)
            idx = self._param_combo.findText(primary)
            if idx >= 0:
                self._param_combo.setCurrentIndex(idx)
            elif cols:
                mean = float(df[cols[0]].mean())
                span = float(df[cols[0]].max() - df[cols[0]].min())
                self._lsl_spin.setValue(mean - span)
                self._usl_spin.setValue(mean + span)
                self._target_spin.setValue(mean)
        self._param_combo.blockSignals(False)

    # ------------------------------------------------------------------ SPC

    @Slot(bool)
    def _toggle_specs(self, enabled: bool) -> None:
        for spin in (self._lsl_spin, self._usl_spin, self._target_spin):
            spin.setEnabled(enabled)

    @Slot()
    def _run_spc(self) -> None:
        df = self._filtered_df
        if df is None or len(df) < 2:
            self._banner.show_message("Need at least 2 tests for SPC.", "warning")
            return

        param = self._param_combo.currentText()
        if not param:
            self._banner.show_message("Select a parameter.", "warning")
            return

        chart = self._chart_type.currentText()
        usl = lsl = target = None
        if self._use_specs.isChecked():
            lsl = self._lsl_spin.value()
            usl = self._usl_spin.value()
            target = self._target_spin.value()

        subgroup = int(self._subgroup_spin.currentText())
        self.status_message.emit(f"Running {chart} on {param}…")
        self._spc_status.setText("Computing…")

        worker = _SPCWorker(df, param, chart, usl, lsl, target, subgroup)
        worker.signals.spc_ready.connect(self._on_spc_ready)
        worker.signals.failed.connect(self._on_spc_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_spc_ready(self, analysis: SPCAnalysis) -> None:
        self._last_spc = analysis
        self._spc_chart.render(analysis)
        self._spc_summary.setPlainText(format_spc_summary(analysis))

        ooc = analysis.get_out_of_control_points()
        self._violations_table.setRowCount(len(ooc))
        for row, pt in enumerate(ooc):
            viols = ", ".join(v.value for v in pt.violations)
            self._violations_table.setItem(row, 0, QTableWidgetItem(pt.test_id))
            self._violations_table.setItem(row, 1, QTableWidgetItem(f"{pt.value:.4g}"))
            self._violations_table.setItem(row, 2, QTableWidgetItem(viols))

        status = "In control" if analysis.n_violations == 0 else f"{analysis.n_violations} violation(s)"
        color = ACCENT_GREEN if analysis.n_violations == 0 else ACCENT_RED
        self._spc_status.setText(status)
        self._spc_status.setStyleSheet(
            f"color: {color}; font-size: {SZ_SM}; font-weight: 600; background: transparent;"
        )
        cpk = ""
        if analysis.capability and analysis.capability.cpk is not None:
            cpk = f", Cpk={analysis.capability.cpk:.2f}"
        self.status_message.emit(f"SPC complete — {status}{cpk}")

    @Slot(str)
    def _on_spc_error(self, error: str) -> None:
        self._banner.show_message(f"SPC failed: {error}", "error")
        self._spc_status.setText(error)
        self.status_message.emit(f"SPC failed: {error}")

    # ------------------------------------------------------------------ reports

    def _export_report(self, mode: str) -> None:
        df = self._filtered_df
        if df is None or df.empty:
            self._banner.show_message("No campaign data to export.", "warning")
            return

        if mode == "html":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save HTML Report",
                str(Path.home() / f"{self._campaign_name}_report.html"),
                "HTML files (*.html)",
            )
        elif mode == "excel":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Excel",
                str(Path.home() / f"{self._campaign_name}.xlsx"),
                "Excel files (*.xlsx)",
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV",
                str(Path.home() / f"{self._campaign_name}.csv"),
                "CSV files (*.csv)",
            )

        if not path:
            return

        params = metric_columns(df)[:6] or [self._param_combo.currentText()]
        params = [p for p in params if p]

        self.status_message.emit(f"Exporting {mode.upper()}…")
        worker = _ReportWorker(
            self._campaign_name, df, params, path, mode, self._info, self._last_spc
        )
        worker.signals.file_saved.connect(self._on_file_saved)
        worker.signals.failed.connect(self._on_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_file_saved(self, path: str) -> None:
        name = Path(path).name
        self._banner.show_message(f"Saved {name}", "success")
        self.status_message.emit(f"Saved {name}")

    # ------------------------------------------------------------------ misc

    def _create_campaign(self) -> None:
        name, ok = QInputDialog.getText(self, "New Campaign", "Campaign name:")
        name = name.strip()
        if not ok or not name:
            return
        ctype, ok2 = QInputDialog.getItem(
            self, "Campaign Type", "Test type:",
            ["cold_flow", "hot_fire"], 0, False,
        )
        if not ok2:
            return
        try:
            create_campaign(name, ctype)
            self._banner.show_message(f"Created campaign {name}", "success")
            self.status_message.emit(f"Created campaign {name}")
            self.refresh_campaigns()
        except Exception as exc:
            self._banner.show_message(f"Create failed: {exc}", "error")

    @Slot(str)
    def _on_error(self, error: str) -> None:
        self._banner.show_message(error, "error")
        self.status_message.emit(error)

    def on_context_changed(self) -> None:
        """Context changes do not affect campaign DB selection."""
        pass
