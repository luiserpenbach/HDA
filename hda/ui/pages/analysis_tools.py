"""Analysis Tools page — anomaly detection, comparison, transient, frequency, envelope."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pyqtgraph as pg
    from hda.ui.style import PLOT_BG, PLOT_FG, configure_pyqtgraph

    configure_pyqtgraph()
    _PG_OK = True
except Exception:
    _PG_OK = False

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.advanced_anomaly import (
    AnomalySeverity,
    format_anomaly_table,
    run_anomaly_detection,
)
from core.campaign_manager_v2 import get_available_campaigns, get_campaign_data
from core.comparison import (
    GoldenReference,
    calculate_correlation_matrix,
    compare_campaigns,
    compare_tests,
    compare_to_golden,
    create_golden_from_campaign,
    format_campaign_comparison,
    linear_regression,
)
from core.frequency_analysis import (
    compute_power_spectral_density,
    detect_harmonics,
    detect_resonance,
)
from core.operating_envelope import calculate_operating_envelope
from core.transient_analysis import (
    TestPhase,
    analyze_shutdown_transient,
    analyze_startup_transient,
    segment_test_phases,
)

from hda.analysis_tools_helpers import (
    detect_time_column,
    metric_columns,
    numeric_columns,
    populate_table,
)
from hda.ui.pages.base import BasePage, InfoBanner, MetricCard
from hda.ui.style import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BORDER,
    PLOT_BG,
    PLOT_FG,
    SZ_SM,
    SZ_XS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_PHASE_LABELS = {
    TestPhase.PRETEST: "Pre-test",
    TestPhase.STARTUP: "Startup",
    TestPhase.TRANSIENT: "Transient",
    TestPhase.STEADY_STATE: "Steady State",
    TestPhase.SHUTDOWN: "Shutdown",
    TestPhase.COOLDOWN: "Cooldown",
}

_PHASE_COLORS = {
    TestPhase.PRETEST: "#a1a1aa",
    TestPhase.STARTUP: "#2563eb",
    TestPhase.TRANSIENT: "#ca8a04",
    TestPhase.STEADY_STATE: "#16a34a",
    TestPhase.SHUTDOWN: "#dc2626",
    TestPhase.COOLDOWN: "#71717a",
}

_ANOMALY_SEVERITY_RGBA = {
    AnomalySeverity.CRITICAL: (220, 38, 38, 60),
    AnomalySeverity.WARNING: (202, 138, 4, 60),
    AnomalySeverity.INFO: (37, 99, 235, 40),
}


def _secondary_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setProperty("secondary", "true")
    return btn


def _checked_items(list_widget: QListWidget) -> List[str]:
    out: List[str] = []
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.checkState() == Qt.Checked:
            out.append(item.text())
    return out


def _set_checklist(list_widget: QListWidget, items: Sequence[str], default_n: int = 5) -> None:
    list_widget.clear()
    for i, name in enumerate(items):
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if i < default_n else Qt.Unchecked)
        list_widget.addItem(item)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class _Sigs(QObject):
    anomaly_ready = Signal(object, object)
    transient_ready = Signal(object, object, object, object)
    psd_ready = Signal(object)
    harmonics_ready = Signal(object, object)
    resonance_ready = Signal(object)
    envelope_ready = Signal(object)
    failed = Signal(str)


class _AnomalyWorker(QRunnable):
    def __init__(
        self,
        df: pd.DataFrame,
        channels: List[str],
        time_col: str,
        sample_rate: float,
        correlation_pairs: Optional[List[Tuple[str, str]]],
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._df = df
        self._channels = channels
        self._time_col = time_col
        self._sample_rate = sample_rate
        self._pairs = correlation_pairs
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            report = run_anomaly_detection(
                df=self._df,
                channels=self._channels,
                timestamp_col=self._time_col,
                sample_rate_hz=self._sample_rate,
                correlation_pairs=self._pairs,
            )
            self.signals.anomaly_ready.emit(report, self._df)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _TransientWorker(QRunnable):
    def __init__(
        self,
        df: pd.DataFrame,
        signal_col: str,
        time_col: str,
        threshold_pct: float,
        cv_threshold: float,
        min_phase_s: float,
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._df = df
        self._signal_col = signal_col
        self._time_col = time_col
        self._threshold_pct = threshold_pct
        self._cv_threshold = cv_threshold
        self._min_phase_s = min_phase_s
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            multi = segment_test_phases(
                self._df,
                signal_col=self._signal_col,
                time_col=self._time_col,
                threshold_pct=self._threshold_pct,
                cv_threshold=self._cv_threshold,
                min_phase_duration_s=self._min_phase_s,
            )
            startup_metrics = None
            shutdown_metrics = None
            phase_types = [p.phase for p in multi.phases]

            if TestPhase.STARTUP in phase_types:
                startup_phase = next(p for p in multi.phases if p.phase == TestPhase.STARTUP)
                mask = (
                    (self._df[self._time_col] >= startup_phase.start_ms / 1000.0)
                    & (self._df[self._time_col] <= startup_phase.end_ms / 1000.0)
                )
                sub = self._df.loc[mask]
                if len(sub) > 2:
                    ss_phases = [p for p in multi.phases if p.phase == TestPhase.STEADY_STATE]
                    ss_val = (
                        ss_phases[0].metrics.get("mean")
                        if ss_phases and ss_phases[0].metrics
                        else None
                    )
                    startup_metrics = analyze_startup_transient(
                        sub, self._signal_col, time_col=self._time_col, steady_value=ss_val
                    )

            if TestPhase.SHUTDOWN in phase_types:
                shutdown_phase = next(p for p in multi.phases if p.phase == TestPhase.SHUTDOWN)
                mask = (
                    (self._df[self._time_col] >= shutdown_phase.start_ms / 1000.0)
                    & (self._df[self._time_col] <= shutdown_phase.end_ms / 1000.0)
                )
                sub = self._df.loc[mask]
                if len(sub) > 2:
                    ss_phases = [p for p in multi.phases if p.phase == TestPhase.STEADY_STATE]
                    ss_val = (
                        ss_phases[0].metrics.get("mean")
                        if ss_phases and ss_phases[0].metrics
                        else None
                    )
                    shutdown_metrics = analyze_shutdown_transient(
                        sub, self._signal_col, time_col=self._time_col, steady_value=ss_val
                    )

            self.signals.transient_ready.emit(
                multi, self._df, startup_metrics, shutdown_metrics
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _PSDWorker(QRunnable):
    def __init__(
        self,
        signal: np.ndarray,
        sample_rate: float,
        method: str,
        window: str,
        nperseg: int,
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._signal = signal
        self._sample_rate = sample_rate
        self._method = method
        self._window = window
        self._nperseg = nperseg
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            result = compute_power_spectral_density(
                self._signal,
                sample_rate_hz=self._sample_rate,
                method=self._method,
                nperseg=min(self._nperseg, len(self._signal)),
                window=self._window,
            )
            self.signals.psd_ready.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _HarmonicsWorker(QRunnable):
    def __init__(
        self,
        signal: np.ndarray,
        sample_rate: float,
        method: str,
        window: str,
        nperseg: int,
        n_harmonics: int,
        tolerance_hz: float,
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._signal = signal
        self._sample_rate = sample_rate
        self._method = method
        self._window = window
        self._nperseg = nperseg
        self._n_harmonics = n_harmonics
        self._tolerance_hz = tolerance_hz
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            psd = compute_power_spectral_density(
                self._signal,
                sample_rate_hz=self._sample_rate,
                method=self._method,
                nperseg=min(self._nperseg, len(self._signal)),
                window=self._window,
            )
            harmonics = detect_harmonics(
                psd, n_harmonics=self._n_harmonics, tolerance_hz=self._tolerance_hz
            )
            self.signals.harmonics_ready.emit(psd, harmonics)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class _ResonanceWorker(QRunnable):
    def __init__(
        self,
        signal: np.ndarray,
        sample_rate: float,
        method: str,
        window: str,
        nperseg: int,
        prominence: float,
    ) -> None:
        super().__init__()
        self.signals = _Sigs()
        self._signal = signal
        self._sample_rate = sample_rate
        self._method = method
        self._window = window
        self._nperseg = nperseg
        self._prominence = prominence
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            psd = compute_power_spectral_density(
                self._signal,
                sample_rate_hz=self._sample_rate,
                method=self._method,
                nperseg=min(self._nperseg, len(self._signal)),
                window=self._window,
            )
            resonances = detect_resonance(psd, prominence=self._prominence)
            self.signals.resonance_ready.emit(resonances)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Plot widget
# ---------------------------------------------------------------------------

class _LinePlotWidget(QWidget):
    def __init__(self, x_label: str = "X", y_label: str = "Y", parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if _PG_OK:
            self.plot = pg.PlotWidget()
            self.plot.setBackground(PLOT_BG)
            self.plot.showGrid(x=True, y=True, alpha=0.35)
            self.plot.getAxis("bottom").setPen(PLOT_FG)
            self.plot.getAxis("left").setPen(PLOT_FG)
            self.plot.getAxis("bottom").setTextPen(PLOT_FG)
            self.plot.getAxis("left").setTextPen(PLOT_FG)
            self.plot.setLabel("bottom", x_label)
            self.plot.setLabel("left", y_label)
            lay.addWidget(self.plot, 1)
        else:
            self.plot = None
            lay.addWidget(QLabel("Install pyqtgraph to view charts."), 1)

    def clear(self) -> None:
        if self.plot is not None:
            self.plot.clear()

    def plot_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        name: str = "",
        color: str = ACCENT_BLUE,
        log_y: bool = False,
    ) -> None:
        if self.plot is None:
            return
        self.plot.clear()
        self.plot.plot(x, y, pen=pg.mkPen(color=color, width=1.6), name=name)
        if log_y:
            self.plot.setLogMode(y=True)
        else:
            self.plot.setLogMode(y=False)


class _ScatterPlotWidget(_LinePlotWidget):
    def plot_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        names: Optional[List[str]] = None,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        if self.plot is None:
            return
        self.plot.clear()
        brush = ACCENT_BLUE
        if colors is not None:
            for xi, yi, ok in zip(x, y, colors):
                c = ACCENT_GREEN if ok else ACCENT_RED
                self.plot.plot(
                    [xi], [yi], pen=None, symbol="o", symbolBrush=c, symbolSize=9,
                )
        else:
            self.plot.plot(
                x, y, pen=None, symbol="o", symbolBrush=brush, symbolSize=8,
            )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

class AnalysisToolsPage(BasePage):
    """Advanced analysis: anomalies, comparison, transients, frequency, envelope."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "Analysis Tools",
            "Anomaly detection, test comparison, transient characterization, "
            "frequency analysis, and operating envelope visualization.",
            parent=parent,
        )

        self._csv_df: Optional[pd.DataFrame] = None
        self._csv_path: str = ""
        self._campaigns: List[Dict[str, Any]] = []
        self._golden: Optional[GoldenReference] = None
        self._last_anomaly_df: Optional[pd.DataFrame] = None
        self._last_anomaly_report = None
        self._last_transient_df: Optional[pd.DataFrame] = None
        self._freq_df: Optional[pd.DataFrame] = None

        self._banner = InfoBanner(parent=self)
        self.content_layout.addWidget(self._banner)

        self._tabs = QTabWidget()
        self.content_layout.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_anomaly_tab(), "Anomaly Detection")
        self._tabs.addTab(self._build_comparison_tab(), "Data Comparison")
        self._tabs.addTab(self._build_transient_tab(), "Transient Analysis")
        self._tabs.addTab(self._build_frequency_tab(), "Frequency Analysis")
        self._tabs.addTab(self._build_envelope_tab(), "Operating Envelope")

        sc = QShortcut(QKeySequence("F5"), self)
        sc.activated.connect(self.refresh_campaigns)

        self.refresh_campaigns()

    def on_context_changed(self) -> None:
        self.refresh_campaigns()

    # ------------------------------------------------------------------ shared

    def refresh_campaigns(self) -> None:
        try:
            self._campaigns = get_available_campaigns()
        except Exception as exc:
            self._campaigns = []
            self._banner.show_message(f"Could not load campaigns: {exc}", "warning")
            return

        names = [c["name"] for c in self._campaigns]
        for combo in (
            self._cmp_campaign,
            self._cmp_campaign_b,
            self._oe_campaign,
        ):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        if names and self._cmp_campaign.currentIndex() < 0:
            self._cmp_campaign.setCurrentIndex(0)
        if len(names) > 1 and self._cmp_campaign_b.currentIndex() < 0:
            self._cmp_campaign_b.setCurrentIndex(1)
        if names and self._oe_campaign.currentIndex() < 0:
            self._oe_campaign.setCurrentIndex(0)
            self._on_envelope_campaign_changed()

    def _browse_csv(self, on_loaded) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self._csv_path = path
            on_loaded(df, Path(path).name)
            self.status_message.emit(f"Loaded {Path(path).name} — {len(df):,} rows.")
        except Exception as exc:
            self._banner.show_message(f"Load failed: {exc}", "error")

    def _csv_info_label(self) -> QLabel:
        lbl = QLabel("No file loaded")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_XS}; background: transparent;"
        )
        return lbl

    # ------------------------------------------------------------------ tab 1

    def _build_anomaly_tab(self) -> QWidget:
        tab = QWidget()
        split = QSplitter(Qt.Horizontal)
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        settings = QWidget()
        settings.setMinimumWidth(240)
        settings.setMaximumWidth(320)
        s_lay = QVBoxLayout(settings)
        s_lay.addWidget(QLabel("Settings"))
        self._ad_sample = QSpinBox()
        self._ad_sample.setRange(1, 100000)
        self._ad_sample.setValue(100)
        self._ad_sample.setSuffix(" Hz")
        self._ad_corr = QCheckBox("Check correlations")
        s_lay.addWidget(QLabel("Sample rate"))
        s_lay.addWidget(self._ad_sample)
        s_lay.addWidget(self._ad_corr)
        s_lay.addStretch()
        split.addWidget(settings)

        main = QWidget()
        m_lay = QVBoxLayout(main)
        row = QHBoxLayout()
        browse = QPushButton("Browse CSV…")
        self._ad_file_lbl = self._csv_info_label()
        browse.clicked.connect(self._load_anomaly_csv)
        row.addWidget(browse)
        row.addWidget(self._ad_file_lbl, 1)
        m_lay.addLayout(row)

        self._ad_channels = QListWidget()
        self._ad_channels.setMaximumHeight(120)
        m_lay.addWidget(QLabel("Channels to analyze"))
        m_lay.addWidget(self._ad_channels)

        run = QPushButton("Detect Anomalies")
        run.clicked.connect(self._run_anomaly_detection)
        m_lay.addWidget(run)

        metrics = QHBoxLayout()
        self._ad_m_total = MetricCard("Anomalies", "—")
        self._ad_m_crit = MetricCard("Critical", "—")
        self._ad_m_warn = MetricCard("Warning", "—")
        self._ad_m_health = MetricCard("Avg Health", "—")
        for c in (self._ad_m_total, self._ad_m_crit, self._ad_m_warn, self._ad_m_health):
            metrics.addWidget(c)
        m_lay.addLayout(metrics)

        self._ad_table = QTableWidget()
        self._ad_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        m_lay.addWidget(self._ad_table, 1)

        viz_row = QHBoxLayout()
        self._ad_viz_channel = QComboBox()
        self._ad_viz_channel.currentIndexChanged.connect(self._render_anomaly_plot)
        viz_row.addWidget(QLabel("Visualization channel"))
        viz_row.addWidget(self._ad_viz_channel, 1)
        m_lay.addLayout(viz_row)

        self._ad_plot = _LinePlotWidget("Time", "Value")
        self._ad_plot.setMinimumHeight(220)
        m_lay.addWidget(self._ad_plot, 1)

        self._ad_summary = QPlainTextEdit()
        self._ad_summary.setReadOnly(True)
        self._ad_summary.setMaximumHeight(120)
        m_lay.addWidget(self._ad_summary)

        split.addWidget(main)
        split.setStretchFactor(1, 1)
        return tab

    def _load_anomaly_csv(self) -> None:
        def on_loaded(df: pd.DataFrame, name: str) -> None:
            self._csv_df = df
            self._last_anomaly_df = df
            self._ad_file_lbl.setText(f"{name} — {len(df):,} rows × {len(df.columns)} cols")
            tcol = detect_time_column(df)
            cols = numeric_columns(df, exclude=[tcol] if tcol else [])
            _set_checklist(self._ad_channels, cols, default_n=5)
            self._banner.show_message(f"Loaded {name}", "success")

        self._browse_csv(on_loaded)

    @Slot()
    def _run_anomaly_detection(self) -> None:
        if self._csv_df is None:
            self._banner.show_message("Load a CSV first.", "warning")
            return
        channels = _checked_items(self._ad_channels)
        if not channels:
            self._banner.show_message("Select at least one channel.", "warning")
            return
        tcol = detect_time_column(self._csv_df) or "timestamp"
        pairs = None
        if self._ad_corr.isChecked() and len(channels) >= 2:
            pairs = [(channels[0], channels[1])]

        self._banner.show_message("Running anomaly detection…", "info")
        worker = _AnomalyWorker(
            self._csv_df, channels, tcol, float(self._ad_sample.value()), pairs
        )
        worker.signals.anomaly_ready.connect(self._on_anomaly_ready)
        worker.signals.failed.connect(self._on_worker_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object)
    def _on_anomaly_ready(self, report, df) -> None:
        self._last_anomaly_report = report
        self._last_anomaly_df = df
        self._ad_m_total.set_value(str(report.total_anomalies))
        self._ad_m_crit.set_value(str(report.critical_count))
        self._ad_m_warn.set_value(str(report.warning_count))
        avg_h = np.mean(list(report.sensor_health.values())) if report.sensor_health else 0.0
        self._ad_m_health.set_value(f"{avg_h:.0%}")

        table_df = format_anomaly_table(report)
        populate_table(self._ad_table, table_df)

        self._ad_viz_channel.blockSignals(True)
        self._ad_viz_channel.clear()
        self._ad_viz_channel.addItems(list(report.channel_reports.keys()))
        self._ad_viz_channel.blockSignals(False)
        self._render_anomaly_plot()
        self._ad_summary.setPlainText(report.summary())
        self._banner.show_message("Anomaly detection complete.", "success")

    def _render_anomaly_plot(self, *_args) -> None:
        if self._last_anomaly_report is None or self._last_anomaly_df is None:
            return
        if self._ad_viz_channel.count() == 0:
            return
        ch = self._ad_viz_channel.currentText()
        if ch not in self._last_anomaly_df.columns:
            return
        df = self._last_anomaly_df
        tcol = detect_time_column(df)
        y = df[ch].values.astype(float)
        x = (
            df[tcol].values.astype(float)
            if tcol and tcol in df.columns
            else np.arange(len(y), dtype=float)
        )
        if self._ad_plot.plot is None:
            return
        self._ad_plot.clear()
        self._ad_plot.plot.plot(x, y, pen=pg.mkPen("#18181b", width=1.2), name=ch)
        for anomaly in self._last_anomaly_report.channel_reports.get(ch, []):
            rgba = _ANOMALY_SEVERITY_RGBA.get(anomaly.severity, (0, 0, 0, 40))
            lo = x[anomaly.start_index]
            hi = x[min(anomaly.end_index, len(x) - 1)]
            region = pg.LinearRegionItem(
                [lo, hi], movable=False,
                brush=pg.mkBrush(*rgba), pen=pg.mkPen(width=0),
            )
            self._ad_plot.plot.addItem(region)

    # ------------------------------------------------------------------ tab 2

    def _build_comparison_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Campaign"))
        self._cmp_campaign = QComboBox()
        self._cmp_campaign.setMinimumWidth(200)
        toolbar.addWidget(self._cmp_campaign, 1)
        refresh = _secondary_btn("Refresh")
        refresh.clicked.connect(self.refresh_campaigns)
        toolbar.addWidget(refresh)
        lay.addLayout(toolbar)

        self._cmp_mode = QComboBox()
        self._cmp_mode.addItems([
            "Test Comparison",
            "Golden Reference",
            "Regression",
            "Correlation",
            "Campaign Comparison",
        ])
        self._cmp_mode.currentIndexChanged.connect(self._on_cmp_mode_changed)
        lay.addWidget(self._cmp_mode)

        self._cmp_stack = QTabWidget()
        self._cmp_stack.addTab(self._build_cmp_test_tab(), "Tests")
        self._cmp_stack.addTab(self._build_cmp_golden_tab(), "Golden")
        self._cmp_stack.addTab(self._build_cmp_regression_tab(), "Regression")
        self._cmp_stack.addTab(self._build_cmp_correlation_tab(), "Correlation")
        self._cmp_stack.addTab(self._build_cmp_campaign_tab(), "Campaigns")
        lay.addWidget(self._cmp_stack, 1)

        self._cmp_result = QTableWidget()
        self._cmp_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self._cmp_result, 1)

        self._cmp_plot = _LinePlotWidget("X", "Y")
        self._cmp_plot.setMinimumHeight(200)
        lay.addWidget(self._cmp_plot, 1)

        self._cmp_text = QPlainTextEdit()
        self._cmp_text.setReadOnly(True)
        self._cmp_text.setMaximumHeight(100)
        lay.addWidget(self._cmp_text)

        return tab

    def _on_cmp_mode_changed(self, index: int) -> None:
        self._cmp_stack.setCurrentIndex(index)

    def _campaign_df(self) -> Optional[pd.DataFrame]:
        name = self._cmp_campaign.currentText()
        if not name:
            return None
        try:
            return get_campaign_data(name)
        except Exception:
            return None

    def _build_cmp_test_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._cmp_test_a = QComboBox()
        self._cmp_test_b = QComboBox()
        self._cmp_params = QListWidget()
        self._cmp_params.setMaximumHeight(100)
        self._cmp_tol = QDoubleSpinBox()
        self._cmp_tol.setRange(1.0, 50.0)
        self._cmp_tol.setValue(5.0)
        self._cmp_tol.setSuffix(" %")
        btn = QPushButton("Compare Tests")
        btn.clicked.connect(self._run_test_comparison)
        form.addRow("Test A", self._cmp_test_a)
        form.addRow("Test B", self._cmp_test_b)
        form.addRow("Parameters", self._cmp_params)
        form.addRow("Tolerance", self._cmp_tol)
        form.addRow(btn)
        self._cmp_campaign.currentIndexChanged.connect(self._refresh_cmp_tests)
        return w

    def _refresh_cmp_tests(self) -> None:
        df = self._campaign_df()
        if df is None or "test_id" not in df.columns:
            return
        ids = df["test_id"].astype(str).tolist()
        metrics = metric_columns(df)
        for combo in (self._cmp_test_a, self._cmp_test_b):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(ids)
            combo.blockSignals(False)
        if len(ids) > 1:
            self._cmp_test_b.setCurrentIndex(1)
        _set_checklist(self._cmp_params, metrics, default_n=5)

    def _run_test_comparison(self) -> None:
        df = self._campaign_df()
        if df is None or "test_id" not in df.columns:
            self._banner.show_message("Select a campaign with test_id.", "warning")
            return
        test_a = self._cmp_test_a.currentText()
        test_b = self._cmp_test_b.currentText()
        if test_a == test_b:
            self._banner.show_message("Select different tests.", "warning")
            return
        params = _checked_items(self._cmp_params)
        if not params:
            self._banner.show_message("Select parameters.", "warning")
            return
        row_a = df[df["test_id"].astype(str) == test_a].iloc[0]
        row_b = df[df["test_id"].astype(str) == test_b].iloc[0]
        data_a = {p: float(row_a[p]) for p in params if pd.notna(row_a.get(p))}
        data_b = {p: float(row_b[p]) for p in params if pd.notna(row_b.get(p))}
        result = compare_tests(
            data_a, data_b, test_a, test_b, default_tolerance=self._cmp_tol.value()
        )
        populate_table(self._cmp_result, result.to_dataframe())
        xs = np.arange(len(result.comparisons))
        self._cmp_plot.clear()
        if self._cmp_plot.plot is not None:
            vals_a = [c.value_a for c in result.comparisons]
            vals_b = [c.value_b for c in result.comparisons]
            self._cmp_plot.plot.plot(xs, vals_a, pen=None, symbol="o", name=test_a)
            self._cmp_plot.plot.plot(xs, vals_b, pen=None, symbol="x", name=test_b)
        status = "PASS" if result.overall_pass else "FAIL"
        self._cmp_text.setPlainText(
            f"{status}: {result.n_within_tolerance}/{result.n_parameters} within tolerance"
        )

    def _build_cmp_golden_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self._golden_name = QComboBox()
        self._golden_name.setEditable(True)
        self._golden_name.addItem("campaign_golden")
        self._golden_params = QListWidget()
        self._golden_params.setMaximumHeight(90)
        self._golden_method = QComboBox()
        self._golden_method.addItems(["mean", "median"])
        self._golden_tol = QDoubleSpinBox()
        self._golden_tol.setRange(1.0, 5.0)
        self._golden_tol.setValue(3.0)
        create_btn = QPushButton("Create Golden Reference")
        create_btn.clicked.connect(self._create_golden)
        compare_btn = QPushButton("Compare Selected Test to Golden")
        compare_btn.clicked.connect(self._compare_golden)
        load_btn = _secondary_btn("Load Golden JSON…")
        load_btn.clicked.connect(self._load_golden_json)
        lay.addWidget(QLabel("Reference name"))
        lay.addWidget(self._golden_name)
        lay.addWidget(QLabel("Parameters"))
        lay.addWidget(self._golden_params)
        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        row.addWidget(self._golden_method)
        row.addWidget(QLabel("Tol ×σ"))
        row.addWidget(self._golden_tol)
        lay.addLayout(row)
        lay.addWidget(create_btn)
        lay.addWidget(load_btn)
        lay.addWidget(compare_btn)
        lay.addStretch()
        self._cmp_campaign.currentIndexChanged.connect(self._refresh_golden_params)
        return w

    def _refresh_golden_params(self) -> None:
        df = self._campaign_df()
        if df is None:
            return
        _set_checklist(self._golden_params, metric_columns(df), default_n=5)

    def _create_golden(self) -> None:
        df = self._campaign_df()
        params = _checked_items(self._golden_params)
        if df is None or not params:
            self._banner.show_message("Select campaign and parameters.", "warning")
            return
        name = self._golden_name.currentText().strip() or "golden"
        try:
            self._golden = create_golden_from_campaign(
                df, name, params,
                tolerance_multiplier=self._golden_tol.value(),
                method=self._golden_method.currentText(),
            )
            self._banner.show_message(f"Created golden reference '{name}'.", "success")
        except Exception as exc:
            self._banner.show_message(str(exc), "error")

    def _load_golden_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load golden reference", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._golden = GoldenReference.from_dict(data)
            self._banner.show_message(f"Loaded golden: {self._golden.name}", "success")
        except Exception as exc:
            self._banner.show_message(str(exc), "error")

    def _compare_golden(self) -> None:
        if self._golden is None:
            self._banner.show_message("Create or load a golden reference first.", "warning")
            return
        df = self._campaign_df()
        test_id = self._cmp_test_a.currentText()
        if df is None or not test_id:
            self._banner.show_message("Select a campaign and test.", "warning")
            return
        row = df[df["test_id"].astype(str) == test_id].iloc[0]
        test_data = {
            p: float(row[p])
            for p in self._golden.parameters
            if p in row.index and pd.notna(row[p])
        }
        result = compare_to_golden(test_data, test_id, self._golden)
        populate_table(self._cmp_result, result.to_dataframe())
        self._cmp_text.setPlainText("PASS" if result.overall_pass else "FAIL")

    def _build_cmp_regression_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._reg_x = QComboBox()
        self._reg_y = QComboBox()
        btn = QPushButton("Run Regression")
        btn.clicked.connect(self._run_regression)
        form.addRow("X (independent)", self._reg_x)
        form.addRow("Y (dependent)", self._reg_y)
        form.addRow(btn)
        self._cmp_campaign.currentIndexChanged.connect(self._refresh_regression_cols)
        return w

    def _refresh_regression_cols(self) -> None:
        df = self._campaign_df()
        if df is None:
            return
        cols = numeric_columns(df)
        for combo in (self._reg_x, self._reg_y):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            combo.blockSignals(False)

    def _run_regression(self) -> None:
        df = self._campaign_df()
        if df is None:
            return
        x_col = self._reg_x.currentText()
        y_col = self._reg_y.currentText()
        if not x_col or not y_col or x_col == y_col:
            self._banner.show_message("Select distinct X and Y columns.", "warning")
            return
        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)
        try:
            result = linear_regression(x, y, x_col, y_col)
        except Exception as exc:
            self._banner.show_message(str(exc), "error")
            return
        self._cmp_plot.plot_xy(x, y, name="data", color=ACCENT_BLUE)
        if self._cmp_plot.plot is not None:
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            y_line = result.predict(x_line)
            self._cmp_plot.plot.plot(
                x_line, y_line, pen=pg.mkPen(ACCENT_RED, width=2), name="fit"
            )
        self._cmp_text.setPlainText(
            f"R²={result.r_squared:.4f}  {result.prediction_equation}\n{result.summary()}"
        )
        populate_table(self._cmp_result, pd.DataFrame())

    def _build_cmp_correlation_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self._corr_params = QListWidget()
        self._corr_params.setMaximumHeight(100)
        self._corr_thresh = QDoubleSpinBox()
        self._corr_thresh.setRange(0.5, 0.95)
        self._corr_thresh.setSingleStep(0.05)
        self._corr_thresh.setValue(0.7)
        btn = QPushButton("Calculate Correlations")
        btn.clicked.connect(self._run_correlation)
        lay.addWidget(QLabel("Parameters"))
        lay.addWidget(self._corr_params)
        lay.addWidget(QLabel("Strong correlation threshold"))
        lay.addWidget(self._corr_thresh)
        lay.addWidget(btn)
        lay.addStretch()
        self._cmp_campaign.currentIndexChanged.connect(self._refresh_corr_params)
        return w

    def _refresh_corr_params(self) -> None:
        df = self._campaign_df()
        if df is None:
            return
        _set_checklist(self._corr_params, metric_columns(df), default_n=8)

    def _run_correlation(self) -> None:
        df = self._campaign_df()
        params = _checked_items(self._corr_params)
        if df is None or not params:
            self._banner.show_message("Select campaign and parameters.", "warning")
            return
        corr = calculate_correlation_matrix(df, params)
        populate_table(self._cmp_result, corr.to_dataframe())
        strong = corr.get_strong_correlations(self._corr_thresh.value())
        lines = [f"{p1} ↔ {p2}: {r:.3f}" for p1, p2, r in strong]
        self._cmp_text.setPlainText(
            "\n".join(lines) if lines else "No strong correlations found."
        )

    def _build_cmp_campaign_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._cmp_campaign_b = QComboBox()
        self._cmp_camp_params = QListWidget()
        self._cmp_camp_params.setMaximumHeight(100)
        btn = QPushButton("Compare Campaigns")
        btn.clicked.connect(self._run_campaign_comparison)
        form.addRow("Compare to", self._cmp_campaign_b)
        form.addRow("Parameters", self._cmp_camp_params)
        form.addRow(btn)
        self._cmp_campaign.currentIndexChanged.connect(self._refresh_cmp_camp_params)
        return w

    def _refresh_cmp_camp_params(self) -> None:
        df = self._campaign_df()
        if df is None:
            return
        _set_checklist(self._cmp_camp_params, metric_columns(df), default_n=5)

    def _run_campaign_comparison(self) -> None:
        df_a = self._campaign_df()
        name_a = self._cmp_campaign.currentText()
        name_b = self._cmp_campaign_b.currentText()
        params = _checked_items(self._cmp_camp_params)
        if df_a is None or not params or not name_b:
            self._banner.show_message("Select two campaigns and parameters.", "warning")
            return
        df_b = get_campaign_data(name_b)
        if df_b is None or df_b.empty:
            self._banner.show_message(f"No data in {name_b}.", "error")
            return
        result = compare_campaigns(df_a, df_b, name_a, name_b, params)
        rows = []
        for param, pdata in result["parameters"].items():
            rows.append({
                "Parameter": param,
                f"Mean ({name_a})": f"{pdata['mean_a']:.4g}",
                f"Mean ({name_b})": f"{pdata['mean_b']:.4g}",
                "Delta %": f"{pdata['mean_diff_pct']:+.2f}%",
                "Status": "Pass" if pdata["means_equivalent"] else "Fail",
            })
        populate_table(self._cmp_result, pd.DataFrame(rows))
        self._cmp_text.setPlainText(format_campaign_comparison(result))

    # ------------------------------------------------------------------ tab 3

    def _build_transient_tab(self) -> QWidget:
        tab = QWidget()
        split = QSplitter(Qt.Horizontal)
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        settings = QWidget()
        settings.setMaximumWidth(280)
        s_lay = QFormLayout(settings)
        self._ta_threshold = QDoubleSpinBox()
        self._ta_threshold.setRange(1.0, 50.0)
        self._ta_threshold.setValue(10.0)
        self._ta_cv = QDoubleSpinBox()
        self._ta_cv.setRange(0.005, 0.10)
        self._ta_cv.setSingleStep(0.005)
        self._ta_cv.setValue(0.02)
        self._ta_min_phase = QDoubleSpinBox()
        self._ta_min_phase.setRange(0.01, 10.0)
        self._ta_min_phase.setValue(0.1)
        s_lay.addRow("Activation %", self._ta_threshold)
        s_lay.addRow("CV threshold", self._ta_cv)
        s_lay.addRow("Min phase (s)", self._ta_min_phase)
        split.addWidget(settings)

        main = QWidget()
        m_lay = QVBoxLayout(main)
        browse = QPushButton("Browse CSV…")
        self._ta_file_lbl = self._csv_info_label()
        browse.clicked.connect(self._load_transient_csv)
        m_lay.addWidget(browse)
        m_lay.addWidget(self._ta_file_lbl)

        row = QHBoxLayout()
        self._ta_time_col = QComboBox()
        self._ta_signal_col = QComboBox()
        row.addWidget(QLabel("Time"))
        row.addWidget(self._ta_time_col, 1)
        row.addWidget(QLabel("Signal"))
        row.addWidget(self._ta_signal_col, 1)
        m_lay.addLayout(row)

        run = QPushButton("Segment Phases")
        run.clicked.connect(self._run_transient)
        m_lay.addWidget(run)

        metrics = QHBoxLayout()
        self._ta_m_phases = MetricCard("Phases", "—")
        self._ta_m_dur = MetricCard("Duration", "—")
        metrics.addWidget(self._ta_m_phases)
        metrics.addWidget(self._ta_m_dur)
        m_lay.addLayout(metrics)

        self._ta_table = QTableWidget()
        self._ta_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        m_lay.addWidget(self._ta_table, 1)

        self._ta_plot = _LinePlotWidget("Time (s)", "Signal")
        self._ta_plot.setMinimumHeight(240)
        m_lay.addWidget(self._ta_plot, 1)

        self._ta_metrics_txt = QPlainTextEdit()
        self._ta_metrics_txt.setReadOnly(True)
        self._ta_metrics_txt.setMaximumHeight(100)
        m_lay.addWidget(self._ta_metrics_txt)

        split.addWidget(main)
        split.setStretchFactor(1, 1)
        return tab

    def _load_transient_csv(self) -> None:
        def on_loaded(df: pd.DataFrame, name: str) -> None:
            self._last_transient_df = df
            self._ta_file_lbl.setText(f"{name} — {len(df):,} rows")
            tcol = detect_time_column(df)
            cols = numeric_columns(df, exclude=[tcol] if tcol else [])
            all_time = list(df.columns)
            self._ta_time_col.clear()
            self._ta_time_col.addItems(all_time)
            if tcol:
                self._ta_time_col.setCurrentText(tcol)
            self._ta_signal_col.clear()
            self._ta_signal_col.addItems(cols)
            self._banner.show_message(f"Loaded {name}", "success")

        self._browse_csv(on_loaded)

    @Slot()
    def _run_transient(self) -> None:
        if self._last_transient_df is None:
            self._banner.show_message("Load a CSV first.", "warning")
            return
        worker = _TransientWorker(
            self._last_transient_df,
            self._ta_signal_col.currentText(),
            self._ta_time_col.currentText(),
            self._ta_threshold.value(),
            self._ta_cv.value(),
            self._ta_min_phase.value(),
        )
        worker.signals.transient_ready.connect(self._on_transient_ready)
        worker.signals.failed.connect(self._on_worker_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object, object, object)
    def _on_transient_ready(self, multi, df, startup, shutdown) -> None:
        self._ta_m_phases.set_value(str(len(multi.phases)))
        self._ta_m_dur.set_value(f"{multi.total_duration_s:.3f} s")

        rows = []
        for phase in multi.phases:
            rows.append({
                "Phase": _PHASE_LABELS.get(phase.phase, phase.phase.value),
                "Start (ms)": f"{phase.start_ms:.1f}",
                "End (ms)": f"{phase.end_ms:.1f}",
                "Duration (s)": f"{phase.duration_s:.4f}",
                "Quality": phase.quality,
            })
        populate_table(self._ta_table, pd.DataFrame(rows))

        time_col = self._ta_time_col.currentText()
        sig_col = self._ta_signal_col.currentText()
        x = df[time_col].values.astype(float)
        y = df[sig_col].values.astype(float)
        if self._ta_plot.plot is not None:
            self._ta_plot.clear()
            self._ta_plot.plot.plot(x, y, pen=pg.mkPen("#18181b", width=1.2))
            for phase in multi.phases:
                hex_color = _PHASE_COLORS.get(phase.phase, "#71717a")
                qc = QColor(hex_color)
                lo = phase.start_ms / 1000.0
                hi = phase.end_ms / 1000.0
                region = pg.LinearRegionItem(
                    [lo, hi], movable=False,
                    brush=pg.mkBrush(qc.red(), qc.green(), qc.blue(), 30),
                    pen=pg.mkPen(width=0),
                )
                self._ta_plot.plot.addItem(region)

        lines = []
        if startup:
            lines.append("Startup: " + ", ".join(f"{k}={v:.4g}" for k, v in startup.items() if isinstance(v, (int, float))))
        if shutdown:
            lines.append("Shutdown: " + ", ".join(f"{k}={v:.4g}" for k, v in shutdown.items() if isinstance(v, (int, float))))
        self._ta_metrics_txt.setPlainText("\n".join(lines))
        self._banner.show_message(f"Detected {len(multi.phases)} phases.", "success")

    # ------------------------------------------------------------------ tab 4

    def _build_frequency_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)

        load_row = QHBoxLayout()
        browse = QPushButton("Browse CSV…")
        self._fa_file_lbl = self._csv_info_label()
        browse.clicked.connect(self._load_freq_csv)
        load_row.addWidget(browse)
        load_row.addWidget(self._fa_file_lbl, 1)
        lay.addLayout(load_row)

        settings = QHBoxLayout()
        self._fa_rate = QSpinBox()
        self._fa_rate.setRange(1, 100000)
        self._fa_rate.setValue(100)
        self._fa_method = QComboBox()
        self._fa_method.addItems(["welch", "periodogram"])
        self._fa_window = QComboBox()
        self._fa_window.addItems(["hann", "hamming", "blackman", "boxcar"])
        self._fa_nperseg = QSpinBox()
        self._fa_nperseg.setRange(32, 8192)
        self._fa_nperseg.setValue(256)
        self._fa_channel = QComboBox()
        for lbl, widget in (
            ("Rate (Hz)", self._fa_rate),
            ("Method", self._fa_method),
            ("Window", self._fa_window),
            ("nperseg", self._fa_nperseg),
            ("Channel", self._fa_channel),
        ):
            settings.addWidget(QLabel(lbl))
            settings.addWidget(widget)
        settings.addStretch()
        lay.addLayout(settings)

        sub = QTabWidget()
        psd_tab = QWidget()
        psd_lay = QVBoxLayout(psd_tab)
        psd_btn = QPushButton("Compute PSD")
        psd_btn.clicked.connect(self._run_psd)
        psd_lay.addWidget(psd_btn)
        self._fa_psd_plot = _LinePlotWidget("Frequency (Hz)", "PSD")
        psd_lay.addWidget(self._fa_psd_plot, 1)
        self._fa_psd_metrics = QHBoxLayout()
        self._fa_m_dom_f = MetricCard("Dominant F", "—")
        self._fa_m_dom_p = MetricCard("Dominant P", "—")
        self._fa_m_total_p = MetricCard("Total P", "—")
        for c in (self._fa_m_dom_f, self._fa_m_dom_p, self._fa_m_total_p):
            self._fa_psd_metrics.addWidget(c)
        psd_lay.addLayout(self._fa_psd_metrics)
        sub.addTab(psd_tab, "PSD")

        harm_tab = QWidget()
        harm_lay = QVBoxLayout(harm_tab)
        self._fa_n_harm = QSpinBox()
        self._fa_n_harm.setRange(1, 15)
        self._fa_n_harm.setValue(5)
        self._fa_harm_tol = QDoubleSpinBox()
        self._fa_harm_tol.setRange(0.1, 10.0)
        self._fa_harm_tol.setValue(1.0)
        harm_row = QHBoxLayout()
        harm_row.addWidget(QLabel("Max harmonics"))
        harm_row.addWidget(self._fa_n_harm)
        harm_row.addWidget(QLabel("Tolerance (Hz)"))
        harm_row.addWidget(self._fa_harm_tol)
        harm_lay.addLayout(harm_row)
        harm_btn = QPushButton("Detect Harmonics")
        harm_btn.clicked.connect(self._run_harmonics)
        harm_lay.addWidget(harm_btn)
        self._fa_harm_table = QTableWidget()
        harm_lay.addWidget(self._fa_harm_table, 1)
        sub.addTab(harm_tab, "Harmonics")

        res_tab = QWidget()
        res_lay = QVBoxLayout(res_tab)
        self._fa_prominence = QDoubleSpinBox()
        self._fa_prominence.setRange(1.0, 20.0)
        self._fa_prominence.setValue(3.0)
        res_lay.addWidget(QLabel("Peak prominence (dB)"))
        res_lay.addWidget(self._fa_prominence)
        res_btn = QPushButton("Detect Resonances")
        res_btn.clicked.connect(self._run_resonance)
        res_lay.addWidget(res_btn)
        self._fa_res_table = QTableWidget()
        res_lay.addWidget(self._fa_res_table, 1)
        sub.addTab(res_tab, "Resonance")

        lay.addWidget(sub, 1)
        return tab

    def _load_freq_csv(self) -> None:
        def on_loaded(df: pd.DataFrame, name: str) -> None:
            self._freq_df = df
            self._fa_file_lbl.setText(f"{name} — {len(df):,} rows")
            tcol = detect_time_column(df)
            cols = numeric_columns(df, exclude=[tcol] if tcol else [])
            self._fa_channel.clear()
            self._fa_channel.addItems(cols)
        self._browse_csv(on_loaded)

    def _freq_signal(self) -> Optional[np.ndarray]:
        if self._freq_df is None:
            self._banner.show_message("Load a CSV first.", "warning")
            return None
        ch = self._fa_channel.currentText()
        if not ch:
            return None
        return self._freq_df[ch].dropna().values.astype(float)

    @Slot()
    def _run_psd(self) -> None:
        sig = self._freq_signal()
        if sig is None or len(sig) < 4:
            return
        worker = _PSDWorker(
            sig, float(self._fa_rate.value()),
            self._fa_method.currentText(),
            self._fa_window.currentText(),
            self._fa_nperseg.value(),
        )
        worker.signals.psd_ready.connect(self._on_psd_ready)
        worker.signals.failed.connect(self._on_worker_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_psd_ready(self, result) -> None:
        self._fa_m_dom_f.set_value(f"{result.dominant_frequency:.2f} Hz")
        self._fa_m_dom_p.set_value(f"{result.dominant_power:.4g}")
        self._fa_m_total_p.set_value(f"{result.total_power:.4g}")
        self._fa_psd_plot.plot_xy(
            result.frequencies, result.power_spectral_density, log_y=True
        )
        if self._fa_psd_plot.plot is not None:
            self._fa_psd_plot.plot.addItem(pg.InfiniteLine(
                pos=result.dominant_frequency, angle=90,
                pen=pg.mkPen(ACCENT_RED, width=1, style=Qt.DashLine),
            ))

    @Slot()
    def _run_harmonics(self) -> None:
        sig = self._freq_signal()
        if sig is None:
            return
        worker = _HarmonicsWorker(
            sig, float(self._fa_rate.value()),
            self._fa_method.currentText(),
            self._fa_window.currentText(),
            self._fa_nperseg.value(),
            self._fa_n_harm.value(),
            self._fa_harm_tol.value(),
        )
        worker.signals.harmonics_ready.connect(self._on_harmonics_ready)
        worker.signals.failed.connect(self._on_worker_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object)
    def _on_harmonics_ready(self, psd, harmonics) -> None:
        rows = [{
            "Harmonic #": h.harmonic_number,
            "Frequency (Hz)": f"{h.frequency:.2f}",
            "Power": f"{h.power:.4g}",
            "Relative": f"{h.relative_power:.1%}",
        } for h in harmonics]
        populate_table(self._fa_harm_table, pd.DataFrame(rows))
        self._fa_psd_plot.plot_xy(psd.frequencies, psd.power_spectral_density, log_y=True)

    @Slot()
    def _run_resonance(self) -> None:
        sig = self._freq_signal()
        if sig is None:
            return
        worker = _ResonanceWorker(
            sig, float(self._fa_rate.value()),
            self._fa_method.currentText(),
            self._fa_window.currentText(),
            self._fa_nperseg.value(),
            self._fa_prominence.value(),
        )
        worker.signals.resonance_ready.connect(self._on_resonance_ready)
        worker.signals.failed.connect(self._on_worker_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_resonance_ready(self, resonances) -> None:
        rows = [{
            "Frequency (Hz)": f"{r['frequency']:.2f}",
            "Q factor": f"{r['q_factor']:.2f}",
            "Bandwidth (Hz)": f"{r['bandwidth']:.2f}",
            "Power": f"{r['power']:.4g}",
        } for r in resonances]
        populate_table(self._fa_res_table, pd.DataFrame(rows))

    # ------------------------------------------------------------------ tab 5

    def _build_envelope_tab(self) -> QWidget:
        tab = QWidget()
        split = QSplitter(Qt.Horizontal)
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        settings = QWidget()
        settings.setMaximumWidth(300)
        s_lay = QFormLayout(settings)
        self._oe_campaign = QComboBox()
        self._oe_of_col = QComboBox()
        self._oe_pc_col = QComboBox()
        self._oe_id_col = QComboBox()
        self._oe_ign_col = QComboBox()
        self._oe_margin = QDoubleSpinBox()
        self._oe_margin.setRange(0.0, 50.0)
        self._oe_margin.setValue(10.0)
        self._oe_filter_success = QCheckBox("Successful ignitions only")
        self._oe_filter_success.setChecked(True)
        self._oe_campaign.currentIndexChanged.connect(self._on_envelope_campaign_changed)
        s_lay.addRow("Campaign", self._oe_campaign)
        s_lay.addRow("O/F column", self._oe_of_col)
        s_lay.addRow("Pc column", self._oe_pc_col)
        s_lay.addRow("Test ID column", self._oe_id_col)
        s_lay.addRow("Ignition column", self._oe_ign_col)
        s_lay.addRow("Safety margin %", self._oe_margin)
        s_lay.addRow(self._oe_filter_success)
        run = QPushButton("Calculate Envelope")
        run.clicked.connect(self._run_envelope)
        s_lay.addRow(run)
        split.addWidget(settings)

        main = QWidget()
        m_lay = QVBoxLayout(main)
        metrics = QHBoxLayout()
        self._oe_m_of_lo = MetricCard("O/F min", "—")
        self._oe_m_of_hi = MetricCard("O/F max", "—")
        self._oe_m_pc_lo = MetricCard("Pc min", "—")
        self._oe_m_pc_hi = MetricCard("Pc max", "—")
        for c in (self._oe_m_of_lo, self._oe_m_of_hi, self._oe_m_pc_lo, self._oe_m_pc_hi):
            metrics.addWidget(c)
        m_lay.addLayout(metrics)
        self._oe_plot = _ScatterPlotWidget("O/F ratio", "Pc (bar)")
        self._oe_plot.setMinimumHeight(320)
        m_lay.addWidget(self._oe_plot, 1)
        split.addWidget(main)
        split.setStretchFactor(1, 1)
        return tab

    def _on_envelope_campaign_changed(self) -> None:
        name = self._oe_campaign.currentText()
        if not name:
            return
        try:
            df = get_campaign_data(name)
        except Exception:
            return
        if df is None or df.empty:
            return
        nums = numeric_columns(df)
        all_cols = list(df.columns)

        def _fill(combo: QComboBox, items: List[str], prefer: Sequence[str]) -> None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            for p in prefer:
                idx = combo.findText(p)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    break
            combo.blockSignals(False)

        _fill(self._oe_of_col, nums, ["avg_of_ratio", "of_ratio", "avg_of"])
        _fill(self._oe_pc_col, nums, ["avg_pc_bar", "pc_bar", "avg_pc"])
        _fill(self._oe_id_col, all_cols, ["test_id", "id", "name"])
        ign_opts = ["(none)"] + all_cols
        self._oe_ign_col.blockSignals(True)
        self._oe_ign_col.clear()
        self._oe_ign_col.addItems(ign_opts)
        for cand in ("ignition_successful", "ignition_success", "ign_success"):
            idx = self._oe_ign_col.findText(cand)
            if idx >= 0:
                self._oe_ign_col.setCurrentIndex(idx)
                break
        self._oe_ign_col.blockSignals(False)

    @Slot()
    def _run_envelope(self) -> None:
        name = self._oe_campaign.currentText()
        if not name:
            self._banner.show_message("Select a campaign.", "warning")
            return
        df = get_campaign_data(name)
        if df is None or df.empty:
            self._banner.show_message("Campaign has no data.", "warning")
            return
        of_col = self._oe_of_col.currentText()
        pc_col = self._oe_pc_col.currentText()
        ign = self._oe_ign_col.currentText()
        ign_col = None if ign == "(none)" else ign
        try:
            envelope = calculate_operating_envelope(
                df,
                of_column=of_col,
                pc_column=pc_col,
                ignition_column=ign_col,
                margin_pct=self._oe_margin.value(),
                filter_successful_only=self._oe_filter_success.isChecked(),
            )
        except Exception as exc:
            self._banner.show_message(str(exc), "error")
            return

        self._oe_m_of_lo.set_value(f"{envelope.of_min:.3f}")
        self._oe_m_of_hi.set_value(f"{envelope.of_max:.3f}")
        self._oe_m_pc_lo.set_value(f"{envelope.pc_min:.2f}")
        self._oe_m_pc_hi.set_value(f"{envelope.pc_max:.2f}")

        plot_df = df[[of_col, pc_col]].dropna()
        x = plot_df[of_col].values.astype(float)
        y = plot_df[pc_col].values.astype(float)
        colors = None
        if ign_col and ign_col in df.columns:
            colors = df.loc[plot_df.index, ign_col].astype(bool).values
        if self._oe_plot.plot is not None:
            self._oe_plot.plot_scatter(x, y, colors=colors)
            rect = pg.PlotDataItem(
                x=[envelope.of_min, envelope.of_max, envelope.of_max, envelope.of_min, envelope.of_min],
                y=[envelope.pc_min, envelope.pc_min, envelope.pc_max, envelope.pc_max, envelope.pc_min],
                pen=pg.mkPen(ACCENT_RED, width=2),
            )
            self._oe_plot.plot.addItem(rect)
        self._banner.show_message(
            f"Envelope from {envelope.n_tests} tests ({envelope.margin_pct:.0f}% margin).",
            "success",
        )

    @Slot(str)
    def _on_worker_failed(self, error: str) -> None:
        self._banner.show_message(error, "error")
        self.status_message.emit(error)
