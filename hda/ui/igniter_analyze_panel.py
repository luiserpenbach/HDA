"""Igniter hot-fire Analyze tab — post-test N₂O/Ethanol analysis panel."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, Slot, QObject
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from hda.ui.pages.base import InfoBanner, MetricCard
from hda.ui.style import (
    BORDER,
    CONTENT_SECONDARY_BG,
    SZ_BASE,
    SZ_SM,
    SZ_XS,
    TEXT_MUTED,
)

from core.igniter_analysis import (
    IgniterAnalysisResult,
    IgniterHardware,
    IgniterTestInputs,
    NOZZLE_SIZES_MM,
    analyze_igniter_post_test,
    inputs_from_steady_state,
    missing_dependencies,
)


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


def _form_row(label: str, widget: QWidget, *, optional: bool = False) -> QWidget:
    row = QWidget()
    col = QVBoxLayout(row)
    col.setContentsMargins(0, 6, 0, 0)
    col.setSpacing(5)
    lbl = QLabel(f"{label} (optional)" if optional else label)
    lbl.setObjectName("FormFieldLabel")
    lbl.setWordWrap(True)
    col.addWidget(lbl)
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    col.addWidget(widget)
    return row


class _IgniterWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _IgniterAnalysisWorker(QRunnable):
    def __init__(self, inputs: IgniterTestInputs, hardware: IgniterHardware) -> None:
        super().__init__()
        self.signals = _IgniterWorkerSignals()
        self._inputs = inputs
        self._hardware = hardware
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            result = analyze_igniter_post_test(self._inputs, self._hardware)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class IgniterAnalyzePanel(QWidget):
    """
    Post-test igniter analysis controls for the Single Test Analysis page.

    Supports manual entry or pulling averages from the steady-state window.
    """

    analysis_finished = Signal(object)
    status_message = Signal(str)
    save_requested = Signal()
    report_requested = Signal()

    PANEL_MIN = 260
    PANEL_MAX = 520

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._get_processed_df: Optional[Callable[[], Any]] = None
        self._get_steady_window: Optional[Callable[[], Tuple[float, float]]] = None
        self._get_sensor_roles: Optional[Callable[[], Dict[str, str]]] = None
        self._get_metadata: Optional[Callable[[], Dict[str, Any]]] = None
        self._last_result: Optional[IgniterAnalysisResult] = None

        outer = QVBoxLayout(self)
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
        inner.setMinimumWidth(self.PANEL_MIN - 24)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 4, 12, 16)
        lay.setSpacing(6)

        self._dep_banner = InfoBanner(parent=inner)
        missing = missing_dependencies()
        if missing:
            self._dep_banner.show_message(
                f"Install optional packages for full analysis: {', '.join(missing)}",
                "warning",
            )
        else:
            self._dep_banner.hide()
        lay.addWidget(self._dep_banner)

        lay.addWidget(_section("Input mode"))
        lay.addWidget(_divider())

        mode_row = QWidget()
        mode_lay = QHBoxLayout(mode_row)
        mode_lay.setContentsMargins(0, 0, 0, 0)
        self._mode_manual = QRadioButton("Manual averages")
        self._mode_steady = QRadioButton("From steady window")
        self._mode_steady.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._mode_manual)
        self._mode_group.addButton(self._mode_steady)
        mode_lay.addWidget(self._mode_manual)
        mode_lay.addWidget(self._mode_steady)
        lay.addWidget(mode_row)

        self._pull_btn = QPushButton("Pull from steady window")
        self._pull_btn.clicked.connect(self._pull_from_steady)
        lay.addWidget(self._pull_btn)

        lay.addWidget(_section("Hardware"))
        lay.addWidget(_divider())

        self._oxidizer = QComboBox()
        self._oxidizer.setEditable(True)
        self._oxidizer.addItems(["N2O", "LOX", "GOX"])
        self._oxidizer.setCurrentText("N2O")
        self._fuel = QComboBox()
        self._fuel.setEditable(True)
        self._fuel.addItems(["Ethanol", "IPA", "Methane", "RP-1"])
        self._fuel.setCurrentText("Ethanol")

        self._throat_mm = self._spin(4.4, 0.5, 20.0, 2, " mm")
        self._cd_n2o = self._spin(0.77, 0.3, 1.0, 3)
        self._cd_eth = self._spin(0.77, 0.3, 1.0, 3)
        self._t_n2o = self._spin(20.0, -40, 60, 1, " °C")
        self._t_eth = self._spin(20.0, -40, 60, 1, " °C")
        self._eta_cstar = self._spin(0.85, 0.5, 1.0, 3)

        self._d_n2o = QComboBox()
        self._d_n2o.addItems([str(d) for d in NOZZLE_SIZES_MM])
        self._d_n2o.setCurrentText("0.6")
        self._d_eth = QComboBox()
        self._d_eth.addItems([str(d) for d in NOZZLE_SIZES_MM])
        self._d_eth.setCurrentText("0.5")

        lay.addWidget(_form_row("Oxidizer", self._oxidizer))
        lay.addWidget(_form_row("Fuel", self._fuel))
        lay.addWidget(_form_row("Throat diameter", self._throat_mm))
        lay.addWidget(_form_row("Cd N₂O", self._cd_n2o))
        lay.addWidget(_form_row("Cd Ethanol", self._cd_eth))
        lay.addWidget(_form_row("N₂O temperature", self._t_n2o))
        lay.addWidget(_form_row("Ethanol temperature", self._t_eth))
        lay.addWidget(_form_row("η_c* target", self._eta_cstar))
        lay.addWidget(_form_row("N₂O orifice", self._d_n2o))
        lay.addWidget(_form_row("Ethanol orifice", self._d_eth))

        lay.addWidget(_section("Measured values"))
        lay.addWidget(_divider())

        self._pc = self._spin(20.0, 0.5, 200, 2, " bar")
        self._mdot_n2o = self._spin(30.0, 0.0, 500, 3, " g/s")
        self._mdot_eth = self._spin(0.0, 0.0, 500, 3, " g/s")
        self._p_eth_up = self._spin(55.0, 1.0, 200, 2, " bar")
        self._p_n2o_up = self._spin(55.0, 5.0, 200, 2, " bar")

        lay.addWidget(_form_row("Chamber Pc", self._pc))
        lay.addWidget(_form_row("N₂O ṁ (meter)", self._mdot_n2o))
        lay.addWidget(_form_row("Ethanol ṁ", self._mdot_eth, optional=True))
        lay.addWidget(
            _form_row(
                "Ethanol upstream P",
                self._p_eth_up,
                optional=True,
            )
        )
        lay.addWidget(
            _form_row(
                "N₂O upstream P (Cd calc)",
                self._p_n2o_up,
                optional=True,
            )
        )

        hint = QLabel(
            "Leave ethanol ṁ at 0 to estimate from upstream P and chamber P "
            "via SPI orifice model."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_XS}; background: transparent;"
        )
        lay.addWidget(hint)

        self._run_btn = QPushButton("Run igniter analysis")
        self._run_btn.clicked.connect(self._run_analysis)
        lay.addWidget(self._run_btn)

        self._save_btn = QPushButton("Save Result to Campaign…")
        self._save_btn.setProperty("secondary", True)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_requested.emit)
        lay.addWidget(self._save_btn)

        self._report_btn = QPushButton("Export Advanced Report…")
        self._report_btn.setProperty("secondary", True)
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self.report_requested.emit)
        lay.addWidget(self._report_btn)

        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._mode_steady.toggled.connect(self._on_mode_changed)
        self._on_mode_changed()

    @staticmethod
    def _spin(
        value: float,
        lo: float,
        hi: float,
        decimals: int,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(decimals)
        sb.setValue(value)
        if suffix:
            sb.setSuffix(suffix)
        sb.setSingleStep(10 ** (-decimals))
        return sb

    def bind_page(
        self,
        *,
        get_processed_df: Callable[[], Any],
        get_steady_window: Callable[[], Tuple[float, float]],
        get_sensor_roles: Callable[[], Dict[str, str]],
        get_metadata: Callable[[], Dict[str, Any]],
    ) -> None:
        self._get_processed_df = get_processed_df
        self._get_steady_window = get_steady_window
        self._get_sensor_roles = get_sensor_roles
        self._get_metadata = get_metadata

    def load_hardware_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        hw = IgniterHardware.from_metadata(metadata)
        self._throat_mm.setValue(hw.throat_diameter_mm)
        self._cd_n2o.setValue(hw.cd_n2o)
        self._cd_eth.setValue(hw.cd_eth)
        self._t_n2o.setValue(hw.t_n2o_c)
        self._t_eth.setValue(hw.t_eth_c)
        self._eta_cstar.setValue(hw.eta_cstar_target)
        self._d_n2o.setCurrentText(f"{hw.d_n2o_orifice_mm:g}")
        self._d_eth.setCurrentText(f"{hw.d_eth_orifice_mm:g}")
        if metadata:
            hw_block = metadata.get("igniter_hardware") or {}
            if hw_block.get("p_n2o_upstream_bar") is not None:
                self._p_n2o_up.setValue(float(hw_block["p_n2o_upstream_bar"]))
            if metadata.get("oxidizer"):
                self._oxidizer.setCurrentText(str(metadata["oxidizer"]))
            if metadata.get("fuel"):
                self._fuel.setCurrentText(str(metadata["fuel"]))

    def _hardware(self) -> IgniterHardware:
        meta = self._get_metadata() if self._get_metadata else {}
        base = IgniterHardware.from_metadata(meta)
        return IgniterHardware(
            throat_diameter_mm=self._throat_mm.value(),
            cd_n2o=self._cd_n2o.value(),
            cd_eth=self._cd_eth.value(),
            t_n2o_c=self._t_n2o.value(),
            t_eth_c=self._t_eth.value(),
            eta_cstar_target=self._eta_cstar.value(),
            d_n2o_orifice_mm=float(self._d_n2o.currentText()),
            d_eth_orifice_mm=float(self._d_eth.currentText()),
            p_amb_bar=base.p_amb_bar,
        )

    def _manual_inputs(self) -> IgniterTestInputs:
        mdot_eth = self._mdot_eth.value()
        return IgniterTestInputs(
            pc_bar=self._pc.value(),
            mdot_n2o_g_s=self._mdot_n2o.value(),
            mdot_eth_g_s=mdot_eth if mdot_eth > 0 else None,
            p_eth_upstream_bar=self._p_eth_up.value(),
            d_eth_orifice_mm=float(self._d_eth.currentText()),
            cd_eth_override=self._cd_eth.value(),
            p_n2o_upstream_bar=self._p_n2o_up.value(),
            d_n2o_orifice_mm=float(self._d_n2o.currentText()),
            oxidizer_name=self._oxidizer.currentText().strip() or "N2O",
            fuel_name=self._fuel.currentText().strip() or "Ethanol",
            input_source="manual",
        )

    def _on_mode_changed(self) -> None:
        steady = self._mode_steady.isChecked()
        self._pull_btn.setEnabled(steady)

    def _pull_from_steady(self) -> None:
        if not all(
            (
                self._get_processed_df,
                self._get_steady_window,
                self._get_sensor_roles,
            )
        ):
            self.status_message.emit("Page bindings not configured.")
            return

        df = self._get_processed_df()
        if df is None or getattr(df, "empty", True):
            self.status_message.emit("Run preprocessing before pulling steady averages.")
            return

        steady = self._get_steady_window()
        roles = self._get_sensor_roles()
        if steady[1] <= steady[0]:
            self.status_message.emit("Set a valid steady-state window first.")
            return

        try:
            inputs = inputs_from_steady_state(
                df,
                roles,
                steady,
                self._hardware(),
            )
        except ValueError as exc:
            self.status_message.emit(str(exc))
            return

        self._pc.setValue(inputs.pc_bar)
        self._mdot_n2o.setValue(inputs.mdot_n2o_g_s)
        if inputs.mdot_eth_g_s is not None:
            self._mdot_eth.setValue(inputs.mdot_eth_g_s)
        if inputs.p_eth_upstream_bar is not None:
            self._p_eth_up.setValue(inputs.p_eth_upstream_bar)

        msg = (
            f"Pulled steady averages ({steady[0]:.2f}–{steady[1]:.2f} s): "
            f"Pc={inputs.pc_bar:.2f} bar, ṁ_N₂O={inputs.mdot_n2o_g_s:.2f} g/s"
        )
        self.status_message.emit(msg)

    def _run_analysis(self) -> None:
        if self._mode_steady.isChecked():
            if not self._get_processed_df or self._get_processed_df() is None:
                self.status_message.emit("Run preprocessing before igniter analysis.")
                return
            if self._get_steady_window and self._get_steady_window()[1] <= self._get_steady_window()[0]:
                self.status_message.emit("Set a valid steady-state window.")
                return

        inputs = self._manual_inputs()
        if self._mode_steady.isChecked() and self._get_processed_df and self._get_sensor_roles:
            try:
                inputs = inputs_from_steady_state(
                    self._get_processed_df(),
                    self._get_sensor_roles(),
                    self._get_steady_window(),
                    self._hardware(),
                )
                inputs.p_n2o_upstream_bar = self._p_n2o_up.value()
                if self._mdot_eth.value() > 0:
                    inputs.mdot_eth_g_s = self._mdot_eth.value()
                inputs.oxidizer_name = self._oxidizer.currentText().strip() or "N2O"
                inputs.fuel_name = self._fuel.currentText().strip() or "Ethanol"
            except ValueError as exc:
                self.status_message.emit(str(exc))
                return

        hardware = self._hardware()
        self._run_btn.setEnabled(False)
        self.status_message.emit("Running igniter post-test analysis…")

        worker = _IgniterAnalysisWorker(inputs, hardware)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_finished(self, result: IgniterAnalysisResult) -> None:
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._report_btn.setEnabled(True)
        self._last_result = result
        self.analysis_finished.emit(result)
        eta = (
            f", η_c*={result.eta_cstar * 100:.1f}%"
            if result.eta_cstar is not None
            else ""
        )
        self.status_message.emit(
            f"Igniter analysis complete — O/F={result.of_ratio:.3f}{eta}"
        )

    def _on_failed(self, error: str) -> None:
        self._run_btn.setEnabled(True)
        self.status_message.emit(f"Igniter analysis failed: {error}")

    def run_analysis(self) -> None:
        """Public trigger used by global shortcuts / parent page."""
        self._run_analysis()

    @property
    def last_result(self) -> Optional[IgniterAnalysisResult]:
        return self._last_result


class IgniterResultsPanel(QWidget):
    """Results display for igniter post-test analysis."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 12)
        lay.setSpacing(8)

        self._banner = InfoBanner(parent=self)
        lay.addWidget(self._banner)

        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        lay.addWidget(self._grid_widget)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setStyleSheet(
            f"QTableWidget {{ font-size: {SZ_SM}; border: 1px solid {BORDER}; "
            f"background: {CONTENT_SECONDARY_BG}; }}"
        )
        lay.addWidget(self._table, 1)
        self.hide()

    def populate(self, result: IgniterAnalysisResult) -> None:
        if result.warnings:
            self._banner.show_message("; ".join(result.warnings), "warning")
        elif result.eta_cstar is not None and result.eta_cstar >= 0.8:
            self._banner.show_message(
                f"c* efficiency {result.eta_cstar * 100:.1f}% — within typical igniter range.",
                "success",
            )
        else:
            self._banner.show_message("Igniter post-test analysis complete.", "info")

        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cards = [
            ("O/F", f"{result.of_ratio:.3f}"),
            ("Pc", f"{result.pc_bar:.2f} bar"),
            ("ṁ total", f"{result.mdot_total_g_s:.2f} g/s"),
            ("c* actual", f"{result.cstar_actual_m_s:.1f} m/s"),
        ]
        if result.eta_cstar is not None:
            cards.append(("η_c*", f"{result.eta_cstar * 100:.1f} %"))
        if result.cd_n2o_back is not None:
            cards.append(("Cd N₂O", f"{result.cd_n2o_back:.4f}"))

        for i, (label, val) in enumerate(cards):
            row, col = divmod(i, 3)
            self._grid.addWidget(MetricCard(label, val), row, col)

        rows = result.to_report_rows()
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(row["Parameter"]))
            self._table.setItem(r, 1, QTableWidgetItem(row["Value"]))

        self.show()
