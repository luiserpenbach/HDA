"""Placeholder pages and MVP implementations for Qt nav items."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.campaign_manager_v2 import get_available_campaigns, get_campaign_data
from hda.ui.pages.base import BasePage, InfoBanner, MetricCard


class _PlaceholderPage(BasePage):
    def __init__(self, title: str, description: str, parent=None):
        super().__init__(title, description, parent=parent)
        banner = InfoBanner(
            f"'{title}' is not yet implemented in the Qt desktop UI. "
            "Use the Streamlit app (streamlit run app.py) for this functionality.",
            "warning",
        )
        self.content_layout.addWidget(banner)
        self.content_layout.addStretch()


class BatchAnalysisPage(_PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            "Batch Analysis",
            "Process multiple test files with a consistent configuration in parallel",
            parent,
        )


class _SystemSignals(QObject):
    loaded = Signal(object, object)  # rows, summary
    failed = Signal(str)


class _SystemLoadWorker(QRunnable):
    def __init__(self, campaign_type: str, search_text: str) -> None:
        super().__init__()
        self.signals = _SystemSignals()
        self._campaign_type = campaign_type
        self._search_text = search_text.strip().lower()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            campaigns = get_available_campaigns()
            rows: List[Dict[str, Any]] = []
            total_tests = 0
            total_qc_rows = 0
            total_qc_pass = 0

            for camp in campaigns:
                name = str(camp.get("name", ""))
                ctype = str(camp.get("type", "unknown"))
                if self._campaign_type != "all" and ctype != self._campaign_type:
                    continue
                if self._search_text and self._search_text not in name.lower():
                    continue

                latest_ts = ""
                mean_pc = None
                mean_of = None
                mean_cd = None
                qc_pass_rate = None

                try:
                    df = get_campaign_data(name)
                except Exception as exc:
                    rows.append(
                        {
                            "campaign": name,
                            "type": ctype,
                            "tests": int(camp.get("test_count", 0) or 0),
                            "qc_pass_rate": None,
                            "latest_test": "",
                            "mean_pc_bar": None,
                            "mean_of_ratio": None,
                            "mean_cd": None,
                            "error": str(exc),
                        }
                    )
                    continue

                if not df.empty:
                    if "test_timestamp" in df.columns:
                        latest_ts = str(df["test_timestamp"].dropna().iloc[0]) if df["test_timestamp"].dropna().size else ""
                    if "avg_pc_bar" in df.columns:
                        s = pd.to_numeric(df["avg_pc_bar"], errors="coerce").dropna()
                        mean_pc = float(s.mean()) if not s.empty else None
                    if "avg_of_ratio" in df.columns:
                        s = pd.to_numeric(df["avg_of_ratio"], errors="coerce").dropna()
                        mean_of = float(s.mean()) if not s.empty else None
                    if "avg_cd_CALC" in df.columns:
                        s = pd.to_numeric(df["avg_cd_CALC"], errors="coerce").dropna()
                        mean_cd = float(s.mean()) if not s.empty else None
                    if "qc_passed" in df.columns:
                        q = pd.to_numeric(df["qc_passed"], errors="coerce").dropna()
                        if not q.empty:
                            total_qc_rows += int(q.size)
                            total_qc_pass += int((q > 0).sum())
                            qc_pass_rate = float((q > 0).mean() * 100.0)

                tests = int(camp.get("test_count", len(df)))
                total_tests += tests
                rows.append(
                    {
                        "campaign": name,
                        "type": ctype,
                        "tests": tests,
                        "qc_pass_rate": qc_pass_rate,
                        "latest_test": latest_ts,
                        "mean_pc_bar": mean_pc,
                        "mean_of_ratio": mean_of,
                        "mean_cd": mean_cd,
                        "error": "",
                    }
                )

            summary = {
                "campaigns": len(rows),
                "tests": total_tests,
                "qc_pass_rate": (100.0 * total_qc_pass / total_qc_rows) if total_qc_rows else None,
                "hot_fire_campaigns": sum(1 for r in rows if r.get("type") == "hot_fire"),
            }
            self.signals.loaded.emit(rows, summary)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class SystemAnalysisPage(BasePage):
    """System-level cross-campaign overview and health table (MVP)."""

    def __init__(self, parent=None):
        super().__init__(
            "System Analysis",
            "Cross-campaign overview for health, QC, and top-level performance trends.",
            parent,
        )

        self._rows: List[Dict[str, Any]] = []

        self._banner = InfoBanner(parent=self)
        self._banner.show_message("System Analysis MVP ready. Click Refresh to load campaigns.", "info")
        self.content_layout.addWidget(self._banner)

        controls = QWidget()
        controls_lay = QHBoxLayout(controls)
        controls_lay.setContentsMargins(0, 0, 0, 0)
        controls_lay.setSpacing(8)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter campaigns by name...")
        self._search.returnPressed.connect(self.refresh_data)
        controls_lay.addWidget(self._search, 1)

        self._type_combo = QComboBox()
        self._type_combo.addItem("All types", "all")
        self._type_combo.addItem("Cold flow", "cold_flow")
        self._type_combo.addItem("Hot fire", "hot_fire")
        controls_lay.addWidget(self._type_combo)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh_data)
        controls_lay.addWidget(self._refresh_btn)
        self.content_layout.addWidget(controls)

        metrics = QWidget()
        metrics_lay = QHBoxLayout(metrics)
        metrics_lay.setContentsMargins(0, 0, 0, 0)
        metrics_lay.setSpacing(8)
        self._card_campaigns = MetricCard("Campaigns", "—")
        self._card_tests = MetricCard("Total Tests", "—")
        self._card_qc = MetricCard("QC Pass Rate", "—")
        self._card_hf = MetricCard("Hot-Fire Campaigns", "—")
        metrics_lay.addWidget(self._card_campaigns)
        metrics_lay.addWidget(self._card_tests)
        metrics_lay.addWidget(self._card_qc)
        metrics_lay.addWidget(self._card_hf)
        self.content_layout.addWidget(metrics)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels(
            [
                "Campaign",
                "Type",
                "Tests",
                "QC pass %",
                "Latest test",
                "Mean Pc [bar]",
                "Mean O/F",
                "Mean Cd",
                "Load error",
            ]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSortingEnabled(True)
        hdr = self._table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        self.content_layout.addWidget(self._table, 1)

    def _fmt(self, value: Any, digits: int = 2) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @Slot()
    def refresh_data(self) -> None:
        self._refresh_btn.setEnabled(False)
        self._banner.show_message("Loading system-level campaign overview…", "info")
        self.status_message.emit("Loading system analysis…")
        worker = _SystemLoadWorker(
            campaign_type=str(self._type_combo.currentData()),
            search_text=self._search.text(),
        )
        worker.signals.loaded.connect(self._on_loaded)
        worker.signals.failed.connect(self._on_failed)
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object)
    def _on_loaded(self, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        self._refresh_btn.setEnabled(True)
        self._rows = rows
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            vals = [
                row.get("campaign", ""),
                row.get("type", ""),
                str(row.get("tests", 0)),
                self._fmt(row.get("qc_pass_rate"), 1),
                row.get("latest_test", ""),
                self._fmt(row.get("mean_pc_bar"), 2),
                self._fmt(row.get("mean_of_ratio"), 3),
                self._fmt(row.get("mean_cd"), 4),
                row.get("error", ""),
            ]
            for c, val in enumerate(vals):
                self._table.setItem(r, c, QTableWidgetItem(val))

        self._table.setSortingEnabled(True)
        self._card_campaigns.set_value(str(summary.get("campaigns", 0)))
        self._card_tests.set_value(str(summary.get("tests", 0)))
        qc_val = summary.get("qc_pass_rate")
        self._card_qc.set_value(f"{qc_val:.1f} %" if qc_val is not None else "—")
        self._card_hf.set_value(str(summary.get("hot_fire_campaigns", 0)))

        self._banner.show_message(
            f"Loaded {summary.get('campaigns', 0)} campaigns across {summary.get('tests', 0)} tests.",
            "success",
        )
        self.status_message.emit("System analysis updated.")

    @Slot(str)
    def _on_failed(self, error: str) -> None:
        self._refresh_btn.setEnabled(True)
        self._banner.show_message(f"System analysis load failed: {error}", "error")
        self.status_message.emit(f"System analysis load failed: {error}")

    def on_context_changed(self) -> None:
        # Context is currently informational for this MVP; we still auto-refresh for convenience.
        self.refresh_data()
