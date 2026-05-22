"""Main application window.

Layout
------
QHBoxLayout
├── NavBar (fixed 230 px)           — left navigation + global context
└── QStackedWidget                  — one widget per navigation page

The NavBar owns Test Root and Program state; it pushes context updates to
whatever page is currently visible (and the next page on switch).

QSettings persistence
---------------------
- main/geometry        — window size + position
- ctx/test_root        — last-used test root path
- ctx/program          — last-used program name
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from hda.ui.nav_bar import NavBar
from hda.ui.pages.base import BasePage
from hda.ui.pages.configurations import ConfigurationsPage
from hda.ui.pages.single_test_analysis import SingleTestAnalysisPage
from hda.ui.pages.placeholders import (
    AnalysisToolsPage,
    BatchAnalysisPage,
    CampaignAnalysisPage,
    SystemAnalysisPage,
)
from hda.ui.pages.test_ingestion import TestIngestionPage
from hda.ui.style import content_stylesheet


class HDAMainWindow(QMainWindow):
    """Navigation-based main window for the HDA Qt desktop application."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = QSettings("HopperPropulsion", "HDA")
        self.setWindowTitle("Hopper Data Studio")
        self.resize(1400, 860)
        self.setMinimumSize(900, 600)

        # ── Central widget ─────────────────────────────────────────────────
        central = QWidget()
        central.setStyleSheet(content_stylesheet())
        root_lay = QHBoxLayout(central)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        self.setCentralWidget(central)

        # ── Nav bar ────────────────────────────────────────────────────────
        self._nav = NavBar()
        root_lay.addWidget(self._nav)

        # ── Page stack ─────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        root_lay.addWidget(self._stack, 1)

        # Build pages in the same order as NAV_ITEMS in nav_bar.py
        self._pages: list[BasePage] = [
            TestIngestionPage(),       # 0 — Test Explorer
            SingleTestAnalysisPage(),  # 1
            BatchAnalysisPage(),       # 2
            CampaignAnalysisPage(),    # 3
            SystemAnalysisPage(),      # 4
            AnalysisToolsPage(),       # 5
            ConfigurationsPage(),      # 6
        ]
        for page in self._pages:
            self._stack.addWidget(page)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setContentsMargins(4, 0, 0, 0)
        bar = QStatusBar()
        bar.addWidget(self._status_lbl, 1)
        self._version_lbl = QLabel("HDA v2.4.0")
        bar.addPermanentWidget(self._version_lbl)
        self.setStatusBar(bar)

        # ── Signals ────────────────────────────────────────────────────────
        self._nav.nav_changed.connect(self._on_nav_changed)
        self._nav.test_root_changed.connect(self._on_test_root_changed)
        self._nav.program_changed.connect(self._on_program_changed)

        # Wire Test Ingestion's "open in analysis" to switch page
        test_page = self._pages[0]
        if isinstance(test_page, TestIngestionPage):
            test_page.open_in_analysis_requested.connect(self._on_open_in_analysis)

        # ── Restore state ──────────────────────────────────────────────────
        self._restore_geometry()
        self._restore_context()

    # ---------------------------------------------------------------- nav

    def _on_nav_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        page = self._pages[index]
        page.set_context(self._nav.test_root(), self._nav.program())

    def _on_test_root_changed(self, path: str) -> None:
        self._settings.setValue("ctx/test_root", path)
        self._push_context_to_active_page()

    def _on_program_changed(self, program: str) -> None:
        self._settings.setValue("ctx/program", program)
        self._push_context_to_active_page()

    def _push_context_to_active_page(self) -> None:
        page = self._pages[self._stack.currentIndex()]
        page.set_context(self._nav.test_root(), self._nav.program())

    def _on_open_in_analysis(self, path: str) -> None:
        """Switch to Single Test Analysis and pre-select the test path."""
        self._nav._on_nav_clicked(1)
        self._status_lbl.setText(f"Test path: {path}")

    # ---------------------------------------------------------------- settings

    def _restore_geometry(self) -> None:
        geo = self._settings.value("main/geometry")
        if geo is not None:
            try:
                self.restoreGeometry(geo)
            except Exception:
                pass

    def _restore_context(self) -> None:
        root = self._settings.value("ctx/test_root", "", type=str)
        program = self._settings.value("ctx/program", "", type=str)
        if root:
            self._nav.set_test_root(root)
        if program:
            self._nav.set_program(program)
        # Push to initial page
        self._push_context_to_active_page()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._settings.setValue("main/geometry", self.saveGeometry())
        super().closeEvent(event)
