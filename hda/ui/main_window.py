"""Main application window.

Layout
------
QSplitter (horizontal)
├── NavBar wrapper (resizable, 180–400 px)
└── QStackedWidget                  — one widget per navigation page

The NavBar owns Test Root and Program state; it pushes context updates to
whatever page is currently visible (and the next page on switch).

QSettings persistence
---------------------
- main/geometry        — window size + position
- main/nav_splitter    — nav bar vs content split
- ctx/test_root        — last-used test root path
- ctx/program          — last-used program name
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from hda.ui.nav_bar import NavBar
from hda.ui.pages.base import BasePage
from hda.ui.pages.campaign_analysis import CampaignAnalysisPage
from hda.ui.pages.analysis_tools import AnalysisToolsPage
from hda.ui.pages.configurations import ConfigurationsPage
from hda.ui.pages.single_test_analysis import SingleTestAnalysisPage
from hda.ui.pages.placeholders import (
    BatchAnalysisPage,
    SystemAnalysisPage,
)
from hda.ui.pages.test_ingestion import TestIngestionPage
from hda.ui.style import content_stylesheet

try:
    from core import __version__ as CORE_VERSION
except ImportError:
    CORE_VERSION = "2.5.0"

NAV_MIN_WIDTH = 200
NAV_MAX_WIDTH = 400
NAV_DEFAULT_WIDTH = 260


class HDAMainWindow(QMainWindow):
    """Navigation-based main window for the HDA Qt desktop application."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = QSettings("HopperPropulsion", "HDA")
        self.setWindowTitle("Hopper Data Studio")
        self.resize(1400, 860)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(content_stylesheet())

        # ── Central widget ─────────────────────────────────────────────────
        central = QWidget()
        root_lay = QHBoxLayout(central)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        self.setCentralWidget(central)

        self._root_splitter = QSplitter(Qt.Horizontal)
        self._root_splitter.setHandleWidth(4)
        self._root_splitter.setChildrenCollapsible(False)
        root_lay.addWidget(self._root_splitter)

        nav_wrap = QWidget()
        nav_wrap.setMinimumWidth(NAV_MIN_WIDTH)
        nav_wrap.setMaximumWidth(NAV_MAX_WIDTH)
        nav_lay = QVBoxLayout(nav_wrap)
        nav_lay.setContentsMargins(0, 0, 0, 0)
        nav_lay.setSpacing(0)
        self._nav = NavBar()
        nav_lay.addWidget(self._nav)
        self._root_splitter.addWidget(nav_wrap)

        # ── Page stack ─────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._root_splitter.addWidget(self._stack)

        self._root_splitter.setStretchFactor(0, 0)
        self._root_splitter.setStretchFactor(1, 1)
        self._root_splitter.splitterMoved.connect(self._save_nav_splitter)
        self._nav_splitter_initialized = False

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
            page.status_message.connect(self._set_status)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setContentsMargins(4, 0, 0, 0)
        bar = QStatusBar()
        bar.addWidget(self._status_lbl, 1)
        self._version_lbl = QLabel(f"HDA v{CORE_VERSION}")
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
        sta_page = self._pages[1]
        if isinstance(sta_page, SingleTestAnalysisPage):
            sta_page.campaign_saved.connect(self._on_campaign_saved)

        # Wire Configurations "Use in Analysis" handoff
        config_page = self._pages[6]
        if isinstance(config_page, ConfigurationsPage):
            config_page.use_in_analysis_requested.connect(self._on_use_in_analysis)

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
        """Switch to Single Test Analysis and load the test folder."""
        self._nav._on_nav_clicked(1)
        sta = self._pages[1]
        if isinstance(sta, SingleTestAnalysisPage):
            sta.load_test_from_path(path)

    def _on_use_in_analysis(self, config_id: str) -> None:
        """Switch to Single Test Analysis with the selected configuration."""
        self._nav._on_nav_clicked(1)
        sta = self._pages[1]
        if isinstance(sta, SingleTestAnalysisPage):
            sta.set_active_config(config_id)
        self._set_status(f"Configuration '{config_id}' selected for analysis.")

    def _on_campaign_saved(self, campaign_name: str) -> None:
        """Switch to Campaign Analysis and preselect saved campaign."""
        self._nav._on_nav_clicked(3)
        page = self._pages[3]
        if isinstance(page, CampaignAnalysisPage):
            page.mark_opened_from_sta(campaign_name)
            page.select_campaign(campaign_name, refresh=True)
        self._set_status(f"Opened Campaign Analysis for '{campaign_name}'.")

    def _set_status(self, message: str) -> None:
        if message:
            self._status_lbl.setText(message)

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

    def _save_nav_splitter(self, *_pos: int) -> None:
        self._settings.setValue("main/nav_splitter", self._root_splitter.saveState())

    def _restore_nav_splitter(self) -> None:
        state = self._settings.value("main/nav_splitter")
        if state is not None:
            try:
                self._root_splitter.restoreState(state)
                return
            except Exception:
                pass
        self._root_splitter.setSizes([NAV_DEFAULT_WIDTH, 1140])

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._nav_splitter_initialized:
            self._restore_nav_splitter()
            self._nav_splitter_initialized = True

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._settings.setValue("main/geometry", self.saveGeometry())
        self._save_nav_splitter()
        super().closeEvent(event)
