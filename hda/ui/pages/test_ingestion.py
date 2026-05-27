"""Test Ingestion page — Qt equivalent of pages/1_Test_Explorer.py.

Three tabs:
  1. Browse Tests   — three-panel file-browser (Systems → Campaigns → Tests)
  2. Ingest New     — self-contained location form with live combos + metadata + create
  3. Edit Metadata  — path picker + structured form + validate / save / revert

Improvements over v1:
  - Ingest tab is self-contained: System and Campaign are editable QComboBox
    widgets populated directly from the filesystem (no longer read-only mirrors
    of the Browse tab selection).
  - Browse → Ingest auto-sync: selecting a campaign in Browse pre-selects it in
    Ingest via the campaign_context_changed signal.
  - Edit tab unsaved-changes guard: dirty tracking + Ctrl+S shortcut +
    confirmation dialog when switching away with unsaved edits.
  - Async directory scanning: _ScanWorker runs scan_test_root in a thread pool
    so large trees do not block the UI.
  - Status column in the tests table with foreground colour-coding.
  - Right-click context menu on tests: Open Folder, Copy Path, Edit Metadata,
    Delete (with confirmation).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import (
    QDate, QObject, QRunnable, Qt, QThreadPool, QTimer, QUrl,
    Signal, Slot,
)
from PySide6.QtGui import (
    QBrush, QColor, QDesktopServices, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.test_metadata import (
    TEST_SUBFOLDERS,
    TestMetadata,
    load_raw_metadata,
    load_test_metadata,
    save_raw_metadata,
    validate_raw_metadata,
)
from core.metadata_manager import create_metadata_template

from hda.ui.pages.base import BasePage, InfoBanner, MetricCard
from hda.ui.style import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_AMBER,
    ACCENT_RED,
    BORDER,
    CONTENT_SECONDARY_BG,
    SZ_BASE,
    SZ_SM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# Status → foreground colour for the tests table
_STATUS_FG: Dict[str, str] = {
    "pending":     ACCENT_AMBER,
    "analyzed":    ACCENT_BLUE,
    "approved":    ACCENT_GREEN,
    "rejected":    ACCENT_RED,
    "no_metadata": TEXT_MUTED,
    "error":       ACCENT_RED,
}


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"font-size: {SZ_SM}; font-weight: 600; color: {TEXT_SECONDARY};"
        "letter-spacing: 0.06em;"
    )
    return lbl


def _secondary_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setProperty("secondary", "true")
    return btn


def _input_dialog(parent: QWidget, title: str, prompt: str) -> Tuple[str, bool]:
    text, ok = QInputDialog.getText(parent, title, prompt)
    return text.strip(), ok


# ---------------------------------------------------------------------------
# Directory scanning helpers
# ---------------------------------------------------------------------------

def scan_test_root(root_path: Path) -> Dict[str, Any]:
    if not root_path.exists():
        return {"programs": [], "program_info": {}, "total_tests": 0}
    programs: List[str] = []
    program_info: Dict[str, Any] = {}
    total_tests = 0
    for item in sorted(root_path.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        name = item.name
        programs.append(name)
        systems, campaigns, tests = [], 0, 0
        for sys_dir in item.iterdir():
            if sys_dir.is_dir() and not sys_dir.name.startswith("."):
                systems.append(sys_dir.name)
                for camp_dir in sys_dir.iterdir():
                    if camp_dir.is_dir() and "-" in camp_dir.name:
                        campaigns += 1
                        for tt_dir in camp_dir.iterdir():
                            if tt_dir.is_dir() and "-" in tt_dir.name:
                                tests += sum(
                                    1 for t in tt_dir.iterdir()
                                    if t.is_dir() and not t.name.startswith(".")
                                )
        program_info[name] = {
            "path": str(item),
            "systems": systems,
            "system_count": len(systems),
            "campaign_count": campaigns,
            "test_count": tests,
        }
        total_tests += tests
    return {"programs": programs, "program_info": program_info, "total_tests": total_tests}


def get_systems_for_program(root_path: Path, program: str) -> List[Dict[str, Any]]:
    systems: List[Dict[str, Any]] = []
    program_dir = root_path / program
    if not program_dir.exists():
        return systems
    for sys_dir in sorted(program_dir.iterdir()):
        if not sys_dir.is_dir() or sys_dir.name.startswith("."):
            continue
        campaigns, tests = 0, 0
        test_types: set = set()
        for camp_dir in sys_dir.iterdir():
            if camp_dir.is_dir() and "-" in camp_dir.name:
                campaigns += 1
                for tt_dir in camp_dir.iterdir():
                    if tt_dir.is_dir() and "-" in tt_dir.name:
                        test_types.add(tt_dir.name.split("-")[-1])
                        tests += sum(
                            1 for t in tt_dir.iterdir()
                            if t.is_dir() and not t.name.startswith(".")
                        )
        systems.append({
            "name": sys_dir.name,
            "path": str(sys_dir),
            "test_types": sorted(test_types),
            "campaign_count": campaigns,
            "test_count": tests,
        })
    return systems


def get_campaigns_for_system(
    root_path: Path, program: str, system: str
) -> List[Dict[str, Any]]:
    campaigns: List[Dict[str, Any]] = []
    system_dir = root_path / program / system
    if not system_dir.exists():
        return campaigns
    for camp_dir in sorted(system_dir.iterdir()):
        if not camp_dir.is_dir() or "-" not in camp_dir.name:
            continue
        campaign_id = camp_dir.name.split("-")[-1]
        test_types: List[str] = []
        test_count = 0
        for tt_dir in camp_dir.iterdir():
            if tt_dir.is_dir() and "-" in tt_dir.name:
                test_types.append(tt_dir.name.split("-")[-1])
                test_count += sum(
                    1 for t in tt_dir.iterdir()
                    if t.is_dir() and not t.name.startswith(".")
                )
        campaigns.append({
            "name": camp_dir.name,
            "campaign_id": campaign_id,
            "path": str(camp_dir),
            "test_types": test_types,
            "test_count": test_count,
        })
    return campaigns


def get_tests_for_campaign(campaign_path: Path) -> List[Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []
    if not campaign_path.exists():
        return tests
    for tt_dir in sorted(campaign_path.iterdir()):
        if not tt_dir.is_dir() or "-" not in tt_dir.name:
            continue
        test_type = tt_dir.name.split("-")[-1]
        for test_dir in sorted(tt_dir.iterdir()):
            if not test_dir.is_dir() or test_dir.name.startswith("."):
                continue
            info: Dict[str, Any] = {
                "test_id": test_dir.name,
                "test_type": test_type,
                "path": str(test_dir),
                "has_metadata": (test_dir / "metadata.json").exists(),
                "has_raw_data": (
                    (test_dir / "raw_data").exists()
                    and any((test_dir / "raw_data").glob("*.csv"))
                ),
                "status": "no_metadata",
                "test_date": "",
                "operator": "",
            }
            if info["has_metadata"]:
                try:
                    md = load_test_metadata(test_dir)
                    if md:
                        info["status"] = md.status
                        info["test_date"] = md.test_date
                        info["operator"] = md.operator
                except Exception:
                    info["status"] = "error"
            tests.append(info)
    return tests


def get_next_test_id(
    test_type_path: Path, system: str, campaign_id: str, test_type: str
) -> str:
    if not test_type_path.exists():
        return f"{system}-{campaign_id}-{test_type}-001"
    existing: List[int] = []
    for d in test_type_path.iterdir():
        if d.is_dir():
            parts = d.name.split("-")
            if len(parts) >= 4:
                try:
                    existing.append(int(parts[-1]))
                except ValueError:
                    pass
    next_num = max(existing, default=0) + 1
    return f"{system}-{campaign_id}-{test_type}-{next_num:03d}"


def create_new_test(
    root_path: Path,
    program: str,
    system: str,
    campaign_id: str,
    test_type: str,
    test_id: str,
    metadata: Optional[TestMetadata] = None,
) -> Tuple[bool, str, Optional[Path]]:
    try:
        campaign_folder = f"{system}-{campaign_id}"
        test_type_folder = f"{system}-{campaign_id}-{test_type}"
        test_folder = (
            root_path / program / system / campaign_folder / test_type_folder / test_id
        )
        if test_folder.exists():
            return False, f"Test folder already exists: {test_id}", None

        test_folder.mkdir(parents=True, exist_ok=True)
        for sub in TEST_SUBFOLDERS:
            (test_folder / sub).mkdir(exist_ok=True)

        if metadata:
            metadata.test_id = test_id
            metadata.program = program
            metadata.system = system
            metadata.campaign_id = campaign_id
            metadata.test_type = test_type
            metadata.save(test_folder)
        else:
            TestMetadata(
                test_id=test_id,
                program=program,
                system=system,
                campaign_id=campaign_id,
                test_type=test_type,
            ).save(test_folder)

        return True, f"Created: {test_id}", test_folder
    except Exception as exc:
        return False, f"Error: {exc}", None


# ===========================================================================
# Async directory scanner
# ===========================================================================

class _ScanSignals(QObject):
    structure_ready = Signal(dict)   # result of scan_test_root
    systems_ready   = Signal(list)   # result of get_systems_for_program
    error           = Signal(str)


class _ScanWorker(QRunnable):
    """Scans the test root in a background thread so large trees don't block the UI."""

    def __init__(self, root_path: Path, program: str) -> None:
        super().__init__()
        self._root = root_path
        self._program = program
        self.signals = _ScanSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            structure = scan_test_root(self._root)
            self.signals.structure_ready.emit(structure)
            if self._program:
                systems = get_systems_for_program(self._root, self._program)
                self.signals.systems_ready.emit(systems)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ===========================================================================
# Browse tab
# ===========================================================================

class _BrowsePanel(QGroupBox):
    """Single column in the three-panel browser."""

    item_selected    = Signal(str)
    action_triggered = Signal()

    def __init__(
        self,
        title: str,
        action_label: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(title, parent)
        self._items: List[Dict[str, Any]] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 12, 8, 8)
        lay.setSpacing(6)

        self._list = QTableWidget(0, 1)
        self._list.horizontalHeader().setVisible(False)
        self._list.verticalHeader().setVisible(False)
        self._list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setAlternatingRowColors(True)
        self._list.horizontalHeader().setStretchLastSection(True)
        self._list.verticalHeader().setDefaultSectionSize(26)
        self._list.itemSelectionChanged.connect(self._on_selection)
        lay.addWidget(self._list, 1)

        self._info = QLabel()
        self._info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        self._info.setWordWrap(True)
        lay.addWidget(self._info)

        if action_label:
            self._action_btn = _secondary_btn(action_label)
            self._action_btn.clicked.connect(self.action_triggered.emit)
            lay.addWidget(self._action_btn)
        else:
            self._action_btn = None

    def populate(
        self, items: List[Dict[str, Any]], name_key: str, info_fn=None
    ) -> None:
        self._items = items
        self._list.setRowCount(len(items))
        for row, item in enumerate(items):
            cell = QTableWidgetItem(item[name_key])
            cell.setData(Qt.UserRole, item)
            self._list.setItem(row, 0, cell)
        if info_fn is None and items:
            self._info.setText(f"{len(items)} item(s)")
        elif info_fn:
            info_fn(self._info, items)

    def clear(self) -> None:
        self._list.setRowCount(0)
        self._items = []
        self._info.clear()

    def selected_data(self) -> Optional[Dict[str, Any]]:
        rows = self._list.selectedItems()
        if not rows:
            return None
        return rows[0].data(Qt.UserRole)

    def _on_selection(self) -> None:
        data = self.selected_data()
        if data:
            self.item_selected.emit(data.get("name", data.get("test_id", "")))

    def set_action_enabled(self, enabled: bool) -> None:
        if self._action_btn:
            self._action_btn.setEnabled(enabled)


class BrowseTab(QWidget):
    open_in_analysis_requested = Signal(str)   # test path
    edit_metadata_requested    = Signal(str)   # test path
    campaign_context_changed   = Signal(str, str)  # (system_name, campaign_id)
    status_message             = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root_path: Optional[Path] = None
        self._program: str = ""
        self._selected_system: Optional[Dict[str, Any]] = None
        self._selected_campaign: Optional[Dict[str, Any]] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)

        # Metric cards row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self._m_programs  = MetricCard("Programs", "—")
        self._m_systems   = MetricCard("Systems", "—")
        self._m_campaigns = MetricCard("Campaigns", "—")
        self._m_tests     = MetricCard("Total Tests", "—")
        for card in (self._m_programs, self._m_systems, self._m_campaigns, self._m_tests):
            metrics_row.addWidget(card)
        lay.addLayout(metrics_row)

        # Three-panel splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        self._panel_systems   = _BrowsePanel("Systems",   "+ New System")
        self._panel_campaigns = _BrowsePanel("Campaigns", "+ New Campaign")
        self._panel_tests     = _BrowsePanel("Tests")     # buttons added separately

        splitter.addWidget(self._panel_systems)
        splitter.addWidget(self._panel_campaigns)
        splitter.addWidget(self._panel_tests)
        splitter.setSizes([1, 1, 1])
        lay.addWidget(splitter, 1)

        # Tests-panel action buttons
        tests_btn_row = QHBoxLayout()
        tests_btn_row.setSpacing(6)
        self._btn_open = _secondary_btn("Open in Analysis")
        self._btn_open.clicked.connect(self._open_in_analysis)
        self._btn_open.setEnabled(False)
        self._btn_edit_meta = _secondary_btn("Edit Metadata")
        self._btn_edit_meta.clicked.connect(self._edit_metadata)
        self._btn_edit_meta.setEnabled(False)
        tests_btn_row.addWidget(self._btn_open)
        tests_btn_row.addWidget(self._btn_edit_meta)
        self._panel_tests.layout().addLayout(tests_btn_row)

        self._banner = InfoBanner(parent=self)
        lay.addWidget(self._banner)

        # Wiring
        self._panel_systems.item_selected.connect(self._on_system_selected)
        self._panel_campaigns.item_selected.connect(self._on_campaign_selected)
        self._panel_systems.action_triggered.connect(self._create_system)
        self._panel_campaigns.action_triggered.connect(self._create_campaign)
        self._panel_tests._list.itemSelectionChanged.connect(self._on_test_selection_changed)

        # Right-click context menu on tests table
        self._panel_tests._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._panel_tests._list.customContextMenuRequested.connect(
            self._show_test_context_menu
        )

    # ---- async refresh ────────────────────────────────────────────────────

    def refresh(self, root_path: Path, program: str) -> None:
        self._root_path = root_path
        self._program = program
        self._selected_system = None
        self._selected_campaign = None
        self._panel_campaigns.clear()
        self._panel_tests.clear()
        self._btn_open.setEnabled(False)
        self._btn_edit_meta.setEnabled(False)
        self._banner.show_message("Scanning…", "info")
        self.status_message.emit(f"Scanning {program}…")

        worker = _ScanWorker(root_path, program)
        worker.signals.structure_ready.connect(self._on_scan_structure)
        worker.signals.systems_ready.connect(self._on_scan_systems)
        worker.signals.error.connect(self._on_scan_error)
        QThreadPool.globalInstance().start(worker)

    def _on_scan_error(self, error: str) -> None:
        self._banner.show_message(f"Scan error: {error}", "error")
        self.status_message.emit(f"Scan error: {error}")

    def _on_scan_structure(self, structure: Dict[str, Any]) -> None:
        info = structure["program_info"].get(self._program, {})
        self._m_programs.set_value(str(len(structure["programs"])))
        self._m_systems.set_value(str(info.get("system_count", 0)))
        self._m_campaigns.set_value(str(info.get("campaign_count", 0)))
        self._m_tests.set_value(str(info.get("test_count", 0)))
        self._banner.clear_message()

    def _on_scan_systems(self, systems: List[Dict[str, Any]]) -> None:
        self._panel_systems.populate(systems, "name", self._system_info)
        self._panel_systems.set_action_enabled(bool(self._program))
        self.status_message.emit(
            f"Loaded {len(systems)} system(s) for {self._program}."
        )

    def _system_info(self, label: QLabel, items: List[Dict[str, Any]]) -> None:
        label.setText(f"{len(items)} system(s)")

    # ---- panel callbacks ──────────────────────────────────────────────────

    def _on_system_selected(self, name: str) -> None:
        if not self._root_path or not self._program:
            return
        data = self._panel_systems.selected_data()
        if not data:
            return
        self._selected_system = data
        self._selected_campaign = None
        self._panel_tests.clear()
        self._btn_open.setEnabled(False)
        self._btn_edit_meta.setEnabled(False)

        campaigns = get_campaigns_for_system(self._root_path, self._program, name)
        self._panel_campaigns.populate(
            campaigns, "campaign_id",
            lambda l, i: l.setText(f"{len(i)} campaign(s)"),
        )
        self._panel_campaigns.set_action_enabled(True)

    def _on_campaign_selected(self, _campaign_id: str) -> None:
        data = self._panel_campaigns.selected_data()
        if not data:
            return
        self._selected_campaign = data
        tests = get_tests_for_campaign(Path(data["path"]))

        tbl = self._panel_tests._list
        tbl.setRowCount(len(tests))
        tbl.setColumnCount(4)
        tbl.setHorizontalHeaderLabels(["Test ID", "Type", "Status", "Date"])
        tbl.horizontalHeader().setVisible(True)
        tbl.horizontalHeader().setStretchLastSection(True)

        for row, test in enumerate(tests):
            id_item = QTableWidgetItem(test["test_id"])
            id_item.setData(Qt.UserRole, test)
            tbl.setItem(row, 0, id_item)
            tbl.setItem(row, 1, QTableWidgetItem(test["test_type"]))

            status = test.get("status", "no_metadata")
            status_item = QTableWidgetItem(status)
            fg = _STATUS_FG.get(status, TEXT_MUTED)
            status_item.setForeground(QBrush(QColor(fg)))
            tbl.setItem(row, 2, status_item)

            tbl.setItem(row, 3, QTableWidgetItem(test.get("test_date", "") or ""))

        self._panel_tests._info.setText(
            f"{len(tests)} test(s) in {data['campaign_id']}"
        )
        self._btn_open.setEnabled(False)
        self._btn_edit_meta.setEnabled(False)

        # Notify Ingest tab (and any other listener) of the new selection
        if self._selected_system:
            self.campaign_context_changed.emit(
                self._selected_system["name"], data["campaign_id"]
            )

    def _on_test_selection_changed(self) -> None:
        has_sel = bool(self._panel_tests._list.selectedItems())
        self._btn_open.setEnabled(has_sel)
        self._btn_edit_meta.setEnabled(has_sel)

    def _selected_test_info(self) -> Optional[Dict[str, Any]]:
        rows = self._panel_tests._list.selectedItems()
        if not rows:
            return None
        return self._panel_tests._list.item(rows[0].row(), 0).data(Qt.UserRole)

    # ---- test actions ─────────────────────────────────────────────────────

    def _open_in_analysis(self) -> None:
        test = self._selected_test_info()
        if test:
            self.open_in_analysis_requested.emit(test["path"])
            self._banner.show_message(
                f"Test path saved: {test['test_id']}. Navigate to Single Test Analysis.",
                "success",
            )

    def _edit_metadata(self) -> None:
        test = self._selected_test_info()
        if test:
            self.edit_metadata_requested.emit(test["path"])

    # ---- right-click context menu ─────────────────────────────────────────

    def _show_test_context_menu(self, pos) -> None:
        test = self._selected_test_info()
        if not test:
            return
        path = test["path"]

        menu = QMenu(self)
        act_open   = menu.addAction("Open Folder")
        act_copy   = menu.addAction("Copy Path")
        menu.addSeparator()
        act_edit   = menu.addAction("Edit Metadata")
        menu.addSeparator()
        act_delete = menu.addAction("Delete Test")

        action = menu.exec(
            self._panel_tests._list.viewport().mapToGlobal(pos)
        )
        if action == act_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        elif action == act_copy:
            QApplication.clipboard().setText(path)
        elif action == act_edit:
            self.edit_metadata_requested.emit(path)
        elif action == act_delete:
            self._delete_test(test)

    def _delete_test(self, test: Dict[str, Any]) -> None:
        reply = QMessageBox.question(
            self, "Delete Test",
            f"Permanently delete test folder:\n\n{test['test_id']}\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            shutil.rmtree(test["path"])
        except Exception as exc:
            QMessageBox.critical(self, "Delete Failed", str(exc))
            return
        if self._selected_campaign:
            self._on_campaign_selected(self._selected_campaign.get("campaign_id", ""))

    # ---- create system / campaign ─────────────────────────────────────────

    def _create_system(self) -> None:
        if not self._root_path or not self._program:
            return
        name, ok = _input_dialog(self._panel_systems, "New System", "System name (e.g. RCS):")
        if ok and name:
            (self._root_path / self._program / name).mkdir(parents=True, exist_ok=True)
            self.refresh(self._root_path, self._program)

    def _create_campaign(self) -> None:
        if not self._selected_system:
            return
        sys_name = self._selected_system["name"]
        cid, ok = _input_dialog(
            self._panel_campaigns, "New Campaign", "Campaign ID (e.g. C01):"
        )
        if ok and cid:
            camp_folder = f"{sys_name}-{cid}"
            (self._root_path / self._program / sys_name / camp_folder).mkdir(
                parents=True, exist_ok=True
            )
            self._on_system_selected(sys_name)


# ===========================================================================
# Metadata form widget (shared by Ingest and Edit tabs)
# ===========================================================================

class MetadataFormWidget(QScrollArea):
    """Structured form for all TestMetadata fields with dirty-change notification."""

    data_changed = Signal()

    def __init__(
        self,
        initial_data: Optional[Dict[str, Any]] = None,
        test_type: str = "",
        read_only_keys: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)

        self._test_type = test_type
        self._read_only = set(read_only_keys or [])
        self._widgets: Dict[str, QWidget] = {}
        self._loading = False

        container = QWidget()
        self._form_lay = QVBoxLayout(container)
        self._form_lay.setContentsMargins(0, 0, 8, 0)
        self._form_lay.setSpacing(10)
        self.setWidget(container)

        self._build_form()

        if initial_data:
            self.set_data(initial_data)

    # ---- change notification ───────────────────────────────────────────────

    def _emit_changed(self) -> None:
        if not self._loading:
            self.data_changed.emit()

    # ---- form construction ─────────────────────────────────────────────────

    def _build_form(self) -> None:
        self._add_section("Identity", [
            ("test_id",     "Test ID",     "line"),
            ("program",     "Program",     "line"),
            ("system",      "System",      "line"),
            ("campaign_id", "Campaign ID", "line"),
            ("test_type",   "Test Type",   "line"),
            ("run_id",      "Run ID",      "line"),
        ])
        self._add_section("Test Article", [
            ("part_name",     "Part Name",     "line"),
            ("part_number",   "Part Number",   "line"),
            ("serial_number", "Serial Number", "line"),
        ])
        self._add_section("Test Info", [
            ("test_date",  "Test Date",  "date"),
            ("test_time",  "Test Time",  "line"),
            ("operator",   "Operator",   "line"),
            ("facility",   "Facility",   "line"),
            ("test_stand", "Test Stand", "line"),
            ("status",     "Status",     "combo:pending,analyzed,approved,rejected"),
        ])
        self._add_section("Cold Flow — Fluid", [
            ("test_fluid",            "Test Fluid",        "combo:,Water,Nitrogen,Helium,Air,Ethanol,IPA,Nitrous Oxide"),
            ("fluid_temperature_K",   "Fluid Temp [K]",    "float"),
            ("fluid_pressure_Pa",     "Fluid Press [Pa]",  "float"),
            ("ambient_temperature_K", "Ambient Temp [K]",  "float"),
            ("ambient_pressure_Pa",   "Ambient Press [Pa]","float"),
        ])
        self._add_section("Hot Fire — Propellants", [
            ("oxidizer",           "Oxidizer",       "combo:,Oxygen,NitrousOxide,Hydrogen Peroxide"),
            ("fuel",               "Fuel",           "combo:,n-Dodecane,Ethanol,Methane,Hydrogen"),
            ("ox_temperature_K",   "Ox Temp [K]",    "float"),
            ("fuel_temperature_K", "Fuel Temp [K]",  "float"),
            ("ox_pressure_Pa",     "Ox Press [Pa]",  "float"),
            ("fuel_pressure_Pa",   "Fuel Press [Pa]","float"),
        ])
        self._add_section("Notes", [
            ("notes",     "Notes",     "text"),
            ("anomalies", "Anomalies", "text"),
        ])
        self._form_lay.addStretch()

    def _add_section(self, title: str, fields: List[Tuple[str, str, str]]) -> None:
        box = QGroupBox(title)
        fl = QFormLayout(box)
        fl.setContentsMargins(8, 8, 8, 8)
        fl.setSpacing(6)
        fl.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for key, label, widget_type in fields:
            w = self._make_widget(key, widget_type)
            if key in self._read_only:
                self._set_readonly(w)
            fl.addRow(label + ":", w)
            self._widgets[key] = w
        self._form_lay.addWidget(box)

    def _make_widget(self, key: str, widget_type: str) -> QWidget:
        if widget_type == "line":
            w = QLineEdit()
            w.textChanged.connect(lambda _t: self._emit_changed())
            return w
        if widget_type == "text":
            w = QPlainTextEdit()
            w.setMaximumHeight(80)
            w.textChanged.connect(self._emit_changed)
            return w
        if widget_type == "date":
            w = QDateEdit()
            w.setCalendarPopup(True)
            w.setDate(QDate.currentDate())
            w.setDisplayFormat("yyyy-MM-dd")
            w.dateChanged.connect(lambda _d: self._emit_changed())
            return w
        if widget_type == "float":
            w = QLineEdit()
            w.setPlaceholderText("0.0")
            w.textChanged.connect(lambda _t: self._emit_changed())
            return w
        if widget_type.startswith("combo:"):
            options = widget_type[6:].split(",")
            w = QComboBox()
            w.addItems(options)
            w.currentTextChanged.connect(lambda _t: self._emit_changed())
            return w
        w = QLineEdit()
        w.textChanged.connect(lambda _t: self._emit_changed())
        return w

    @staticmethod
    def _set_readonly(w: QWidget) -> None:
        if isinstance(w, QLineEdit):
            w.setReadOnly(True)
        elif isinstance(w, QComboBox):
            w.setEnabled(False)

    # ---- data access ───────────────────────────────────────────────────────

    def get_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            if isinstance(w, QLineEdit):
                data[key] = w.text().strip()
            elif isinstance(w, QPlainTextEdit):
                data[key] = w.toPlainText().strip()
            elif isinstance(w, QDateEdit):
                data[key] = w.date().toString("yyyy-MM-dd")
            elif isinstance(w, QComboBox):
                data[key] = w.currentText().strip()
        return data

    def set_data(self, data: Dict[str, Any]) -> None:
        self._loading = True
        try:
            for key, w in self._widgets.items():
                val = data.get(key, "")
                if isinstance(w, QLineEdit):
                    w.setText(str(val) if val not in (None, "") else "")
                elif isinstance(w, QPlainTextEdit):
                    w.setPlainText(str(val) if val not in (None, "") else "")
                elif isinstance(w, QDateEdit):
                    if val:
                        try:
                            dt = QDate.fromString(str(val)[:10], "yyyy-MM-dd")
                            if dt.isValid():
                                w.setDate(dt)
                        except Exception:
                            pass
                elif isinstance(w, QComboBox):
                    idx = w.findText(str(val))
                    w.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._loading = False

    def set_test_type(self, test_type: str) -> None:
        self._test_type = test_type

    def populate_identity(
        self,
        test_id: str = "",
        program: str = "",
        system: str = "",
        campaign_id: str = "",
        test_type: str = "",
    ) -> None:
        mapping = {
            "test_id": test_id, "program": program,
            "system": system, "campaign_id": campaign_id,
            "test_type": test_type,
        }
        self._loading = True
        try:
            for key, val in mapping.items():
                w = self._widgets.get(key)
                if isinstance(w, QLineEdit) and val:
                    w.setText(val)
        finally:
            self._loading = False


# ===========================================================================
# Ingest New Test tab
# ===========================================================================

TEST_TYPES = [
    ("CF", "CF — Cold Flow"),
    ("HF", "HF — Hot Fire"),
    ("LK", "LK — Leak Test"),
    ("PR", "PR — Pressure Test"),
]
TEST_TYPE_CODES = [t[0] for t in TEST_TYPES]


class IngestTab(QWidget):
    test_created = Signal(str)   # emits test_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root_path: Optional[Path] = None
        self._program: str = ""
        self._raw_data_bytes: Optional[bytes] = None
        self._raw_data_name: str = ""
        self._json_metadata: Optional[Dict[str, Any]] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setHandleWidth(1)
        top_splitter.setChildrenCollapsible(False)

        # ── Left: location ─────────────────────────────────────────────────
        loc_box = QGroupBox("Test Location")
        loc_lay = QFormLayout(loc_box)
        loc_lay.setContentsMargins(12, 12, 12, 12)
        loc_lay.setSpacing(8)
        loc_lay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Program: read-only — comes from global nav context
        self._f_program = QLineEdit()
        self._f_program.setReadOnly(True)
        self._f_program.setPlaceholderText("(set in sidebar)")

        # System: editable combo populated from filesystem
        self._sys_combo = QComboBox()
        self._sys_combo.setEditable(True)
        self._sys_combo.setInsertPolicy(QComboBox.NoInsert)
        self._sys_combo.lineEdit().setPlaceholderText("Select or type system name")
        self._sys_combo.currentTextChanged.connect(self._on_system_changed)

        # Campaign: editable combo populated from filesystem
        self._camp_combo = QComboBox()
        self._camp_combo.setEditable(True)
        self._camp_combo.setInsertPolicy(QComboBox.NoInsert)
        self._camp_combo.lineEdit().setPlaceholderText("Select or type campaign ID")
        self._camp_combo.currentTextChanged.connect(self._refresh_suggested_id)

        # Test type
        self._f_test_type = QComboBox()
        for code, label in TEST_TYPES:
            self._f_test_type.addItem(label, code)
        self._f_test_type.currentIndexChanged.connect(self._refresh_suggested_id)

        # Suggested / custom ID
        self._suggested_lbl = QLabel("—")
        self._suggested_lbl.setStyleSheet(
            f"font-weight: 600; color: {ACCENT_BLUE}; font-size: {SZ_BASE};"
        )
        self._use_suggested = QCheckBox("Use suggested ID")
        self._use_suggested.setChecked(True)
        self._use_suggested.toggled.connect(self._on_use_suggested_toggled)
        self._f_test_id = QLineEdit()
        self._f_test_id.setPlaceholderText("Custom test ID")
        self._f_test_id.setVisible(False)

        loc_lay.addRow("Program:", self._f_program)
        loc_lay.addRow("System:", self._sys_combo)
        loc_lay.addRow("Campaign:", self._camp_combo)
        loc_lay.addRow("Test Type:", self._f_test_type)
        loc_lay.addRow("Suggested ID:", self._suggested_lbl)
        loc_lay.addRow("", self._use_suggested)
        loc_lay.addRow("Custom ID:", self._f_test_id)
        top_splitter.addWidget(loc_box)

        # ── Right: metadata ────────────────────────────────────────────────
        meta_box = QGroupBox("Metadata")
        meta_lay = QVBoxLayout(meta_box)
        meta_lay.setContentsMargins(8, 12, 8, 8)
        meta_lay.setSpacing(6)

        source_row = QHBoxLayout()
        self._rb_manual = QRadioButton("Manual Input")
        self._rb_manual.setChecked(True)
        self._rb_upload = QRadioButton("Upload JSON")
        self._rb_manual.toggled.connect(self._on_meta_source_changed)
        source_row.addWidget(self._rb_manual)
        source_row.addWidget(self._rb_upload)
        source_row.addStretch()
        meta_lay.addLayout(source_row)

        self._meta_stack = QStackedWidget()

        self._manual_form = MetadataFormWidget(
            read_only_keys=["test_id", "program", "system", "campaign_id", "test_type"]
        )
        self._meta_stack.addWidget(self._manual_form)   # index 0

        upload_widget = QWidget()
        upload_lay = QVBoxLayout(upload_widget)
        upload_lay.setContentsMargins(0, 0, 0, 0)
        upload_lay.setSpacing(6)
        self._upload_btn = _secondary_btn("Choose metadata.json …")
        self._upload_btn.clicked.connect(self._browse_metadata_json)
        self._upload_status = QLabel("No file selected")
        self._upload_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        self._json_preview = QPlainTextEdit()
        self._json_preview.setReadOnly(True)
        self._json_preview.setPlaceholderText("JSON preview will appear here …")
        self._json_preview.setMaximumHeight(200)
        upload_lay.addWidget(self._upload_btn)
        upload_lay.addWidget(self._upload_status)
        upload_lay.addWidget(QLabel("Preview:"))
        upload_lay.addWidget(self._json_preview)
        upload_lay.addStretch()
        self._meta_stack.addWidget(upload_widget)       # index 1

        meta_lay.addWidget(self._meta_stack, 1)
        top_splitter.addWidget(meta_box)
        top_splitter.setSizes([1, 1])
        lay.addWidget(top_splitter, 1)

        # ── Bottom: raw data + create ──────────────────────────────────────
        bottom_box = QGroupBox("Raw Data (optional)")
        bottom_lay = QHBoxLayout(bottom_box)
        bottom_lay.setContentsMargins(12, 12, 12, 12)
        bottom_lay.setSpacing(8)

        self._raw_status = QLabel("No file selected")
        self._raw_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        raw_btn = _secondary_btn("Choose CSV …")
        raw_btn.clicked.connect(self._browse_raw_csv)
        raw_clear = _secondary_btn("✕")
        raw_clear.setFixedWidth(32)
        raw_clear.clicked.connect(self._clear_raw_data)
        bottom_lay.addWidget(raw_btn)
        bottom_lay.addWidget(raw_clear)
        bottom_lay.addWidget(self._raw_status, 1)
        lay.addWidget(bottom_box)

        self._banner = InfoBanner(parent=self)
        lay.addWidget(self._banner)

        self._create_btn = QPushButton("Create New Test")
        self._create_btn.setFixedHeight(36)
        self._create_btn.setEnabled(False)
        self._create_btn.clicked.connect(self._do_create)
        lay.addWidget(self._create_btn)

    # ---- context / combo population ────────────────────────────────────────

    def set_context(
        self,
        root_path: Optional[Path],
        program: str,
        system: str = "",
        campaign_id: str = "",
    ) -> None:
        self._root_path = root_path
        self._program = program
        self._f_program.setText(program)

        self._refresh_systems_combo()
        if system:
            self._sys_combo.setCurrentText(system)

        self._refresh_campaigns_combo()
        if campaign_id:
            self._camp_combo.setCurrentText(campaign_id)

        self._manual_form.populate_identity(
            program=program, system=system, campaign_id=campaign_id
        )
        self._refresh_suggested_id()
        self._banner.clear_message()

    def _refresh_systems_combo(self) -> None:
        if not self._root_path or not self._program:
            self._sys_combo.clear()
            return
        systems = get_systems_for_program(self._root_path, self._program)
        prev = self._sys_combo.currentText()
        self._sys_combo.blockSignals(True)
        self._sys_combo.clear()
        for s in systems:
            self._sys_combo.addItem(s["name"])
        if prev and self._sys_combo.findText(prev) >= 0:
            self._sys_combo.setCurrentText(prev)
        self._sys_combo.blockSignals(False)

    def _refresh_campaigns_combo(self) -> None:
        system = self._sys_combo.currentText().strip()
        if not self._root_path or not self._program or not system:
            self._camp_combo.clear()
            return
        campaigns = get_campaigns_for_system(self._root_path, self._program, system)
        prev = self._camp_combo.currentText()
        self._camp_combo.blockSignals(True)
        self._camp_combo.clear()
        for c in campaigns:
            self._camp_combo.addItem(c["campaign_id"])
        if prev and self._camp_combo.findText(prev) >= 0:
            self._camp_combo.setCurrentText(prev)
        self._camp_combo.blockSignals(False)

    def _on_system_changed(self, _text: str) -> None:
        self._refresh_campaigns_combo()
        self._refresh_suggested_id()

    def _current_test_type(self) -> str:
        idx = self._f_test_type.currentIndex()
        return self._f_test_type.itemData(idx) or ""

    def _refresh_suggested_id(self) -> None:
        system = self._sys_combo.currentText().strip()
        campaign_id = self._camp_combo.currentText().strip()
        if not all([self._root_path, self._program, system, campaign_id]):
            self._suggested_lbl.setText("—")
            self._create_btn.setEnabled(False)
            return
        tt = self._current_test_type()
        camp_folder = f"{system}-{campaign_id}"
        tt_folder = f"{system}-{campaign_id}-{tt}"
        tt_path = self._root_path / self._program / system / camp_folder / tt_folder
        suggested = get_next_test_id(tt_path, system, campaign_id, tt)
        self._suggested_lbl.setText(suggested)
        self._f_test_id.setText(suggested)
        self._create_btn.setEnabled(True)
        self._manual_form.populate_identity(
            program=self._program,
            system=system,
            campaign_id=campaign_id,
            test_type=tt,
            test_id=suggested,
        )

    def _on_use_suggested_toggled(self, checked: bool) -> None:
        self._f_test_id.setVisible(not checked)

    def _on_meta_source_changed(self) -> None:
        self._meta_stack.setCurrentIndex(0 if self._rb_manual.isChecked() else 1)

    def _browse_metadata_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata.json", str(Path.home()), "JSON files (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as fh:
                self._json_metadata = json.load(fh)
            self._upload_status.setText(Path(path).name)
            self._upload_status.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: {SZ_SM};")
            self._json_preview.setPlainText(json.dumps(self._json_metadata, indent=2))
        except Exception as exc:
            self._banner.show_message(f"Could not read JSON: {exc}", "error")

    def _browse_raw_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select raw data CSV", str(Path.home()), "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "rb") as fh:
            self._raw_data_bytes = fh.read()
        self._raw_data_name = Path(path).name
        self._raw_status.setText(self._raw_data_name)
        self._raw_status.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: {SZ_SM};")

    def _clear_raw_data(self) -> None:
        self._raw_data_bytes = None
        self._raw_data_name = ""
        self._raw_status.setText("No file selected")
        self._raw_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")

    def _do_create(self) -> None:
        if not self._root_path:
            self._banner.show_message("No test root selected.", "error")
            return
        system = self._sys_combo.currentText().strip()
        campaign_id = self._camp_combo.currentText().strip()
        if not system or not campaign_id:
            self._banner.show_message("Select a System and Campaign first.", "error")
            return

        test_id = (
            self._suggested_lbl.text()
            if self._use_suggested.isChecked()
            else self._f_test_id.text().strip()
        )
        if not test_id or test_id == "—":
            self._banner.show_message("Could not determine test ID.", "error")
            return

        metadata: Optional[TestMetadata] = None
        if self._rb_upload.isChecked() and self._json_metadata:
            try:
                metadata = TestMetadata.from_dict(self._json_metadata)
            except Exception as exc:
                self._banner.show_message(f"Bad metadata JSON: {exc}", "error")
                return
        else:
            raw = self._manual_form.get_data()
            raw["test_id"] = test_id
            raw["program"] = self._program
            raw["system"] = system
            raw["campaign_id"] = campaign_id
            raw["test_type"] = self._current_test_type()
            raw["status"] = "pending"
            metadata = TestMetadata.from_dict(raw)

        success, message, test_folder = create_new_test(
            self._root_path, self._program, system, campaign_id,
            self._current_test_type(), test_id, metadata,
        )

        if success and test_folder:
            if self._raw_data_bytes and self._raw_data_name:
                dest = test_folder / "raw_data" / self._raw_data_name
                dest.write_bytes(self._raw_data_bytes)
            self._banner.show_message(
                f"✓ Created {test_id}  →  {test_folder}", "success"
            )
            self.test_created.emit(test_id)
        else:
            self._banner.show_message(message, "error")


# ===========================================================================
# Edit Metadata tab
# ===========================================================================

class EditMetadataTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._test_folder: Optional[Path] = None
        self._original_data: Optional[Dict[str, Any]] = None
        self._has_existing_file = False
        self._is_dirty = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)

        # Path row
        path_row = QHBoxLayout()
        path_lbl = QLabel("Test folder:")
        path_lbl.setStyleSheet(f"font-weight: 500; color: {TEXT_SECONDARY};")
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText(
            "Select a test in Browse tab, or paste a path here …"
        )
        self._path_input.editingFinished.connect(self._load_from_path)
        browse_btn = _secondary_btn("Browse …")
        browse_btn.clicked.connect(self._browse_path)
        path_row.addWidget(path_lbl)
        path_row.addWidget(self._path_input, 1)
        path_row.addWidget(browse_btn)
        lay.addLayout(path_row)

        self._banner = InfoBanner(parent=self)
        lay.addWidget(self._banner)

        # Create-from-template section (shown when no metadata.json exists)
        self._template_box = QGroupBox("No metadata.json — Create from Template")
        tmpl_lay = QHBoxLayout(self._template_box)
        tmpl_lay.setContentsMargins(12, 12, 12, 12)
        tmpl_lay.setSpacing(8)
        tmpl_lay.addWidget(QLabel("Template:"))
        self._tmpl_combo = QComboBox()
        self._tmpl_combo.addItems(["cold_flow", "hot_fire"])
        tmpl_lay.addWidget(self._tmpl_combo)
        create_tmpl_btn = QPushButton("Create from Template")
        create_tmpl_btn.clicked.connect(self._create_from_template)
        tmpl_lay.addWidget(create_tmpl_btn)
        tmpl_lay.addStretch()
        self._template_box.setVisible(False)
        lay.addWidget(self._template_box)

        # Metadata form
        self._form = MetadataFormWidget()
        self._form.data_changed.connect(self._on_form_changed)
        lay.addWidget(self._form, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._validate_btn = _secondary_btn("Validate")
        self._validate_btn.clicked.connect(self._validate)
        self._save_btn = QPushButton("Save Changes")
        self._save_btn.clicked.connect(self._save)
        self._revert_btn = _secondary_btn("Revert to Saved")
        self._revert_btn.clicked.connect(self._revert)
        btn_row.addStretch()
        btn_row.addWidget(self._validate_btn)
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._revert_btn)
        lay.addLayout(btn_row)

        self._set_form_enabled(False)

        # Ctrl+S shortcut (scoped to this tab widget)
        shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut.activated.connect(self._save)

    # ---- dirty tracking ────────────────────────────────────────────────────

    def _on_form_changed(self) -> None:
        if not self._is_dirty:
            self._is_dirty = True
            self._save_btn.setText("Save Changes *")

    def _clear_dirty(self) -> None:
        self._is_dirty = False
        self._save_btn.setText("Save Changes")

    def check_unsaved_ok(self) -> bool:
        """Return True if safe to leave (no unsaved edits, or user confirms discard)."""
        if not self._is_dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes in Edit Metadata.\n\nDiscard them and leave?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Discard

    # ---- helpers ───────────────────────────────────────────────────────────

    def _set_form_enabled(self, enabled: bool) -> None:
        self._form.setEnabled(enabled)
        self._validate_btn.setEnabled(enabled)
        self._save_btn.setEnabled(enabled)
        self._revert_btn.setEnabled(enabled)

    def set_test_path(self, path: str) -> None:
        self._path_input.setText(path)
        self._load_from_path()

    def _browse_path(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Test Folder", str(Path.home())
        )
        if folder:
            self._path_input.setText(folder)
            self._load_from_path()

    def _load_from_path(self) -> None:
        path_str = self._path_input.text().strip()
        if not path_str:
            return
        self._test_folder = Path(path_str)

        if not self._test_folder.exists():
            self._banner.show_message(f"Folder does not exist: {path_str}", "error")
            self._set_form_enabled(False)
            return

        metadata_file = self._test_folder / "metadata.json"
        self._has_existing_file = metadata_file.exists()

        if self._has_existing_file:
            try:
                data = load_raw_metadata(self._test_folder) or {}
                self._original_data = data
                self._form.set_data(data)
                self._clear_dirty()
                self._set_form_enabled(True)
                self._template_box.setVisible(False)
                self._banner.show_message(
                    f"Loaded metadata.json from {self._test_folder.name}/", "success"
                )
            except Exception as exc:
                self._banner.show_message(f"Error loading metadata: {exc}", "error")
                self._set_form_enabled(False)
        else:
            self._banner.show_message(
                f"No metadata.json in {self._test_folder.name}/ — create from template below.",
                "warning",
            )
            self._template_box.setVisible(True)
            self._set_form_enabled(False)
            self._original_data = None
            self._clear_dirty()

    def _create_from_template(self) -> None:
        if not self._test_folder:
            return
        tmpl_type = self._tmpl_combo.currentText()
        try:
            template = create_metadata_template(tmpl_type, include_examples=True)
        except Exception:
            template = {}
        folder_name = self._test_folder.name
        template["test_id"] = folder_name
        parsed = TestMetadata.from_test_id(folder_name)
        template.setdefault("system", parsed.system)
        template.setdefault("campaign_id", parsed.campaign_id)
        template.setdefault("test_type", parsed.test_type)
        template.setdefault("status", "pending")

        self._original_data = None
        self._form.set_data(template)
        self._clear_dirty()
        self._set_form_enabled(True)
        self._template_box.setVisible(False)
        self._banner.show_message(
            "Template loaded — fill in the fields and click Save.", "info"
        )

    def _validate(self) -> None:
        data = self._form.get_data()
        try:
            is_valid, warnings = validate_raw_metadata(data)
        except Exception as exc:
            self._banner.show_message(f"Validation error: {exc}", "error")
            return
        if is_valid and not warnings:
            self._banner.show_message("Metadata is valid.", "success")
        else:
            msgs = "; ".join(warnings) if warnings else "Unknown validation failure."
            self._banner.show_message(f"Warnings: {msgs}", "warning")

    def _save(self) -> None:
        if not self._test_folder or not self._save_btn.isEnabled():
            return
        data = self._form.get_data()
        try:
            _, warnings = validate_raw_metadata(data)
            if warnings:
                reply = QMessageBox.question(
                    self, "Validation warnings",
                    "There are validation warnings:\n\n"
                    + "\n".join(f"• {w}" for w in warnings)
                    + "\n\nSave anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
        except Exception:
            pass
        try:
            save_raw_metadata(self._test_folder, data)
            self._original_data = json.loads(json.dumps(data, default=str))
            self._has_existing_file = True
            self._clear_dirty()
            self._banner.show_message(
                f"Saved metadata.json to {self._test_folder.name}/", "success"
            )
        except Exception as exc:
            self._banner.show_message(f"Save failed: {exc}", "error")

    def _revert(self) -> None:
        if not self._test_folder:
            return
        if self._has_existing_file:
            try:
                data = load_raw_metadata(self._test_folder) or {}
                self._original_data = data
                self._form.set_data(data)
                self._clear_dirty()
                self._banner.show_message("Reverted to saved file.", "info")
            except Exception as exc:
                self._banner.show_message(f"Revert failed: {exc}", "error")
        else:
            self._form.set_data({})
            self._clear_dirty()
            self._banner.show_message("Form cleared.", "info")


# ===========================================================================
# Top-level page
# ===========================================================================

class TestIngestionPage(BasePage):
    """Qt equivalent of pages/1_Test_Explorer.py."""

    open_in_analysis_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            title="Test Explorer",
            description="Browse test data structure, create new tests, and manage metadata",
            parent=parent,
        )
        self._selected_system: str = ""
        self._selected_campaign_id: str = ""
        self._last_tab_idx: int = 0

        # ── No-context placeholder ────────────────────────────────────────────
        self._no_ctx_widget = self._build_no_context_widget()
        self.content_layout.addWidget(self._no_ctx_widget)

        # ── Main content (tabs) ───────────────────────────────────────────────
        self._main_widget = QWidget()
        main_lay = QVBoxLayout(self._main_widget)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        self._tabs = QTabWidget()
        self._browse_tab = BrowseTab()
        self._ingest_tab = IngestTab()
        self._edit_tab   = EditMetadataTab()

        self._tabs.addTab(self._browse_tab, "Browse Tests")
        self._tabs.addTab(self._ingest_tab, "Ingest New Test")
        self._tabs.addTab(self._edit_tab,   "Edit Metadata")

        main_lay.addWidget(self._tabs)
        self.content_layout.addWidget(self._main_widget, 1)
        self._main_widget.setVisible(False)

        # ── Cross-tab wiring ──────────────────────────────────────────────────
        self._browse_tab.edit_metadata_requested.connect(self._on_edit_metadata_requested)
        self._browse_tab.open_in_analysis_requested.connect(
            self.open_in_analysis_requested.emit
        )
        self._browse_tab.campaign_context_changed.connect(self._on_campaign_context_changed)
        self._browse_tab.status_message.connect(self.status_message.emit)
        self._ingest_tab.test_created.connect(self._on_test_created)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        sc_refresh = QShortcut(QKeySequence("F5"), self)
        sc_refresh.setContext(Qt.WidgetWithChildrenShortcut)
        sc_refresh.activated.connect(self._refresh_browse)

    # ---- no-context placeholder ─────────────────────────────────────────────

    def _build_no_context_widget(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 16, 0, 0)
        lay.setSpacing(16)

        info = InfoBanner(
            "Select a Test Root folder and Program in the left sidebar to get started.",
            "info",
        )
        lay.addWidget(info)

        steps_box = QGroupBox("Getting Started")
        steps_lay = QVBoxLayout(steps_box)
        steps_lay.setContentsMargins(12, 12, 12, 12)
        for step in [
            "1. Click '…' next to Test Root in the sidebar and pick a folder",
            "2. Select a Test Program from the dropdown",
            "3. Browse Systems, Campaigns, and Tests in the Browse tab",
            "4. Create new tests with the Ingest New Test tab",
        ]:
            lbl = QLabel(step)
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {SZ_BASE};")
            steps_lay.addWidget(lbl)
        lay.addWidget(steps_box)

        create_box = QGroupBox("Create New Test Root")
        create_lay = QHBoxLayout(create_box)
        create_lay.setContentsMargins(12, 12, 12, 12)
        create_lay.setSpacing(8)
        create_lay.addWidget(QLabel("Path:"))
        self._quick_path = QLineEdit()
        self._quick_path.setPlaceholderText("/path/to/create/test_data")
        create_lay.addWidget(self._quick_path, 1)
        create_btn = QPushButton("Create Test Root")
        create_btn.clicked.connect(self._create_test_root)
        create_lay.addWidget(create_btn)
        lay.addWidget(create_box)

        lay.addStretch()
        return w

    def _create_test_root(self) -> None:
        path_str = self._quick_path.text().strip()
        if not path_str:
            return
        try:
            Path(path_str).mkdir(parents=True, exist_ok=True)
            QMessageBox.information(
                self, "Created",
                f"Created test root: {path_str}\n\nSet it as the Test Root in the sidebar.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    # ---- context lifecycle ───────────────────────────────────────────────────

    def on_context_changed(self) -> None:
        has_ctx = bool(self._test_root and self._program)
        self._no_ctx_widget.setVisible(not has_ctx)
        self._main_widget.setVisible(has_ctx)

        if has_ctx:
            root = Path(self._test_root)
            self._browse_tab.refresh(root, self._program)
            self._ingest_tab.set_context(
                root, self._program,
                self._selected_system, self._selected_campaign_id,
            )

    # ---- tab-switch guard ───────────────────────────────────────────────────

    def _on_tab_changed(self, new_idx: int) -> None:
        edit_idx = self._tabs.indexOf(self._edit_tab)
        if self._last_tab_idx == edit_idx and new_idx != edit_idx:
            if not self._edit_tab.check_unsaved_ok():
                # Can't call setCurrentIndex inside currentChanged; defer it.
                QTimer.singleShot(0, lambda: self._tabs.setCurrentIndex(edit_idx))
        self._last_tab_idx = new_idx

    # ---- cross-tab actions ───────────────────────────────────────────────────

    def _on_edit_metadata_requested(self, path: str) -> None:
        self._edit_tab.set_test_path(path)
        self._tabs.setCurrentWidget(self._edit_tab)

    def _on_campaign_context_changed(self, system: str, campaign_id: str) -> None:
        """Browse tab selected a campaign — pre-select it in Ingest tab."""
        self._selected_system = system
        self._selected_campaign_id = campaign_id
        if self._test_root and self._program:
            self._ingest_tab.set_context(
                Path(self._test_root), self._program, system, campaign_id
            )

    def _on_test_created(self, _test_id: str) -> None:
        if self._test_root and self._program:
            self._browse_tab.refresh(Path(self._test_root), self._program)

    def _refresh_browse(self) -> None:
        """Refresh browse tab (F5)."""
        if self._test_root and self._program:
            self.status_message.emit("Refreshing test folder…")
            self._browse_tab.refresh(Path(self._test_root), self._program)

    # ---- public helpers ─────────────────────────────────────────────────────

    def preselect_from_browse(self, system: str, campaign_id: str) -> None:
        self._selected_system = system
        self._selected_campaign_id = campaign_id
        if self._test_root and self._program:
            self._ingest_tab.set_context(
                Path(self._test_root), self._program, system, campaign_id
            )
