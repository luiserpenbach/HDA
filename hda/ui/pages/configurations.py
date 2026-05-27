"""Configurations page — list, edit, create, and import testbench hardware configs.

Two-panel layout:
  Left  — scrollable config list with filter, + New, and Import buttons
  Right — structured editor with Details / Channels / Uncertainties / Settings / JSON tabs

Built-in configs are read-only; Clone creates an editable copy in saved_configs/.
The JSON tab shows a diff view (original | current) when there are unsaved changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.saved_configs import (
    BUILTIN_SAVED_CONFIGS,
    SavedConfig,
    SavedConfigManager,
)

from hda.ui.pages.base import BasePage, InfoBanner
from hda.ui.style import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BORDER,
    CONTENT_SECONDARY_BG,
    SZ_BASE,
    SZ_SM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "saved_configs")


def _secondary_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setProperty("secondary", "true")
    return btn


# ---------------------------------------------------------------------------
# Generic editable table with add / remove row controls
# ---------------------------------------------------------------------------

class _TableEditor(QWidget):
    """Editable table for key→value style config sections."""

    data_changed = Signal()

    def __init__(
        self,
        columns: List[str],
        placeholder_row: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._columns = columns
        self._placeholder = placeholder_row or [""] * len(columns)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._tbl = QTableWidget(0, len(columns))
        self._tbl.setHorizontalHeaderLabels(columns)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.verticalHeader().setDefaultSectionSize(26)
        self._tbl.horizontalHeader().setHighlightSections(False)
        self._tbl.horizontalHeader().setStretchLastSection(True)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.itemChanged.connect(lambda _i: self.data_changed.emit())
        lay.addWidget(self._tbl, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._add_btn = _secondary_btn("+ Add Row")
        self._add_btn.clicked.connect(self._add_row)
        self._del_btn = _secondary_btn("− Remove Row")
        self._del_btn.clicked.connect(self._remove_row)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._del_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

    def _add_row(self) -> None:
        row = self._tbl.rowCount()
        self._tbl.insertRow(row)
        for col, val in enumerate(self._placeholder):
            self._tbl.setItem(row, col, QTableWidgetItem(val))
        self.data_changed.emit()

    def _remove_row(self) -> None:
        items = self._tbl.selectedItems()
        if items:
            self._tbl.removeRow(items[0].row())
            self.data_changed.emit()

    def get_rows(self) -> List[List[str]]:
        result = []
        for r in range(self._tbl.rowCount()):
            cells = []
            for c in range(self._tbl.columnCount()):
                item = self._tbl.item(r, c)
                cells.append(item.text().strip() if item else "")
            result.append(cells)
        return result

    def set_rows(self, rows: List[List[str]]) -> None:
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                if c < self._tbl.columnCount():
                    self._tbl.setItem(r, c, QTableWidgetItem(str(val)))
        self._tbl.blockSignals(False)

    def set_editable(self, editable: bool) -> None:
        self._tbl.setEditTriggers(
            QAbstractItemView.AllEditTriggers
            if editable
            else QAbstractItemView.NoEditTriggers
        )
        self._add_btn.setEnabled(editable)
        self._del_btn.setEnabled(editable)


# ---------------------------------------------------------------------------
# Config list panel (left side)
# ---------------------------------------------------------------------------

class _ConfigListPanel(QWidget):
    config_selected      = Signal(str)   # config_id
    new_config_requested = Signal()
    import_requested     = Signal()

    def __init__(
        self, manager: SavedConfigManager, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._all: List[Dict[str, Any]] = []

        self.setFixedWidth(270)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter configs…")
        self._search.textChanged.connect(self._apply_filter)
        lay.addWidget(self._search)

        self._tbl = QTableWidget(0, 2)
        self._tbl.setHorizontalHeaderLabels(["Name", "Type"])
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.verticalHeader().setDefaultSectionSize(26)
        self._tbl.horizontalHeader().setHighlightSections(False)
        self._tbl.horizontalHeader().setStretchLastSection(True)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.itemSelectionChanged.connect(self._on_selection)
        lay.addWidget(self._tbl, 1)

        new_btn = QPushButton("+ New Config")
        new_btn.clicked.connect(self.new_config_requested.emit)
        import_btn = _secondary_btn("Import from File…")
        import_btn.clicked.connect(self.import_requested.emit)
        lay.addWidget(new_btn)
        lay.addWidget(import_btn)

        self.refresh()

    def refresh(self, select_id: Optional[str] = None) -> None:
        self._all = self._manager.list_templates()
        self._apply_filter()
        if select_id:
            self.select_config(select_id)

    def _apply_filter(self) -> None:
        q = self._search.text().strip().lower()
        configs = [
            c for c in self._all
            if not q or q in c["name"].lower() or q in c["test_type"].lower()
        ]
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(len(configs))
        for row, cfg in enumerate(configs):
            name_item = QTableWidgetItem(cfg["name"])
            name_item.setData(Qt.UserRole, cfg["id"])
            if cfg.get("builtin"):
                name_item.setForeground(QBrush(QColor(TEXT_MUTED)))
            self._tbl.setItem(row, 0, name_item)

            tt = cfg["test_type"]
            type_item = QTableWidgetItem(tt.replace("_", " ").title())
            type_item.setForeground(QBrush(QColor(
                ACCENT_BLUE if tt == "cold_flow" else ACCENT_AMBER
            )))
            self._tbl.setItem(row, 1, type_item)
        self._tbl.blockSignals(False)

    def _on_selection(self) -> None:
        items = self._tbl.selectedItems()
        if items:
            config_id = self._tbl.item(items[0].row(), 0).data(Qt.UserRole)
            if config_id:
                self.config_selected.emit(config_id)

    def select_config(self, config_id: str) -> None:
        for row in range(self._tbl.rowCount()):
            item = self._tbl.item(row, 0)
            if item and item.data(Qt.UserRole) == config_id:
                self._tbl.selectRow(row)
                return

    def selected_id(self) -> Optional[str]:
        items = self._tbl.selectedItems()
        if items:
            return self._tbl.item(items[0].row(), 0).data(Qt.UserRole)
        return None


# ---------------------------------------------------------------------------
# Config editor panel (right side)
# ---------------------------------------------------------------------------

class _ConfigEditorWidget(QWidget):
    config_saved   = Signal(str)   # config_id (also fires after clone)
    config_deleted = Signal(str)   # config_id
    use_in_analysis_requested = Signal(str)   # config_id

    def __init__(
        self, manager: SavedConfigManager, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._config_id: Optional[str] = None
        self._is_builtin: bool = False
        self._is_dirty: bool = False
        self._original_dict: Optional[Dict[str, Any]] = None
        self._json_tab_idx: int = 4

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Placeholder shown when nothing is selected
        self._placeholder = QLabel("← Select a configuration to view or edit")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_BASE};"
        )
        lay.addWidget(self._placeholder)

        # Editor container (hidden until a config is selected)
        self._editor = QWidget()
        editor_lay = QVBoxLayout(self._editor)
        editor_lay.setContentsMargins(0, 0, 0, 0)
        editor_lay.setSpacing(8)
        self._editor.setVisible(False)

        # ── Header row ──────────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)

        self._hdr_name = QLabel()
        self._hdr_name.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {TEXT_PRIMARY};"
        )
        self._hdr_badge = QLabel()
        hdr_row.addWidget(self._hdr_name)
        hdr_row.addWidget(self._hdr_badge)
        hdr_row.addStretch()

        self._btn_save   = QPushButton("Save Changes")
        self._btn_revert = _secondary_btn("Revert")
        self._btn_use    = _secondary_btn("Use in Analysis")
        self._btn_use.setToolTip("Open Single Test Analysis with this configuration selected")
        self._btn_clone  = _secondary_btn("Clone")
        self._btn_export = _secondary_btn("Export…")
        self._btn_delete = _secondary_btn("Delete")
        self._btn_delete.setStyleSheet(
            f"QPushButton[secondary='true'] {{ color: {ACCENT_RED}; border-color: {ACCENT_RED}; }}"
            f"QPushButton[secondary='true']:hover {{ background: #fef2f2; }}"
        )

        for btn in (self._btn_save, self._btn_revert, self._btn_use, self._btn_clone,
                    self._btn_export, self._btn_delete):
            hdr_row.addWidget(btn)

        self._btn_save.clicked.connect(self._save)
        self._btn_revert.clicked.connect(self._revert)
        self._btn_use.clicked.connect(self._use_in_analysis)
        self._btn_clone.clicked.connect(self._clone)
        self._btn_export.clicked.connect(self._export)
        self._btn_delete.clicked.connect(self._delete)
        editor_lay.addLayout(hdr_row)

        self._banner = InfoBanner(parent=self._editor)
        editor_lay.addWidget(self._banner)

        # ── Tab widget ───────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        editor_lay.addWidget(self._tabs, 1)

        # -- Details tab
        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setFrameShape(QScrollArea.NoFrame)
        details_inner = QWidget()
        details_lay = QFormLayout(details_inner)
        details_lay.setContentsMargins(4, 8, 4, 4)
        details_lay.setSpacing(8)
        details_lay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._f_name    = QLineEdit()
        self._f_version = QLineEdit()
        self._f_version.setFixedWidth(100)
        self._f_desc    = QLineEdit()
        self._f_type    = QComboBox()
        self._f_type.addItems(["cold_flow", "hot_fire"])
        self._f_author  = QLineEdit()
        self._f_tags    = QLineEdit()
        self._f_tags.setPlaceholderText("comma-separated (e.g. injector, mtb)")

        self._f_parent  = QLineEdit()
        self._f_parent.setReadOnly(True)
        self._f_parent.setStyleSheet(
            f"background: {CONTENT_SECONDARY_BG}; color: {TEXT_MUTED};"
        )

        details_lay.addRow("Name:", self._f_name)
        details_lay.addRow("Version:", self._f_version)
        details_lay.addRow("Description:", self._f_desc)
        details_lay.addRow("Test Type:", self._f_type)
        details_lay.addRow("Author:", self._f_author)
        details_lay.addRow("Tags:", self._f_tags)
        details_lay.addRow("Parent Config:", self._f_parent)
        details_scroll.setWidget(details_inner)
        self._tabs.addTab(details_scroll, "Details")

        # -- Channels tab
        ch_container = QWidget()
        ch_lay = QVBoxLayout(ch_container)
        ch_lay.setContentsMargins(0, 4, 0, 0)
        ch_info = QLabel(
            "Map DAQ acquisition channel IDs to sensor names used in analysis."
        )
        ch_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        ch_info.setWordWrap(True)
        self._tbl_channels = _TableEditor(
            ["DAQ Channel ID", "Sensor Name"],
            placeholder_row=["", ""],
        )
        ch_lay.addWidget(ch_info)
        ch_lay.addWidget(self._tbl_channels, 1)
        self._tabs.addTab(ch_container, "Channels")

        # -- Uncertainties tab
        unc_container = QWidget()
        unc_lay = QVBoxLayout(unc_container)
        unc_lay.setContentsMargins(0, 4, 0, 0)
        unc_info = QLabel(
            "Sensor measurement uncertainties. Type: 'rel' = relative (fraction), "
            "'abs' = absolute (engineering units)."
        )
        unc_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        unc_info.setWordWrap(True)
        self._tbl_uncertainties = _TableEditor(
            ["Sensor / Category", "Type (rel/abs)", "Value"],
            placeholder_row=["", "rel", "0.005"],
        )
        unc_lay.addWidget(unc_info)
        unc_lay.addWidget(self._tbl_uncertainties, 1)
        self._tabs.addTab(unc_container, "Uncertainties")

        # -- Settings tab
        set_container = QWidget()
        set_lay = QVBoxLayout(set_container)
        set_lay.setContentsMargins(0, 4, 0, 0)
        set_info = QLabel(
            "Sampling, processing, and QC settings. Numeric values are "
            "auto-detected as int or float."
        )
        set_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        set_info.setWordWrap(True)
        self._tbl_settings = _TableEditor(
            ["Key", "Value"],
            placeholder_row=["", ""],
        )
        set_lay.addWidget(set_info)
        set_lay.addWidget(self._tbl_settings, 1)
        self._tabs.addTab(set_container, "Settings")

        # -- JSON tab (single panel, or side-by-side diff when dirty)
        json_container = QWidget()
        json_lay = QVBoxLayout(json_container)
        json_lay.setContentsMargins(0, 4, 0, 0)
        json_lay.setSpacing(4)

        json_btn_row = QHBoxLayout()
        copy_btn = _secondary_btn("Copy JSON")
        copy_btn.clicked.connect(self._copy_json)
        self._diff_lbl = QLabel()
        self._diff_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_SM};")
        json_btn_row.addWidget(copy_btn)
        json_btn_row.addWidget(self._diff_lbl)
        json_btn_row.addStretch()
        json_lay.addLayout(json_btn_row)

        self._json_stack = QStackedWidget()

        # Index 0: single panel (no unsaved changes)
        self._json_single = QPlainTextEdit()
        self._json_single.setReadOnly(True)
        self._json_single.setFont(self._json_single.document().defaultFont())
        self._json_stack.addWidget(self._json_single)

        # Index 1: side-by-side diff
        diff_splitter = QSplitter(Qt.Horizontal)
        diff_splitter.setHandleWidth(1)

        orig_grp = QGroupBox("Saved")
        orig_inner = QVBoxLayout(orig_grp)
        orig_inner.setContentsMargins(4, 8, 4, 4)
        self._json_orig = QPlainTextEdit()
        self._json_orig.setReadOnly(True)
        orig_inner.addWidget(self._json_orig)

        curr_grp = QGroupBox("Current (unsaved)")
        curr_inner = QVBoxLayout(curr_grp)
        curr_inner.setContentsMargins(4, 8, 4, 4)
        self._json_curr = QPlainTextEdit()
        self._json_curr.setReadOnly(True)
        curr_inner.addWidget(self._json_curr)

        diff_splitter.addWidget(orig_grp)
        diff_splitter.addWidget(curr_grp)
        self._json_stack.addWidget(diff_splitter)

        json_lay.addWidget(self._json_stack, 1)
        self._json_tab_idx = self._tabs.addTab(json_container, "JSON")

        # ── Connect change signals ───────────────────────────────────────────
        for w in (self._f_name, self._f_version, self._f_desc, self._f_author, self._f_tags):
            w.textChanged.connect(lambda _t: self._mark_dirty())
        self._f_type.currentTextChanged.connect(lambda _t: self._mark_dirty())
        for tbl in (self._tbl_channels, self._tbl_uncertainties, self._tbl_settings):
            tbl.data_changed.connect(self._mark_dirty)

        lay.addWidget(self._editor, 1)

        # Ctrl+S
        shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut.activated.connect(self._save)

    # ── Load / clear ────────────────────────────────────────────────────────

    def load_config(self, config_id: str) -> None:
        if self._is_dirty and not self.check_unsaved_ok():
            return
        cfg = self._manager.get_template(config_id)
        if cfg is None:
            self._banner.show_message(f"Config not found: {config_id}", "error")
            return

        self._config_id = config_id
        self._is_builtin = config_id in BUILTIN_SAVED_CONFIGS
        self._original_dict = cfg.to_dict()

        self._populate_form(cfg)
        self._clear_dirty()
        self._update_header()
        self._update_edit_mode()
        self._update_json_panel()

        self._placeholder.setVisible(False)
        self._editor.setVisible(True)

    def clear(self) -> None:
        self._config_id = None
        self._is_builtin = False
        self._is_dirty = False
        self._original_dict = None
        self._placeholder.setVisible(True)
        self._editor.setVisible(False)

    def check_unsaved_ok(self) -> bool:
        if not self._is_dirty or self._is_builtin:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            f"'{self._config_id}' has unsaved changes.\n\nDiscard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Discard

    # ── Form population ─────────────────────────────────────────────────────

    def _populate_form(self, cfg: SavedConfig) -> None:
        # Block signals during bulk load
        for w in (self._f_name, self._f_version, self._f_desc, self._f_author, self._f_tags):
            w.blockSignals(True)
        self._f_type.blockSignals(True)

        self._f_name.setText(cfg.name)
        self._f_version.setText(cfg.version)
        self._f_desc.setText(cfg.description)
        idx = self._f_type.findText(cfg.test_type)
        self._f_type.setCurrentIndex(idx if idx >= 0 else 0)
        self._f_author.setText(cfg.author or "")
        self._f_tags.setText(", ".join(cfg.tags))
        self._f_parent.setText(cfg.parent_config or "")

        for w in (self._f_name, self._f_version, self._f_desc, self._f_author, self._f_tags):
            w.blockSignals(False)
        self._f_type.blockSignals(False)

        # Channels table
        self._tbl_channels.set_rows(
            [[ch_id, sensor] for ch_id, sensor in sorted(cfg.channel_config.items())]
        )

        # Uncertainties table
        unc_rows = []
        for sensor, spec in cfg.uncertainties.items():
            if isinstance(spec, dict):
                unc_rows.append([
                    sensor,
                    spec.get("type", "rel"),
                    str(spec.get("value", "")),
                ])
            else:
                unc_rows.append([sensor, "rel", str(spec)])
        self._tbl_uncertainties.set_rows(unc_rows)

        # Settings table (merge settings + qc into one flat view)
        rows = [[k, str(v)] for k, v in cfg.settings.items()]
        rows += [[k, str(v)] for k, v in cfg.qc.items()]
        self._tbl_settings.set_rows(rows)

    # ── Serialise current form state ────────────────────────────────────────

    def _current_dict(self) -> Dict[str, Any]:
        tags = [t.strip() for t in self._f_tags.text().split(",") if t.strip()]
        channels = {
            row[0]: row[1]
            for row in self._tbl_channels.get_rows()
            if row[0]
        }
        uncertainties: Dict[str, Any] = {}
        for row in self._tbl_uncertainties.get_rows():
            if row[0]:
                try:
                    val: Any = float(row[2]) if row[2] else 0.0
                except ValueError:
                    val = row[2]
                uncertainties[row[0]] = {"type": row[1] or "rel", "value": val}

        settings: Dict[str, Any] = {}
        for row in self._tbl_settings.get_rows():
            if row[0]:
                val_str = row[1]
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                settings[row[0]] = val

        base = self._original_dict or {}
        return {
            "config_name": self._f_name.text().strip() or "Unnamed",
            "version": self._f_version.text().strip() or "1.0.0",
            "test_type": self._f_type.currentText(),
            "description": self._f_desc.text().strip(),
            "author": self._f_author.text().strip(),
            "tags": tags,
            "channel_config": channels,
            "uncertainties": uncertainties,
            "settings": settings,
            "qc": base.get("qc", {}),
            "created_date": base.get("created_date", ""),
            "parent_config": self._f_parent.text().strip() or None,
        }

    # ── Dirty tracking ───────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        if not self._is_dirty:
            self._is_dirty = True
            self._btn_save.setText("Save Changes *")
            self._update_header()
        if self._tabs.currentIndex() == self._json_tab_idx:
            self._update_json_panel()

    def _clear_dirty(self) -> None:
        self._is_dirty = False
        self._btn_save.setText("Save Changes")

    # ── Header / edit-mode helpers ───────────────────────────────────────────

    def _update_header(self) -> None:
        saved_name = (
            self._original_dict.get("config_name", "Unnamed")
            if self._original_dict
            else "Unnamed"
        )
        self._hdr_name.setText(saved_name + (" *" if self._is_dirty else ""))

        tt = (self._original_dict or {}).get("test_type", "cold_flow")
        if tt == "cold_flow":
            self._hdr_badge.setText("Cold Flow")
            self._hdr_badge.setStyleSheet(
                f"font-size: {SZ_SM}; font-weight: 600; padding: 2px 8px; "
                f"border-radius: 3px; background: #eff6ff; color: {ACCENT_BLUE}; "
                "border: 1px solid #bfdbfe;"
            )
        else:
            self._hdr_badge.setText("Hot Fire")
            self._hdr_badge.setStyleSheet(
                f"font-size: {SZ_SM}; font-weight: 600; padding: 2px 8px; "
                f"border-radius: 3px; background: #fffbeb; color: {ACCENT_AMBER}; "
                "border: 1px solid #fde68a;"
            )

    def _update_edit_mode(self) -> None:
        editable = not self._is_builtin
        for w in (self._f_name, self._f_version, self._f_desc, self._f_author, self._f_tags):
            w.setReadOnly(not editable)
        self._f_type.setEnabled(editable)
        for tbl in (self._tbl_channels, self._tbl_uncertainties, self._tbl_settings):
            tbl.set_editable(editable)

        self._btn_save.setEnabled(editable)
        self._btn_revert.setEnabled(editable)
        self._btn_delete.setEnabled(editable)
        self._btn_use.setEnabled(bool(self._config_id))
        # Clone and Export always available
        self._btn_clone.setEnabled(True)
        self._btn_export.setEnabled(True)

        if self._is_builtin:
            self._banner.show_message(
                "Built-in config — read only. Click Clone to create an editable copy.",
                "info",
            )
        else:
            self._banner.clear_message()

    # ── JSON panel ───────────────────────────────────────────────────────────

    def _on_tab_changed(self, idx: int) -> None:
        if idx == self._json_tab_idx:
            self._update_json_panel()

    def _update_json_panel(self) -> None:
        current_json = json.dumps(self._current_dict(), indent=2)
        if self._is_dirty and self._original_dict:
            self._json_stack.setCurrentIndex(1)
            self._json_orig.setPlainText(json.dumps(self._original_dict, indent=2))
            self._json_curr.setPlainText(current_json)
            self._diff_lbl.setText("Diff view — unsaved changes highlighted")
        else:
            self._json_stack.setCurrentIndex(0)
            self._json_single.setPlainText(current_json)
            self._diff_lbl.setText("")

    def _copy_json(self) -> None:
        QApplication.clipboard().setText(json.dumps(self._current_dict(), indent=2))
        self._banner.show_message("JSON copied to clipboard.", "success")

    # ── Actions ──────────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._is_builtin or not self._config_id or not self._btn_save.isEnabled():
            return
        d = self._current_dict()
        cfg = SavedConfig.from_dict(d)
        try:
            self._manager.save_template(cfg, self._config_id)
            self._original_dict = d
            self._clear_dirty()
            self._update_header()
            self._update_json_panel()
            self._banner.show_message(f"Saved  {self._config_id}.json", "success")
            self.config_saved.emit(self._config_id)
        except Exception as exc:
            self._banner.show_message(f"Save failed: {exc}", "error")

    def _revert(self) -> None:
        if self._is_builtin or not self._config_id or not self._is_dirty:
            return
        reply = QMessageBox.question(
            self, "Revert",
            "Discard all unsaved changes and reload from disk?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        cfg = self._manager.get_template(self._config_id)
        if cfg:
            self._original_dict = cfg.to_dict()
            self._populate_form(cfg)
            self._clear_dirty()
            self._update_header()
            self._update_json_panel()
            self._banner.show_message("Reverted to saved version.", "info")

    def _use_in_analysis(self) -> None:
        if not self._config_id:
            return
        if self._is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "This configuration has unsaved changes. Save before opening in analysis?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Save:
                self._save()
                if self._is_dirty:
                    return
        self.use_in_analysis_requested.emit(self._config_id)

    def _clone(self) -> None:
        if not self._config_id:
            return
        src_name = (self._original_dict or {}).get("config_name", self._config_id)
        name, ok = QInputDialog.getText(
            self, "Clone Config", "Name for the cloned config:",
            text=f"Copy of {src_name}",
        )
        name = name.strip()
        if not ok or not name:
            return
        d = self._current_dict()
        d["config_name"] = name
        d["parent_config"] = self._config_id
        cfg = SavedConfig.from_dict(d)
        try:
            new_id = self._manager.save_template(cfg)
            self._banner.show_message(f"Cloned as {new_id}", "success")
            self.config_saved.emit(new_id)
        except Exception as exc:
            self._banner.show_message(f"Clone failed: {exc}", "error")

    def _export(self) -> None:
        if not self._config_id:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Config",
            str(Path.home() / f"{self._config_id}.json"),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w") as fh:
                json.dump(self._current_dict(), fh, indent=2)
            self._banner.show_message(f"Exported to {Path(path).name}", "success")
        except Exception as exc:
            self._banner.show_message(f"Export failed: {exc}", "error")

    def _delete(self) -> None:
        if self._is_builtin or not self._config_id:
            return
        reply = QMessageBox.question(
            self, "Delete Config",
            f"Permanently delete '{self._config_id}.json'?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            deleted_id = self._config_id
            self._manager.delete_template(self._config_id)
            self.clear()
            self.config_deleted.emit(deleted_id)
        except Exception as exc:
            self._banner.show_message(f"Delete failed: {exc}", "error")


# ---------------------------------------------------------------------------
# Top-level page
# ---------------------------------------------------------------------------

class ConfigurationsPage(BasePage):
    """Manage testbench hardware configurations stored in saved_configs/."""

    use_in_analysis_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            title="Configurations",
            description="Manage testbench hardware configurations and sensor calibration data",
            parent=parent,
        )
        self._manager = SavedConfigManager(_CONFIGS_DIR)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        self._list_panel = _ConfigListPanel(self._manager)
        splitter.addWidget(self._list_panel)

        self._editor = _ConfigEditorWidget(self._manager)
        splitter.addWidget(self._editor)

        splitter.setSizes([270, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.content_layout.addWidget(splitter, 1)

        # Wiring
        self._list_panel.config_selected.connect(self._editor.load_config)
        self._list_panel.new_config_requested.connect(self._new_config)
        self._list_panel.import_requested.connect(self._import_config)
        self._editor.config_saved.connect(
            lambda cid: self._list_panel.refresh(select_id=cid)
        )
        self._editor.config_saved.connect(
            lambda cid: self.status_message.emit(f"Saved configuration '{cid}'.")
        )
        self._editor.config_deleted.connect(lambda _cid: self._list_panel.refresh())
        self._editor.use_in_analysis_requested.connect(self._on_use_in_analysis)

    def _on_use_in_analysis(self, config_id: str) -> None:
        self.status_message.emit(f"Opening analysis with config '{config_id}'…")
        self.use_in_analysis_requested.emit(config_id)

    # ---- new / import ───────────────────────────────────────────────────────

    def _new_config(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New Configuration", "Configuration name:"
        )
        name = name.strip()
        if not ok or not name:
            return

        test_type, ok2 = QInputDialog.getItem(
            self, "Test Type", "Select test type:",
            ["cold_flow", "hot_fire"], 0, False,
        )
        if not ok2:
            return

        builtin_id = "cold_flow_default" if test_type == "cold_flow" else "hot_fire_default"
        try:
            cfg = self._manager.create_from_parent(builtin_id, name, {})
            new_id = self._manager.save_template(cfg)
            self._list_panel.refresh(select_id=new_id)
            self._editor.load_config(new_id)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _import_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Config", str(Path.home()), "JSON files (*.json)"
        )
        if not path:
            return
        try:
            new_id = self._manager.import_template(path)
            self._list_panel.refresh(select_id=new_id)
            self._editor.load_config(new_id)
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))
