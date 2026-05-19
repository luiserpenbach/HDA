"""Left navigation bar with global test-root / program context.

The NavBar is a fixed-width QWidget that lives on the left side of the
main window. It owns:
  - App title + subtitle
  - Global context selectors (Test Root path + Program combo)
  - Navigation item buttons (one per page)
  - Version label at the bottom

Signals
-------
nav_changed(int)
    Emitted when the user clicks a nav item. The int is the page index.
test_root_changed(str)
    Emitted when the test-root path changes (user edits or picks a folder).
program_changed(str)
    Emitted when the user picks a different program from the combo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from hda.ui.style import nav_stylesheet


# ---------------------------------------------------------------------------
# Page registry — single source of truth for id, label, and page index
# ---------------------------------------------------------------------------

NAV_ITEMS: list[tuple[str, str]] = [
    ("test_ingestion",       "Test Explorer"),
    ("single_test",          "Single Test Analysis"),
    ("batch_analysis",       "Batch Analysis"),
    ("campaign_analysis",    "Campaign Analysis"),
    ("system_analysis",      "System Analysis"),
    ("analysis_tools",       "Analysis Tools"),
    ("configurations",       "Configurations"),
]


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setObjectName("NavDivider")
    line.setFixedHeight(1)
    return line


class NavBar(QWidget):
    nav_changed = Signal(int)
    test_root_changed = Signal(str)
    program_changed = Signal(str)

    WIDTH = 230

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("NavBar")
        self.setFixedWidth(self.WIDTH)
        self.setStyleSheet(nav_stylesheet())
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._active_index: int = 0
        self._nav_buttons: list[QPushButton] = []
        self._programs: list[str] = []

        self._build()

    # ---------------------------------------------------------------- build

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── App identity ──────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("NavBar")
        h_lay = QVBoxLayout(header)
        h_lay.setContentsMargins(16, 16, 16, 12)
        h_lay.setSpacing(2)

        title = QLabel("Hopper Data Studio")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Rocket propulsion data analysis")
        subtitle.setObjectName("AppSubtitle")
        h_lay.addWidget(title)
        h_lay.addWidget(subtitle)
        outer.addWidget(header)

        outer.addWidget(_divider())

        # ── Global context ─────────────────────────────────────────────────
        ctx = QWidget()
        ctx.setObjectName("NavBar")
        ctx_lay = QVBoxLayout(ctx)
        ctx_lay.setContentsMargins(12, 10, 12, 10)
        ctx_lay.setSpacing(6)

        root_label = QLabel("TEST ROOT")
        root_label.setObjectName("NavSectionLabel")
        ctx_lay.addWidget(root_label)

        root_row = QHBoxLayout()
        root_row.setSpacing(4)
        self._root_input = QLineEdit()
        self._root_input.setObjectName("NavInput")
        self._root_input.setPlaceholderText("/path/to/test_data")
        self._root_input.setToolTip("Root folder that contains test program folders")
        self._root_input.editingFinished.connect(self._on_root_edited)
        root_row.addWidget(self._root_input, 1)

        browse_btn = QPushButton("…")
        browse_btn.setObjectName("NavMicroBtn")
        browse_btn.setToolTip("Browse for test root folder")
        browse_btn.clicked.connect(self._on_browse_root)
        root_row.addWidget(browse_btn)
        ctx_lay.addLayout(root_row)

        program_label = QLabel("PROGRAM")
        program_label.setObjectName("NavSectionLabel")
        ctx_lay.addWidget(program_label)

        self._program_combo = QComboBox()
        self._program_combo.setObjectName("NavCombo")
        self._program_combo.setToolTip("Select the active test program")
        self._program_combo.currentTextChanged.connect(self._on_program_changed)
        ctx_lay.addWidget(self._program_combo)

        outer.addWidget(ctx)
        outer.addWidget(_divider())

        # ── Navigation items ───────────────────────────────────────────────
        nav_container = QWidget()
        nav_container.setObjectName("NavBar")
        nav_lay = QVBoxLayout(nav_container)
        nav_lay.setContentsMargins(8, 8, 8, 8)
        nav_lay.setSpacing(1)

        nav_label = QLabel("NAVIGATION")
        nav_label.setObjectName("NavSectionLabel")
        nav_label.setContentsMargins(4, 0, 0, 4)
        nav_lay.addWidget(nav_label)

        for idx, (key, label) in enumerate(NAV_ITEMS):
            btn = QPushButton(label)
            btn.setObjectName("NavItem")
            btn.setCheckable(False)
            btn.setProperty("active", "false")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _checked, i=idx: self._on_nav_clicked(i))
            self._nav_buttons.append(btn)
            nav_lay.addWidget(btn)

        outer.addWidget(nav_container)

        # ── Spacer ─────────────────────────────────────────────────────────
        outer.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        outer.addWidget(_divider())

        # ── Footer ─────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setObjectName("NavBar")
        f_lay = QVBoxLayout(footer)
        f_lay.setContentsMargins(16, 8, 16, 12)
        version_lbl = QLabel("v2.4.0")
        version_lbl.setObjectName("NavVersion")
        f_lay.addWidget(version_lbl)
        outer.addWidget(footer)

        # Activate first item
        self._set_active(0)

    # ---------------------------------------------------------------- helpers

    def _set_active(self, index: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._active_index = index

    def _on_nav_clicked(self, index: int) -> None:
        self._set_active(index)
        self.nav_changed.emit(index)

    def _on_browse_root(self) -> None:
        current = self._root_input.text().strip()
        start = current if current and Path(current).exists() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select Test Root Folder", start
        )
        if folder:
            self._root_input.setText(folder)
            self._emit_root_changed(folder)

    def _on_root_edited(self) -> None:
        self._emit_root_changed(self._root_input.text().strip())

    def _emit_root_changed(self, path: str) -> None:
        self.test_root_changed.emit(path)
        self._refresh_programs(path)

    def _on_program_changed(self, text: str) -> None:
        if text:
            self.program_changed.emit(text)

    def _refresh_programs(self, root_path: str) -> None:
        """Scan root directory and populate program combo."""
        root = Path(root_path)
        if not root.exists():
            self._program_combo.clear()
            return

        programs = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        prev = self._program_combo.currentText()
        self._program_combo.blockSignals(True)
        self._program_combo.clear()
        self._program_combo.addItems(programs)
        if prev in programs:
            self._program_combo.setCurrentText(prev)
        self._program_combo.blockSignals(False)
        # Emit after unblocking so the page can react
        if self._program_combo.currentText():
            self.program_changed.emit(self._program_combo.currentText())

    # ---------------------------------------------------------------- public API

    def set_test_root(self, path: str) -> None:
        """Pre-populate the test root (e.g. from persisted QSettings)."""
        self._root_input.setText(path)
        self._refresh_programs(path)

    def set_program(self, program: str) -> None:
        """Pre-select a program (e.g. from persisted QSettings)."""
        if self._program_combo.findText(program) >= 0:
            self._program_combo.setCurrentText(program)

    def test_root(self) -> str:
        return self._root_input.text().strip()

    def program(self) -> str:
        return self._program_combo.currentText()

    def active_index(self) -> int:
        return self._active_index
