"""Left navigation bar.

Visual design: light zinc-50 sidebar — a subtle off-white, not a dark rail.
The background is set via QPalette (most reliable) rather than stylesheet alone.
A thin border-right is drawn in paintEvent.

Signals
-------
nav_changed(int)        — user clicks a nav item (page index)
test_root_changed(str)  — test-root path edited or browsed
program_changed(str)    — program selected or created
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hda.ui.style import (
    BORDER,
    SIDEBAR_BG,
    SIDEBAR_BORDER,
    SZ_XS,
    SZ_SM,
    SZ_BASE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    nav_stylesheet,
)


# ---------------------------------------------------------------------------
# Page registry — order matches QStackedWidget pages in main_window.py
# ---------------------------------------------------------------------------

NAV_ITEMS: list[tuple[str, str]] = [
    ("test_ingestion",    "Test Explorer"),
    ("single_test",       "Single Test Analysis"),
    ("batch_analysis",    "Batch Analysis"),
    ("campaign_analysis", "Campaign Analysis"),
    ("system_analysis",   "System Analysis"),
    ("analysis_tools",    "Analysis Tools"),
    ("configurations",    "Configurations"),
]


def _divider() -> QFrame:
    f = QFrame()
    f.setObjectName("NavDivider")
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none;")
    return f


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("NavSectionLabel")
    lbl.setStyleSheet(
        f"color: {TEXT_MUTED}; font-size: {SZ_XS}; font-weight: 700; "
        f"letter-spacing: 0.08em; background: transparent;"
    )
    return lbl


class NavBar(QWidget):
    nav_changed     = Signal(int)
    test_root_changed = Signal(str)
    program_changed = Signal(str)

    _MIN_WIDTH = 180

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(self._MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Reliable background via QPalette (stylesheet alone can leave gaps)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(SIDEBAR_BG))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Child styling via stylesheet (does NOT affect this widget's bg rect)
        self.setStyleSheet(nav_stylesheet())

        self._active_index = 0
        self._nav_buttons: list[QPushButton] = []

        self._build()

    # ---------------------------------------------------------------- paint

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        # Draw a thin right border separating nav from content
        p = QPainter(self)
        p.setPen(QColor(SIDEBAR_BORDER))
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())

    # ---------------------------------------------------------------- build

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── App identity ────────────────────────────────────────────────────
        lay.addWidget(self._build_header())
        lay.addWidget(_divider())

        # ── Global context ──────────────────────────────────────────────────
        lay.addWidget(self._build_context())
        lay.addWidget(_divider())

        # ── Navigation items ────────────────────────────────────────────────
        lay.addWidget(self._build_nav(), 1)  # stretch=1 so it fills remaining space

        lay.addWidget(_divider())

        # ── Footer ──────────────────────────────────────────────────────────
        lay.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setAutoFillBackground(False)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 12)
        lay.setSpacing(2)

        title = QLabel("Hopper Data Studio")
        title.setObjectName("AppTitle")
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 700; background: transparent;"
        )
        subtitle = QLabel("Rocket propulsion data analysis")
        subtitle.setObjectName("AppSubtitle")
        subtitle.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {SZ_SM}; background: transparent;"
        )
        lay.addWidget(title)
        lay.addWidget(subtitle)
        return w

    def _build_context(self) -> QWidget:
        w = QWidget()
        w.setAutoFillBackground(False)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        # TEST ROOT
        lay.addWidget(_section_label("TEST ROOT"))

        root_row = QHBoxLayout()
        root_row.setSpacing(4)

        self._root_input = QLineEdit()
        self._root_input.setObjectName("NavInput")
        self._root_input.setPlaceholderText("/path/to/test_data")
        self._root_input.setToolTip(
            "Root folder containing test program sub-folders.\n"
            "Each immediate sub-folder is treated as a Test Program."
        )
        self._root_input.editingFinished.connect(self._on_root_edited)
        root_row.addWidget(self._root_input, 1)

        browse_btn = QPushButton("…")
        browse_btn.setObjectName("NavMicroBtn")
        browse_btn.setToolTip("Browse for test root folder")
        browse_btn.clicked.connect(self._browse_root)
        root_row.addWidget(browse_btn)
        lay.addLayout(root_row)

        # PROGRAM
        lay.addWidget(_section_label("PROGRAM"))

        prog_row = QHBoxLayout()
        prog_row.setSpacing(4)

        self._program_combo = QComboBox()
        self._program_combo.setObjectName("NavCombo")
        self._program_combo.setToolTip(
            "Active test program.\n"
            "Each program is a top-level folder under the Test Root."
        )
        self._program_combo.currentTextChanged.connect(self._on_program_changed)
        prog_row.addWidget(self._program_combo, 1)

        new_prog_btn = QPushButton("+")
        new_prog_btn.setObjectName("NavMicroBtn")
        new_prog_btn.setToolTip("Create a new Test Program folder")
        new_prog_btn.clicked.connect(self._create_program)
        prog_row.addWidget(new_prog_btn)
        lay.addLayout(prog_row)

        return w

    def _build_nav(self) -> QWidget:
        w = QWidget()
        w.setAutoFillBackground(False)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(1)

        lbl = _section_label("NAVIGATION")
        lbl.setContentsMargins(4, 0, 0, 4)
        lay.addWidget(lbl)

        for idx, (_key, label) in enumerate(NAV_ITEMS):
            btn = QPushButton(label)
            btn.setObjectName("NavItem")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("active", "false")
            btn.clicked.connect(lambda _chk=False, i=idx: self._on_nav_clicked(i))
            lay.addWidget(btn)
            self._nav_buttons.append(btn)

        # Push nav items to the top of this section, spacer fills the rest
        lay.addStretch(1)
        return w

    def _build_footer(self) -> QWidget:
        w = QWidget()
        w.setAutoFillBackground(False)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 6, 14, 10)
        ver = QLabel("v2.4.0")
        ver.setObjectName("NavVersion")
        ver.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {SZ_XS}; background: transparent;")
        lay.addWidget(ver)
        return w

    # ---------------------------------------------------------------- active state

    def _set_active(self, index: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            is_active = i == index
            btn.setProperty("active", "true" if is_active else "false")
            # Force style re-evaluation
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._active_index = index

    # ---------------------------------------------------------------- slots

    def _on_nav_clicked(self, index: int) -> None:
        self._set_active(index)
        self.nav_changed.emit(index)

    def _browse_root(self) -> None:
        current = self._root_input.text().strip()
        start = current if current and Path(current).is_dir() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Test Root Folder", start)
        if folder:
            self._root_input.setText(folder)
            self._emit_root_changed(folder)

    def _on_root_edited(self) -> None:
        self._emit_root_changed(self._root_input.text().strip())

    def _emit_root_changed(self, path: str) -> None:
        self._refresh_programs(path)
        self.test_root_changed.emit(path)

    def _on_program_changed(self, text: str) -> None:
        if text:
            self.program_changed.emit(text)

    def _create_program(self) -> None:
        root = self._root_input.text().strip()
        if not root:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Test Root", "Set a Test Root folder first.")
            return
        name, ok = QInputDialog.getText(
            self, "New Test Program", "Program folder name (e.g. Engine-A):"
        )
        name = name.strip()
        if not ok or not name:
            return
        try:
            (Path(root) / name).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", str(exc))
            return
        self._refresh_programs(root)
        self._program_combo.setCurrentText(name)

    def _refresh_programs(self, root_path: str) -> None:
        root = Path(root_path)
        if not root.is_dir():
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

        current = self._program_combo.currentText()
        if current:
            self.program_changed.emit(current)

    # ---------------------------------------------------------------- public API

    def set_test_root(self, path: str) -> None:
        self._root_input.setText(path)
        self._refresh_programs(path)

    def set_program(self, program: str) -> None:
        if self._program_combo.findText(program) >= 0:
            self._program_combo.setCurrentText(program)

    def test_root(self) -> str:
        return self._root_input.text().strip()

    def program(self) -> str:
        return self._program_combo.currentText()

    def active_index(self) -> int:
        return self._active_index

    # ── Activate first item on startup ─────────────────────────────────────
    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._set_active(0)
