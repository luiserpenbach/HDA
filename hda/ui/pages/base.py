"""Base page infrastructure for the HDA Qt desktop UI.

Every navigation page inherits from BasePage, which provides:
  - A standard header (title + description only — no priority badges)
  - A context-update hook so the nav bar pushes root/program changes
  - A consistent layout scaffold
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from hda.ui.style import (
    SZ_2XL,
    SZ_BASE,
    SZ_SM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_AMBER,
    ACCENT_RED,
    BORDER,
    CONTENT_SECONDARY_BG,
    RADIUS_SM,
)


class PageHeader(QWidget):
    """Title + description row. No priority badges — those are internal dev labels."""

    def __init__(
        self,
        title: str,
        description: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: {SZ_2XL}; font-weight: 700; color: {TEXT_PRIMARY}; background: transparent;"
        )
        lay.addWidget(title_lbl)

        if description:
            desc = QLabel(description)
            desc.setStyleSheet(
                f"font-size: {SZ_BASE}; color: {TEXT_MUTED}; background: transparent;"
            )
            desc.setWordWrap(True)
            lay.addWidget(desc)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {BORDER}; border: none; margin-top: 6px;")
        lay.addWidget(divider)


class MetricCard(QWidget):
    """Small numeric metric widget — label above, large value below."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setStyleSheet(
            f"#MetricCard {{ border: 1px solid {BORDER}; border-radius: 6px; "
            f"background: {CONTENT_SECONDARY_BG}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(3)

        self._label_w = QLabel(label)
        self._label_w.setStyleSheet(
            f"font-size: {SZ_SM}; color: {TEXT_MUTED}; font-weight: 500; "
            f"border: none; background: transparent;"
        )
        lay.addWidget(self._label_w)

        self._value_w = QLabel(value)
        self._value_w.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {TEXT_PRIMARY}; "
            f"border: none; background: transparent;"
        )
        lay.addWidget(self._value_w)

    def set_value(self, value: str) -> None:
        self._value_w.setText(value)

    def set_label(self, label: str) -> None:
        self._label_w.setText(label)


class InfoBanner(QLabel):
    """Single-line info / success / warning / error banner."""

    _KINDS = {
        "info":    (ACCENT_BLUE,  TEXT_PRIMARY, CONTENT_SECONDARY_BG),
        "success": (ACCENT_GREEN, TEXT_PRIMARY, CONTENT_SECONDARY_BG),
        "warning": (ACCENT_AMBER, TEXT_PRIMARY, CONTENT_SECONDARY_BG),
        "error":   (ACCENT_RED,   TEXT_PRIMARY, CONTENT_SECONDARY_BG),
    }

    def __init__(
        self,
        text: str = "",
        kind: str = "info",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self._apply(kind)
        self.setVisible(bool(text))

    def _apply(self, kind: str) -> None:
        accent, fg, bg = self._KINDS.get(kind, self._KINDS["info"])
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {accent}; "
            f"border-left: 3px solid {accent}; border-radius: {RADIUS_SM}; "
            f"padding: 7px 12px; font-size: {SZ_SM};"
        )

    def show_message(self, text: str, kind: str = "info") -> None:
        self._apply(kind)
        self.setText(text)
        self.setVisible(True)

    def clear_message(self) -> None:
        self.clear()
        self.hide()


class BasePage(QWidget):
    """
    Abstract base for all navigation pages.

    Subclasses should:
      1. Call super().__init__(title, description, ...)
      2. Add content widgets to self.content_layout
      3. Override on_context_changed() to react to root/program updates

    Emit ``status_message`` for activity updates shown in the main window status bar.
    """

    status_message = Signal(str)

    def __init__(
        self,
        title: str,
        description: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._test_root: str = ""
        self._program: str = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(16)

        outer.addWidget(PageHeader(title, description))

        self.content_layout = outer

    # ---------------------------------------------------------------- context

    def set_context(self, test_root: str, program: str) -> None:
        changed = (test_root != self._test_root) or (program != self._program)
        self._test_root = test_root
        self._program = program
        if changed:
            self.on_context_changed()

    def on_context_changed(self) -> None:
        """Override to react when test_root or program changes."""

    @property
    def test_root(self) -> str:
        return self._test_root

    @property
    def program(self) -> str:
        return self._program
