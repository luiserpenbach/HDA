"""Base page infrastructure for the HDA Qt desktop UI.

Every page in the navigation inherits from BasePage. It provides:
  - A standard header (title, badge, description)
  - A context-update hook so the nav bar can push root/program changes
  - A consistent layout scaffold
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hda.ui.style import (
    ACCENT_BLUE,
    SZ_2XL,
    SZ_BASE,
    SZ_SM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    badge_style,
)


class PageHeader(QWidget):
    """Title + optional badge + description row."""

    def __init__(
        self,
        title: str,
        description: str = "",
        badge_text: str = "",
        badge_kind: str = "neutral",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 8)
        lay.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: {SZ_2XL}; font-weight: 700; color: {TEXT_PRIMARY};"
        )
        title_row.addWidget(title_lbl)

        if badge_text:
            badge = QLabel(badge_text)
            badge.setStyleSheet(badge_style(badge_kind))
            badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            title_row.addWidget(badge)

        title_row.addStretch()
        lay.addLayout(title_row)

        if description:
            desc_lbl = QLabel(description)
            desc_lbl.setStyleSheet(f"font-size: {SZ_BASE}; color: {TEXT_MUTED};")
            desc_lbl.setWordWrap(True)
            lay.addWidget(desc_lbl)


class MetricCard(QWidget):
    """Small numeric-metric card (label + value) matching the Streamlit st.metric style."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("card", "true")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(2)

        self._label = QLabel(label)
        self._label.setStyleSheet(f"font-size: {SZ_SM}; color: {TEXT_MUTED}; font-weight: 500;")
        lay.addWidget(self._label)

        self._value = QLabel(value)
        self._value.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {TEXT_PRIMARY};"
        )
        lay.addWidget(self._value)

    def set_value(self, value: str) -> None:
        self._value.setText(value)

    def set_label(self, label: str) -> None:
        self._label.setText(label)


class InfoBanner(QLabel):
    """Single-line info / warning / error banner."""

    KINDS = {
        "info":    ("#dbeafe", "#1d4ed8", "#eff6ff"),
        "success": ("#dcfce7", "#15803d", "#f0fdf4"),
        "warning": ("#fef3c7", "#92400e", "#fffbeb"),
        "error":   ("#fee2e2", "#991b1b", "#fef2f2"),
    }

    def __init__(self, text: str = "", kind: str = "info", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        border, fg, bg = self.KINDS.get(kind, self.KINDS["info"])
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {border}; "
            f"border-radius: 4px; padding: 6px 12px; font-size: {SZ_SM};"
        )
        self.setWordWrap(True)
        self.setVisible(bool(text))

    def show_message(self, text: str, kind: str = "info") -> None:
        border, fg, bg = self.KINDS.get(kind, self.KINDS["info"])
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {border}; "
            f"border-radius: 4px; padding: 6px 12px; font-size: {SZ_SM};"
        )
        self.setText(text)
        self.setVisible(bool(text))

    def clear_message(self) -> None:
        self.clear()
        self.hide()


class BasePage(QWidget):
    """
    Abstract base for all navigation pages.

    Subclasses should:
      1. Call ``super().__init__(title, description, ...)``
      2. Add content widgets to ``self.content_layout``
      3. Override ``on_context_changed`` to react to root/program updates
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        badge_text: str = "",
        badge_kind: str = "neutral",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._test_root: str = ""
        self._program: str = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(12)

        header = PageHeader(title, description, badge_text, badge_kind)
        outer.addWidget(header)

        # Subclasses add widgets here
        self.content_layout = outer

    # ---------------------------------------------------------------- context

    def set_context(self, test_root: str, program: str) -> None:
        """Called by MainWindow when root or program changes."""
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
