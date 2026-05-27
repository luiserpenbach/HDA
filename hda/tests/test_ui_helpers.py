"""Tests for UI input helpers."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from hda.ui.style import resolve_app_font_family
from hda.ui.wheel_guard import WheelGuardFilter


def test_resolve_app_font_family_returns_non_empty_string():
    family = resolve_app_font_family()
    assert isinstance(family, str)
    assert family.strip()


def test_wheel_guard_blocks_unfocused_spinbox_wheel():
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox

    app = QApplication.instance() or QApplication([])
    guard = WheelGuardFilter(app)
    spin = QDoubleSpinBox()
    spin.setValue(10.0)
    spin.clearFocus()

    event = QWheelEvent(
        QPointF(0, 0),
        QPointF(0, 0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    blocked = guard.eventFilter(spin, event)
    assert blocked is True
    assert spin.value() == 10.0
