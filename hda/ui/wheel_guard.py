"""Block mouse-wheel changes on inputs unless the control has keyboard focus.

Without this, scrolling a panel accidentally changes combo boxes and spin boxes
under the cursor — a common Qt desktop UX pitfall.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDateEdit,
    QSlider,
    QTimeEdit,
)


class WheelGuardFilter(QObject):
    """Forward wheel events to scroll areas when inputs are not focused."""

    _BLOCK_TYPES = (
        QAbstractSpinBox,
        QComboBox,
        QSlider,
        QDateEdit,
        QTimeEdit,
    )

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel:
            return False
        if not isinstance(obj, self._BLOCK_TYPES):
            return False
        if obj.hasFocus():
            return False

        parent = obj.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                viewport = parent.viewport()
                if viewport is not None:
                    QApplication.sendEvent(viewport, event)
                return True
            parent = parent.parentWidget()
        event.ignore()
        return True


def install_wheel_guard(app: QApplication) -> WheelGuardFilter:
    """Install the global wheel guard on ``app``. Returns the filter (keep alive)."""
    guard = WheelGuardFilter(app)
    app.installEventFilter(guard)
    return guard
