"""Application entry point.

Launches the new navigation-based HDA desktop UI (HDAMainWindow).

The legacy Workspace-based window is still importable as ``LegacyMainWindow``
for use by existing tests and the analytics subsystem until those pages are
ported to the new navigation architecture.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

from hda.ui.logging_setup import get_logger, setup_logging
from hda.ui.main_window import HDAMainWindow
from hda.ui.style import apply_app_font
from hda.ui.wheel_guard import install_wheel_guard


_log = get_logger("ui.main")


def main(
    log_dir: Optional[Path] = None,
    argv: Optional[list[str]] = None,
) -> int:
    """Launch the Qt application. Returns the Qt exit code."""
    log_dir = log_dir or (Path.home() / ".hda" / "logs")
    setup_logging(log_dir=log_dir)
    _log.info("starting HDA desktop ui")

    app = QApplication(argv if argv is not None else sys.argv)
    apply_app_font(app)
    install_wheel_guard(app)
    app.setApplicationName("Hopper Data Studio")
    app.setOrganizationName("Hopper Propulsion Systems")
    app.setApplicationVersion("2.4.0")

    window = HDAMainWindow()
    window.show()
    return app.exec()
