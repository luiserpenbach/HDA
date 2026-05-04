"""Application entry point.

Single-instance lock + structured logging + main-window construction.
The CLI in ``hda/__main__.py`` is the canonical way to launch; this
module is also importable so other Python entry points can drive it.

Single-instance: a ``QLockFile`` next to the database protects against
two Qt processes opening the same ``hda.db``. SQLite WAL handles
in-process concurrency, but two separate processes editing the same
file can still race; the lock is the belt to that suspenders.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QLockFile
from PySide6.QtWidgets import QApplication, QMessageBox

from hda.domain.errors import HDAError
from hda.ui.logging_setup import get_logger, setup_logging
from hda.ui.main_window import MainWindow
from hda.ui.workspace import Workspace, build_default_workspace


_log = get_logger("ui.main")


def main(
    db_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    campaign_id: str = "DEMO-C1",
    argv: Optional[list[str]] = None,
) -> int:
    """Launch the Qt app. Returns the Qt exit code."""
    db_path = db_path or (Path.home() / ".hda" / "hda.db")
    log_dir = log_dir or (Path.home() / ".hda" / "logs")
    setup_logging(log_dir=log_dir)
    _log.info("starting hda v3 ui; db=%s", db_path)

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Hopper Data Studio")
    app.setOrganizationName("Hopper Propulsion Systems")

    lock_path = db_path.with_suffix(db_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(lock_path))
    lock.setStaleLockTime(0)
    if not lock.tryLock(50):
        QMessageBox.critical(
            None,
            "Already running",
            f"Another HDA instance is already using {db_path}.\n"
            "Close it first or pick a different database.",
        )
        return 1

    try:
        try:
            workspace: Workspace = build_default_workspace(
                db_path=db_path, log_dir=log_dir, campaign_id=campaign_id
            )
        except HDAError as e:
            _log.exception("workspace setup failed")
            QMessageBox.critical(None, "Workspace setup failed", str(e))
            return 2

        window = MainWindow(workspace, default_campaign_id=campaign_id)
        window.show()
        return app.exec()
    finally:
        lock.unlock()
