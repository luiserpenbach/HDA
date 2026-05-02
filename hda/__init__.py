"""Hopper Data Studio v3 — Qt-targeted, batch-first analysis app.

Layered:
    domain/       pure-python core. No I/O, no Qt, no DB.
    persistence/  SQLite (WAL) repositories.
    services/     orchestration, threading, watch-folder.
    ui/           PySide6 widgets (added in a later commit).

Hard rules:
    - domain may not import persistence, services, or ui.
    - services may not import ui.
    - ui owns the Qt event loop; nothing else may.
"""

__version__ = "3.0.0-dev"
