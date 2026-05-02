"""SQLite connection management.

Replaces the per-campaign-DB design of the Streamlit app with a single
``hda.db`` so cross-campaign analytics (e.g., tracking part PN-X across every
campaign it appeared in) becomes a normal indexed query rather than a
multi-database UNION.

Hardening applied vs. the previous implementation:
    - WAL mode: concurrent readers, single writer, durable across crashes.
    - busy_timeout 5s: avoids spurious "database is locked" under contention.
    - foreign_keys ON.
    - Per-thread connections via threading.local.
    - Single writer Lock so no two threads attempt SQLite writes simultaneously.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from hda.domain.errors import DBError


_BUSY_TIMEOUT_MS = 5000


class Database:
    """Thread-aware SQLite handle.

    One ``Database`` instance per ``hda.db`` file, shared across threads.
    Each thread gets its own connection (SQLite connection objects are not
    safe to share across threads). Writes serialize through ``_write_lock``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._init_pragmas()

    def _init_pragmas(self) -> None:
        conn = self._raw_connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.commit()
        finally:
            conn.close()

    def _raw_connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(
                str(self.path),
                timeout=_BUSY_TIMEOUT_MS / 1000.0,
                isolation_level=None,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
        except sqlite3.Error as e:
            raise DBError(f"Failed to open database at {self.path}: {e}") from e
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def connect(self) -> sqlite3.Connection:
        """Get the calling thread's connection, creating it on first use."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._raw_connect()
            self._local.conn = conn
        return conn

    def close_thread(self) -> None:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @property
    def write_lock(self) -> threading.RLock:
        return self._write_lock


@contextmanager
def transaction(db: Database, write: bool = False) -> Iterator[sqlite3.Connection]:
    """Context-managed transaction.

    Writes acquire the database-wide write lock so SQLite never sees concurrent
    writers from this process. Reads do not block reads.

    Args:
        db: Database handle.
        write: True for write transactions (acquires write_lock + IMMEDIATE).
    """
    conn = db.connect()
    if write:
        with db.write_lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
    else:
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
