"""Transactional migration runner.

Differences from the Streamlit-era ``campaign_manager_v2.py`` migrations:
    - Each migration runs inside a single IMMEDIATE transaction; partial
      failures roll back fully. The schema_version row is updated in the same
      transaction as the DDL it represents, so a crash mid-migration cannot
      leave the database "upgraded but unmarked".
    - Migration list is the authoritative source of truth. ``SCHEMA_VERSION``
      in ``schema.py`` is checked against ``MIGRATIONS`` at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from hda.domain.errors import DBError
from hda.persistence.db import Database, transaction
from hda.persistence.schema import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    apply: Callable[..., None]


def _ensure_meta_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def current_version(db: Database) -> int:
    conn = db.connect()
    _ensure_meta_table(conn)
    row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def _set_version(conn, version: int) -> None:
    conn.execute(
        """
        INSERT INTO _meta(key, value) VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(version),),
    )


def apply_migrations(db: Database) -> int:
    """Apply pending migrations. Returns the resulting schema version."""
    from hda.persistence.migrations.m001_initial import migration as m001

    migrations: Sequence[Migration] = sorted(MIGRATIONS, key=lambda m: m.version)
    if not migrations:
        raise DBError("No migrations registered")
    if migrations[-1].version != SCHEMA_VERSION:
        raise DBError(
            f"SCHEMA_VERSION ({SCHEMA_VERSION}) does not match last migration "
            f"({migrations[-1].version}). Bump SCHEMA_VERSION when adding migrations."
        )

    start = current_version(db)
    target = SCHEMA_VERSION
    if start > target:
        raise DBError(
            f"Database schema is newer ({start}) than this app supports ({target}). "
            "Refusing to downgrade."
        )

    for m in migrations:
        if m.version <= start:
            continue
        with transaction(db, write=True) as conn:
            m.apply(conn)
            _set_version(conn, m.version)
    return current_version(db)


from hda.persistence.migrations.m001_initial import migration as _m001  # noqa: E402

MIGRATIONS: List[Migration] = [_m001]
