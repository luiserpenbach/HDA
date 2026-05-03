"""Hardware repository.

Hardware (part_number, serial_number) is normalized into its own table so
analytics can filter "all tests of part PN-X across every campaign" with a
single indexed join. The repository is upsert-style: ``get_or_create`` is the
only entry point used by the ingest pipeline.
"""

from __future__ import annotations

from typing import List, Optional

from hda.domain.errors import DBError
from hda.domain.types import Hardware
from hda.persistence.db import Database, transaction


class HardwareRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get_or_create(self, hardware: Hardware) -> int:
        """Return the hardware id, inserting the row if needed."""
        existing = self._lookup_id(hardware)
        if existing is not None:
            return existing
        with transaction(self._db, write=True) as conn:
            existing = self._lookup_id_inside(conn, hardware)
            if existing is not None:
                return existing
            try:
                cursor = conn.execute(
                    "INSERT INTO hardware(part_number, serial_number) VALUES (?, ?)",
                    (hardware.part_number, hardware.serial_number),
                )
            except Exception as e:
                raise DBError(
                    f"Failed to insert hardware {hardware.part_number}/{hardware.serial_number}: {e}"
                ) from e
            new_id = cursor.lastrowid
            if new_id is None:
                raise DBError("INSERT returned no rowid")
            return int(new_id)

    def get(self, hardware_id: int) -> Optional[Hardware]:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT part_number, serial_number FROM hardware WHERE id = ?",
            (hardware_id,),
        ).fetchone()
        if row is None:
            return None
        return Hardware(
            part_number=row["part_number"], serial_number=row["serial_number"]
        )

    def find_by_part(self, part_number: str) -> List[Hardware]:
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT part_number, serial_number FROM hardware "
            "WHERE part_number = ? ORDER BY serial_number",
            (part_number,),
        ).fetchall()
        return [
            Hardware(part_number=r["part_number"], serial_number=r["serial_number"])
            for r in rows
        ]

    def _lookup_id(self, hardware: Hardware) -> Optional[int]:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT id FROM hardware WHERE part_number = ? AND serial_number = ?",
            (hardware.part_number, hardware.serial_number),
        ).fetchone()
        return int(row["id"]) if row else None

    @staticmethod
    def _lookup_id_inside(conn, hardware: Hardware) -> Optional[int]:
        row = conn.execute(
            "SELECT id FROM hardware WHERE part_number = ? AND serial_number = ?",
            (hardware.part_number, hardware.serial_number),
        ).fetchone()
        return int(row["id"]) if row else None
