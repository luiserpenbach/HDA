"""Measurements repository.

Long-form storage: one row per (test_run, name). This is what makes
cross-campaign analytics — "Cd of part PN-X across every test it appeared
in" — a single indexed join rather than a multi-DB UNION over the
per-campaign legacy DBs.

``write_all`` replaces all measurements for a test run inside a single
transaction so a partial write never persists.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

import pandas as pd

from hda.domain.errors import DBError
from hda.domain.types import MeasurementWithUncertainty, Provenance
from hda.persistence.db import Database, transaction


class MeasurementsRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def write_all(
        self,
        test_run_id: str,
        measurements: Mapping[str, MeasurementWithUncertainty],
    ) -> None:
        with transaction(self._db, write=True) as conn:
            try:
                conn.execute(
                    "DELETE FROM measurements WHERE test_run_id = ?",
                    (test_run_id,),
                )
                conn.executemany(
                    """
                    INSERT INTO measurements(
                        test_run_id, name, value, uncertainty, unit, provenance
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            test_run_id,
                            m.name,
                            float(m.value),
                            float(m.uncertainty),
                            m.unit,
                            m.provenance.value,
                        )
                        for m in measurements.values()
                    ],
                )
            except Exception as e:
                raise DBError(
                    f"Failed to write measurements for {test_run_id}: {e}"
                ) from e

    def get_for_run(
        self, test_run_id: str
    ) -> List[MeasurementWithUncertainty]:
        conn = self._db.connect()
        rows = conn.execute(
            """
            SELECT name, value, uncertainty, unit, provenance
              FROM measurements
             WHERE test_run_id = ?
             ORDER BY name
            """,
            (test_run_id,),
        ).fetchall()
        return [
            MeasurementWithUncertainty(
                name=r["name"],
                value=r["value"],
                uncertainty=r["uncertainty"],
                unit=r["unit"],
                provenance=Provenance(r["provenance"]),
            )
            for r in rows
        ]

    def hardware_history(
        self,
        part_number: str,
        measurement_name: str,
        serial_number: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return time-ordered measurements of ``measurement_name`` for every
        test of ``part_number`` across every campaign — the cross-campaign
        analytics primitive.
        """
        conn = self._db.connect()
        sql = """
            SELECT tr.id           AS test_run_id,
                   tr.campaign_id,
                   h.part_number,
                   h.serial_number,
                   tr.persisted_at,
                   tr.discovered_at,
                   m.value,
                   m.uncertainty,
                   m.unit
              FROM measurements m
              JOIN test_runs tr ON tr.id = m.test_run_id
              JOIN hardware h    ON h.id = tr.hardware_id
             WHERE m.name = ?
               AND h.part_number = ?
        """
        params: list = [measurement_name, part_number]
        if serial_number is not None:
            sql += " AND h.serial_number = ?"
            params.append(serial_number)
        sql += " ORDER BY COALESCE(tr.persisted_at, tr.discovered_at)"
        rows = conn.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def list_parts_with_measurements(self) -> List[str]:
        """Distinct part numbers that have at least one persisted measurement."""
        conn = self._db.connect()
        rows = conn.execute(
            """
            SELECT DISTINCT h.part_number
              FROM hardware h
              JOIN test_runs tr ON tr.hardware_id = h.id
              JOIN measurements m ON m.test_run_id = tr.id
             ORDER BY h.part_number
            """
        ).fetchall()
        return [r["part_number"] for r in rows]

    def list_serials_for_part(self, part_number: str) -> List[str]:
        conn = self._db.connect()
        rows = conn.execute(
            """
            SELECT DISTINCT h.serial_number
              FROM hardware h
              JOIN test_runs tr ON tr.hardware_id = h.id
              JOIN measurements m ON m.test_run_id = tr.id
             WHERE h.part_number = ?
             ORDER BY h.serial_number
            """,
            (part_number,),
        ).fetchall()
        return [r["serial_number"] for r in rows]

    def list_measurement_names_for_part(
        self, part_number: str, serial_number: Optional[str] = None
    ) -> List[str]:
        conn = self._db.connect()
        sql = """
            SELECT DISTINCT m.name
              FROM measurements m
              JOIN test_runs tr ON tr.id = m.test_run_id
              JOIN hardware h    ON h.id = tr.hardware_id
             WHERE h.part_number = ?
        """
        params: list = [part_number]
        if serial_number is not None:
            sql += " AND h.serial_number = ?"
            params.append(serial_number)
        sql += " ORDER BY m.name"
        rows = conn.execute(sql, params).fetchall()
        return [r["name"] for r in rows]
