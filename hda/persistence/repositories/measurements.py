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
