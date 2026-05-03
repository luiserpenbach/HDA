"""QC findings repository."""

from __future__ import annotations

from typing import List, Sequence

from hda.domain.errors import DBError
from hda.domain.types import QCFinding, QCStatus
from hda.persistence.db import Database, transaction


class QCFindingsRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def write_all(
        self, test_run_id: str, findings: Sequence[QCFinding], qc_passed: bool
    ) -> None:
        with transaction(self._db, write=True) as conn:
            try:
                conn.execute(
                    "DELETE FROM qc_findings WHERE test_run_id = ?",
                    (test_run_id,),
                )
                conn.executemany(
                    """
                    INSERT INTO qc_findings(
                        test_run_id, check_name, status, message, blocking
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            test_run_id,
                            f.check_name,
                            f.status.value,
                            f.message,
                            1 if f.blocking else 0,
                        )
                        for f in findings
                    ],
                )
                conn.execute(
                    "UPDATE test_runs SET qc_passed = ? WHERE id = ?",
                    (1 if qc_passed else 0, test_run_id),
                )
            except Exception as e:
                raise DBError(
                    f"Failed to write QC findings for {test_run_id}: {e}"
                ) from e

    def get_for_run(self, test_run_id: str) -> List[QCFinding]:
        conn = self._db.connect()
        rows = conn.execute(
            """
            SELECT check_name, status, message, blocking
              FROM qc_findings
             WHERE test_run_id = ?
             ORDER BY id
            """,
            (test_run_id,),
        ).fetchall()
        return [
            QCFinding(
                check_name=r["check_name"],
                status=QCStatus(r["status"]),
                message=r["message"] or "",
                blocking=bool(r["blocking"]),
            )
            for r in rows
        ]
