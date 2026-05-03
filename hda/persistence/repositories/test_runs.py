"""TestRun repository.

Stores the per-test snapshot at the current state. The first row is written
at ingest time with the file/metadata/preprocessing fields populated; the
analysis fields (steady window, qc, traceability) are filled by subsequent
state-update calls as the TestRun progresses through the state machine.

Updates always go through ``update_state`` so the new state is validated
against the current state's transition DAG before the row is touched. This
means a buggy caller cannot silently jump from PREPROCESSED to PERSISTED.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Mapping, Optional

from hda.domain.errors import DBError
from hda.domain.state import TestState, transition
from hda.domain.types import TestRun
from hda.persistence.db import Database, transaction


class TestRunRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def insert_initial(
        self,
        run: TestRun,
        hardware_id: Optional[int],
        metadata_values: Mapping[str, Any],
        metadata_hash: str,
    ) -> None:
        """Insert a freshly-ingested TestRun row.

        The TestRun must be in DISCOVERED, INGESTING, AWAITING_METADATA, or
        PREPROCESSED state. Analysis fields are left NULL.
        """
        if run.state in (
            TestState.PERSISTED,
            TestState.QC_FAILED,
            TestState.ERROR,
        ):
            raise DBError(
                f"Refusing to insert TestRun in terminal state {run.state.value}"
            )
        with transaction(self._db, write=True) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO test_runs(
                        id, campaign_id, hardware_id, file_path, file_hash,
                        state, test_id_label, operator, fluid,
                        metadata_json, metadata_hash, discovered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.campaign_id,
                        hardware_id,
                        str(run.file_path),
                        run.file_hash,
                        run.state.value,
                        run.metadata.test_id if run.metadata else None,
                        run.metadata.operator if run.metadata else None,
                        run.metadata.fluid if run.metadata else None,
                        json.dumps(dict(metadata_values), sort_keys=True),
                        metadata_hash,
                        (run.discovered_at or datetime.utcnow()).isoformat(),
                    ),
                )
            except Exception as e:
                raise DBError(f"Failed to insert TestRun {run.id}: {e}") from e

    def update_state(
        self,
        test_run_id: str,
        new_state: TestState,
        error_message: Optional[str] = None,
    ) -> None:
        """Validate the transition then write the new state."""
        with transaction(self._db, write=True) as conn:
            row = conn.execute(
                "SELECT state FROM test_runs WHERE id = ?", (test_run_id,)
            ).fetchone()
            if row is None:
                raise DBError(f"TestRun {test_run_id} does not exist")
            current = TestState(row["state"])
            transition(current, new_state)
            persisted_at = (
                datetime.utcnow().isoformat()
                if new_state is TestState.PERSISTED
                else None
            )
            conn.execute(
                """
                UPDATE test_runs
                   SET state = ?,
                       error_message = COALESCE(?, error_message),
                       persisted_at = COALESCE(?, persisted_at)
                 WHERE id = ?
                """,
                (new_state.value, error_message, persisted_at, test_run_id),
            )

    def get_state(self, test_run_id: str) -> Optional[TestState]:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT state FROM test_runs WHERE id = ?", (test_run_id,)
        ).fetchone()
        return TestState(row["state"]) if row else None

    def find_by_file_hash(self, file_hash: str) -> List[str]:
        """Return ids of all TestRuns sharing a file hash (duplicate detection)."""
        conn = self._db.connect()
        rows = conn.execute(
            "SELECT id FROM test_runs WHERE file_hash = ?", (file_hash,)
        ).fetchall()
        return [r["id"] for r in rows]

    def list_for_campaign(self, campaign_id: str) -> List[Mapping[str, Any]]:
        conn = self._db.connect()
        rows = conn.execute(
            """
            SELECT id, state, file_path, test_id_label, operator,
                   discovered_at, persisted_at
              FROM test_runs
             WHERE campaign_id = ?
             ORDER BY COALESCE(persisted_at, discovered_at) DESC
            """,
            (campaign_id,),
        ).fetchall()
        return [dict(r) for r in rows]
