"""Campaign repository.

Minimal first slice: create / get / list / archive. The full repository set
(test_runs, hardware, measurements) lands with the ingest service.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from hda.domain.errors import DBError
from hda.domain.types import Campaign
from hda.persistence.db import Database, transaction


class CampaignRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, campaign: Campaign) -> None:
        with transaction(self._db, write=True) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO campaigns(id, name, test_type, created_at, archived)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        campaign.id,
                        campaign.name,
                        campaign.test_type,
                        campaign.created_at.isoformat(),
                        1 if campaign.archived else 0,
                    ),
                )
            except Exception as e:
                raise DBError(f"Failed to create campaign {campaign.id}: {e}") from e

    def get(self, campaign_id: str) -> Optional[Campaign]:
        conn = self._db.connect()
        row = conn.execute(
            "SELECT id, name, test_type, created_at, archived FROM campaigns WHERE id = ?",
            (campaign_id,),
        ).fetchone()
        if row is None:
            return None
        return Campaign(
            id=row["id"],
            name=row["name"],
            test_type=row["test_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            archived=bool(row["archived"]),
        )

    def list(self, include_archived: bool = False) -> List[Campaign]:
        conn = self._db.connect()
        if include_archived:
            rows = conn.execute(
                "SELECT id, name, test_type, created_at, archived FROM campaigns ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, test_type, created_at, archived "
                "FROM campaigns WHERE archived = 0 ORDER BY created_at DESC"
            ).fetchall()
        return [
            Campaign(
                id=r["id"],
                name=r["name"],
                test_type=r["test_type"],
                created_at=datetime.fromisoformat(r["created_at"]),
                archived=bool(r["archived"]),
            )
            for r in rows
        ]

    def archive(self, campaign_id: str) -> None:
        with transaction(self._db, write=True) as conn:
            conn.execute(
                "UPDATE campaigns SET archived = 1 WHERE id = ?", (campaign_id,)
            )
