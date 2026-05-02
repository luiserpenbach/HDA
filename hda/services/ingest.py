"""Ingest service interface.

Every entry point — watch folder, drag-and-drop, file-dialog, folder-dialog,
CLI, scripted — converges on ``IngestService.enqueue``. There are no special
paths; the source is recorded for traceability but does not change behavior.

Concrete implementation lands in a follow-up commit alongside the file
hashing, preprocessing, and metadata-resolution pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


class IngestSource(str, Enum):
    WATCH_FOLDER = "watch_folder"
    DRAG_DROP = "drag_drop"
    FILE_DIALOG = "file_dialog"
    FOLDER_DIALOG = "folder_dialog"
    CLI = "cli"
    SCRIPTED = "scripted"


@dataclass(frozen=True, slots=True)
class IngestRequest:
    file_path: Path
    campaign_id: str
    source: IngestSource
    sidecar_metadata: Optional[Mapping[str, Any]] = None
    operator: Optional[str] = None


class IngestService(Protocol):
    """Frictionless ingest entry point. All sources call ``enqueue``."""

    def enqueue(self, request: IngestRequest) -> str:
        """Enqueue a file for ingest. Returns the ``test_run_id`` assigned."""
        ...
