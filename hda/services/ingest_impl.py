"""Concrete IngestService.

Wires the building blocks — file hash, sidecar load, metadata resolution,
preprocessing (incl. derived channels), hardware upsert, TestRun insert —
into a single ``enqueue`` call. Today this runs synchronously inside the
caller's thread; the Qt layer will wrap it in a ``QThreadPool`` worker so
``enqueue`` returns immediately and progress arrives via signals. The
synchronous entry point ``process`` exists for that wrapper to call.

Idempotency: if a file with the same SHA-256 has already been ingested
into the requested campaign, ``enqueue`` returns the existing TestRun id
without re-processing. Watch-folder events that fire twice on the same
file are therefore safe.

Preprocessing-vs-metadata: preprocessing is data-only and never depends
on metadata, so we always run it and cache the result. That way the
operator can preview a freshly-dropped CSV in the detail panel
*immediately*, even when required metadata fields are still missing —
analysis is the only thing that's deferred to ``AWAITING_METADATA``.
``complete_metadata`` later fills the missing fields and the analysis
service runs against the same cached preprocessed data.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from hda.domain.derived import FormulaLibrary, standard_library
from hda.domain.errors import ConfigError, IngestError
from hda.domain.metadata import (
    MetadataLayer,
    MetadataSchema,
    ResolvedMetadata,
    canonical_metadata_hash,
    load_sidecar,
    resolve_metadata,
)
from hda.domain.state import TestState
from hda.domain.types import Hardware, TestMetadata, TestRun
from hda.persistence.db import Database, transaction
from hda.persistence.repositories import (
    CampaignRepository,
    HardwareRepository,
    TestRunRepository,
)
from hda.services.hashing import hash_file
from hda.services.ingest import IngestRequest, IngestSource
from hda.services.preprocessing import (
    PreprocessedData,
    PreprocessingConfig,
    preprocess,
)


@dataclass(frozen=True, slots=True)
class IngestPipeline:
    """Per-campaign ingest configuration.

    The pipeline carries the metadata schema (plugin-declared fields), the
    preprocessing config (channel map, resample, derived channels), and any
    campaign-level metadata defaults that fill fields the sidecar omitted.
    """

    metadata_schema: MetadataSchema
    preprocessing: PreprocessingConfig
    campaign_metadata_defaults: Mapping[str, Any] = field(default_factory=dict)
    sidecar_filename: str = "metadata.json"


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    """What ingest produced for a single file."""

    test_run_id: str
    state: TestState
    file_hash: str
    duplicate_of: Optional[str] = None
    missing_metadata: tuple[str, ...] = ()
    preprocessed: Optional[PreprocessedData] = None
    metadata: Optional[TestMetadata] = None
    metadata_hash: str = ""


@dataclass(frozen=True, slots=True)
class CompleteMetadataOutcome:
    test_run_id: str
    metadata: TestMetadata
    metadata_hash: str


class IngestServiceImpl:
    """Synchronous ingest service.

    Construction-time deps:
        db: opened, migrated Database.
        pipelines: campaign_id -> IngestPipeline. Lookup fails loudly if a
            request arrives for an unconfigured campaign.
        library: optional FormulaLibrary; defaults to ``standard_library()``.
        csv_reader: optional injection seam for tests / non-CSV formats.
    """

    def __init__(
        self,
        db: Database,
        pipelines: Mapping[str, IngestPipeline],
        library: Optional[FormulaLibrary] = None,
        csv_reader=None,
    ) -> None:
        self._db = db
        self._pipelines = dict(pipelines)
        self._library = library if library is not None else standard_library()
        self._read = csv_reader if csv_reader is not None else _default_csv_read
        self._campaigns = CampaignRepository(db)
        self._hardware = HardwareRepository(db)
        self._test_runs = TestRunRepository(db)

    def enqueue(self, request: IngestRequest) -> str:
        """Public entry point. Returns the resulting TestRun id."""
        return self.process(request).test_run_id

    def process(self, request: IngestRequest) -> IngestOutcome:
        pipeline = self._pipelines.get(request.campaign_id)
        if pipeline is None:
            raise ConfigError(
                f"No ingest pipeline registered for campaign '{request.campaign_id}'"
            )
        if self._campaigns.get(request.campaign_id) is None:
            raise ConfigError(
                f"Campaign '{request.campaign_id}' does not exist; create it first"
            )
        path = Path(request.file_path)
        if not path.exists():
            raise IngestError(f"Ingest source file not found: {path}")

        file_hash = hash_file(path)
        existing = self._test_runs.find_by_file_hash(file_hash)
        if existing:
            return IngestOutcome(
                test_run_id=existing[0],
                state=self._test_runs.get_state(existing[0]) or TestState.PERSISTED,
                file_hash=file_hash,
                duplicate_of=existing[0],
            )

        sidecar = self._load_sidecar(path, pipeline, request.sidecar_metadata)
        operator_payload = dict(request.sidecar_metadata or {})
        if request.operator and "operator" not in operator_payload:
            operator_payload["operator"] = request.operator

        resolved = resolve_metadata(
            schema=pipeline.metadata_schema,
            sidecar=sidecar,
            campaign_defaults=pipeline.campaign_metadata_defaults,
            operator=operator_payload,
        )
        if resolved.errors:
            raise ConfigError(
                "Metadata validation errors: "
                + "; ".join(f"{e.field_name}: {e.message}" for e in resolved.errors)
            )

        try:
            raw = self._read(path)
        except IngestError:
            raise
        except Exception as e:
            raise IngestError(f"Failed to read {path}: {e}") from e

        # Preprocessing depends only on the data + the campaign's
        # PreprocessingConfig — never on metadata. Always run it so the
        # detail panel can preview the time-series even when required
        # metadata is still missing.
        preprocessed = preprocess(raw, pipeline.preprocessing, self._library)
        new_state = (
            TestState.PREPROCESSED if resolved.complete else TestState.AWAITING_METADATA
        )

        metadata_obj = (
            self._build_metadata(resolved.values) if resolved.complete else None
        )
        test_run_id = _new_test_run_id()
        run = TestRun(
            id=test_run_id,
            campaign_id=request.campaign_id,
            file_path=path,
            file_hash=file_hash,
            state=new_state,
            metadata=metadata_obj,
            discovered_at=datetime.utcnow(),
        )
        hardware_id = (
            self._hardware.get_or_create(metadata_obj.hardware)
            if metadata_obj is not None
            else None
        )
        metadata_hash = canonical_metadata_hash(_for_hash(resolved.values, resolved.sources))
        self._test_runs.insert_initial(
            run,
            hardware_id=hardware_id,
            metadata_values=resolved.values,
            metadata_hash=metadata_hash,
        )
        return IngestOutcome(
            test_run_id=test_run_id,
            state=new_state,
            file_hash=file_hash,
            missing_metadata=tuple(resolved.missing_required),
            preprocessed=preprocessed,
            metadata=metadata_obj,
            metadata_hash=metadata_hash,
        )

    def complete_metadata(
        self,
        test_run_id: str,
        operator_metadata: Mapping[str, Any],
    ) -> CompleteMetadataOutcome:
        """Fill the missing required fields on an AWAITING_METADATA run.

        Re-validates the merged set against the campaign's schema; raises
        ``ConfigError`` if required fields are still missing or any
        supplied value is invalid. On success, updates the test_runs row
        with the new resolved values + hash, persists the hardware row
        (now that part_number/serial_number are known), and transitions
        the state to PREPROCESSED — ready for the analysis service.
        """
        conn = self._db.connect()
        row = conn.execute(
            "SELECT campaign_id, metadata_json, state FROM test_runs WHERE id = ?",
            (test_run_id,),
        ).fetchone()
        if row is None:
            raise ConfigError(f"TestRun {test_run_id} does not exist")
        current_state = TestState(row["state"])
        if current_state is not TestState.AWAITING_METADATA:
            raise ConfigError(
                f"TestRun {test_run_id} is in state {current_state.value}; "
                "complete_metadata only applies to AWAITING_METADATA"
            )

        pipeline = self._pipelines.get(row["campaign_id"])
        if pipeline is None:
            raise ConfigError(
                f"No ingest pipeline registered for campaign '{row['campaign_id']}'"
            )

        previously_resolved: dict[str, Any] = (
            json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        )
        # Previously-resolved values keep their (sidecar / campaign-default)
        # source priority; the new operator dict only fills gaps.
        resolved = resolve_metadata(
            schema=pipeline.metadata_schema,
            sidecar=previously_resolved or None,
            campaign_defaults=None,
            operator=dict(operator_metadata),
        )
        if resolved.errors:
            raise ConfigError(
                "Metadata validation errors: "
                + "; ".join(f"{e.field_name}: {e.message}" for e in resolved.errors)
            )
        if resolved.missing_required:
            raise ConfigError(
                "Required fields still missing: "
                + ", ".join(resolved.missing_required)
            )

        metadata_obj = self._build_metadata(resolved.values)
        hardware_id = self._hardware.get_or_create(metadata_obj.hardware)
        metadata_hash = canonical_metadata_hash(
            _for_hash(resolved.values, resolved.sources)
        )

        with transaction(self._db, write=True) as conn:
            conn.execute(
                """
                UPDATE test_runs
                   SET state = ?,
                       hardware_id = ?,
                       test_id_label = ?,
                       operator = ?,
                       fluid = ?,
                       metadata_json = ?,
                       metadata_hash = ?
                 WHERE id = ?
                """,
                (
                    TestState.PREPROCESSED.value,
                    hardware_id,
                    metadata_obj.test_id or None,
                    metadata_obj.operator or None,
                    metadata_obj.fluid or None,
                    json.dumps(dict(resolved.values), sort_keys=True),
                    metadata_hash,
                    test_run_id,
                ),
            )

        return CompleteMetadataOutcome(
            test_run_id=test_run_id,
            metadata=metadata_obj,
            metadata_hash=metadata_hash,
        )

    def metadata_schema_for_run(
        self, test_run_id: str
    ) -> MetadataSchema:
        """Return the metadata schema for the campaign owning ``test_run_id``."""
        conn = self._db.connect()
        row = conn.execute(
            "SELECT campaign_id FROM test_runs WHERE id = ?", (test_run_id,)
        ).fetchone()
        if row is None:
            raise ConfigError(f"TestRun {test_run_id} does not exist")
        pipeline = self._pipelines.get(row["campaign_id"])
        if pipeline is None:
            raise ConfigError(
                f"No ingest pipeline registered for campaign '{row['campaign_id']}'"
            )
        return pipeline.metadata_schema

    def existing_metadata_for_run(
        self, test_run_id: str
    ) -> Mapping[str, Any]:
        """Return the previously-resolved metadata values for a run."""
        conn = self._db.connect()
        row = conn.execute(
            "SELECT metadata_json FROM test_runs WHERE id = ?", (test_run_id,)
        ).fetchone()
        if row is None or not row["metadata_json"]:
            return {}
        try:
            return json.loads(row["metadata_json"])
        except (ValueError, TypeError):
            return {}

    def _load_sidecar(
        self,
        data_path: Path,
        pipeline: IngestPipeline,
        request_payload: Optional[Mapping[str, Any]],
    ) -> Optional[Mapping[str, Any]]:
        sidecar_path = data_path.with_name(pipeline.sidecar_filename)
        if sidecar_path.exists():
            return load_sidecar(sidecar_path)
        if request_payload is not None:
            return None
        return None

    def _build_metadata(self, values: Mapping[str, Any]) -> TestMetadata:
        try:
            return TestMetadata(
                hardware=Hardware(
                    part_number=str(values["part_number"]),
                    serial_number=str(values["serial_number"]),
                ),
                fluid=str(values.get("fluid", "")),
                operator=str(values.get("operator", "")),
                test_id=str(values.get("test_id", "")),
                geometry=dict(values.get("geometry", {}) or {}),
                notes=str(values.get("notes", "")),
                extra={
                    k: v
                    for k, v in values.items()
                    if k
                    not in {
                        "part_number",
                        "serial_number",
                        "fluid",
                        "operator",
                        "test_id",
                        "geometry",
                        "notes",
                    }
                },
            )
        except (KeyError, ValueError) as e:
            raise ConfigError(f"Cannot build TestMetadata from resolved values: {e}") from e


def _new_test_run_id() -> str:
    return uuid.uuid4().hex


def _default_csv_read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise IngestError(f"CSV at {path} is empty")
    return df


def _for_hash(
    values: Mapping[str, Any],
    sources: Mapping[str, MetadataLayer],
) -> Mapping[str, Any]:
    """Hash payload includes both the resolved values and their layer of origin
    so that a sidecar with value X and a campaign default with value X produce
    distinct hashes — provenance is part of the audit chain.
    """
    return {
        "values": dict(values),
        "sources": {k: v.value for k, v in sources.items()},
    }
