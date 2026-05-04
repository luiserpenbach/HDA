"""Workspace assembly.

A Workspace is the runtime composition root: an opened+migrated database,
a plugin registry, ingest pipelines and analysis profiles per campaign,
and a formula library. The UI consumes a fully-built Workspace so it never
sees the wiring details.

``build_default_workspace`` is a convenience factory for "I just want to
drop a CSV in and see results" — it creates an ad-hoc campaign, registers
the basic_means plugin, and uses sensible defaults so the user can start
testing fast and customize later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

from hda.domain.derived import FormulaLibrary, standard_library
from hda.domain.metadata import FieldType, MetadataField, MetadataSchema
from hda.domain.plugin_modules import BasicMeansPlugin
from hda.domain.plugins import PluginRegistry
from hda.domain.qc import QCConfig
from hda.domain.types import Campaign
from hda.persistence import Database, apply_migrations
from hda.persistence.repositories import CampaignRepository
from hda.services import (
    AnalysisProfile,
    AnalysisServiceImpl,
    IngestPipeline,
    IngestServiceImpl,
    NaNPolicy,
    PreprocessingConfig,
)
from hda.services.preprocessed_cache import PreprocessedDataCache


@dataclass(slots=True)
class Workspace:
    db: Database
    plugins: PluginRegistry
    ingest_pipelines: dict[str, IngestPipeline] = field(default_factory=dict)
    analysis_profiles: dict[str, AnalysisProfile] = field(default_factory=dict)
    formula_library: FormulaLibrary = field(default_factory=standard_library)
    log_dir: Optional[Path] = None
    ingest_service: Optional[IngestServiceImpl] = None
    analysis_service: Optional[AnalysisServiceImpl] = None
    preprocessed_cache: PreprocessedDataCache = field(
        default_factory=PreprocessedDataCache
    )

    def configure_services(self) -> None:
        """Construct the service instances. Call after pipelines/profiles
        are populated and the DB has been migrated."""
        self.ingest_service = IngestServiceImpl(
            db=self.db,
            pipelines=self.ingest_pipelines,
            library=self.formula_library,
        )
        self.analysis_service = AnalysisServiceImpl(
            db=self.db,
            plugins=self.plugins,
            profiles=self.analysis_profiles,
            formula_library=self.formula_library,
        )


def _default_metadata_schema() -> MetadataSchema:
    return MetadataSchema(
        fields=(
            MetadataField("part_number", FieldType.STRING, required=True),
            MetadataField("serial_number", FieldType.STRING, required=True),
            MetadataField("operator", FieldType.STRING, required=True),
            MetadataField("test_id", FieldType.STRING),
            MetadataField("fluid", FieldType.STRING),
            MetadataField("notes", FieldType.STRING),
        )
    )


def build_default_workspace(
    db_path: Path,
    campaign_id: str = "DEMO-C1",
    campaign_name: str = "Demo Campaign",
    test_type: str = "cold_flow",
    log_dir: Optional[Path] = None,
    auto_confirm_confidence: float = 0.0,
) -> Workspace:
    """Return a Workspace ready for the UI.

    Creates the campaign in the DB if it doesn't exist. Registers the
    basic_means plugin under the requested campaign so any CSV (with a
    ``timestamp`` column and at least one numeric channel) ingests and
    analyzes successfully out of the box.
    """
    db = Database(db_path)
    apply_migrations(db)

    campaigns = CampaignRepository(db)
    if campaigns.get(campaign_id) is None:
        campaigns.create(
            Campaign(
                id=campaign_id,
                name=campaign_name,
                test_type=test_type,
                created_at=datetime.utcnow(),
            )
        )

    plugins = PluginRegistry()
    plugins.register(BasicMeansPlugin())

    ingest_pipeline = IngestPipeline(
        metadata_schema=_default_metadata_schema(),
        preprocessing=PreprocessingConfig(
            timestamp_column="timestamp",
            resample_freq_hz=None,
            nan_policy=NaNPolicy.INTERPOLATE,
        ),
    )

    analysis_profile = AnalysisProfile(
        plugin_name="basic_means",
        qc_config=QCConfig(),
        steady_state_signal=None,
        steady_state_cv_threshold=0.05,
        steady_state_window_s=0.5,
        steady_state_min_duration_s=1.0,
        auto_confirm_confidence=auto_confirm_confidence,
        sensor_calibrations={},
    )

    ws = Workspace(
        db=db,
        plugins=plugins,
        ingest_pipelines={campaign_id: ingest_pipeline},
        analysis_profiles={campaign_id: analysis_profile},
        log_dir=log_dir,
    )
    ws.configure_services()
    return ws
