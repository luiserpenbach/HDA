"""Initial schema.

Single ``hda.db`` shape supporting the v3 workflow:

    campaigns          one row per campaign (incl. ad-hoc single-test campaigns)
    hardware           normalized (part_number, serial_number) so analytics can
                       filter / track a part across campaigns by indexed FK.
    test_runs          per-test snapshot at PERSISTED time. metadata + traceability
                       carried as JSON for forward compatibility; queryable fields
                       (campaign, hardware, state, dates) are columnar + indexed.
    measurements       long-form (test_run_id, name, value, uncertainty, unit,
                       provenance). Cross-campaign analytics ("show me Cd of
                       part PN-X across all campaigns") becomes a normal join.
    qc_findings        per-check QC result rows, joined to test_run.
    derived_specs      per-campaign derived-channel / derived-measurement specs,
                       so the formula library applied to a test is recoverable.
"""

from __future__ import annotations

import sqlite3

from hda.persistence.migrations.runner import Migration


_DDL: tuple[str, ...] = (
    """
    CREATE TABLE campaigns (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        test_type     TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        archived      INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE hardware (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        part_number   TEXT NOT NULL,
        serial_number TEXT NOT NULL,
        UNIQUE(part_number, serial_number)
    )
    """,
    "CREATE INDEX idx_hardware_part   ON hardware(part_number)",
    "CREATE INDEX idx_hardware_serial ON hardware(serial_number)",
    """
    CREATE TABLE test_runs (
        id                  TEXT PRIMARY KEY,
        campaign_id         TEXT NOT NULL REFERENCES campaigns(id),
        hardware_id         INTEGER REFERENCES hardware(id),
        file_path           TEXT NOT NULL,
        file_hash           TEXT NOT NULL,
        state               TEXT NOT NULL,
        test_id_label       TEXT,
        operator            TEXT,
        fluid               TEXT,
        metadata_json       TEXT,
        metadata_hash       TEXT,
        steady_start_s      REAL,
        steady_end_s        REAL,
        steady_method       TEXT,
        steady_confidence   REAL,
        qc_passed           INTEGER,
        confidence          REAL,
        processing_version  TEXT,
        plugin_name         TEXT,
        plugin_version      TEXT,
        config_hash         TEXT,
        traceability_json   TEXT,
        error_message       TEXT,
        discovered_at       TEXT,
        persisted_at        TEXT
    )
    """,
    "CREATE INDEX idx_test_runs_campaign  ON test_runs(campaign_id)",
    "CREATE INDEX idx_test_runs_hardware  ON test_runs(hardware_id)",
    "CREATE INDEX idx_test_runs_state     ON test_runs(state)",
    "CREATE INDEX idx_test_runs_persisted ON test_runs(persisted_at)",
    """
    CREATE TABLE measurements (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        test_run_id        TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
        name               TEXT NOT NULL,
        value              REAL NOT NULL,
        uncertainty        REAL NOT NULL,
        unit               TEXT NOT NULL DEFAULT '',
        provenance         TEXT NOT NULL DEFAULT 'sensor'
    )
    """,
    "CREATE INDEX idx_measurements_run_name ON measurements(test_run_id, name)",
    "CREATE INDEX idx_measurements_name     ON measurements(name)",
    """
    CREATE TABLE qc_findings (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        test_run_id   TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
        check_name    TEXT NOT NULL,
        status        TEXT NOT NULL,
        message       TEXT NOT NULL DEFAULT '',
        blocking      INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX idx_qc_findings_run ON qc_findings(test_run_id)",
    """
    CREATE TABLE derived_specs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
        kind          TEXT NOT NULL CHECK(kind IN ('channel','measurement')),
        name          TEXT NOT NULL,
        unit          TEXT NOT NULL DEFAULT '',
        formula       TEXT NOT NULL,
        inputs_json   TEXT NOT NULL,
        params_json   TEXT NOT NULL DEFAULT '{}',
        uncertainty_method TEXT NOT NULL DEFAULT 'propagate',
        UNIQUE(campaign_id, kind, name)
    )
    """,
    "CREATE INDEX idx_derived_specs_campaign ON derived_specs(campaign_id)",
)


def _apply(conn: sqlite3.Connection) -> None:
    for stmt in _DDL:
        conn.execute(stmt)


migration = Migration(
    version=1,
    description="initial schema: campaigns, hardware, test_runs, measurements, qc, derived_specs",
    apply=_apply,
)
