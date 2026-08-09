# Metron — Design Document

**Status**: Draft for review
**Date**: 2026-08-09
**Author**: Claude (with Luis Erpenbach)
**Scope**: Full redesign of Hopper Data Studio (HDA) as **Metron** (μέτρον, "measure") — a single internal web application replacing the Streamlit app, the Qt nav app, and the legacy `core/` engine. Repo: `amphora-space/metron`.

---

## Table of Contents

1. [Purpose & Context](#1-purpose--context)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [Constraints & Decision Record](#3-constraints--decision-record)
4. [System Architecture](#4-system-architecture)
5. [Domain Model & Database Schema](#5-domain-model--database-schema)
6. [Time-Series Storage](#6-time-series-storage)
7. [Ingestion](#7-ingestion)
8. [Analysis Pipeline](#8-analysis-pipeline)
9. [Configuration Model](#9-configuration-model)
10. [API Surface](#10-api-surface)
11. [Frontend](#11-frontend)
12. [Auth, Identity & Audit](#12-auth-identity--audit)
13. [Reporting & Export](#13-reporting--export)
14. [Spark Studio Integration Contract](#14-spark-studio-integration-contract)
15. [Code Reuse Map](#15-code-reuse-map)
16. [Testing & Numeric Parity](#16-testing--numeric-parity)
17. [Delivery Phases & Retirement Criteria](#17-delivery-phases--retirement-criteria)
18. [Risks](#18-risks)
19. [Open Decisions](#19-open-decisions)

---

## 1. Purpose & Context

HDA today is three generations of the same application in one repo (~61k LOC):

1. **`core/` + Streamlit** — the legacy engine and web UI. Sound analytics math, failing connective tissue: silent uncertainty-zeroing bugs, a ceremonial plugin system, non-transactional per-campaign SQLite, ~40 interactions for one ingest-to-report pass, zero caching.
2. **Qt nav app (`hda/ui/`)** — a better UI (draggable regions, dockable plots, background workers) built on the same broken engine, doubling maintenance.
3. **The v3 stack (`hda/domain` + `hda/persistence` + `hda/services`)** — a clean-room re-architecture with ~290 tests that fixed the engine's defects, covers ~15% of the functional surface, and was never wired to a UI.

Meanwhile, the test bench side has been rebuilt as **Spark Studio** (`amphora-space/spark-studio`, internally `tbctl`): a Raspberry-Pi-hosted FastAPI + React application that records runs as self-contained HDF5 files with content-hash provenance (`channels_sha256`, `sequence_sha256`), a JSON-lines event log, a post-run review report, and per-rate-group time series (50 Hz default, 1 kHz reserved).

**Metron** (μέτρον — "measure"; the name honors the tool's creed that no number ships without its uncertainty) is the fourth and final generation: one web application, hosted on a single internal server, that ingests Spark Studio runs (and legacy CSVs), and owns everything after the test: analysis, uncertainty, QC, campaigns, SPC, comparison, reporting. It lifts the v3 stack as its engine core, ports the validated analytics from `core/`, and retires everything else.

### Division of responsibility

| | Spark Studio (bench) | Metron (analysis) |
|---|---|---|
| Role | Testbench control, live monitoring | Post-test truth |
| Data | Live streams, HDF5 run files | Archived runs, permanent analytical record |
| Anomaly | Real-time "something is wrong now" | Forensic (drift, spikes, correlation breaks, sensor health) |
| Channel config | Owns the DAQ/channel definition (`channels.yaml`) | Consumes it, keyed by content hash |
| Plotting | Live strip charts (uPlot) | Deep analysis plots (uPlot + statistical charts) |

Neither application depends on the other at runtime. Metron pulls; Spark Studio never needs to know Metron exists.

---

## 2. Goals & Non-Goals

### Goals

- **G1 — Speed of the core loop.** Ingest → analyze → save → report for a Spark Studio run in **under 2 minutes and under 10 interactions** (today: ~40 interactions across disconnected pages). For runs with a known channel config and a reviewed uncertainty overlay, the target is near-zero manual mapping.
- **G2 — Engineering integrity, strengthened.** The three P0 principles survive intact and get stronger:
  - *Traceability* — end-to-end provenance chain: bench channel-config hash → sequence hash → raw data hash → analysis config hash → processing version → result, with a verified analyst identity.
  - *Uncertainty* — every metric carries error bars; the system makes it **structurally impossible** to persist a measurement without an uncertainty (unlike today, where key mismatches silently produce ±0.000).
  - *QC* — a visible gate with an explicit, recorded, attributed override.
- **G3 — One data model.** One database. "Campaign" means exactly one thing. Systems, campaigns, and runs are related entities, not folder paths or filename prefixes. Cross-campaign analytics is a query.
- **G4 — Effective features, not all features.** The core loop, campaign/SPC analysis, comparison, operating envelope, and system views are first-class. Everything else must justify itself.
- **G5 — Multi-user on one server.** Multiple engineers concurrently, simple username/password auth, per-user identity in every record.
- **G6 — Full retirement.** Streamlit and Qt are retired workflow-by-workflow as their replacements ship; `core/` is deleted at the end.

### Non-Goals

- No live/streaming view of tests in progress (Spark Studio's job).
- No acquisition, control, or sequence features.
- No cloud deployment, horizontal scaling, or multi-tenant concerns. One server, one org.
- No plugin marketplace / runtime-loadable third-party plugins. Test types are code, reviewed like code.
- No editing of bench channel configs (they are imported, immutable, hash-keyed).

---

## 3. Constraints & Decision Record

Confirmed with Luis:

| # | Decision | Rationale |
|---|---|---|
| D1 | Single shared internal server, Docker Compose deployment | Matches actual usage; no orchestration overhead |
| D2 | Simple auth: username + password, server sessions, roles `engineer` / `admin` | Enough to give traceability a real identity |
| D3 | Data volumes: ≤ ~2 GB worst case (legacy CSVs), few kHz max sample rate; Spark Studio runs are MB-scale | Sets the time-series storage design (§6) |
| D4 | Spark Studio is the primary data source; its HDF5 format + `channels.yaml` are the integration contract | See §14 |
| D5 | Comparison, operating envelope, and system analysis are **core scope**, not deferred | They are part of basic analysis workflow |
| D6 | Qt and Streamlit retire as soon as each workflow is replaced | Per-workflow retirement, not big-bang |
| D7 | Backend engine = v3 stack (`hda/domain`, `hda/persistence` design) + ported `core/` analytics | The v3 design is validated by ~290 tests; the analytics math is validated by use |
| D8 | **PostgreSQL** for the relational store | Multi-user writes on one server; removes the SQLite write-lock problem class |
| D9 | **No legacy data migration.** Old `campaigns/*.db` and the CSV archive stay where they are; legacy CSVs can be ingested manually on demand if ever needed | HDA is not currently in use; nothing depends on the old records |
| D10 | **No Phase 0 legacy hotfixes.** The legacy app is not in use, so its bugs are documented (this doc, §16) but not fixed | Effort goes straight into the new implementation |
| D11 | **Fresh repository** under `amphora-space`; this repo is archived after retirement | Clean history, new stack, no legacy weight |
| D12 | **Watcher transport: REST pull from the tbctl node** (poll `GET /api/runs`, download new `.h5` + sidecar, verify, re-sync mutable metadata periodically). Filesystem-mount mode kept as an optional alternative | No shared-filesystem coupling to the Pi, survives network interruptions with plain retry, and the node already serves the API. Polling every ~30 s is plenty |
| D13 | **UX prototyping runs in parallel with Phase 1** (§17): the core-loop and inbox workflows are validated with clickable prototypes before production frontend code | Workflow design is the hard part; Phase 1 is workflow-agnostic and need not wait |
| D14 | **Product name: Metron** (μέτρον, "measure"); repo `amphora-space/metron` | Names the tool after its creed — nothing ships without its measure |

---

## 4. System Architecture

### 4.1 Deployment

One Docker Compose stack on the internal server:

```
┌─────────────────────────────────────────────────────────────┐
│ internal server                                             │
│                                                             │
│  ┌──────────┐   ┌───────────────┐   ┌───────────────────┐   │
│  │ caddy /  │──▶│ hda-api       │──▶│ postgres          │   │
│  │ nginx    │   │ FastAPI       │   └───────────────────┘   │
│  │ (TLS,    │   │ (uvicorn)     │   ┌───────────────────┐   │
│  │  static  │   │               │──▶│ data volume       │   │
│  │  SPA)    │   └───────┬───────┘   │  /data/raw        │   │
│  └──────────┘           │           │  /data/series     │   │
│                 ┌───────▼───────┐   │  /data/pyramids   │   │
│                 │ hda-worker    │──▶│  /data/reports    │   │
│                 │ (job runner)  │   └───────────────────┘   │
│                 └───────────────┘                           │
└─────────────────────────────────────────────────────────────┘
          ▲ pulls .h5 / .annotations.json / channels.yaml
          │ (tbctl-node REST API; filesystem share optional)
┌─────────┴─────────┐
│ Spark Studio node │  (Raspberry Pi 5, test cabinet)
└───────────────────┘
```

Components:

- **hda-api** — FastAPI application: REST + SSE, auth, all synchronous reads. Stateless; safe to restart.
- **hda-worker** — same codebase, separate process consuming a Postgres-backed job queue (ingest conversion, pyramid builds, analysis runs, report generation). One worker process with a small thread/process pool is sufficient at this scale; the queue abstraction leaves room for more.
- **postgres** — all relational data (§5).
- **data volume** — immutable raw files, Parquet series, downsample pyramids, generated reports. Backed up together with a Postgres dump; that pair is the complete system state.
- **frontend** — static SPA build served by the reverse proxy; no server-side rendering.

### 4.2 Process rules

- Anything that can exceed ~1 s runs as a **job**: HDF5/CSV conversion, pyramid build, analysis execution, report generation, campaign export. Jobs report progress; the client subscribes over SSE.
- All numeric work happens **server-side over full-resolution data**. The browser only ever receives decimated series for display (§6.3) and final numbers.
- A bounded in-process LRU cache keeps recently used preprocessed arrays warm (per-analysis alignment results, steady-window candidates), so interactive steady-window adjustment answers in milliseconds without re-reading files. The cache is never a source of truth (lifted from `hda/services/preprocessed_cache.py`).

---

## 5. Domain Model & Database Schema

### 5.1 Entity overview

```
Program ─┬─ System ─┬─ Campaign ─┬─ TestRun ─┬─ Analysis ─┬─ Measurement
         │          │            │           │            ├─ QCFinding
         │          │            │           │            └─ Report
         │          │            │           ├─ RunChannel
         │          │            │           ├─ RunEvent
         │          │            │           └─ SeriesFile / PyramidFile
         │          │            ├─ DerivedSpec
         │          │            └─ GoldenReference
         ChannelConfig (hash-keyed) ◀── TestRun
         ConfigOverlay (versioned)  ◀── Analysis
         User ◀── every mutating record
         Job, AuditLog
```

Key inversions vs. today:

- **A test run's identity is its `run_id` + content hashes.** Program/system/campaign are *assignments* (foreign keys, editable, audited) — not filesystem locations. The archive becomes pure storage.
- **A run can have multiple analyses** (reanalysis with a different window, config, or processing version), exactly one of which is `is_current`. History is never overwritten — today's silent result-overwrite disappears.
- **Measurements are rows, not columns.** Adding a metric or a test type never requires a schema migration. This kills the `to_database_record` hardcoding that made the legacy plugin system unable to persist plugin metrics.

### 5.2 Schema (DDL sketch, PostgreSQL)

Types and defaults abridged; the real migrations live in versioned files under `migrations/` executed by the transactional runner (design from `hda/persistence/migrations/runner.py` — a crash mid-migration can never leave the DB "upgraded but unmarked").

```sql
-- Identity ---------------------------------------------------------------
CREATE TABLE users (
  id            uuid PRIMARY KEY,
  username      text UNIQUE NOT NULL,
  display_name  text NOT NULL,
  password_hash text NOT NULL,             -- argon2id
  role          text NOT NULL CHECK (role IN ('engineer','admin')),
  is_active     boolean NOT NULL DEFAULT true,
  created_at    timestamptz NOT NULL
);

-- Organizational hierarchy (assignments, not folders) ---------------------
CREATE TABLE programs (
  id uuid PRIMARY KEY, name text UNIQUE NOT NULL, created_at timestamptz NOT NULL
);
CREATE TABLE systems (
  id uuid PRIMARY KEY,
  program_id uuid NOT NULL REFERENCES programs(id),
  name text NOT NULL,
  UNIQUE (program_id, name)
);
CREATE TABLE campaigns (
  id uuid PRIMARY KEY,
  system_id   uuid NOT NULL REFERENCES systems(id),
  name        text NOT NULL,
  test_type   text NOT NULL,               -- 'cold_flow' | 'hot_fire' | 'igniter_hot_fire' | ...
  status      text NOT NULL DEFAULT 'active',  -- 'active' | 'closed'
  description text,
  created_by  uuid NOT NULL REFERENCES users(id),
  created_at  timestamptz NOT NULL,
  UNIQUE (system_id, name)
);

-- Bench channel configs, immutable, content-addressed ---------------------
CREATE TABLE channel_configs (
  sha256      text PRIMARY KEY,            -- spark-studio channels_sha256
  source      text NOT NULL,               -- 'spark_studio' | 'manual' | 'legacy_csv'
  body_yaml   text NOT NULL,               -- verbatim channels.yaml (or synthesized equivalent)
  body_json   jsonb NOT NULL,              -- parsed form for querying
  imported_by uuid REFERENCES users(id),
  imported_at timestamptz NOT NULL
);

-- Metron-side analysis config: uncertainty + roles overlay (see §9) ----------
CREATE TABLE config_overlays (
  id          uuid PRIMARY KEY,
  name        text NOT NULL,
  version     integer NOT NULL,            -- monotonically increasing per name
  test_type   text NOT NULL,
  body        jsonb NOT NULL,              -- roles, uncertainties, settings (schema-validated)
  reviewed_for_channel_sha text REFERENCES channel_configs(sha256),
  created_by  uuid NOT NULL REFERENCES users(id),
  created_at  timestamptz NOT NULL,
  UNIQUE (name, version)
);

-- Test runs ---------------------------------------------------------------
CREATE TABLE test_runs (
  id             uuid PRIMARY KEY,
  run_id         text UNIQUE NOT NULL,     -- canonical source ID (spark-studio root attr, or synthesized for CSV)
  origin         text NOT NULL,            -- 'spark_studio' | 'csv'
  name           text NOT NULL,
  campaign_id    uuid REFERENCES campaigns(id),   -- nullable: unassigned runs live in the inbox
  state          text NOT NULL,            -- run lifecycle, §7.3
  started_at     timestamptz,
  ended_at       timestamptz,
  raw_data_hash  text NOT NULL,            -- SHA-256 of the source file(s)
  source_path    text NOT NULL,            -- path of the untouched original in /data/raw
  channels_sha   text REFERENCES channel_configs(sha256),
  sequence_sha   text,                     -- spark-studio sequence_sha256 (opaque provenance)
  bench_report   jsonb,                    -- spark-studio report_json (mutable, re-syncable)
  user_meta      jsonb,                    -- spark-studio user_meta (mutable, re-syncable)
  annotations    jsonb,                    -- .annotations.json sidecar (mutable, re-syncable)
  meta           jsonb NOT NULL DEFAULT '{}',  -- part/serial/fluid/geometry — resolved metadata, §8.3
  ingested_by    uuid REFERENCES users(id),
  ingested_at    timestamptz NOT NULL,
  metadata_synced_at timestamptz
);

-- Per-run channel snapshot (from the run's own config, so recalibrations never lie)
CREATE TABLE run_channels (
  run_id      uuid NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
  tag         text NOT NULL,               -- 'pressure.chamber', 'calc.dp_ox_inj', ...
  kind        text NOT NULL,               -- 'analog_in' | 'thermocouple_k' | 'derived' | ...
  unit        text,
  rate_group  text NOT NULL,
  rate_hz     real NOT NULL,
  sample_count bigint NOT NULL,
  cal         jsonb,                       -- calibration record as acquired
  PRIMARY KEY (run_id, tag)
);

-- Event log (valve commands, FIRE, operator notes) ------------------------
CREATE TABLE run_events (
  run_id   uuid NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
  seq      integer NOT NULL,
  t        double precision NOT NULL,      -- epoch seconds, as recorded
  type     text NOT NULL,
  payload  jsonb NOT NULL,
  PRIMARY KEY (run_id, seq)
);

-- Series + pyramid file registry (§6) -------------------------------------
CREATE TABLE series_files (
  run_id     uuid NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
  rate_group text NOT NULL,
  path       text NOT NULL,                -- Parquet under /data/series
  t0         double precision NOT NULL,
  t1         double precision NOT NULL,
  n_samples  bigint NOT NULL,
  PRIMARY KEY (run_id, rate_group)
);
CREATE TABLE pyramid_files (
  run_id     uuid NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
  rate_group text NOT NULL,
  level      integer NOT NULL,             -- 0 = coarsest
  path       text NOT NULL,
  n_buckets  bigint NOT NULL,
  PRIMARY KEY (run_id, rate_group, level)
);

-- Analyses ----------------------------------------------------------------
CREATE TABLE analyses (
  id                 uuid PRIMARY KEY,
  run_id             uuid NOT NULL REFERENCES test_runs(id),
  state              text NOT NULL,        -- analysis lifecycle DAG, §8.1
  is_current         boolean NOT NULL DEFAULT false,
  test_type          text NOT NULL,
  overlay_id         uuid REFERENCES config_overlays(id),
  overlay_snapshot   jsonb NOT NULL,       -- frozen copy: results never depend on a mutable row
  role_map           jsonb NOT NULL,       -- tag → role, as resolved for this analysis
  align_spec         jsonb NOT NULL,       -- timebase alignment: target rate, method, per-group treatment
  steady_t0          double precision,
  steady_t1          double precision,
  detection_method   text,                 -- 'events+cv' | 'cv' | 'ml' | 'derivative' | 'manual'
  qc_passed          boolean,
  qc_overridden      boolean NOT NULL DEFAULT false,
  qc_override_reason text,
  processing_version text NOT NULL,
  analyst_id         uuid NOT NULL REFERENCES users(id),
  created_at         timestamptz NOT NULL,
  completed_at       timestamptz,
  superseded_by      uuid REFERENCES analyses(id)
);
CREATE UNIQUE INDEX one_current_analysis ON analyses(run_id) WHERE is_current;

-- Long-form measurements: a new metric is a row, never a migration --------
CREATE TABLE measurements (
  analysis_id     uuid NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  name            text NOT NULL,           -- 'cd', 'mass_flow', 'c_star', 'of_ratio', ...
  value           double precision NOT NULL,
  uncertainty     double precision NOT NULL,  -- NOT NULL: no naked numbers, enforced by schema
  unit            text NOT NULL,
  rel_uncertainty_pct double precision,
  kind            text NOT NULL,           -- 'measured' | 'derived'
  method          jsonb,                   -- propagation method, formula id/version, inputs
  PRIMARY KEY (analysis_id, name)
);

CREATE TABLE qc_findings (
  analysis_id uuid NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  check_name  text NOT NULL,
  status      text NOT NULL,               -- 'pass' | 'warn' | 'fail'
  blocking    boolean NOT NULL,
  detail      jsonb NOT NULL,
  PRIMARY KEY (analysis_id, check_name)
);

-- Campaign-level artifacts -------------------------------------------------
CREATE TABLE derived_specs (                -- v3 declarative derived-measurement system
  id          uuid PRIMARY KEY,
  campaign_id uuid NOT NULL REFERENCES campaigns(id),
  spec        jsonb NOT NULL,               -- formula name+version, kwarg→source map
  created_by  uuid NOT NULL REFERENCES users(id),
  created_at  timestamptz NOT NULL
);
CREATE TABLE golden_references (            -- finally persisted, not session-bound
  id          uuid PRIMARY KEY,
  campaign_id uuid NOT NULL REFERENCES campaigns(id),
  name        text NOT NULL,
  source_analysis_id uuid REFERENCES analyses(id),
  snapshot    jsonb NOT NULL,               -- frozen measurement set
  created_by  uuid NOT NULL REFERENCES users(id),
  created_at  timestamptz NOT NULL,
  UNIQUE (campaign_id, name)
);

-- Reports, jobs, audit -----------------------------------------------------
CREATE TABLE reports (
  id         uuid PRIMARY KEY,
  scope      text NOT NULL,                -- 'analysis' | 'campaign' | 'qualification' | 'comparison'
  ref_id     uuid NOT NULL,
  format     text NOT NULL,                -- 'html' | 'zip' | 'xlsx' | 'csv' | 'json'
  path       text NOT NULL,                -- stored file; download links never vanish
  created_by uuid NOT NULL REFERENCES users(id),
  created_at timestamptz NOT NULL
);
CREATE TABLE jobs (
  id         uuid PRIMARY KEY,
  type       text NOT NULL,
  status     text NOT NULL,                -- 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
  progress   real NOT NULL DEFAULT 0,
  payload    jsonb NOT NULL,
  result     jsonb,
  error      text,
  created_by uuid REFERENCES users(id),
  created_at timestamptz NOT NULL,
  started_at timestamptz,
  finished_at timestamptz
);
CREATE TABLE audit_log (
  id         bigserial PRIMARY KEY,
  at         timestamptz NOT NULL,
  user_id    uuid REFERENCES users(id),
  action     text NOT NULL,                -- 'qc.override', 'run.assign', 'analysis.supersede', ...
  entity     text NOT NULL,
  entity_id  text NOT NULL,
  detail     jsonb
);
```

### 5.3 What this schema fixes, explicitly

| Legacy defect | Fix |
|---|---|
| Two unrelated "campaign" concepts (folders vs `.db` files) | One `campaigns` table; runs are assigned via FK |
| System analysis by DB-filename string-splitting | `systems` table + joins |
| Plugin metrics cannot be persisted (`to_database_record` hardcoding) | Long-form `measurements` |
| ±0.000 uncertainties persisted silently | `uncertainty NOT NULL` + write-path validation that rejects zero-uncertainty measured values without an explicit `method` justification |
| Results overwritten on reanalysis | Append-only `analyses` with `is_current` + `superseded_by` |
| Config snapshot as a JSON string blob | `overlay_snapshot` frozen per analysis; `channel_configs` content-addressed |
| Golden references die with the session | `golden_references` table |
| `qc_summary` JSON counts, findings discarded | `qc_findings` rows |
| Non-transactional migrations, empty version steps | Versioned transactional migration runner |

---

## 6. Time-Series Storage

### 6.1 Design targets

- Spark Studio runs: ~20 channels at 50 Hz + up to a few 1 kHz channels; minutes long → MB-scale. Must feel instant.
- Legacy/other CSVs: up to ~2 GB, few kHz. Must be usable without holding gigabytes in server memory per user.
- Multiple concurrent users on one server.

### 6.2 Layout

- **Raw originals** (`/data/raw/…`) — the ingested `.h5` or `.csv`, byte-for-byte, never modified. The SHA-256 in `test_runs.raw_data_hash` is computed over this file. This is the traceability anchor.
- **Canonical series** (`/data/series/{run}/{rate_group}.parquet`) — one Parquet file per rate group: column `t` (float64 epoch seconds) plus one column per channel tag, in acquired engineering units. **Rate groups are preserved, never flattened**: the 50 Hz group and a 1 kHz group live side by side. Alignment onto a common timebase is a per-analysis operation (§8.2), recorded in `align_spec` — never baked into storage.
- **Downsample pyramids** (`/data/pyramids/{run}/{rate_group}/L{n}.parquet`) — per-level min/max/mean buckets per channel (`t_bucket, tag_min, tag_max, tag_mean`). Levels shrink by 4× per step from full resolution down to ~1k buckets. Min/max preservation means spikes never disappear from an overview plot.

### 6.3 Serving plots

`GET /api/runs/{id}/series?tags=…&t0=…&t1=…&px=1800` picks the coarsest level with ≥ ~2 buckets/px for the requested window, and switches to raw samples when the window is small enough (≤ ~4k points/channel). The browser never receives more than a few thousand points per channel per request; zoom is a new fetch. Responses are cacheable (immutable data + deterministic parameters → strong ETags).

**Rule: the pyramid is for eyes only.** Every number that appears in a result — steady-window statistics, QC, metrics — is computed server-side from the canonical full-resolution series.

### 6.4 Memory policy

Legacy 2 GB CSVs are converted to Parquet once at ingest (columnar, compressed, typically 5–10× smaller) and thereafter read with column/row-group projection — analysis touches only the channels and the time window it needs. No full-file DataFrames held per session; the warm cache (§4.2) holds bounded per-analysis working sets only.

---

## 7. Ingestion

### 7.1 Sources

1. **Spark Studio pull (primary).** A watcher polls the tbctl node's REST API (~30 s interval), downloads new `.h5` files + sidecars, and re-syncs mutable metadata on a slower cycle. A filesystem-mount mode exists as an optional alternative. Discovery is passive and continuous; runs appear in the **Inbox** (§11.2) without anyone doing anything.
2. **Manual upload (secondary).** `.h5` or `.csv` via the browser, for legacy data and odd sources. CSV ingest synthesizes a single `default` rate group, requires a time-column/unit confirmation step (auto-detected via the magnitude heuristic from `hda/preprocessing.py:detect_time_unit`), and gets `origin='csv'`. This is also the only path for old HDA-era CSVs, ingested on demand — there is no bulk legacy migration (D9).

### 7.2 The immutable / mutable split

A Spark Studio run file contains both immutable acquisition data and mutable review metadata. Ingestion treats them differently:

- **Immutable, hashed once:** `/groups/*` time series, `/events`, `run_id`, timing attrs, `channels_sha256`, `sequence_sha256`. If a re-read finds these changed for an existing `run_id`, that is an **integrity alarm**, surfaced loudly — never silently re-imported.
- **Mutable, re-synced:** `report_json`, `user_meta`, `.annotations.json`. Re-ingesting an already-known run refreshes these fields (`metadata_synced_at`), and the watcher re-syncs them periodically. Prior values are kept in the audit log.

### 7.3 Run lifecycle state machine

```
DISCOVERED ──▶ FETCHING ──▶ CONVERTING ──▶ READY ──▶ (assigned to campaign)
     │             │             │
     └─────────────┴─────────────┴──▶ INGEST_FAILED (retryable, error surfaced)

READY + mutable-metadata change ──▶ READY (metadata_synced_at bumped)
```

- **DISCOVERED** — watcher has seen the file; row exists with source path only.
- **FETCHING** — copying the original into `/data/raw`, computing `raw_data_hash`.
- **CONVERTING** — job: parse HDF5 (h5py, `tools/tbctl_load.py` semantics), write Parquet series + pyramids, extract `run_channels`, `run_events`, root attrs; register unknown `channels_sha256` (fetch the YAML if reachable, else flag "config needed").
- **READY** — visible in the Inbox, plottable, analyzable.

Assignment to a campaign is an explicit user action (single dropdown in the Inbox, with a suggestion inferred from run name/tags and the campaign's test type) — or automatic when a campaign has a configured match rule (e.g. "runs named `*-hot` from this bench → campaign IGN-HF-C3").

### 7.4 What ingestion deletes from the current workflow

Channel mapping (tags are canonical), time-unit guessing (epoch seconds in the file), metadata retyping (pulled from `user_meta` + campaign template), config selection (keyed by `channels_sha256`), and the entire "upload the same CSV three times on three pages" problem (one run entity, every tool binds to it).

---

## 8. Analysis Pipeline

### 8.1 Analysis lifecycle DAG

Adapted from `hda/domain/state.py` (the 11-state DAG), split cleanly between the *run* lifecycle (§7.3) and the *analysis* lifecycle:

```
DRAFT ──▶ ALIGNED ──▶ WINDOWED ──▶ QC_RUN ──┬─▶ ANALYZED ──▶ SAVED
  ▲          │            ▲                 ├─▶ QC_FAILED ──┐
  │          │            └─────────────────┴───────────────┘
  └──────────┴── (config/window change reverts to the appropriate earlier state)
```

- **DRAFT** — analysis created for a run; test type + overlay resolved (auto-suggested from campaign + `channels_sha256`).
- **ALIGNED** — timebase alignment computed per `align_spec` (default: analysis at the primary group's native rate; other groups interpolated onto it only where a metric needs them; recorded per-channel).
- **WINDOWED** — steady window set. Default seeding: `fire_window()` from run events, then CV-based refinement *within* that window; manual drag always available. Every change re-enters here and invalidates downstream state — the state machine makes today's stale-result bugs (results surviving a config change) structurally impossible.
- **QC_RUN** — QC suite over the windowed data. Blocking failures gate progression. Override is explicit: requires a typed reason, records `qc_overridden + qc_override_reason + analyst`, and is audit-logged. (The v3 stance was "no override"; operational reality wants the labelled escape hatch — but attributed and permanent.)
- **ANALYZED** — plugin computed metrics; uncertainties propagated; derived measurements evaluated from campaign `derived_specs`.
- **SAVED** — measurements + findings persisted; `is_current` flipped; prior current analysis marked superseded.

Transitions are validated in the repository layer before any row is touched (`update_state` validates against the DAG — from `hda/persistence`), so the orchestrator physically cannot skip a phase.

### 8.2 Preprocessing & alignment

One implementation, server-side, replacing today's three divergent copies:

1. (CSV only) time normalization, sort, dedupe.
2. NaN policy per overlay settings (`interpolate` | `drop` | `leave`), recorded.
3. Alignment per `align_spec` (§8.1); resampling method and target recorded.
4. Optional trim (rare — the fire window mostly replaces it), with the dim-preview interaction (§11.4).

Every step's parameters land in the analysis record. Same input + same spec ⇒ same output, reproducible from the record alone.

### 8.3 Metadata resolution

The v3 three-layer resolver (`hda/domain/metadata/resolver.py`), adapted to the new sources — for each required field (part, serial, fluid, geometry, …), first-set-wins with the source layer recorded per field:

1. Run-level: Spark Studio `user_meta` + annotations.
2. Campaign template: defaults configured on the campaign (e.g. the article under test for the whole campaign).
3. Operator input: filled in the UI, prompted only for what is still missing.

Incomplete required metadata puts the analysis in a visible "awaiting metadata" condition instead of erroring or silently defaulting. **The water-density class of bug is eliminated here**: fluid is a required, resolved, displayed field — never a silent fallback.

### 8.4 Test-type plugins

The v3 plugin protocol (`hda/domain/plugins.py`): class-level `name`/`version`, explicit registration (no filesystem auto-discovery), immutable `AnalysisContext` in, measurement set out. A plugin declares:

- required/optional **roles** (e.g. `chamber_pressure`, `upstream_pressure`, `mass_flow`) — resolved against channel tags via the overlay's role map, with namespace-based auto-suggestion (`pressure.chamber` → `chamber_pressure` is deterministic, not heuristic);
- its metric computations with uncertainty propagation (closed-form where validated in `core/uncertainty.py`, numerical Jacobian / Monte Carlo from `hda/domain/uncertainty.py` otherwise);
- test-type-specific QC checks (wired through the plugin, not hardcoded in the QC runner — fixing the legacy leak).

Initial plugin set: `cold_flow`, `hot_fire`, `igniter_hot_fire` (the igniter physics from `core/igniter_analysis.py` finally gets first-class uncertainties instead of the silent 0.0).

### 8.5 Derived measurements

The v3 declarative system (`hda/domain/derived/`): campaign-scoped `derived_specs` name a formula from a versioned `FormulaLibrary` and map kwargs to measurement/channel names. Chains propagate uncertainty (`mf_fuel` → `of_ratio` → `c_star`). Spark Studio's `calc.*` channels are imported as-is for display, and can additionally be re-derived and verified by the formula system (a QC check flags disagreement between bench-computed and Metron-recomputed derived channels).

### 8.6 QC

The `QCReport`/`blocking` model from `core/qc_checks.py`, rebuilt on the v3 types: generic checks (timestamps, gaps, NaN ratio, flatline, saturation, range, sensor correlation) run for every analysis; test-type checks come from the plugin. New inputs the legacy system never had: the bench `report_json` verdicts are displayed alongside (bench review answers "did the run execute correctly", Metron QC answers "is this data analyzable" — both visible, neither substitutes).

### 8.7 The readiness check

The Parameter Requirements table — the most valuable feature in the current app — becomes a first-class API object: for a given analysis, every required input (role → resolved tag → present in data? • metadata field → resolved from which layer? • config value → set?) with satisfied/missing status. The UI renders it as the gate in front of "Run analysis"; nothing runs while it shows red, and it shows *exactly what* is missing and where it will come from.

---

## 9. Configuration Model

Two artifacts, cleanly split (replacing four overlapping legacy modules and three validation systems):

1. **Channel configs (theirs).** Imported Spark Studio `channels.yaml`, content-addressed by `channels_sha256`, immutable. Owns: hardware mapping, units, calibration, rate groups, tag namespace. Metron never edits these. Unknown hash on ingest ⇒ prompt to import that YAML version (or auto-fetch from the node); known forever after.
2. **Config overlays (ours).** Versioned Metron documents, schema-validated with **Pydantic as a hard dependency** (the legacy app's optional-pydantic-silently-validates-nothing failure is not carried forward). An overlay contains:
   - **roles**: role → channel tag map (auto-suggested from the namespace, editable);
   - **uncertainties**: per-tag specs (`{type: rel|abs, value, unit}`) — the calibration knowledge Spark Studio doesn't carry;
   - **settings**: NaN policy, alignment defaults, steady-detection parameters, QC thresholds.

An overlay records `reviewed_for_channel_sha`. When a run arrives with a different `channels_sha256` than the overlay was reviewed against (bench recalibrated / channels changed), the analysis proceeds but carries a visible "overlay not reviewed against this bench config" flag until an engineer confirms or revises — recalibrations can never silently invalidate uncertainty assumptions again.

(No automated migration of legacy `saved_configs/*.json` — per D9 there is no legacy migration; the handful of useful values in those files, mainly sensor uncertainties, are re-entered into overlays by hand.)

---

## 10. API Surface

REST, JSON, cookie-session auth. Sketch of the surface (v1):

```
Auth
  POST   /api/auth/login | /logout
  GET    /api/auth/me

Organization
  GET/POST        /api/programs, /api/systems, /api/campaigns
  GET/PATCH       /api/campaigns/{id}                # incl. metadata template, match rule
  GET             /api/campaigns/{id}/results        # long-form measurements, filterable
  GET/POST        /api/campaigns/{id}/derived-specs
  GET/POST        /api/campaigns/{id}/golden-refs

Runs
  GET    /api/runs                                   # filter: state, campaign, unassigned, origin, date, text
  POST   /api/runs/ingest                            # upload; or {source_path} for watcher-visible files
  POST   /api/runs/{id}/resync                       # refresh mutable metadata
  PATCH  /api/runs/{id}                              # campaign assignment, meta edits (audited)
  GET    /api/runs/{id}                              # full detail incl. channels, bench report, provenance
  GET    /api/runs/{id}/series?tags&t0&t1&px         # decimated or raw, §6.3
  GET    /api/runs/{id}/events
  GET    /api/runs/{id}/export.csv?tags&t0&t1        # full-resolution export (mirrors tbctl's endpoint shape)

Analyses
  POST   /api/runs/{id}/analyses                     # create draft (test_type, overlay)
  GET    /api/analyses/{id}                          # incl. readiness object (§8.7)
  PATCH  /api/analyses/{id}                          # window, role_map, align_spec, overlay → state reverts per DAG
  POST   /api/analyses/{id}/window/suggest           # events+CV seeding
  GET    /api/analyses/{id}/window/stats             # instant steady-window statistics (warm cache)
  POST   /api/analyses/{id}/qc                       # run QC
  POST   /api/analyses/{id}/qc/override              # {reason} — audited
  POST   /api/analyses/{id}/run                      # job: compute + propagate + derive
  POST   /api/analyses/{id}/save                     # persist, flip is_current
  POST   /api/analyses/{id}/sweep                    # parameter sweep over windows → comparison table

Analytics
  GET    /api/campaigns/{id}/spc?param&chart=imr|xbar_r|cusum|ewma&usl&lsl
  GET    /api/systems/{id}/spc?param                  # cross-campaign, boundary-annotated
  POST   /api/compare                                 # {a, b} analyses/campaigns/golden, tolerance
  GET    /api/campaigns/{id}/envelope?x&y&filters
  POST   /api/tools/{transient|frequency|anomaly}     # bound to a run + window, job-backed

Configs
  GET/POST  /api/channel-configs                      # import YAML; GET by sha
  GET/POST  /api/overlays;  POST /api/overlays/{id}/review  # confirm against a channel sha

Reports & jobs
  POST   /api/reports                                 # {scope, ref_id, format} → job
  GET    /api/reports/{id}/download                   # stored artifact, link never dies
  GET    /api/jobs/{id};  GET /api/jobs/{id}/events   # SSE progress
  GET    /api/export/campaign/{id}?format=csv|xlsx|json|qualification_zip
```

Design rules: every list endpoint paginates and filters server-side; every response carrying results includes uncertainties and provenance fields (no API-level naked numbers); mutating endpoints write `audit_log`; SSE only for job progress and inbox updates (no websocket state to manage).

---

## 11. Frontend

### 11.1 Stack

React + TypeScript + Vite. **TanStack Query** for all server state (caching, invalidation, optimistic updates); a small client store (Zustand) for UI-only state. **uPlot** for time-series plots — deliberately the same library Spark Studio uses: engineers get identical pan/zoom/cursor behavior on the bench and in analysis, and it comfortably handles the point volumes in §6.3. **ECharts** for statistical charts (SPC, histograms, box/violin, correlation, envelope). URL-addressable everything — `/runs/{id}`, `/analyses/{id}`, `/campaigns/{id}/spc?param=cd` — so any view is a shareable link (the one thing Streamlit got right, kept).

### 11.2 Page map

```
/login
/                      Dashboard: inbox (unassigned READY runs), recent analyses,
                       campaign health tiles, my pending QC overrides
/runs                  Run explorer: filterable table (state, campaign, origin, date, text)
                       + Miller-column browse (Program → System → Campaign → Runs)
/runs/{id}             Run detail: plot workspace, events timeline, bench review,
                       channels, provenance card, [Analyze] [Assign] [Export]
/analyses/{id}         THE CORE LOOP (§11.3)
/campaigns             Campaign list + create
/campaigns/{id}        Tabs: Summary · Results · SPC · Trends · Compare · Envelope · Reports
/systems/{id}          Cross-campaign: summary, trends with campaign boundaries, SPC
/tools                 Transient · Frequency · Anomaly — each binds to a selected run
                       (never re-upload; run picker + window from its current analysis)
/configs               Channel configs (imported, by sha) · Overlays (versioned, review status)
/admin                 Users, watcher settings, migration status
```

Batch analysis is **not a page**: the run explorer supports multi-select → "Analyze N runs with overlay X", which fans out the §8 pipeline as jobs and lands results in a batch review table. Same pipeline, same code path, no drift.

### 11.3 The core loop screen (`/analyses/{id}`)

Layout lifted conceptually from the Qt STA page — left controls, right persistent plot workspace; the plot never unmounts while stages change:

- **Left rail — pipeline stepper** (the current app's status checklist, made structural): Align → Window → QC → Analyze → Save, each with state, each clickable to revisit; downstream steps visibly invalidate when an upstream one changes.
- **Window stage**: steady region as a draggable overlay, bidirectionally bound to numeric fields (single `{t0, t1}` store, `source` discriminator — the Qt reentrancy-guard pattern done the React way). Event markers (FIRE, valve commands) drawn on the plot; "Suggest" seeds from the fire window. Live window statistics (mean/CV per role channel) update as you drag, served from the warm cache.
- **QC stage**: findings table with pass/warn/fail chips, blocking items pinned on top; bench review verdicts in a separate labelled panel; override behind a reason-required dialog.
- **Analyze stage**: the readiness table (§8.7) followed by results — metric cards with `value ± u (rel %)`, measured/derived badges, and the provenance card (all five hashes, processing version, analyst, timestamp).
- **Sweep** (kept from Streamlit's Quick Iteration): run the analysis across a set of windows, get a `%Δ vs current` table, promote any row to the current window.
- Keyboard: `F5` re-run stage, `Ctrl+Enter` advance, `Ctrl+S` save — honoring the Qt app's habit set.

### 11.4 Interaction patterns carried from Qt

- Dim-before-discard on trim (out-of-window data ghosted at low alpha before commit).
- Split render paths: overlay layer never re-renders the data layer during drag.
- Pull-then-edit: auto-populated values stay editable and show what was pulled from where.
- Auto-suggest never clobbers a manual choice; validation reports what's still missing.
- Wheel-guard on numeric inputs (`onWheel → blur`) so scrolling a form never changes a value.
- Two-channel error surfacing: contextual banner + global toast/status.

### 11.5 Design system

Dark-first, dense, engineering-native — continuous with the Qt app's VS Code Dark+ direction and visually adjacent to Spark Studio. The Qt token layer (`hda/ui/style.py`) and the state-color map (`hda/ui/dashboard.py`) transfer to CSS custom properties as the starting palette; a light theme is derived from the same tokens. Tabular numerals everywhere digits align. No emoji, no priority badges, information density over whitespace (the Qt design rules, kept verbatim).

---

## 12. Auth, Identity & Audit

- Username + password (argon2id), server-side sessions in Postgres, HTTP-only secure cookies. No OAuth/SSO in v1 (internal network); the session layer is thin enough to swap later.
- Roles: `engineer` (everything analytical) and `admin` (user management, watcher/config administration, campaign closure). Deliberately no finer-grained permissions — the audit trail, not access control, is the accountability mechanism.
- Every mutating action records the user; `analyses.analyst_id` is a verified identity, replacing the legacy free-text `analyst_username`.
- `audit_log` captures the sensitive verbs: QC overrides (with reason), run reassignment, analysis supersession, metadata edits, overlay review confirmations, migration actions.

---

## 13. Reporting & Export

- **Test report** (per analysis): HTML, server-rendered from templates (Jinja2, autoescaped — the legacy 0-escaping string concatenation is not ported). Sections: identity & provenance chain, plots (rendered server-side from the same series API), measurements with uncertainties, QC findings + any override with reason, bench review summary, event timeline.
- **Campaign report**: summary stats, trend + SPC charts with violations, capability indices, results table.
- **Qualification package** (kept — it maps to a real process): ZIP of summary CSV, full long-form CSV, JSON archive, traceability report, MANIFEST.json — now including the full provenance chain per run and the auth identity of the preparer.
- **Exports**: CSV/XLSX/JSON at campaign and run scope; full-resolution series CSV per run (mirroring tbctl's export shape).
- All generation runs as jobs; artifacts are stored in `/data/reports` and registered in `reports` — a download link is permanent, never a byproduct of transient UI state.

---

## 14. Spark Studio Integration Contract

What Metron depends on (and nothing more):

| Contract item | Metron's use |
|---|---|
| HDF5 layout: `/groups/<rg>/{t,data}` + group attrs `tags`, `rate_hz` | Series ingestion |
| Root attrs: `run_id`, `t_stop*`, `sample_counts`, `channels_sha256`, `sequence_sha256`, `report_json`, `user_meta` | Identity, provenance, review sync |
| `/events` JSON-lines | Event timeline, fire-window seeding, phase ground truth |
| `.annotations.json` sidecar | Review sync |
| `configs/channels.yaml` structure (tags, kinds, units, cal, rate_groups, `calc.*` exprs) | Channel-config import, role auto-mapping, derived verification |
| File naming `<yyyymmdd-hhmmss>-<name>.h5`; `run_id` canonical | Discovery; identity comes from attrs, not filename |
| Node REST (`GET /api/runs/…`) | Primary watcher transport (D12); filesystem share optional |

Compatibility posture: the ingest parser validates against this contract and **quarantines** (visible error state, raw file kept) anything that deviates, rather than guessing. When tbctl v2's UI editors arrive, nothing changes for Metron — they round-trip through the same YAML and the sha-based provenance holds (their frozen decision D12).

Deliberately out of scope: Metron writing anything back to the bench; shared auth; a "send to analysis" button in Spark Studio (trivial later: one POST to `/api/runs/ingest`).

---

## 15. Code Reuse Map

From the codebase assessment — what moves, what's rewritten, what dies:

### Carried forward (port, don't rewrite)

| Source | Destination | Notes |
|---|---|---|
| `hda/domain/` (types, state DAG, uncertainty, QC model, plugins, derived, metadata resolver) | backend engine core | Near-wholesale, with its ~290 tests |
| `hda/persistence/` migration runner + repository pattern + normalized-schema design | Postgres layer | Schema per §5; transactional runner kept |
| `core/traceability.py` | provenance module | Nearly verbatim; extended with the bench hashes |
| `core/spc.py` | analytics | Math untouched, typed interface |
| `core/transient_analysis.py`, `core/frequency_analysis.py`, `core/advanced_anomaly.py` | analytics | Math untouched |
| `core/steady_state_detection.py` | window seeding | Drop the >95%-window rejection; add event seeding |
| `core/igniter_analysis.py` | igniter plugin | Best domain code in the repo; gains real uncertainties |
| `core/fluid_properties.py` | fluid module | Delete the `test_metadata.py` twin |
| `core/uncertainty.py` closed-form propagations (Cd, Isp, c*, O/F) | plugin formulas + parity fixtures | Formulas validated; plumbing replaced by v3 machinery |
| `core/comparison.py`, `core/operating_envelope.py` (calc half) | analytics | Envelope's plotly half stays behind |
| `core/qc_checks.py` check implementations | QC suite | Rehung on v3 types; role hardcoding removed |
| `core/batch_analysis.py` injected-callable runner concept | job fan-out | |
| `hda/preprocessing.py` (`detect_time_unit`, pipeline shape) | CSV ingest path | The Qt copy is the most correct of the three |
| Qt design tokens, state-color map, interaction patterns | frontend | §11.4–11.5 |
| Streamlit report *layouts*, qualification package *content* | report templates | Rebuilt as escaped Jinja2 |

### Rewritten (concept kept, code not)

Config/metadata cluster (`config_validation`, `config_manager`, `saved_configs`, `metadata_manager`) → §9. `campaign_manager_v2` → §5. `plugins.py` + `plugin_modules/` → v3 protocol. `integrated_analysis.py` → the §8 orchestrator. `reporting.py` → templated renderer.

### Deleted at the end

All Streamlit `pages/` + `app.py`; all Qt `hda/ui/`; the remainder of `core/`; the duplicate helpers (`analysis_tools_helpers`, `campaign_helpers` — absorbed); the v3 `services/` Qt-facing glue. Net: ~6k LOC carried, ~40k+ retired.

---

## 16. Testing & Numeric Parity

- **Golden-file verification suite (gate for Phase 2).** A fixture set of real runs (cold flow, hot fire, igniter) with **hand-verified expected values** — computed independently (spreadsheet / manual propagation), optionally cross-checked against the legacy code *where the legacy code is known-correct*. The new engine must reproduce every measurement and uncertainty within tight tolerance. Known-wrong legacy behaviors (water-density fallback, ±0.000 uncertainties, ms/s window ambiguity) are documented here as anti-targets: fixtures explicitly assert the *correct* value, never bug-compatible output.
- **Engine tests**: the v3 suite comes along; ported analytics keep their existing tests, rehoused.
- **Property tests** for uncertainty propagation (closed-form vs Jacobian vs Monte Carlo agreement on the flagship formulas).
- **Contract tests** for Spark Studio ingest against fixture `.h5` files (including a malformed set → quarantine behavior).
- **API tests** over a real Postgres (testcontainer); DAG transition enforcement tested at the repository layer.
- **E2E smoke** (Playwright): login → inbox → assign → analyze → QC → save → report download, run against the compose stack in CI.

---

## 17. Delivery Phases & Retirement Criteria

The legacy apps are not in daily use (D10), so there is no transition period to protect — phases optimize for reaching a usable core loop fast, with the workflow validated before it is built.

### Phase 1 — Spine  ∥  UX track

Two parallel streams:

**1a — Spine (backend, workflow-agnostic).** Fresh repo; compose stack; Postgres schema + migration runner; auth; ingest pipeline (HDF5 + CSV → raw + Parquet + pyramids); Spark Studio watcher (REST pull); channel-config import; series API; engine core lifted from `hda/domain` with its tests.
**Exit**: new runs appear in the inbox automatically and are plottable via the series API.

**1b — UX track (frontend, workflow-critical).** 2–3 rounds of clickable prototypes of the **inbox → assign → analyze → QC → save** loop and the campaign SPC view, populated with realistic data (Spark Studio `--sim` runs). Each round: Luis walks the workflow, interaction counts and dead ends are measured against the G1 target, the flow is revised. No production frontend code before this converges.
**Exit**: the core-loop screen design (§11.3) is validated or revised; §11 is updated to match; the API surface is adjusted where the prototypes demanded it.

### Phase 2 — The core loop

`/runs`, `/runs/{id}`, `/analyses/{id}` built to the validated design: alignment, event-seeded windowing, QC + override, plugin analysis with uncertainties, derived measurements, save, test report, sweep. Cold-flow + hot-fire + igniter plugins. Golden-file verification suite green (§16).
**Exit / retirement**: the new app is the tool for single-test work; Streamlit pages 1–2 and Qt Test Explorer + STA have no reason to exist.

### Phase 3 — Campaign, system & comparison

`/campaigns/{id}` full tab set: results, SPC (I-MR, X̄-R, CUSUM, EWMA, WE rules, capability), trends, comparison (test/campaign/golden with persisted goldens), operating envelope; `/systems/{id}` cross-campaign views with boundary annotations; campaign reports + qualification package + exports.
**Exit / retirement**: Streamlit pages 4–7 and Qt Campaign/Configurations superseded.

### Phase 4 — Long tail & full retirement

Multi-select batch fan-out; `/tools` (transient, frequency, anomaly) bound to runs; overlay review workflow polish; admin; watcher hardening; documentation. A final UX iteration pass over the real workflows with real usage behind them.
**Exit**: the old repo (Streamlit, Qt, `core/`) is archived; one application remains.

---

## 18. Risks

| Risk | Mitigation |
|---|---|
| Numeric regressions during the port | Golden-file verification suite as a hard gate (§16); math ported, not rewritten |
| Building the wrong workflow (the legacy apps' actual failure mode) | UX track (Phase 1b): prototypes validated against interaction-count targets before production frontend code; final iteration pass in Phase 4 |
| Spark Studio format drift before v2 stabilizes | Narrow contract (§14), validating parser, quarantine path; contract fixtures pinned |
| Scope creep re-importing all legacy features | §2 G4; the Phase-4 cut list requires a usage argument per feature |
| Single-server durability | Nightly Postgres dump + `/data` snapshot as one backup unit; restore rehearsed once |
| One-person-team bus factor on a new stack | Boring, mainstream choices (FastAPI, Postgres, React, TanStack, uPlot); no exotic infra |
| Interactive window-stats latency on 2 GB legacy runs | Warm cache + column-projected Parquet reads; worst case degrades to a visible sub-second spinner, never a rerun-the-world |

---

## 19. Open Decisions

None. All decisions are recorded in §3 (D1–D14). The product is named **Metron** (D14); the repo is `amphora-space/metron`.

---

*Companion references: `CLAUDE.md` (legacy conventions), `PLUGIN_ARCHITECTURE.md` (legacy plugin system), `hda/README.md` (Qt app), `amphora-space/spark-studio` — `docs/analysis.md`, `configs/channels.yaml`, `tools/tbctl_load.py` (integration contract).*
