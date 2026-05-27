# HDA v3 — Qt6 / PySide6 rewrite

This package is the new Hopper Data Studio, built alongside the existing
Streamlit app in `core/` and `pages/`. The Streamlit code is **untouched**;
v3 will be developed in parallel and replace it when feature-complete.

## Why a rewrite

A deep audit of the Streamlit app surfaced architectural issues that cannot
be fixed incrementally: implicit session-state coupling, full-page reruns,
business logic in pages, race-prone state transitions, per-campaign SQLite
files that block cross-campaign analytics, missing structured logging, and
SPC bugs that affect every campaign chart. v3 addresses these from the
ground up.

## Layered architecture

```
domain/       pure-python core. No I/O, no Qt, no DB.
persistence/  SQLite (WAL) + migrations + repositories. One hda.db.
services/     orchestration, threading, watch-folder.
ui/           PySide6 widgets (added in a later commit).
```

**Hard rules** (enforced by import discipline + a future CI check):

- `domain/` may not import `persistence/`, `services/`, or `ui/`.
- `services/` may not import `ui/`.
- Only `ui/` owns the Qt event loop.

## Workflow primitives

| Concept | Where | Notes |
|---|---|---|
| Single test belongs to a campaign | `domain.types.Campaign` | Single-test "ad-hoc" campaigns are first-class. |
| Cross-campaign part tracking | `persistence` schema | Single `hda.db` with `hardware(part_number, serial_number)` indexed for filtering across campaigns. |
| Plugin-aware metadata at ingest | `domain.types.TestMetadata.extra` | Plugin-declared fields persist into traceability. |
| Frictionless ingest | `services.IngestService.enqueue` | Watch folder, drag-drop, dialog, CLI all converge here. |
| Calculated channels & measurements | `domain.derived` | Declarative specs + a `FormulaLibrary` registry. Participate in QC and traceability. |
| TestRun lifecycle | `domain.state.TestState` + `transition()` | DAG of legal transitions. Replaces session-state races. |

## Two Qt stacks (nav app vs v3)

The `hda/` package contains **two UI architectures** that coexist:

| | **Nav app** (current entry point) | **v3 stack** (tested, not wired to entry) |
|---|---|---|
| **Launch** | `python -m hda` or `hda` | Legacy widgets: `dashboard.py`, `detail_panel.py`, `analytics_window.py` |
| **Shell** | `HDAMainWindow` — sidebar nav + page stack | Campaign dashboard + detail panel |
| **Analysis** | `core/` (`integrated_analysis`, `campaign_manager_v2`) | `hda/domain` + `hda/services` |
| **Campaign DB** | Per-campaign SQLite in `campaigns/` | Single `hda.db` (WAL) |
| **Status** | 3 pages implemented (Test Explorer, Single Test Analysis, Configurations) | ~260 unit tests; ingest/reanalyze pipeline complete |

**Recommended path:** extend the **nav app** using `core/` for analysis until Campaign Analysis is built; selectively reuse v3 widgets (steady preview, hardware analytics) as components.

## Launching the desktop app (nav app)

```bash
pip install -e .          # from repo root; installs PySide6 + pyqtgraph
python -m hda             # or: hda
hda --log-dir /var/log/hda   # optional log directory override
```

The nav app opens **Hopper Data Studio** with Test Root / Program context in the sidebar. Implemented pages:

- **Test Explorer** — browse/ingest/edit metadata (filesystem via `core.test_metadata`)
- **Single Test Analysis** — CSV load, pyqtgraph steady window, full P0 analysis
- **Configurations** — edit `saved_configs/` JSON; **Use in Analysis** handoff

Placeholder pages (Batch, Campaign, System, Tools) direct users to Streamlit until implemented.

Logs: `~/.hda/logs/hda.log` (rotating). Version shown in status bar matches `core.__version__` (currently 2.5.0). Package semver is `3.0.0.dev0`.

### Archived v3 dashboard launcher

An earlier prototype used `hda --db PATH --campaign NAME` to open a campaign-scoped test list (`dashboard.py` + `detail_panel.py`). That CLI is **not** exposed by `hda/__main__.py` today. The v3 stack remains in the repo for unit tests and future integration; see sections below for hardware analytics and steady-state preview behaviour.

## Running the tests

```bash
QT_QPA_PLATFORM=offscreen python -m pytest hda/tests/
```

260 tests passing as of this commit (one skipped where Qt widgets
cannot load — desktops are fine, libEGL-less containers skip
gracefully).

## Hardware analytics (cross-campaign)

Toolbar action **Hardware Analytics… (Ctrl+H)** opens a separate window
that filters by `part_number` / `serial_number` / `measurement_name` and
plots the measurement value with error bars across **every campaign that
hardware appeared in**. Powered by a single indexed join on the v3
single-DB schema (``MeasurementsRepository.hardware_history``) — the
multi-database UNION the legacy app needed is gone.

The status bar shows ``n / mean / std / cv / range`` for the currently
filtered set; the table below the plot lists each test_run with its
campaign, serial, persisted timestamp, value, and uncertainty.

## Interactive steady-state preview

Selecting a persisted test in the dashboard opens the **steady-state
window** preview in the detail panel: the chosen channel plotted with a
draggable shaded region marking the steady window. Drag either handle
and a stats readout updates live (mean, std, n, CV%) for every channel
— microsecond-fast because ``window_stats`` is pure numpy and the
preprocessed DataFrame is held in an in-memory LRU cache.

Click **Apply window** to commit the new bounds. The detail panel
spawns a ``ReanalyzeWorker`` that re-runs QC + plugin compute + derived
measurements and atomically replaces the persisted measurements + qc
findings + steady-window fields. State machine has dedicated edges
``PERSISTED → STEADY_DETECTED`` and ``QC_FAILED → STEADY_DETECTED``
gated behind ``AnalysisService.reanalyze`` so accidental code paths can't
trigger a re-run.

## Next commits (in order)

### Nav app (priority)

1. **Campaign Analysis page** — see plan below (wraps `core.spc` + `campaign_manager_v2`)
2. **Batch Analysis page** — wraps `core.batch_analysis.run_batch_analysis()`
3. **System Analysis** / **Analysis Tools** — port Streamlit page logic

### v3 stack (selective adoption)

1. Watch folder + drag-and-drop ingest in the UI.
2. SPC charts on the analytics screen (I-MR control limits + Western
   Electric rules), built on the existing cross-campaign query.
3. Date-range / campaign multi-select filters on analytics.
4. Reuse `steady_state_preview.py` + `ReanalyzeWorker` inside Single Test Analysis.

## Campaign Analysis — implementation plan

**Decision:** the nav app uses **per-campaign SQLite** (`campaigns/*.db` via `core.campaign_manager_v2`) to match Streamlit and existing campaign data. Migration to single `hda.db` is deferred until SPC parity is proven on the legacy path.

**Page:** replace `CampaignAnalysisPage` placeholder in `hda/ui/pages/placeholders.py` with `campaign_analysis.py`.

**UI sections** (mirror `pages/4_Analysis_by_Campaign.py`):

| Tab | Core modules | Qt widgets |
|-----|--------------|------------|
| Campaign picker | `get_available_campaigns`, `get_campaign_data` | Combo + refresh worker |
| Summary | campaign metadata, test table | `QTableView` + export actions |
| SPC | `create_imr_chart`, `create_xbar_r_chart`, CUSUM/EWMA | pyqtgraph or matplotlib embed |
| Reports | `generate_campaign_report`, `export_campaign_excel` | Background `QRunnable` + file dialog |

**Workers:** `_CampaignLoadWorker`, `_SPCWorker`, `_ReportWorker` — all call `core/` only (no Streamlit).

**Acceptance criteria:**

- Select campaign from `campaigns/` directory; load test results without blocking UI
- I-MR chart with control limits and Western Electric violation markers
- Export HTML report and Excel via existing `core.export` / `core.reporting`
- Status bar feedback during load and report generation
