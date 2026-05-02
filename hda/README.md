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

## Running the new tests

```bash
python -m pytest hda/tests/ -v
```

93 tests passing as of this commit; UI is not yet implemented.

## Next commits (in order)

1. Concrete `IngestService` + preprocessing pipeline (file hash, parse,
   resample, sidecar+campaign+operator metadata resolution, derived-channel
   evaluation during preprocessing).
2. Concrete `AnalysisService` + plugin port (cold flow, hot fire), with
   QC, steady-state, measurement+derived-measurement evaluation, and
   traceability persisted via the new repositories.
3. Repositories for `test_runs`, `hardware`, `measurements`, `qc_findings`,
   `derived_specs` — with cross-campaign hardware queries.
4. Uncertainty propagation for derived measurements (numerical Jacobian +
   Monte Carlo, plumbed into ``UncertaintyMethod``).
5. PySide6 shell — single window, dashboard model, watch folder.
6. Test-detail screen with interactive steady-state preview.
7. Campaign analytics with the cross-campaign hardware filter.
