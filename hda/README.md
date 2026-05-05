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

## Launching the desktop app (PySide6)

```bash
pip install -e .                          # one-time, from the repo root
hda                                       # default db at ~/.hda/hda.db
hda --db /tmp/hda.db --campaign INJ-CF-C1 # override location / campaign
```

`pip install -e .` installs the package in editable mode and creates an
`hda` console script. You can also still use `python -m hda` once the
package is on the Python path (either via the editable install above or
by running from the repo root).

The app opens a single window with a campaign-scoped test list on the
left, a detail panel (measurements + QC findings) on the right, and an
"Add Test…" toolbar action (Ctrl+O) that runs ingest + analysis on a
background QThreadPool worker so the UI never blocks. Logs land in
``~/.hda/logs/hda.log`` (rotating). A ``QLockFile`` next to ``hda.db``
prevents two app instances from racing on the same database.

The first launch creates a ``DEMO-C1`` campaign with the
``basic_means`` plugin so any CSV with a ``timestamp`` column ingests
and analyzes immediately. Drop your hot-fire / cold-flow CSV in via the
file dialog to test the pipeline end-to-end.

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

1. Watch folder + drag-and-drop ingest in the UI.
2. SPC charts on the analytics screen (I-MR control limits + Western
   Electric rules), built on the existing cross-campaign query.
3. Date-range / campaign multi-select filters on analytics.
