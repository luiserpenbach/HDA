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
pip install PySide6 numpy pandas pytest                # one-time
python -m hda                                          # default db at ~/.hda/hda.db
python -m hda --db /tmp/hda.db --campaign INJ-CF-C1    # override location / campaign
```

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

210 tests passing as of this commit (one skipped where Qt widgets
cannot load — desktops are fine, libEGL-less containers skip
gracefully).

## Next commits (in order)

1. Hot-fire plugin port — chamber pressure, thrust, mass flows, OF, Isp,
   c* with chained uncertainty over the SensorUncertainty model.
2. Watch folder + drag-and-drop ingest in the UI; auto-confirm threshold
   surfacing.
3. Interactive steady-state preview in the detail panel (drag-handle
   window, live mean/std/n).
4. Cross-campaign analytics screen with the hardware filter.
