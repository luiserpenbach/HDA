# HDA Desktop (Qt / PySide6)

Native desktop UI for Hopper Data Studio. It runs **alongside** the Streamlit app
(`app.py` + `pages/`) and uses the same **`core/`** analysis engine, campaign
databases, and saved configs.

## Quick start

From the repo root:

```bash
pip install -e ".[dev]"    # PySide6, pyqtgraph, scipy, pytest
python -m hda                # or: hda
hda --log-dir C:/logs/hda    # optional log directory
```

Logs default to `~/.hda/logs/hda.log` (rotating). Status bar shows `core.__version__`
(analysis engine); package semver is `hda.__version__` (`3.0.0-dev`).

**Streamlit-only install** (no desktop): `pip install -r requirements.txt`

## Architecture

Two UI stacks coexist in `hda/`:

| | **Nav app** (entry point) | **v3 stack** (parallel, tested) |
|---|---|---|
| Launch | `python -m hda` | Legacy: `dashboard.py`, `detail_panel.py`, `analytics_window.py` |
| Analysis | `core/` (`integrated_analysis`, `campaign_manager_v2`, …) | `hda/domain` + `hda/services` |
| Campaign DB | Per-campaign SQLite in `campaigns/` | Single `hda.db` (WAL) in v3 schema |
| Status | **Primary daily UI** | ~260 unit tests; ingest/reanalyze pipeline |

**Recommended:** extend the nav app with `core/` until v3 migration is planned.
Reuse v3 widgets (steady preview, hardware analytics) as components where useful.

```
hda/
├── __main__.py              CLI → hda.ui.main
├── preprocessing.py         Shared CSV time/resample/trim pipeline (STA)
├── plot_utils.py            Plot window helpers
├── campaign_helpers.py      Campaign page filters / metric discovery
├── analysis_tools_helpers.py Table + column helpers for Analysis Tools
├── ui/
│   ├── main_window.py       Resizable nav + page stack
│   ├── nav_bar.py           Test root, program, navigation
│   ├── style.py             VS Code Dark+ tokens + stylesheets
│   ├── plot_panels.py       Dockable multi-panel pyqtgraph workspace
│   └── pages/
│       ├── test_ingestion.py      Test Explorer
│       ├── single_test_analysis.py
│       ├── campaign_analysis.py
│       ├── analysis_tools.py
│       ├── configurations.py
│       └── placeholders.py        Batch + System Analysis stubs
├── domain/                  Pure Python (v3)
├── persistence/             SQLite repositories (v3)
└── services/                Ingest / analysis orchestration (v3)
```

**Import rules (v3 layers):** `domain/` → no I/O/Qt; `services/` → no Qt; only `ui/` owns the event loop.

## Implemented pages

| Nav item | Module | Summary |
|---|---|---|
| Test Explorer | `test_ingestion.py` | Browse / ingest / edit metadata via `core.test_metadata` |
| Single Test Analysis | `single_test_analysis.py` | Tabbed preprocess → steady state → analysis; dockable plot panels; trim lines; SVG export per panel |
| Batch Analysis | `placeholders.py` | Stub → use Streamlit or upcoming port |
| Campaign Analysis | `campaign_analysis.py` | SPC (I-MR, X-bar/R), filters, HTML/Excel/CSV reports |
| System Analysis | `placeholders.py` | Stub |
| Analysis Tools | `analysis_tools.py` | Anomaly detection, comparison, transient, frequency, operating envelope |
| Configurations | `configurations.py` | Edit `saved_configs/`; **Use in Analysis** handoff to STA |

### Single Test Analysis highlights

- **Preprocessing tab:** time unit auto-detect (incl. Unix ms), gap fill, resample, trim with red draggable bounds + dimmed preview; save processed CSV
- **Plot workspace:** add unlimited dockable panels; per-panel sensor toggles; synced trim/steady overlays; SVG export
- **Steady State tab:** auto-detect, draggable region, sensor roles, run analysis via `core.integrated_analysis`
- Handoffs: Test Explorer → STA; Configurations → STA with active config

### Analysis Tools highlights

Wraps P2 `core/` modules with background workers and pyqtgraph charts:

1. **Anomaly Detection** — CSV upload, multi-channel, severity plot bands  
2. **Data Comparison** — campaign test compare, golden reference, regression, correlation, campaign-vs-campaign  
3. **Transient Analysis** — phase segmentation, startup/shutdown metrics  
4. **Frequency Analysis** — PSD, harmonics, resonance  
5. **Operating Envelope** — O/F vs Pc scatter + envelope rectangle  

Press **F5** on Analysis Tools or Campaign Analysis to refresh the campaign list.

## Design system

Theme: **VS Code Dark+** (`hda/ui/style.py`).

- Call `content_stylesheet()` once on the main window central widget  
- Call `nav_stylesheet()` + `QPalette` on `NavBar`  
- Call `configure_pyqtgraph()` before creating plots  
- Call `apply_app_font()` once in `main.py`  
- No P0/P1/P2 badges in user-facing UI  
- Heavy work in `QRunnable` workers — never block the UI thread  

Scroll wheel on spinboxes/combos is suppressed globally (`main.py`) to avoid accidental value changes while scrolling.

## Tests

```bash
# Desktop package (needs PySide6)
pip install -e ".[dev]"
python -m pytest hda/tests/ -q

# Headless CI
QT_QPA_PLATFORM=offscreen python -m pytest hda/tests/ -q
```

Also run the shared integrity suite from repo root:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## v3 stack (not the nav entry point)

The v3 dashboard/analytics windows support:

- **Hardware Analytics (Ctrl+H)** — cross-campaign part/serial history from single `hda.db`
- **Steady-state preview + reanalyze** — draggable window, `ReanalyzeWorker`, state machine edges

These remain available for tests and selective reuse; they are not opened by `python -m hda` today.

## Roadmap

1. **Batch Analysis** — `core.batch_analysis.run_batch_analysis()`  
2. **System Analysis** — cross-campaign system metrics  
3. STA tabs: Analyze / Results / Export (currently placeholders inside STA)  
4. Optional: adopt v3 steady preview inside STA; v3 single-DB migration (long term)
