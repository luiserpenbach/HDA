# Hopper Data Studio

Engineering-grade analysis for rocket propulsion test data — cold flow, hot fire, and campaign SPC — with traceability, uncertainty, and QC built in.

Two UIs share the same **`core/`** engine:

| UI | Launch | Best for |
|---|---|---|
| **Desktop (Qt)** | `pip install -e .` then `python -m hda` | Daily test-stand work, dense plots, keyboard workflows |
| **Streamlit (web)** | `pip install -r requirements.txt` then `streamlit run app.py` | Remote access, sharing via URL |

See [`hda/README.md`](hda/README.md) for the desktop app (pages, plot panels, workers, tests).

## Engineering integrity (P0)

- **Traceability** — SHA-256 hashing and audit trails on every analysis
- **Uncertainty** — error propagation for all metrics (no naked numbers)
- **QC** — blocking checks before analysis runs
- **Config validation** — dataclass schemas for hardware + metadata
- **Campaign manager** — SQLite per campaign with full traceability fields

## Features

### Analysis & reporting (P1)

- SPC control charts and capability indices
- HTML / Excel / CSV campaign reports and export
- Batch multi-file processing (Streamlit; Qt stub)

### Advanced (P2)

- Anomaly detection, test comparison, transient & frequency analysis
- Operating envelope visualization
- Available in Streamlit **Analysis Tools** and Qt **Analysis Tools** page

## Installation

### Streamlit app

```bash
git clone <repo-url>
cd HDA
pip install -r requirements.txt
streamlit run app.py
```

### Desktop app

```bash
pip install -e ".[dev]"
python -m hda
```

Optional fluid properties: `pip install CoolProp>=6.4.1`

## Project structure

```
HDA/
├── app.py                      # Streamlit entry
├── requirements.txt            # Streamlit + core dependencies
├── pyproject.toml              # Desktop package (hda), PySide6, pyqtgraph
├── core/                       # Shared analysis engine (P0/P1/P2)
│   ├── integrated_analysis.py
│   ├── campaign_manager_v2.py
│   ├── spc.py, reporting.py, batch_analysis.py
│   ├── advanced_anomaly.py, comparison.py, transient_analysis.py
│   └── plugin_modules/         # cold_flow, hot_fire plugins
├── pages/                      # Streamlit pages (1_Test_Explorer … 7_Configurations)
├── hda/                        # Qt desktop UI + v3 domain stack
│   ├── ui/pages/               # Test Explorer, STA, Campaign, Tools, Configs
│   ├── preprocessing.py        # STA preprocess pipeline
│   └── ui/plot_panels.py       # Dockable plot workspace
├── saved_configs/              # Testbench JSON configs
├── campaigns/                  # Per-campaign SQLite (*.db)
└── tests/                      # core/ test suite
```

## Quick start (Python API)

```python
from core.integrated_analysis import analyze_cold_flow_test
from core.campaign_manager_v2 import create_campaign, save_to_campaign

result = analyze_cold_flow_test(
    df=df,
    config=config,
    steady_window=(1.5, 5.0),  # seconds on time_s axis
    test_id="INJ-CF-001",
    file_path="test_data.csv",
)

print(result.passed_qc, result.measurements["Cd"])
record = result.to_database_record("cold_flow")
save_to_campaign("INJ_Acceptance_Q1", record)
```

## Configuration

- **Active config** (`saved_configs/`) — testbench hardware, channels, uncertainties
- **Test metadata** (`metadata.json` per test folder) — geometry, fluid, part/serial

See `CLAUDE.md` for full conventions and both UI architectures.

## Testing

```bash
# Core integrity + analysis
python -m pytest tests/ -q

# Desktop package (requires PySide6)
pip install -e ".[dev]"
python -m pytest hda/tests/ -q
```

## Version

| Component | Version |
|-----------|---------|
| `core` (analysis engine) | 2.5.0 |
| Streamlit app | tracks `core` |
| Qt package (`hda`) | 3.0.0-dev |
| Campaign DB schema | 2 (`SCHEMA_VERSION` in `campaign_manager_v2.py`) |

## License

Internal use only — Hopper Propulsion Systems
