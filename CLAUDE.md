# CLAUDE.md - AI Assistant Guide for Hopper Data Studio

**Version**: 2.3.0
**Last Updated**: 2026-01-16
**Purpose**: Comprehensive guide for AI assistants working with the HDA codebase

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Codebase Architecture](#codebase-architecture)
4. [Development Workflows](#development-workflows)
5. [Key Conventions](#key-conventions)
6. [Common Tasks](#common-tasks)
7. [Testing Guidelines](#testing-guidelines)
8. [Configuration Management](#configuration-management)
9. [Data Flow](#data-flow)
10. [Safety and Integrity Rules](#safety-and-integrity-rules)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is Hopper Data Studio?

Hopper Data Studio (HDA) is a **Streamlit-based web application** for analyzing rocket propulsion test data with built-in engineering integrity systems. It's used internally by Hopper Propulsion Systems for cold flow and hot fire testing.

### Core Philosophy: Engineering Integrity First

The system is built on three **non-negotiable principles**:

1. **Traceability** - Every result traceable to source data via SHA-256 hashing
2. **Uncertainty Quantification** - Every metric has error bars (no naked numbers)
3. **Quality Control** - Bad data is blocked, not silently processed

### Priority Levels

The codebase uses a P0/P1/P2 priority system:

- **P0 (Core/Foundation)**: Traceability, Uncertainty, QC, Config Validation, Campaign Manager
- **P1 (High Priority)**: SPC, Reporting, Batch Analysis, Export
- **P2 (Advanced)**: Anomaly Detection, Comparison, Saved Configurations

**IMPORTANT**: When making changes, P0 components require extra scrutiny. Breaking P0 breaks the entire integrity system.

---

## Technology Stack

### Core Dependencies

```
Python 3.8+
├── numpy>=1.20.0           # Numerical computing
├── pandas>=1.3.0           # Data manipulation
├── scipy>=1.7.0            # Scientific computing
├── matplotlib>=3.4.0       # Plotting
├── plotly>=5.0.0           # Interactive plots
├── streamlit>=1.20.0       # Web UI framework
└── openpyxl>=3.0.0         # Excel export
```

### Optional Dependencies

```
CoolProp>=6.4.1  # Highly recommended for accurate fluid properties
```

Install with: `pip install CoolProp>=6.4.1`

**Note**: CoolProp is optional but recommended. The app gracefully degrades to simplified calculations if unavailable.

### No package.json

This is a **Python-only project**. There is no Node.js, npm, or package.json. All dependencies are in `requirements.txt`.

---

## Codebase Architecture

### Directory Structure

```
HDA/
├── app.py                      # Main Streamlit entry point (landing page)
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit config (max upload 1GB)
├── core/                       # Business logic (NO UI code here)
│   ├── [P0] traceability.py       # SHA-256, audit trails
│   ├── [P0] uncertainty.py        # Error propagation
│   ├── [P0] qc_checks.py          # Quality control
│   ├── [P0] config_validation.py  # Schema validation
│   ├── [P0] campaign_manager_v2.py # SQLite database
│   ├── [P0] integrated_analysis.py # High-level API
│   ├── [P1] spc.py                # Statistical Process Control
│   ├── [P1] reporting.py          # HTML reports
│   ├── [P1] batch_analysis.py     # Multi-file processing
│   ├── [P1] export.py             # Data export
│   ├── [P2] advanced_anomaly.py   # Anomaly detection
│   ├── [P2] comparison.py         # Test comparison
│   └── [P2] saved_configs.py      # Config management
├── pages/                      # Streamlit pages (UI only)
│   ├── 1_Test_Explorer.py
│   ├── 2_Single_Test_Analysis.py  # Most complex (85k lines)
│   ├── 3_Batch_Test_Analysis.py
│   ├── 4_Analysis_by_Campaign.py
│   ├── 5_Analysis_by_System.py
│   ├── 6_Analysis_Tools.py
│   ├── 7_Configurations.py
│   ├── _shared_sidebar.py         # Global context selector
│   └── _shared_widgets.py         # Reusable UI components
├── tests/                      # Test suite
│   ├── test_p0_components.py      # Core integrity tests
│   ├── test_p1_components.py      # Analysis/reporting tests
│   ├── test_p2_components.py      # Advanced features
│   └── test_fluid_properties.py
├── saved_configs/              # Testbench hardware configs
├── campaigns/                  # SQLite campaign databases
└── scripts/                    # Utility scripts
```

### Key Architectural Patterns

#### 1. Separation of Concerns

**CRITICAL**: Maintain strict separation:
- `core/` modules: Business logic, calculations, database operations (NO Streamlit code)
- `pages/` modules: UI, user interaction, display (calls core modules)

**Bad Example**:
```python
# In core/integrated_analysis.py - DON'T DO THIS
import streamlit as st
st.write("Processing...")  # ❌ Never import streamlit in core/
```

**Good Example**:
```python
# In pages/2_Single_Test_Analysis.py
from core.integrated_analysis import analyze_cold_flow_test

result = analyze_cold_flow_test(df, config, ...)  # ✅ Core does work
st.write(result.measurements)  # ✅ Page displays results
```

#### 2. Dataclass-Heavy Design

All data structures use Python dataclasses with type hints:

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AnalysisResult:
    test_id: str
    measurements: Dict[str, MeasurementWithUncertainty]
    qc_report: QCReport
    traceability: DataTraceability
    passed_qc: bool

    def to_database_record(self, test_type: str) -> Dict[str, Any]:
        """Convert to flat dict for SQLite storage."""
        # Serialization logic...
```

**Key Classes**:
- `AnalysisResult`: Complete analysis output
- `MeasurementWithUncertainty`: Metric with error bars
- `QCReport`: Quality control results
- `DataTraceability`: SHA-256 hashes and audit trail
- `SPCAnalysis`: Control chart results
- `TestMetadata`: Test article properties

#### 3. High-Level API Pattern

`integrated_analysis.py` provides a clean facade:

```python
# Single function call does: QC → analysis → uncertainty → traceability
result = analyze_cold_flow_test(
    df=df,
    config=config,
    steady_window=(1500, 5000),
    test_id="INJ-CF-001",
    file_path="test_data.csv",
    metadata={'part': 'INJ-V1', 'serial_num': 'SN-001'}
)

# Returns complete AnalysisResult with all integrity checks
print(result.passed_qc)  # True/False
print(result.measurements['Cd'])  # MeasurementWithUncertainty object
```

**When to use**:
- Use `analyze_cold_flow_test()` or `analyze_hot_fire_test()` for standard workflows
- Only call lower-level functions (`calculate_cd_uncertainty()`, etc.) for custom workflows

---

## Development Workflows

### Workflow 1: Single Test Analysis (Most Common)

```
1. Upload CSV → pd.read_csv()
2. Load config (from saved_configs/ or manual)
3. Preprocess data:
   - Convert timestamps (ms → s)
   - Resample to uniform rate (typically 100 Hz)
   - Handle NaN values
4. Detect steady state window (auto or manual)
5. Run QC checks → assert_qc_passed() or display warnings
6. Calculate metrics + uncertainties
7. Create traceability record (SHA-256 hashes)
8. Package as AnalysisResult
9. Save to campaign (SQLite)
10. Generate report (optional)
```

**Entry point**: `pages/2_Single_Test_Analysis.py`
**Core function**: `core.integrated_analysis.analyze_cold_flow_test()`

### Workflow 2: Campaign Analysis & SPC

```
1. Load campaign → get_campaign_data(campaign_name)
2. Filter tests (by part, date, etc.)
3. Create control charts:
   - I-MR charts for individual measurements
   - X-bar/R charts for subgroups
4. Check Western Electric rules (violations)
5. Calculate capability indices (Cp, Cpk)
6. Export results
7. Generate campaign report
```

**Entry point**: `pages/4_Analysis_by_Campaign.py`
**Core functions**: `core.spc.create_imr_chart()`, `core.reporting.generate_campaign_report()`

### Workflow 3: Batch Processing

```
1. Select folder with multiple test files
2. Apply consistent config to all
3. Process in parallel (ThreadPoolExecutor)
4. Aggregate results
5. Save batch to campaign
6. Generate batch report
```

**Entry point**: `pages/3_Batch_Test_Analysis.py`
**Core function**: `core.batch_analysis.run_batch_analysis()`

### Git Workflow

**Branch naming**: `claude/add-<feature>-<session-id>`

Example: `claude/add-claude-documentation-3Ze7B`

**CRITICAL Git Rules**:
1. Always develop on designated feature branch (never main/master)
2. Push with: `git push -u origin <branch-name>`
3. Branch MUST start with `claude/` and match session ID (403 error otherwise)
4. On network failures, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s)
5. Never force push to main/master
6. Never skip hooks (`--no-verify`) unless explicitly requested

**Commit message style**:
```bash
git commit -m "$(cat <<'EOF'
feat: Add global Test Root and Program selector to sidebar

- Created _shared_sidebar.py for global context
- Added session state for test_root and program
- Updated all pages to use global context
EOF
)"
```

### Running the Application

```bash
# Start Streamlit app
streamlit run app.py

# Runs on http://localhost:8501
# Auto-reloads on file changes (development mode)
```

---

## Key Conventions

### Naming Conventions

#### Files
- Core modules: `snake_case.py` (e.g., `steady_state_detection.py`)
- Pages: `#_Title_Case.py` (e.g., `2_Single_Test_Analysis.py`)
- Private/shared pages: `_prefix.py` (e.g., `_shared_sidebar.py`)
- Tests: `test_*.py` (e.g., `test_p0_components.py`)

#### Variables
- General: `snake_case` (`steady_window`, `test_id`, `qc_report`)
- DataFrames: `df`, `df_processed`, `steady_df`, `campaign_df`
- Configs: `config`, `active_config`, `validated_config`
- Constants: `UPPER_SNAKE_CASE` (`PROCESSING_VERSION`, `SCHEMA_VERSION`)

#### Classes
- PascalCase: `AnalysisResult`, `QCReport`, `MeasurementWithUncertainty`
- Enums: `QCStatus`, `UncertaintyType`, `ViolationType`, `ControlChartType`

#### Functions
- Verbs first: `calculate_cd_uncertainty()`, `detect_steady_state_auto()`, `create_campaign()`
- Validators: `validate_config()`, `verify_data_integrity()`, `check_timestamp_monotonic()`
- Getters: `get_campaign_data()`, `get_available_campaigns()`, `get_test_metadata()`

### Code Style Patterns

#### 1. Type Hints Everywhere

```python
def analyze_cold_flow_test(
    df: pd.DataFrame,
    config: Dict[str, Any],
    steady_window: Tuple[float, float],
    test_id: str,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """
    Analyze a cold flow test with full integrity checks.

    Args:
        df: Processed test data
        config: Active configuration
        steady_window: (start_ms, end_ms) for steady state
        test_id: Unique test identifier
        file_path: Original CSV path (for hash)
        metadata: Test article properties

    Returns:
        Complete analysis results with QC, uncertainties, traceability

    Raises:
        ValueError: If QC checks fail
    """
    ...
```

#### 2. Session State Initialization (Streamlit pages)

```python
# Always check before accessing
if 'selected_config' not in st.session_state:
    st.session_state.selected_config = None

if 'global_test_root' not in st.session_state:
    st.session_state.global_test_root = None
```

#### 3. Error Handling with Context

```python
if not qc_report.passed:
    blocking_failures = [c.name for c in qc_report.blocking_failures]
    raise ValueError(
        f"QC checks failed for {test_id}. "
        f"Blocking failures: {blocking_failures}. "
        f"Fix these issues before analysis."
    )
```

#### 4. Optional Dependencies with Graceful Degradation

```python
try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    # Use simplified calculations

def get_density(fluid: str, pressure: float, temp: float) -> float:
    if COOLPROP_AVAILABLE:
        return CP.PropsSI('D', 'P', pressure, 'T', temp, fluid)
    else:
        # Fallback to ideal gas law or lookup tables
        return simplified_density(fluid, pressure, temp)
```

#### 5. Comprehensive Docstrings

Use Google-style docstrings for all public functions:

```python
def create_imr_chart(
    df: pd.DataFrame,
    parameter: str,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
) -> SPCAnalysis:
    """
    Create Individual-Moving Range control chart.

    Analyzes process stability using Western Electric rules and
    calculates capability indices if spec limits provided.

    Args:
        df: Campaign data with parameter column
        parameter: Column name to analyze (e.g., 'avg_cd_CALC')
        usl: Upper specification limit (optional)
        lsl: Lower specification limit (optional)

    Returns:
        SPCAnalysis with control limits, violations, capability

    Raises:
        ValueError: If parameter not in df

    Example:
        >>> df = get_campaign_data("INJ_Acceptance_Q1")
        >>> analysis = create_imr_chart(df, 'avg_cd_CALC', usl=0.70, lsl=0.60)
        >>> print(f"Cpk: {analysis.capability.cpk:.2f}")
    """
    ...
```

### Version Tracking

**Three separate version numbers**:

1. **Application Version**: `__version__ = "2.3.0"` in `core/__init__.py`
   - Semantic versioning (major.minor.patch)
   - Update when adding features or fixing bugs

2. **Processing Version**: `PROCESSING_VERSION = "2.0.0"` in `core/traceability.py`
   - Stored in traceability records
   - Update when analysis algorithms change
   - Ensures results can be traced to calculation method

3. **Schema Version**: `SCHEMA_VERSION = 2` in `core/campaign_manager_v2.py`
   - SQLite database schema version
   - Update when database structure changes
   - Includes migration system

**When to update**:
- Application version: Any user-facing change
- Processing version: Any change to calculation logic
- Schema version: Any database schema change (requires migration)

---

## Common Tasks

### Task 1: Adding a New Analysis Metric

**Example**: Add "average pressure drop" to cold flow analysis

1. **Update uncertainty calculation** (`core/uncertainty.py`):
```python
def calculate_cold_flow_uncertainties(...):
    # ... existing code ...

    # Add new metric
    measurements['p_drop_bar'] = MeasurementWithUncertainty(
        value=p_up - p_down,
        uncertainty=np.sqrt(u_p_up**2 + u_p_down**2),
        unit='bar',
        rel_uncertainty_pct=100 * uncertainty / value
    )

    return measurements
```

2. **Update database schema** (`core/campaign_manager_v2.py`):
```python
# Increment SCHEMA_VERSION
SCHEMA_VERSION = 3

# Add column to CREATE TABLE
CREATE TABLE IF NOT EXISTS test_results (
    ...
    avg_p_drop_bar REAL,
    u_p_drop_bar REAL,
    ...
)

# Write migration function
def migrate_v2_to_v3(conn):
    conn.execute("ALTER TABLE test_results ADD COLUMN avg_p_drop_bar REAL")
    conn.execute("ALTER TABLE test_results ADD COLUMN u_p_drop_bar REAL")
```

3. **Update AnalysisResult serialization** (`core/integrated_analysis.py`):
```python
def to_database_record(self, test_type: str) -> Dict[str, Any]:
    record = {...}

    if 'p_drop_bar' in self.measurements:
        record['avg_p_drop_bar'] = self.measurements['p_drop_bar'].value
        record['u_p_drop_bar'] = self.measurements['p_drop_bar'].uncertainty

    return record
```

4. **Update tests** (`tests/test_p0_components.py`):
```python
def test_pressure_drop_uncertainty():
    result = analyze_cold_flow_test(...)
    assert 'p_drop_bar' in result.measurements
    assert result.measurements['p_drop_bar'].uncertainty > 0
```

5. **Update PROCESSING_VERSION** if algorithm changed:
```python
# core/traceability.py
PROCESSING_VERSION = "2.1.0"  # Increment minor version
```

### Task 2: Adding a New Streamlit Page

**Example**: Create "8_Advanced_Filtering.py"

1. **Create file**: `pages/8_Advanced_Filtering.py`

2. **Follow standard page structure**:
```python
import streamlit as st
from pages._shared_sidebar import render_global_context
from core.campaign_manager_v2 import get_campaign_data

# Page config
st.set_page_config(
    page_title="Advanced Filtering",
    page_icon="🔍",
    layout="wide"
)

# Session state
if 'filter_params' not in st.session_state:
    st.session_state.filter_params = {}

# Sidebar - global context
with st.sidebar:
    context = render_global_context()

# Main content
st.title("Advanced Filtering")

with st.expander("Filter Settings", expanded=True):
    # UI for filter parameters
    min_cd = st.number_input("Min Cd", value=0.0)
    max_cd = st.number_input("Max Cd", value=1.0)

# Load data (call core module)
if st.button("Apply Filter"):
    df = get_campaign_data(context.campaign_name)
    filtered_df = df[(df['avg_cd_CALC'] >= min_cd) & (df['avg_cd_CALC'] <= max_cd)]
    st.dataframe(filtered_df)
```

3. **Add to navigation** (automatic via Streamlit numbering)

4. **Test manually**:
```bash
streamlit run app.py
# Navigate to new page in sidebar
```

### Task 3: Fixing a QC Check Issue

**Example**: QC falsely flagging good data as flatline

1. **Locate the check** (`core/qc_checks.py`):
```python
def check_flatline(df: pd.DataFrame, sensor: str, window_size: int = 100) -> QCCheckResult:
    # ... existing code ...
```

2. **Understand the logic**:
- What threshold is being used?
- What window size?
- Is it appropriate for this test type?

3. **Make conservative changes**:
```python
# BEFORE
threshold = 0.001  # Too strict

# AFTER
threshold = 0.01   # More appropriate for this sensor
```

4. **Update tests**:
```python
def test_flatline_detection():
    # Test with known-good data that was failing
    df = create_test_dataframe()
    result = check_flatline(df, 'PT-01')
    assert result.passed, "Known-good data should pass"
```

5. **Document the change**:
```python
def check_flatline(...):
    """
    Check for flatline (constant value) in sensor data.

    Uses rolling standard deviation. Threshold of 0.01 chosen to
    avoid false positives on steady-state data while catching
    true sensor failures.

    Updated 2026-01-16: Increased threshold from 0.001 to 0.01
    to reduce false positives on low-noise sensors.
    """
```

---

## Testing Guidelines

### Test Organization

```
tests/
├── test_p0_components.py      # Core integrity (CRITICAL)
├── test_p1_components.py      # Analysis & reporting
├── test_p2_components.py      # Advanced features
└── test_fluid_properties.py   # Fluid calculations
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific priority level
python tests/test_p0_components.py

# Single test
pytest tests/test_p0_components.py::TestTraceability::test_file_hash

# With coverage
pytest tests/ --cov=core --cov-report=html
```

### Testing Conventions

1. **One test class per module**:
```python
class TestTraceability:
    def test_file_hash(self):
        ...

    def test_config_hash(self):
        ...

class TestUncertainty:
    def test_cd_uncertainty(self):
        ...
```

2. **Helper functions** for test data:
```python
def create_test_dataframe() -> pd.DataFrame:
    """Create synthetic test data for testing."""
    ...

def create_test_config() -> Dict[str, Any]:
    """Create valid test configuration."""
    ...
```

3. **Assertions with messages**:
```python
assert result.passed_qc, f"Expected QC to pass, got: {result.qc_report}"
assert abs(cd - 0.65) < 0.01, f"Expected Cd ≈ 0.65, got {cd}"
```

4. **Test both happy path and error cases**:
```python
def test_qc_passes_good_data():
    df = create_good_test_dataframe()
    report = run_qc_checks(df, config)
    assert report.passed

def test_qc_fails_bad_data():
    df = create_bad_test_dataframe()  # Has NaNs, gaps, etc.
    report = run_qc_checks(df, config)
    assert not report.passed
    assert len(report.blocking_failures) > 0
```

### What to Test

**P0 (Must test thoroughly)**:
- Hash reproducibility (same data → same hash)
- Uncertainty propagation (correct formulas)
- QC check sensitivity (catches real issues)
- Config validation (rejects invalid configs)
- Database integrity (no data loss)

**P1 (Test key functionality)**:
- SPC calculations (control limits, Cpk)
- Report generation (produces valid HTML)
- Batch processing (handles errors gracefully)
- Export formats (valid CSV/Excel)

**P2 (Test major features)**:
- Anomaly detection (catches known anomalies)
- Comparison logic (correct deviations)

---

## Configuration Management

### Two Types of Configuration (v2.3.0+)

#### 1. Active Configuration (Testbench Hardware)

**Location**: `saved_configs/` directory
**Persistence**: Stored as JSON files
**When to update**: Only when testbench is modified or recalibrated

**Contents**:
- Sensor channel mappings (`"10001": "IG-PT-01"`)
- Calibration uncertainties (`{"type": "rel", "value": 0.005}`)
- Processing settings (resample frequency, filters)
- Reference values (atmospheric pressure, etc.)

**Example**:
```json
{
  "config_name": "MTB_Injector_CF_Config",
  "test_type": "cold_flow",
  "channel_config": {
    "10001": "IG-PT-01",
    "10002": "IG-PT-02",
    "10010": "FM-01"
  },
  "uncertainties": {
    "IG-PT-01": {"type": "rel", "value": 0.005},
    "IG-PT-02": {"type": "rel", "value": 0.005},
    "FM-01": {"type": "abs", "value": 0.1, "unit": "g/s"}
  },
  "settings": {
    "resample_freq_ms": 10,
    "steady_state_cv_threshold": 0.02,
    "p_atm_bar": 1.01325
  }
}
```

**How to access**:
```python
from core.saved_configs import SavedConfigManager

manager = SavedConfigManager()
configs = manager.list_configs()
active_config = manager.load_config("MTB_Injector_CF_Config")
```

#### 2. Test Metadata (Test Article Properties)

**Location**: Per-test `metadata.json` files in test folders
**Persistence**: Stored with test data
**When to provide**: For every test (varies per test article)

**Contents**:
- Part/serial numbers
- Geometry (orifice area, throat area)
- Fluid properties (name, gamma, molecular weight)

**Example**:
```json
{
  "part_number": "INJ-B1-03",
  "serial_number": "SN-2024-045",
  "geometry": {
    "orifice_area_mm2": 3.14159,
    "throat_area_mm2": 12.566
  },
  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4,
    "molecular_weight": 28.014
  }
}
```

**How to access**:
```python
from core.metadata_manager import MetadataManager

manager = MetadataManager()
metadata = manager.load_from_file("path/to/metadata.json")
# OR
metadata = manager.load_from_ui_inputs(st.session_state)
```

### Merging Configurations

**Never store test-specific data in active configs** and vice versa:

```python
from core.config_validation import merge_config_and_metadata

# Merge at analysis time
merged_config = merge_config_and_metadata(active_config, metadata)

# Now merged_config has both hardware and article properties
result = analyze_cold_flow_test(df, merged_config, ...)
```

---

## Data Flow

### Complete Analysis Pipeline

```
CSV File (raw data)
  ↓
1. UPLOAD & PARSE
  pd.read_csv() → raw DataFrame
  ↓
2. PREPROCESSING
  - Convert timestamps (ms → s)
  - Shift time to zero
  - Sort & deduplicate
  - Resample to uniform rate (100 Hz)
  - Interpolate/drop NaN values
  → processed DataFrame
  ↓
3. STEADY STATE DETECTION
  - Auto detection (CV-based, ML, derivative)
  - Manual adjustment (via UI)
  → steady_window (start_ms, end_ms)
  → steady_df (subset of processed DataFrame)
  ↓
4. QUALITY CONTROL
  - check_timestamp_monotonic()
  - check_sampling_rate()
  - check_sensor_range()
  - check_flatline()
  - check_nan_ratio()
  → QCReport (passed: True/False)
  ↓
5. ANALYSIS (if QC passed)
  - Calculate base metrics (Cd, mass flow, etc.)
  - Parse uncertainty config
  - Propagate uncertainties (analytical + Monte Carlo)
  → measurements: Dict[str, MeasurementWithUncertainty]
  ↓
6. TRACEABILITY
  - compute_file_hash() or compute_dataframe_hash()
  - compute_config_hash()
  - AnalysisContext (analyst, timestamp, version)
  - ProcessingRecord (steady window, method)
  → DataTraceability
  ↓
7. PACKAGING
  → AnalysisResult(
      test_id, measurements, qc_report,
      traceability, passed_qc
    )
  ↓
8. STORAGE
  - result.to_database_record()
  - save_to_campaign(campaign_name, record)
  → SQLite database (campaigns/*.db)
  ↓
9. REPORTING (optional)
  - generate_test_report(result)
  → HTML report with full traceability
```

### Database Schema

**Cold Flow Test Record** (25+ fields):

```python
{
  # Identity
  'test_id': str,
  'test_date': str,
  'part': str,
  'serial_num': str,
  'operator': str,
  'fluid': str,

  # Measurements (value + uncertainty for each)
  'avg_p_up_bar': float,
  'u_p_up_bar': float,
  'avg_p_down_bar': float,
  'u_p_down_bar': float,
  'avg_mf_g_s': float,
  'u_mf_g_s': float,
  'avg_cd_CALC': float,
  'u_cd_CALC': float,
  'cd_rel_uncertainty_pct': float,

  # Traceability (P0 - cannot be null)
  'raw_data_hash': str,          # SHA-256 of source data
  'config_hash': str,            # SHA-256 of config
  'config_snapshot': str,        # JSON of full config
  'analyst_username': str,       # Who ran analysis
  'analysis_timestamp': str,     # When
  'processing_version': str,     # What version

  # Processing details
  'steady_window_start_ms': float,
  'steady_window_end_ms': float,
  'steady_duration_s': float,
  'detection_method': str,
  'resample_freq_ms': float,

  # QC
  'qc_passed': bool,
  'qc_summary': str              # JSON of QC results
}
```

---

## Safety and Integrity Rules

### Critical Rules - NEVER Violate These

#### 1. Traceability is Sacred

**NEVER**:
- Calculate a result without creating a traceability record
- Modify data after hashing without updating hash
- Save to campaign without SHA-256 hash

**ALWAYS**:
- Call `create_full_traceability_record()` for every analysis
- Store `raw_data_hash`, `config_hash`, `analyst_username` in database
- Include `processing_version` to track algorithm changes

```python
# ✅ CORRECT
traceability = create_full_traceability_record(
    file_path=file_path,
    config=config,
    processing_record=processing_record
)
result = AnalysisResult(..., traceability=traceability)

# ❌ WRONG - No traceability
result = AnalysisResult(test_id, measurements, None, None, True)
```

#### 2. Uncertainty is Mandatory

**NEVER**:
- Return a metric without uncertainty
- Display a number without error bars
- Export data without uncertainty columns

**ALWAYS**:
- Use `MeasurementWithUncertainty` dataclass
- Propagate uncertainties through calculations
- Include both absolute uncertainty and relative percentage

```python
# ✅ CORRECT
cd = MeasurementWithUncertainty(
    value=0.654,
    uncertainty=0.018,
    unit='',
    rel_uncertainty_pct=2.8
)

# ❌ WRONG - Naked number
cd = 0.654  # Where are the error bars?
```

#### 3. QC Checks are Non-Negotiable

**NEVER**:
- Skip QC checks to "just get results"
- Silently ignore QC failures
- Let users bypass blocking failures without fixing data

**ALWAYS**:
- Call `run_qc_checks()` before analysis
- Call `assert_qc_passed()` to enforce blocking failures
- Display QC warnings even if passed overall

```python
# ✅ CORRECT
qc_report = run_qc_checks(df, config, sensors)
if not qc_report.passed:
    raise ValueError(f"QC failed: {qc_report.blocking_failures}")

# ❌ WRONG - Bypassing QC
result = analyze_cold_flow_test(df, config, ...)  # Hope it's good data
```

#### 4. No Business Logic in UI

**NEVER**:
- Put calculations in Streamlit pages
- Import Streamlit in core modules
- Mix UI code with business logic

**ALWAYS**:
- Keep `core/` modules UI-agnostic
- Pages only orchestrate and display
- Write unit tests for core modules (can't unit test Streamlit)

```python
# ✅ CORRECT
# pages/2_Single_Test_Analysis.py
result = analyze_cold_flow_test(df, config, ...)  # Core does work
st.write(result)  # Page displays

# ❌ WRONG
# pages/2_Single_Test_Analysis.py
cd = calculate_cd(p_up, p_down, mf, area)  # Business logic in page
st.write(cd)
```

#### 5. Version Tracking

**NEVER**:
- Change calculation algorithm without updating `PROCESSING_VERSION`
- Change database schema without updating `SCHEMA_VERSION` + migration
- Update version without documenting change

**ALWAYS**:
- Increment version when algorithms change
- Write migration functions for schema changes
- Document what changed in comments

### Common Pitfalls to Avoid

#### Pitfall 1: Forgetting to Update Schema Version

```python
# ❌ WRONG
# Added new column to database but forgot to:
# 1. Increment SCHEMA_VERSION
# 2. Write migration function
# 3. Update existing databases

# ✅ CORRECT
SCHEMA_VERSION = 3  # Incremented

def migrate_v2_to_v3(conn):
    """Add p_drop column - 2026-01-16."""
    conn.execute("ALTER TABLE test_results ADD COLUMN avg_p_drop_bar REAL")
```

#### Pitfall 2: Inconsistent Units

**ALWAYS** use these standard units:
- Pressure: `bar` (not psi, Pa, kPa)
- Mass flow: `g/s` (not kg/s, lb/hr)
- Temperature: `K` (not C, F)
- Time: `s` (not ms) - except for window specifications
- Area: `mm²` (not m², in²)

```python
# ✅ CORRECT
p_up_bar = df['IG-PT-01'].mean()  # Already in bar

# ❌ WRONG
p_up_psi = df['IG-PT-01'].mean() * 14.5038  # Don't convert
```

#### Pitfall 3: Hardcoded Values

**NEVER** hardcode sensor names, thresholds, or test-specific values:

```python
# ❌ WRONG
if df['IG-PT-01'].mean() > 10:  # Hardcoded sensor name

# ✅ CORRECT
p_up_sensor = config['sensors']['p_upstream']
if df[p_up_sensor].mean() > config['settings']['p_max_bar']:
```

---

## Troubleshooting

### Common Issues

#### Issue 1: QC Checks Failing

**Symptom**: `ValueError: QC checks failed for test_id`

**Diagnosis**:
```python
qc_report = run_qc_checks(df, config, sensors)
print(qc_report.blocking_failures)  # What failed?
print(qc_report.warnings)           # What warnings?
```

**Solutions**:
- **Flatline**: Sensor disconnected or saturated → Check hardware
- **NaN ratio too high**: Missing data → Interpolate or trim
- **Sensor out of range**: Pressure spike → Check test conditions
- **Sampling rate inconsistent**: Data acquisition issue → Resample

#### Issue 2: Uncertainty Too Large

**Symptom**: `rel_uncertainty_pct > 10%` (unusually high)

**Diagnosis**:
```python
print(measurements['Cd'].uncertainty)  # Absolute value
print(measurements['Cd'].rel_uncertainty_pct)  # Percentage

# Check individual sensor uncertainties
print(config['uncertainties'])
```

**Solutions**:
- **Calibration uncertainty too high**: Recalibrate sensors
- **Measurement near zero**: Relative uncertainty explodes → Use absolute
- **Error propagation amplification**: Check correlation terms

#### Issue 3: Streamlit App Won't Start

**Symptom**: `ModuleNotFoundError` or import errors

**Diagnosis**:
```bash
# Check Python version
python --version  # Need 3.8+

# Check dependencies
pip list | grep streamlit
pip list | grep pandas

# Try importing core modules
python -c "from core.integrated_analysis import analyze_cold_flow_test"
```

**Solutions**:
- Install dependencies: `pip install -r requirements.txt`
- Check virtual environment activated
- Clear Streamlit cache: `streamlit cache clear`

#### Issue 4: Database Corruption

**Symptom**: `sqlite3.DatabaseError` or missing data

**Diagnosis**:
```bash
# Check database integrity
sqlite3 campaigns/campaign_name.db "PRAGMA integrity_check;"

# Check schema version
sqlite3 campaigns/campaign_name.db "PRAGMA user_version;"
```

**Solutions**:
- **Old schema**: Run migrations manually
- **Corrupted file**: Restore from backup
- **Version mismatch**: Export to CSV, recreate database, reimport

#### Issue 5: Steady State Detection Fails

**Symptom**: No steady state found or incorrect window

**Diagnosis**:
```python
# Try different methods
window_cv = detect_steady_state_cv(df, sensor, threshold=0.02)
window_ml = detect_steady_state_ml(df, sensor)
window_deriv = detect_steady_state_derivative(df, sensor)

# Plot to visualize
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time_s'], y=df[sensor], mode='lines'))
fig.add_vrect(x0=window_cv[0]/1000, x1=window_cv[1]/1000, fillcolor="green", opacity=0.2)
fig.show()
```

**Solutions**:
- **CV threshold too strict**: Increase from 0.01 to 0.02 or 0.05
- **Short test**: Reduce `min_duration_s` requirement
- **No steady state**: Test may be purely transient → Manual window

---

## Quick Reference

### Key Files to Know

| File | Purpose | When to Modify |
|------|---------|----------------|
| `core/integrated_analysis.py` | High-level API | Adding new test types |
| `core/uncertainty.py` | Error propagation | New metrics, new sensors |
| `core/qc_checks.py` | Quality control | Tuning thresholds, new checks |
| `core/campaign_manager_v2.py` | SQLite database | Schema changes |
| `core/traceability.py` | SHA-256 hashing | Never (unless fixing bug) |
| `pages/2_Single_Test_Analysis.py` | Main analysis UI | UI improvements |
| `pages/_shared_sidebar.py` | Global context | Adding global settings |
| `tests/test_p0_components.py` | Core tests | After modifying P0 |

### Essential Commands

```bash
# Run app
streamlit run app.py

# Run tests
pytest tests/ -v

# Run specific test file
python tests/test_p0_components.py

# Clear Streamlit cache
streamlit cache clear

# Check database schema
sqlite3 campaigns/campaign.db "SELECT sql FROM sqlite_master WHERE type='table';"

# List campaigns
ls -lh campaigns/*.db
```

### Key Functions

```python
# Analysis
from core.integrated_analysis import analyze_cold_flow_test, analyze_hot_fire_test

# Campaign
from core.campaign_manager_v2 import create_campaign, save_to_campaign, get_campaign_data

# QC
from core.qc_checks import run_qc_checks, assert_qc_passed

# Uncertainty
from core.uncertainty import calculate_cold_flow_uncertainties

# Traceability
from core.traceability import create_full_traceability_record, compute_file_hash

# SPC
from core.spc import create_imr_chart, create_xbar_r_chart

# Reporting
from core.reporting import generate_test_report, generate_campaign_report

# Config
from core.saved_configs import SavedConfigManager
from core.metadata_manager import MetadataManager
```

---

## When in Doubt

1. **Check tests**: `tests/test_p0_components.py` shows correct usage patterns
2. **Check existing pages**: See how others implemented similar features
3. **Preserve integrity**: When unsure, favor stricter QC and more traceability
4. **Ask before changing P0**: Any change to traceability, uncertainty, or QC needs careful review
5. **Document decisions**: Add comments explaining "why" for future developers

---

**Last Updated**: 2026-01-16
**Codebase Version**: 2.3.0
**Maintained by**: AI assistants working with HDA

For questions or issues, refer to recent git commits and test files for examples.
