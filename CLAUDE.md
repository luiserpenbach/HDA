# CLAUDE.md - AI Assistant Guide for Hopper Data Studio

## Repository Overview

**Hopper Data Studio** (HDA) is a Streamlit-based web application for analyzing rocket propulsion test data with built-in engineering integrity systems. The platform provides comprehensive analysis tools for cold flow and hot fire testing with emphasis on data traceability, uncertainty quantification, and quality control.

**Core Purpose**: Ensure that all propulsion test data analysis is traceable, uncertainties are properly quantified, and quality control is rigorous.

**Tech Stack**:
- Python 3.x (primary language)
- Streamlit (web UI framework)
- NumPy, Pandas, SciPy (data processing)
- Plotly (visualization)
- SQLite (campaign data storage)

**Version**: 2.0.0 (Core Version with Engineering Integrity)

---

## Directory Structure

```
/home/user/HDA/
├── app.py                      # Main Streamlit entry point (home page)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── core/                       # Engineering integrity modules
│   ├── __init__.py            # Core module exports
│   ├── README.md              # Core module documentation
│   │
│   ├── # Core Engineering Integrity
│   ├── traceability.py        # SHA-256 hashing, audit trails, data integrity
│   ├── uncertainty.py         # Error propagation, measurement uncertainty
│   ├── qc_checks.py           # Pre-analysis quality control validation
│   ├── config_validation.py   # Pydantic-based config schema validation
│   ├── campaign_manager_v2.py # SQLite database with full traceability
│   ├── integrated_analysis.py # High-level analysis API
│   │
│   ├── # Analysis & Reporting
│   ├── spc.py                 # Statistical Process Control (SPC)
│   ├── reporting.py           # HTML report generation
│   ├── batch_analysis.py      # Multi-file batch processing
│   ├── export.py              # Data export (CSV, Excel, JSON)
│   │
│   ├── # Advanced Features
│   ├── advanced_anomaly.py    # Multi-type anomaly detection
│   ├── comparison.py          # Test-to-test and golden reference comparison
│   ├── templates.py           # Configuration template management
│   │
│   └── # Supporting Modules
│       ├── fluid_properties.py # Thermodynamic fluid properties
│       └── test_metadata.py    # Test metadata structures
│
├── pages/                      # Streamlit multi-page app pages
│   ├── 1_Single_Test_Analysis.py  # Individual test analysis
│   ├── 2_Campaign_Management.py   # Campaign tracking and viewing
│   ├── 3_SPC_Analysis.py          # Statistical Process Control
│   ├── 4_Batch_Processing.py      # Multi-file batch analysis
│   ├── 5_Reports_Export.py        # Report generation and data export
│   ├── 6_Anomaly_Detection.py     # Advanced anomaly detection
│   ├── 7_Data_Comparison.py       # Test comparison tools
│   └── 8_Config_Templates.py      # Template management UI
│
├── data_lib/                   # Legacy/utility libraries
│   ├── campaign_manager.py    # Original campaign manager (v1)
│   ├── cf_analytics.py        # Cold flow analytics helpers
│   ├── hf_analytics.py        # Hot fire analytics helpers
│   ├── propulsion_physics.py  # Physics calculations
│   ├── sensor_data_tools.py   # Sensor data utilities
│   ├── spc_analysis.py        # Legacy SPC analysis
│   ├── report_generator.py    # Legacy report generator
│   ├── config_loader.py       # Configuration file loader
│   ├── db_manager.py          # Database utilities
│   └── transient_analysis.py  # Transient detection
│
├── configs/                    # Test configuration files (JSON)
│   ├── IGN_C1_HotFire.json
│   └── LCSC_B1_SingSide_ColdFlow.json
│
├── config_templates/           # Reusable configuration templates
│   └── lcsc_b1_injectors___cold_flow_standard.json
│
├── campaigns/                  # SQLite databases for test campaigns
│   ├── INJ-CF-C1.db
│   ├── IGN-HF-C1-DEMO.db
│   └── INJ-CF-C1-Var2_Characterization.db
│
└── tests/                      # Test suite
    ├── test_p0_components.py  # Core integrity tests
    ├── test_p1_components.py  # Analysis & reporting tests
    └── test_p2_components.py  # Advanced features tests
```

---

## Architecture & Design Principles

### Core Engineering Integrity

The platform is built around ensuring data quality and traceability:

- **Traceability**: Every data point must be traceable to its source via SHA-256 hashing
- **Uncertainty**: All measurements must include uncertainty quantification
- **QC Checks**: Data must pass quality checks before analysis
- **Config Validation**: All configurations must be schema-validated
- **Campaign Manager**: Database storage with full audit trails
- **Integrated Analysis**: Unified API that enforces integrity requirements

### Key Features

- **Statistical Process Control**: Monitor process stability with control charts and capability indices
- **Reporting**: Professional HTML reports with full traceability
- **Batch Analysis**: Parallel processing of multiple test files
- **Data Export**: Rich export capabilities (CSV, Excel, JSON) with metadata
- **Anomaly Detection**: Multi-algorithm detection for data quality
- **Data Comparison**: Test-to-test and golden reference comparisons
- **Configuration Templates**: Reusable template system with inheritance

### Key Design Patterns

1. **Dataclass-based models**: Use `@dataclass` for structured data (e.g., `AnalysisResult`, `QCReport`, `SPCAnalysis`)
2. **Pydantic validation**: Configuration validation uses Pydantic for schema enforcement
3. **Uncertainty propagation**: All calculated metrics include uncertainty via error propagation
4. **Cryptographic hashing**: SHA-256 hashing for data integrity verification
5. **SQLite with schema versioning**: Campaign databases use versioned schemas for migration support

### Data Flow

```
Raw Data (CSV) → QC Checks → Analysis → Uncertainty Calculation → Database Storage
                                  ↓
                          Traceability Record (SHA-256 hashes)
                                  ↓
                          Campaign Database (SQLite)
                                  ↓
                          SPC Analysis / Reporting / Export
```

---

## Development Workflows

### Making Code Changes

1. **Always read before editing**: Never modify files without reading them first
2. **Maintain engineering integrity**: When modifying analysis code, ensure core components (traceability, uncertainty, QC) remain intact
3. **Follow existing patterns**: Match the coding style and patterns already in use
4. **Update tests**: If changing core logic, update relevant tests in `tests/`
5. **Preserve backward compatibility**: Database schema changes require migration logic

### Adding New Features

1. **Use existing infrastructure**: Leverage `core/integrated_analysis.py` for analysis workflows
2. **Add uncertainty**: All new metrics must include uncertainty calculation
3. **Add QC checks**: If analyzing new data types, add appropriate QC checks
4. **Update exports**: If adding new fields, update `core/export.py` to include them
5. **Update documentation**: Keep this file and docstrings current

### Working with Streamlit Pages

1. **Page numbering**: Pages are numbered (1_, 2_, etc.) for ordering in sidebar
2. **Import from core**: Pages should import from `core/` module, not `data_lib/`
3. **Session state**: Use `st.session_state` for cross-page data sharing
4. **Error handling**: Always wrap analysis calls in try/except blocks for user-friendly errors
5. **Styling**: Use the CSS classes defined in `app.py` for consistent styling

### Configuration Files

**Structure**:
```json
{
  "sensor_mapping": {
    "timestamp": "Time",
    "pressure_upstream": "P_upstream",
    "pressure_downstream": "P_downstream",
    "temperature": "T_amb",
    "mass_flow": "mf_g_s"
  },
  "sensor_uncertainties": {
    "pressure_upstream": {"value": 0.05, "unit": "psi", "type": "absolute"},
    "temperature": {"value": 1.0, "unit": "K", "type": "absolute"}
  },
  "geometry": {
    "orifice_area_mm2": 3.14159,
    "throat_area_mm2": 12.566
  },
  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4
  }
}
```

**Validation**: All configs are validated via `core/config_validation.py` before use.

---

## Key Conventions

### Naming Conventions

**Files**:
- Snake_case for Python files: `campaign_manager.py`
- Streamlit pages: `N_Page_Name.py` (numbered for ordering)

**Functions**:
- Snake_case: `calculate_cold_flow_uncertainties()`
- Verb-based names: `run_qc_checks()`, `analyze_campaign_spc()`

**Classes**:
- PascalCase: `AnalysisResult`, `QCReport`, `SPCAnalysis`
- Use `@dataclass` for data structures

**Variables**:
- Snake_case: `test_id`, `steady_window`, `campaign_name`
- Descriptive names over abbreviations (except common physics: `Cd`, `Isp`, `c_star`)

### Code Style

**Imports**:
```python
# Standard library
import sys
import json
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import streamlit as st

# Local - prefer core over data_lib
from core.integrated_analysis import analyze_cold_flow_test
from core.qc_checks import run_qc_checks
```

**Docstrings**:
```python
def analyze_cold_flow_test(
    df: pd.DataFrame,
    config: dict,
    steady_window: tuple[int, int],
    test_id: str,
    file_path: str,
    metadata: dict | None = None,
) -> AnalysisResult:
    """
    Analyze cold flow test with full engineering integrity.

    Args:
        df: Test data DataFrame with timestamp column
        config: Validated configuration dict
        steady_window: (start_idx, end_idx) for steady-state region
        test_id: Unique test identifier
        file_path: Path to original data file
        metadata: Optional metadata (part number, serial, etc.)

    Returns:
        AnalysisResult with measurements, uncertainties, QC, traceability
    """
```

**Error Handling**:
```python
# Explicit validation
if not isinstance(steady_window, tuple) or len(steady_window) != 2:
    raise ValueError("steady_window must be (start_idx, end_idx)")

# User-friendly errors in Streamlit
try:
    result = analyze_cold_flow_test(...)
except Exception as e:
    st.error(f"Analysis failed: {str(e)}")
    st.stop()
```

### Testing Conventions

**Test file structure**:
```python
"""
Test Suite for [Component Name]
================================
Brief description.

Run with: python -m pytest tests/test_*.py -v
Or:       python tests/test_*.py
"""

def test_feature_name():
    """Test specific feature with descriptive name."""
    # Arrange
    test_data = create_test_data()

    # Act
    result = function_under_test(test_data)

    # Assert
    assert result.passed
    assert abs(result.value - expected) < tolerance
```

**Run tests**:
```bash
# All tests
python -m pytest tests/ -v

# Individual test suites
python tests/test_p0_components.py  # Core integrity tests
python tests/test_p1_components.py  # Analysis & reporting tests
python tests/test_p2_components.py  # Advanced features tests
```

---

## Git Workflow

### Branch Naming

**Pattern**: `claude/<feature-description>-<session-id>`

Examples:
- `claude/add-uncertainty-propagation-ABC123`
- `claude/fix-spc-control-limits-XYZ789`
- `claude/implement-batch-analysis-DEF456`

**Important**:
- Branches MUST start with `claude/`
- Branches MUST end with the session ID
- Push failures with 403 errors indicate branch naming issues

### Commit Messages

**Format**:
```
<type>: <short summary>

<optional detailed description>
```

**Types**:
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code restructuring without behavior change
- `docs:` - Documentation updates
- `test:` - Test additions/modifications
- `style:` - Formatting, no code change

**Examples**:
```
feat: Add batch processing with parallel execution

Implemented batch_analysis.py with ThreadPoolExecutor for parallel
processing of multiple test files. Includes progress callbacks and
error handling for individual file failures.

fix: Correct Cd uncertainty calculation for compressible flow

Previous calculation didn't account for temperature uncertainty
propagation through density calculation.

docs: Update CLAUDE.md with git workflow conventions
```

### Push Workflow

```bash
# Always use -u flag for new branches
git push -u origin claude/feature-name-SESSION_ID

# On network failures, the system automatically retries with exponential backoff
# Manual retry sequence if needed:
git push -u origin branch-name  # Wait 2s if failed
git push -u origin branch-name  # Wait 4s if failed
git push -u origin branch-name  # Wait 8s if failed
git push -u origin branch-name  # Final attempt
```

---

## Database Schema

### Campaign Database (Schema Version 3)

**campaigns/[campaign_name].db**

**Tables**:

1. **metadata** - Campaign information
   - `id`, `campaign_name`, `test_type`, `created_at`, `created_by`, `description`

2. **tests** - Test results
   - Measurement columns: `test_id`, `test_datetime`, `avg_*`, `std_*`, `unc_*`
   - Traceability: `file_hash`, `data_hash`, `config_hash`
   - Metadata: `analyst`, `processing_version`
   - QC: `qc_passed`, `qc_summary`

3. **traceability** - Full audit trail
   - `test_id`, `raw_data_hash`, `processed_data_hash`, `config_hash`
   - `analyst`, `timestamp`, `processing_version`, `steady_window`

4. **qc_results** - Detailed QC check results
   - `test_id`, `check_name`, `status`, `message`, `value`, `threshold`

### Migration

When schema changes are needed:
```python
from core.campaign_manager_v2 import migrate_database

migrate_database("campaigns/old_campaign.db")
```

---

## Common Tasks for AI Assistants

### Task 1: Add a New Measurement to Analysis

1. **Update uncertainty calculation** in `core/uncertainty.py`:
   ```python
   def calculate_new_metric_uncertainty(inputs, config):
       # Use error propagation
       ...
       return MeasurementWithUncertainty(value, uncertainty, unit)
   ```

2. **Update integrated analysis** in `core/integrated_analysis.py`:
   ```python
   measurements['new_metric'] = calculate_new_metric_uncertainty(...)
   ```

3. **Update database schema** if needed (requires migration logic)

4. **Update export** in `core/export.py` to include new field

5. **Add tests** in `tests/test_p0_components.py`

### Task 2: Add a New QC Check

1. **Add check function** in `core/qc_checks.py`:
   ```python
   def check_new_condition(df: pd.DataFrame, config: dict) -> QCCheckResult:
       """Check for specific condition."""
       # Implementation
       return QCCheckResult(
           name="New Condition Check",
           status=QCStatus.PASSED,
           message="Condition met",
           value=measured_value,
           threshold=threshold_value
       )
   ```

2. **Add to run_qc_checks()** in same file:
   ```python
   checks.append(check_new_condition(df, config))
   ```

3. **Add test** in `tests/test_p0_components.py`

### Task 3: Create a New Streamlit Page

1. **Create file** `pages/N_Page_Name.py` (choose appropriate number N)

2. **Use standard structure**:
   ```python
   import streamlit as st
   from core.module import function

   st.set_page_config(page_title="Page Name", layout="wide")
   st.title("Page Name")

   # Page content
   ```

3. **Import from core**, not data_lib

4. **Handle errors gracefully** with try/except and `st.error()`

### Task 4: Fix a Bug

1. **Reproduce the issue** - Understand the exact failure mode

2. **Read relevant code** - Don't guess at implementation

3. **Check tests** - Are there tests that should have caught this?

4. **Fix with minimal changes** - Don't refactor unrelated code

5. **Add regression test** - Prevent this bug from returning

6. **Update documentation** if the bug revealed unclear behavior

---

## Performance Considerations

### Large Datasets

- Use `pandas` vectorized operations, avoid Python loops
- For batch processing, use `max_workers` parameter in `batch_analysis.py`
- Consider chunking very large files (>1GB)

### Streamlit Optimization

- Use `@st.cache_data` for expensive computations:
  ```python
  @st.cache_data
  def load_and_process_data(file_path):
      return pd.read_csv(file_path)
  ```

- Avoid recomputing on every widget interaction
- Use `st.session_state` to persist data across reruns

### Database Performance

- Campaign databases use indexes on `test_id` and `test_datetime`
- For large campaigns (>1000 tests), consider periodic cleanup of old tests
- Use `PRAGMA journal_mode=WAL` for concurrent read/write (already enabled in v2)

---

## Troubleshooting

### Common Issues

**Issue**: "Config validation failed"
- **Cause**: JSON config doesn't match schema
- **Fix**: Use `core/config_validation.py` to see exact error
- **Prevention**: Use templates from `config_templates/`

**Issue**: "QC checks failed"
- **Cause**: Data doesn't meet quality thresholds
- **Fix**: Examine `qc_report.checks` to see which check failed
- **Prevention**: Verify data collection setup and sensor calibration

**Issue**: "Test ID already exists in campaign"
- **Cause**: Attempting to save duplicate test_id
- **Fix**: Use unique test IDs or update existing record
- **Prevention**: Implement test ID generation scheme

**Issue**: "Import error from core module"
- **Cause**: Importing removed/renamed function
- **Fix**: Check `core/__init__.py` for available exports
- **Prevention**: Refer to this CLAUDE.md or `core/README.md`

### Debug Mode

Enable detailed logging in Streamlit:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Security & Data Integrity

### Traceability Requirements

**Every analysis must include**:
- SHA-256 hash of raw data file
- SHA-256 hash of processed DataFrame
- SHA-256 hash of configuration
- Analyst identifier
- Processing timestamp
- Processing software version

### Data Validation

**Before analysis**:
1. Config validation (schema check)
2. QC checks (data quality)
3. Column existence verification

**After analysis**:
1. Uncertainty validation (no NaN/Inf)
2. Physical bounds checking (e.g., 0 < Cd < 1)
3. Traceability record creation

### Sensitive Data

- **DO NOT** commit actual test data to git
- **DO NOT** commit campaign databases with proprietary data
- **DO** use `.gitignore` to exclude data files
- **DO** use synthetic data for examples and tests

---

## Dependencies & Installation

### Core Dependencies

```
numpy>=1.20.0          # Numerical computing
pandas>=1.3.0          # Data manipulation
scipy>=1.7.0           # Scientific computing
matplotlib>=3.4.0      # Plotting
plotly>=5.0.0          # Interactive plots
streamlit>=1.20.0      # Web UI framework
openpyxl>=3.0.0        # Excel export
```

### Optional Dependencies

```
CoolProp>=6.4.1        # High-accuracy fluid properties (recommended)
```

### Installation

```bash
# Clone repository
git clone <repo-url>
cd HDA

# Install dependencies
pip install -r requirements.txt

# Optional: Install CoolProp for accurate fluid properties
pip install CoolProp>=6.4.1

# Run application
streamlit run app.py
```

---

## Best Practices for AI Assistants

### Before Making Changes

1. ✅ **READ the relevant files first** - Never modify code you haven't read
2. ✅ **Understand the architecture** - Review the core engineering integrity components
3. ✅ **Check for existing patterns** - Match the style already in use
4. ✅ **Review tests** - Understand expected behavior from tests

### When Writing Code

1. ✅ **Maintain engineering integrity** - Never break traceability, uncertainty, or QC
2. ✅ **Use dataclasses** - For structured data, use `@dataclass`
3. ✅ **Add docstrings** - Document function purpose, args, returns
4. ✅ **Handle errors gracefully** - Provide helpful error messages
5. ✅ **Include type hints** - Use modern Python type annotations
6. ✅ **Propagate uncertainty** - All new metrics need uncertainty calculation

### When Testing

1. ✅ **Run existing tests** - Ensure you didn't break anything
2. ✅ **Add new tests** - Cover your new functionality
3. ✅ **Test edge cases** - Empty data, invalid configs, etc.
4. ✅ **Test in Streamlit** - Run the actual app, not just unit tests

### When Committing

1. ✅ **Use descriptive commit messages** - Follow the convention above
2. ✅ **Keep commits focused** - One logical change per commit
3. ✅ **Don't commit data files** - Check `.gitignore`
4. ✅ **Use correct branch naming** - `claude/<description>-<session_id>`

### When Stuck

1. ✅ **Read the documentation** - `README.md`, `core/README.md`, this file
2. ✅ **Examine similar code** - Find analogous functionality
3. ✅ **Check test files** - Tests show expected usage patterns
4. ✅ **Ask clarifying questions** - Better to ask than to guess wrong

---

## Module Reference Quick Guide

### Core Modules

```python
# Engineering Integrity (Core)
from core.integrated_analysis import analyze_cold_flow_test, analyze_hot_fire_test
from core.qc_checks import run_qc_checks, assert_qc_passed
from core.traceability import create_full_traceability_record
from core.uncertainty import calculate_cold_flow_uncertainties
from core.config_validation import validate_config
from core.campaign_manager_v2 import save_to_campaign, get_campaign_data

# Analysis & Reporting
from core.spc import create_imr_chart, analyze_campaign_spc
from core.reporting import generate_test_report, generate_campaign_report
from core.batch_analysis import run_batch_analysis, batch_cold_flow_analysis
from core.export import export_campaign_excel, export_for_qualification

# Advanced Features
from core.advanced_anomaly import run_anomaly_detection
from core.comparison import compare_tests, compare_to_golden
from core.templates import TemplateManager, create_config_from_template
```

### Legacy Modules (Avoid in New Code)

```python
# data_lib modules are legacy - use core/ equivalents instead
# Only use data_lib for:
# - Maintaining existing code that hasn't been migrated
# - Utility functions not yet available in core/
```

---

## Version History

- **v2.0.0** - Engineering Integrity System (current)
  - Core engineering integrity components (traceability, uncertainty, QC)
  - Full traceability with SHA-256 hashing
  - Comprehensive uncertainty propagation
  - Schema version 3 database
  - Advanced analysis features (SPC, reporting, batch processing)

- **v1.x** - Initial implementation
  - Basic analysis capabilities
  - Original data_lib modules
  - Campaign manager v1

---

## Contributing Guidelines

When contributing to this repository:

1. **Follow the conventions** documented in this file
2. **Maintain backward compatibility** unless explicitly breaking changes
3. **Update tests** for any changed functionality
4. **Update documentation** (this file, docstrings, README.md)
5. **Use type hints** for all new functions
6. **Run tests** before pushing (`python -m pytest tests/ -v`)
7. **Test in Streamlit** to ensure UI works correctly

---

## Resources

- **Main README**: `/home/user/HDA/README.md` - User-facing documentation
- **Core README**: `/home/user/HDA/core/README.md` - Core module API reference
- **Config Examples**: `/home/user/HDA/configs/` - Sample configuration files
- **Templates**: `/home/user/HDA/config_templates/` - Reusable templates

---

## Contact & Support

For questions about this codebase:
1. Review this CLAUDE.md file
2. Check the README files in root and core/
3. Examine test files for usage examples
4. Review existing code for patterns

---

**Last Updated**: 2025-12-30
**Core Version**: 2.0.0
**Schema Version**: 3
**Processing Version**: 2.0.0+integrity
