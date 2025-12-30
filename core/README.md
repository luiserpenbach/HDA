# Hopper Data Studio - Engineering Integrity Components

## Overview

This package implements the engineering integrity systems for propulsion test data analysis, split into two priority levels:

### P0 Components (Non-negotiable)
1. **Traceability** - Cryptographic verification and audit trails
2. **Uncertainty** - Error propagation for all metrics
3. **QC Checks** - Pre-analysis data quality validation
4. **Config Validation** - Schema validation for configurations
5. **Campaign Manager v2** - Enhanced database with traceability
6. **Integrated Analysis** - High-level analysis API

### P1 Components (High Priority)
1. **SPC** - Statistical Process Control, control charts, capability indices
2. **Reporting** - HTML reports with full traceability
3. **Batch Analysis** - Multi-file processing
4. **Export** - Enhanced data export with metadata

## Installation

Copy the `core/` directory into your Hopper Data Studio project:

```
hopper_studio/
├── core/                    # Engineering integrity modules
│   ├── __init__.py
│   ├── traceability.py      # P0: Data hashing, audit trails
│   ├── uncertainty.py       # P0: Error propagation
│   ├── qc_checks.py         # P0: Pre-analysis validation
│   ├── config_validation.py # P0: Config schema validation
│   ├── campaign_manager_v2.py # P0: Enhanced database
│   ├── integrated_analysis.py # P0: High-level API
│   ├── spc.py               # P1: Statistical Process Control
│   ├── reporting.py         # P1: HTML report generation
│   ├── batch_analysis.py    # P1: Multi-file processing
│   └── export.py            # P1: Enhanced data export
├── tests/
│   ├── test_p0_components.py
│   └── test_p1_components.py
└── pages/                   # Streamlit pages
```

## Quick Start

### Complete Cold Flow Analysis

```python
from core.integrated_analysis import analyze_cold_flow_test
from core.campaign_manager_v2 import save_to_campaign, create_campaign

# Create campaign (once)
create_campaign("INJ_Acceptance_Q1", "cold_flow")

# Analyze a test
result = analyze_cold_flow_test(
    df=df,
    config=config,
    steady_window=(1500, 5000),
    test_id="INJ-CF-001",
    file_path="test_data.csv",
    metadata={'part': 'INJ-V1', 'serial_num': 'SN-001'}
)

# Check results
print(f"QC Passed: {result.passed_qc}")
print(f"Cd: {result.measurements['Cd']}")  # 0.654 ± 0.018 (2.8%)

# Save to campaign
record = result.to_database_record('cold_flow')
save_to_campaign("INJ_Acceptance_Q1", record)
```

### Statistical Process Control (P1)

```python
from core.spc import create_imr_chart, format_spc_summary
from core.campaign_manager_v2 import get_campaign_data

# Load campaign data
df = get_campaign_data("INJ_Acceptance_Q1")

# Create control chart
analysis = create_imr_chart(
    df,
    parameter='avg_cd_CALC',
    usl=0.70,  # Upper spec limit
    lsl=0.60,  # Lower spec limit
)

# Check results
print(f"In Control: {analysis.n_violations == 0}")
print(f"Cpk: {analysis.capability.cpk:.2f}")
print(format_spc_summary(analysis))

# Get out-of-control points
for point in analysis.get_out_of_control_points():
    print(f"  {point.test_id}: {[v.value for v in point.violations]}")
```

### Generate Reports (P1)

```python
from core.reporting import generate_test_report, generate_campaign_report, save_report
from core.spc import analyze_campaign_spc

# Single test report
html = generate_test_report(
    test_id='CF-001',
    test_type='cold_flow',
    measurements=result.measurements,
    traceability=result.traceability,
    qc_report={'passed': True, 'summary': {'passed': 5}, 'checks': []},
)
save_report(html, 'reports/CF-001_report.html')

# Campaign report with SPC
spc_results = analyze_campaign_spc(df, ['avg_cd_CALC', 'avg_mf_g_s'])
html = generate_campaign_report(
    campaign_name='INJ_Acceptance_Q1',
    df=df,
    parameters=['avg_cd_CALC', 'avg_mf_g_s'],
    spc_analyses=spc_results,
)
save_report(html, 'reports/campaign_summary.html')
```

### Batch Analysis (P1)

```python
from core.batch_analysis import (
    discover_test_files,
    run_batch_analysis,
    save_batch_to_campaign,
    export_batch_results,
    load_csv_with_timestamp,
)
from pathlib import Path

# Discover test files
files = discover_test_files('/path/to/tests', '*.csv')
print(f"Found {len(files)} test files")

# Run batch analysis
def my_analyze_func(df, config, test_id, file_path):
    # Your analysis logic here
    return analyze_cold_flow_test(...)

report = run_batch_analysis(
    files=files,
    config=config,
    load_func=load_csv_with_timestamp,
    analyze_func=my_analyze_func,
    max_workers=4,  # Parallel processing
    progress_callback=lambda i, n, f: print(f"Processing {i+1}/{n}")
)

print(report.summary())

# Save to campaign
saved, skipped = save_batch_to_campaign(report, "INJ_Acceptance_Q1")

# Export results
export_batch_results(report, 'batch_results.csv', format='csv')
```

### Export Data (P1)

```python
from core.export import (
    export_campaign_excel,
    export_campaign_json,
    export_for_qualification,
)

# Excel with multiple sheets (Data, Metadata, QC, SPC)
export_campaign_excel(
    df=df,
    output_path='campaign_data.xlsx',
    campaign_info={'name': 'INJ_Acceptance_Q1', 'type': 'cold_flow'},
    spc_summary=spc_results,
)

# Qualification-ready package
outputs = export_for_qualification(
    df=df,
    campaign_info=campaign_info,
    output_dir='qualification_docs/',
)
# Creates: summary CSV, full Excel, traceability report, JSON archive
```

## API Reference

### P0: Traceability

```python
from core.traceability import (
    compute_file_hash,           # SHA-256 hash of file
    compute_dataframe_hash,      # Hash of DataFrame contents
    create_full_traceability_record,  # Complete audit record
)

# Create traceability record
record = create_full_traceability_record(
    raw_data_path="test.csv",
    df=df,
    config=config,
    steady_window=(1500, 5000),
)
# Returns dict with hashes, analyst info, timestamps, processing version
```

### P0: Uncertainty

```python
from core.uncertainty import (
    MeasurementWithUncertainty,
    calculate_cold_flow_uncertainties,
    calculate_hot_fire_uncertainties,
)

# Full uncertainty propagation
measurements = calculate_cold_flow_uncertainties(
    p_up=25.0,
    p_down=24.5,
    temp=293.15,
    mass_flow=12.5,
    area=1e-6,
    config=config,  # Contains sensor and geometry uncertainties
)

cd = measurements['Cd']
print(f"Cd = {cd.value:.4f} ± {cd.uncertainty:.4f} ({cd.relative_uncertainty_percent:.1f}%)")
```

### P0: QC Checks

```python
from core.qc_checks import run_qc_checks, assert_qc_passed

# Run all QC checks
report = run_qc_checks(df, config)

print(f"Passed: {report.passed}")
print(f"Summary: {report.summary}")

for check in report.checks:
    print(f"  {check.name}: {check.status.name} - {check.message}")

# Raise exception if QC fails
assert_qc_passed(report, strict=True)
```

### P1: SPC

```python
from core.spc import (
    create_imr_chart,        # Individual-Moving Range chart
    calculate_capability,     # Cp, Cpk, Pp, Ppk
    check_western_electric_rules,  # Out-of-control detection
    detect_trend,            # Trend analysis
)

# Control chart analysis
analysis = create_imr_chart(df, 'avg_cd_CALC', usl=0.70, lsl=0.60)

# Results include:
# - analysis.limits (UCL, LCL, center line)
# - analysis.points (with violations flagged)
# - analysis.capability (Cpk, Ppk, etc.)
# - analysis.has_trend, trend_direction
```

## Testing

Run the test suite:

```bash
# P0 tests
python tests/test_p0_components.py

# P1 tests
python tests/test_p1_components.py
```

## Version

- Core Version: 2.0.0
- Schema Version: 3 (campaign database)
- Processing Version: 2.0.0+integrity

## License

Internal use only - Hopper Propulsion Systems
