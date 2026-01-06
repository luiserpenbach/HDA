# Hopper Data Studio

A Streamlit-based web application for analyzing rocket propulsion test data with built-in engineering integrity systems.

## Overview

Hopper Data Studio provides comprehensive analysis tools for cold flow and hot fire testing with emphasis on:
- **Data Traceability** - SHA-256 hashing and audit trails
- **Uncertainty Quantification** - Full error propagation for all metrics
- **Quality Control** - Pre-analysis data validation

## Features

### Core Engineering Integrity (P0)
- **Traceability** - Cryptographic verification and audit trails
- **Uncertainty** - Error propagation for all metrics
- **QC Checks** - Pre-analysis data quality validation
- **Config Validation** - Schema validation for configurations
- **Campaign Manager** - SQLite database with full traceability

### Analysis & Reporting (P1)
- **SPC** - Statistical Process Control, control charts, capability indices
- **Reporting** - HTML reports with full traceability
- **Batch Analysis** - Multi-file parallel processing
- **Export** - CSV, Excel, JSON with metadata

### Advanced Features (P2)
- **Anomaly Detection** - Multi-type detection with sensor health
- **Data Comparison** - Test-to-test and golden reference comparison
- **Saved Configurations** - Reusable testbench configurations

## Installation

```bash
# Clone repository
git clone <repo-url>
cd HDA

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Optional: CoolProp for accurate fluid properties
```bash
pip install CoolProp>=6.4.1
```

## Project Structure

```
HDA/
├── app.py                      # Main Streamlit entry point
├── requirements.txt            # Python dependencies
├── core/                       # Engineering integrity modules
│   ├── traceability.py        # SHA-256 hashing, audit trails
│   ├── uncertainty.py         # Error propagation
│   ├── qc_checks.py           # Quality control validation
│   ├── config_validation.py   # Config schema validation
│   ├── campaign_manager_v2.py # SQLite database with traceability
│   ├── integrated_analysis.py # High-level analysis API
│   ├── spc.py                 # Statistical Process Control
│   ├── reporting.py           # HTML report generation
│   ├── batch_analysis.py      # Multi-file processing
│   ├── export.py              # Data export
│   ├── advanced_anomaly.py    # Anomaly detection
│   ├── comparison.py          # Test comparison
│   └── saved_configs.py       # Configuration management
├── pages/                      # Streamlit pages
│   ├── 1_Single_Test_Analysis.py
│   ├── 2_Campaign_Management.py
│   ├── 3_SPC_Analysis.py
│   ├── 4_Batch_Processing.py
│   ├── 5_Reports_Export.py
│   ├── 6_Anomaly_Detection.py
│   ├── 7_Data_Comparison.py
│   └── 8_Saved_Configurations.py
├── configs/                    # Example configuration files
├── saved_configs/              # Saved testbench configurations
├── campaigns/                  # SQLite campaign databases
└── tests/                      # Test suite
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

### Statistical Process Control

```python
from core.spc import create_imr_chart
from core.campaign_manager_v2 import get_campaign_data

df = get_campaign_data("INJ_Acceptance_Q1")

analysis = create_imr_chart(
    df,
    parameter='avg_cd_CALC',
    usl=0.70,
    lsl=0.60,
)

print(f"In Control: {analysis.n_violations == 0}")
print(f"Cpk: {analysis.capability.cpk:.2f}")
```

### Generate Reports

```python
from core.reporting import generate_test_report, save_report

html = generate_test_report(
    test_id='CF-001',
    test_type='cold_flow',
    measurements=result.measurements,
    traceability=result.traceability,
    qc_report={'passed': True, 'summary': {'passed': 5}, 'checks': []},
)
save_report(html, 'reports/CF-001_report.html')
```

## Configuration

### Active Configuration (Testbench Hardware)

Stored in `saved_configs/`, contains:
- Sensor mappings and channel IDs
- Calibration uncertainties
- Processing settings

Changes only when testbench is modified or recalibrated.

### Test Metadata (Test Article Properties)

Provided per test via `metadata.json` or UI entry:
- Geometry (orifice area, throat area)
- Fluid properties
- Part/serial numbers

```json
{
  "part_number": "INJ-B1-03",
  "serial_number": "SN-2024-045",
  "geometry": {
    "orifice_area_mm2": 3.14159
  },
  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4
  }
}
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test suites
python tests/test_p0_components.py  # Core integrity
python tests/test_p1_components.py  # Analysis & reporting
python tests/test_p2_components.py  # Advanced features
```

## Version

- **Application Version**: 2.3.0
- **Schema Version**: 3 (campaign database)
- **Processing Version**: 2.0.0+integrity

## License

Internal use only - Hopper Propulsion Systems
