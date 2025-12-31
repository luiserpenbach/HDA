# Test Metadata Guide - Hopper Data Studio v2.3.0

## Overview

Starting in v2.3.0, Hopper Data Studio separates **testbench hardware configuration** from **test article properties**:

- **Active Configuration**: Testbench hardware (sensors, uncertainties, channel mappings) - load once
- **Test Metadata**: Per-test properties (geometry, fluid, part numbers) - provide for each test

This guide explains how to create and use test metadata files.

---

## Quick Start

### Basic Workflow

1. **Select Active Configuration** (testbench hardware) - one time setup
2. **Provide Test Metadata** for each test:
   - **Option A**: Create `metadata.json` in your test data folder → auto-loads
   - **Option B**: Enter metadata via UI

### Example Directory Structure

```
test_data/
├── TEST-001/
│   ├── test_data.csv          # Your test data
│   └── metadata.json          # Auto-loaded metadata
├── TEST-002/
│   ├── test_data.csv
│   └── metadata.json
└── TEST-003/
    ├── test_data.csv
    └── metadata.json
```

---

## Creating metadata.json

### Cold Flow Example

```json
{
  "part_number": "INJ-B1-03",
  "serial_number": "SN-2024-045",
  "test_datetime": "2024-03-15T14:30:00",
  "analyst": "J. Smith",

  "geometry": {
    "orifice_area_mm2": 3.14159,
    "orifice_diameter_mm": 2.0,
    "upstream_diameter_mm": 6.35,
    "downstream_diameter_mm": 6.35
  },

  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4,
    "molecular_weight": 28.014
  },

  "test_conditions": {
    "ambient_temp_K": 293.15,
    "ambient_pressure_kPa": 101.325,
    "expected_pressure_psi": 250.0
  },

  "notes": "Baseline characterization test for injector element B1-03"
}
```

### Hot Fire Example

```json
{
  "part_number": "IGN-C1-12",
  "serial_number": "SN-2024-089",
  "test_datetime": "2024-03-20T10:15:00",
  "analyst": "M. Johnson",

  "geometry": {
    "throat_area_mm2": 12.566,
    "throat_diameter_mm": 4.0,
    "expansion_ratio": 3.5,
    "chamber_volume_cc": 150.0
  },

  "oxidizer_fluid": {
    "name": "oxygen",
    "gamma": 1.4,
    "molecular_weight": 32.0
  },

  "fuel_fluid": {
    "name": "kerosene",
    "gamma": 1.25,
    "molecular_weight": 170.0
  },

  "test_conditions": {
    "target_of_ratio": 2.5,
    "target_chamber_pressure_psi": 300.0,
    "burn_duration_sec": 5.0
  },

  "notes": "Igniter characterization at nominal O/F ratio"
}
```

### Minimal Example (Analysis Only)

For quick analysis without campaign tracking:

```json
{
  "geometry": {
    "orifice_area_mm2": 3.14159
  },
  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4
  }
}
```

---

## Field Reference

### Identification Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `part_number` | string | For campaign | Part/design identifier (e.g., "INJ-B1-03") |
| `serial_number` | string | Optional | Unique serial number (e.g., "SN-2024-045") |
| `test_id` | string | Optional | Test identifier (auto-generated if not provided) |
| `test_datetime` | ISO 8601 | Optional | Test date/time (e.g., "2024-03-15T14:30:00") |
| `analyst` | string | Optional | Engineer who conducted test |

### Geometry (Cold Flow)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `orifice_area_mm2` | number | Yes | Orifice flow area in mm² |
| `orifice_diameter_mm` | number | Optional | Orifice diameter (if circular) |
| `upstream_diameter_mm` | number | Optional | Upstream pipe diameter |
| `downstream_diameter_mm` | number | Optional | Downstream pipe diameter |
| `length_mm` | number | Optional | Orifice length/thickness |
| `edge_type` | string | Optional | "sharp" or "rounded" |

### Geometry (Hot Fire)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `throat_area_mm2` | number | Yes | Nozzle throat area in mm² |
| `throat_diameter_mm` | number | Optional | Throat diameter |
| `expansion_ratio` | number | Optional | Exit area / throat area |
| `chamber_volume_cc` | number | Optional | Combustion chamber volume |
| `contraction_ratio` | number | Optional | Injector area / throat area |
| `nozzle_type` | string | Optional | "conical", "bell", etc. |

### Fluid Properties (Cold Flow)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Fluid name (e.g., "nitrogen", "water") |
| `gamma` | number | Yes* | Specific heat ratio (for gases) |
| `molecular_weight` | number | Optional | Molecular weight (g/mol) |
| `temperature_K` | number | Optional | Reference temperature |

*Required for compressible flow (gases)

### Fluid Properties (Hot Fire)

For hot fire, use separate `oxidizer_fluid` and `fuel_fluid` objects with same structure as above.

### Test Conditions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ambient_temp_K` | number | Optional | Ambient temperature |
| `ambient_pressure_kPa` | number | Optional | Ambient pressure |
| `expected_pressure_psi` | number | Optional | Target operating pressure |
| `target_of_ratio` | number | Optional | Target O/F ratio (hot fire) |
| `burn_duration_sec` | number | Optional | Expected burn time (hot fire) |

### Additional Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `notes` | string | Optional | Test notes, observations, comments |
| `configuration_variant` | string | Optional | Hardware configuration variant |
| `test_objectives` | array | Optional | List of test objectives |
| `pre_test_checks` | object | Optional | Pre-test verification results |

---

## When is Metadata Required?

### Analysis Only (Optional)

For single-test analysis or reporting:
- Minimum: `geometry` and `fluid`
- Can be entered via UI if no metadata.json file
- Other fields optional

### Campaign Entry (Required)

For saving to campaign database:
- **Required**: `part_number` (or `test_id`)
- **Required**: `geometry`
- **Required**: `fluid`
- **Recommended**: `serial_number`, `test_datetime`, `analyst`

The system will check `metadata.is_complete_for_campaign()` before allowing campaign save.

---

## How Metadata is Loaded

### Auto-Loading from File

1. Upload test CSV file from folder (e.g., `TEST-001/test_data.csv`)
2. System automatically looks for `TEST-001/metadata.json`
3. If found → metadata auto-loads and populates UI
4. If not found → UI entry fields are shown

### UI Entry

If no `metadata.json` file is found:
1. Metadata section appears in UI
2. Enter geometry (required for analysis)
3. Enter fluid properties (required for analysis)
4. Optionally enter identification fields
5. Click "Analyze" (for analysis) or "Save to Campaign" (requires complete metadata)

### Merging File + UI

If both file and UI entries exist:
- UI entries **override** file values (by default)
- Allows quick corrections without editing file
- Original file preserved

---

## Creating Metadata Templates

### Using the Script

Generate a template for your test type:

```python
from core.metadata_manager import create_metadata_template, save_metadata_template

# Create cold flow template
template = create_metadata_template(test_type="cold_flow", include_examples=True)
save_metadata_template(template, "my_test_metadata.json")
```

Or use the UI (Page 1 - Single Test Analysis):
1. Click "Download Metadata Template"
2. Select test type (cold flow / hot fire)
3. Fill in template
4. Save as `metadata.json` in test folder

### Copying from Previous Test

If testing similar hardware:
```bash
# Copy metadata from previous test
cp TEST-001/metadata.json TEST-002/metadata.json

# Edit only what changed (part number, serial, etc.)
nano TEST-002/metadata.json
```

---

## Migration from v2.0-v2.2

### Automatic Migration

Old configs (v2.0-v2.2) that contained geometry and fluid are automatically split:
1. **Active Configuration**: Sensors, uncertainties → `saved_configs/`
2. **Test Metadata**: Geometry, fluid → shown as example

**Important**: Old configs still work via auto-migration. No action required.

### Using the Migration Script

To permanently migrate old configs:

```bash
# Preview what will be migrated
python scripts/migrate_configs_v2.3.py --dry-run

# Execute migration
python scripts/migrate_configs_v2.3.py

# Custom directories
python scripts/migrate_configs_v2.3.py \
  --config-dir old_configs/ \
  --saved-configs-dir new_configs/ \
  --metadata-dir metadata_examples/
```

**Output**:
- `saved_configs/[name]_active_config.json` - Testbench hardware
- `example_metadata/[name]_metadata_example.json` - Example metadata

**Next Steps**:
1. Review migrated active configs
2. Copy metadata examples to test folders
3. Customize for each specific test

---

## Best Practices

### Organizing Test Data

```
campaign_data/
├── injector_b1/
│   ├── TEST-001-baseline/
│   │   ├── data.csv
│   │   └── metadata.json
│   ├── TEST-002-high_pressure/
│   │   ├── data.csv
│   │   └── metadata.json
│   └── TEST-003-low_pressure/
│       ├── data.csv
│       └── metadata.json
└── injector_b2/
    └── ...
```

### Metadata Versioning

Track changes to test articles:

```json
{
  "part_number": "INJ-B1-03",
  "serial_number": "SN-2024-045",
  "configuration_variant": "v2",
  "geometry": {
    "orifice_area_mm2": 3.14159,
    "modification_notes": "Increased diameter from 1.8mm to 2.0mm"
  },
  "notes": "Modified orifice based on TEST-002 results"
}
```

### Reusing Metadata

For test series with same hardware:

```bash
# Base metadata
cp base_metadata.json TEST-001/metadata.json
cp base_metadata.json TEST-002/metadata.json
cp base_metadata.json TEST-003/metadata.json

# Update only test-specific fields (test_datetime, notes, etc.)
```

### Documentation

Include metadata in qualification packages:

```bash
# Export campaign with metadata
# (Page 5 - Reports & Export)
# Check "Include Metadata in Export"
# Download campaign_export.xlsx
```

All metadata fields are included in Excel export for traceability.

---

## Troubleshooting

### "Metadata not found"

**Cause**: No `metadata.json` in test folder, UI fields not filled
**Solution**: Either create `metadata.json` or enter metadata via UI

### "Metadata incomplete for campaign save"

**Cause**: Missing required fields (part_number, geometry, fluid)
**Solution**: Add missing fields to `metadata.json` or enter via UI

### "Geometry validation failed"

**Cause**: Geometry values out of range or wrong units
**Solution**: Check units (mm² for area, mm for diameter) and realistic values

### "Fluid not recognized"

**Cause**: Fluid name doesn't match known fluids
**Solution**: Use standard names ("nitrogen", "oxygen", "water", "kerosene") or provide gamma manually

### Metadata not auto-loading

**Checklist**:
1. Is file named exactly `metadata.json`? (case-sensitive)
2. Is it in the same folder as your CSV file?
3. Is it valid JSON? (Use online JSON validator)
4. Does it have minimum required fields?

---

## Schema Validation

Metadata is validated using Pydantic schemas. Common validation errors:

### "Field required"
Missing required field. Add it to your metadata.json.

### "Value is not a valid float"
Non-numeric value in numeric field. Check your numbers.

### "Extra fields not permitted"
Unknown field name. Check spelling against field reference above.

### "Value out of range"
Value outside physical bounds (e.g., negative area). Check units and values.

---

## Examples by Use Case

### Quick Analysis (No Campaign)

```json
{
  "geometry": {"orifice_area_mm2": 3.14},
  "fluid": {"name": "nitrogen", "gamma": 1.4}
}
```

### Production Testing

```json
{
  "part_number": "INJ-B1-03",
  "serial_number": "SN-2024-045",
  "test_datetime": "2024-03-15T14:30:00",
  "analyst": "J. Smith",
  "geometry": {
    "orifice_area_mm2": 3.14159,
    "orifice_diameter_mm": 2.0
  },
  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4
  },
  "test_conditions": {
    "ambient_temp_K": 293.15,
    "expected_pressure_psi": 250.0
  },
  "configuration_variant": "production",
  "notes": "Acceptance test per procedure ATP-001"
}
```

### Research & Development

```json
{
  "part_number": "DEV-001",
  "test_id": "EXPERIMENT-2024-03-15-001",
  "test_datetime": "2024-03-15T14:30:00",
  "analyst": "J. Smith",
  "geometry": {
    "orifice_area_mm2": 3.14159,
    "edge_type": "sharp",
    "length_mm": 1.5
  },
  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4,
    "temperature_K": 293.15
  },
  "test_objectives": [
    "Characterize discharge coefficient vs Reynolds number",
    "Validate CFD model predictions",
    "Assess manufacturing repeatability"
  ],
  "notes": "First test of new sharp-edge design. Compare to rounded-edge baseline."
}
```

---

## API Reference

### Python API

```python
from core.metadata_manager import (
    load_metadata_from_folder,
    create_metadata_template,
    save_metadata_template,
    MetadataManager
)

# Load metadata from folder
metadata, source = load_metadata_from_folder("TEST-001/")
if metadata:
    print(f"Loaded from {source.source_type}")
    print(f"Part number: {metadata.part_number}")
    print(f"Complete for campaign: {metadata.is_complete_for_campaign()}")

# Create template
template = create_metadata_template(test_type="cold_flow")
save_metadata_template(template, "template.json")

# Use MetadataManager for advanced operations
manager = MetadataManager()
metadata = manager.load_from_folder("TEST-001/")
merged = manager.merge_file_and_ui(file_metadata, ui_metadata)
```

### Validation

```python
from core.config_validation import validate_test_metadata

# Validate metadata dict
metadata_dict = {
    "geometry": {"orifice_area_mm2": 3.14},
    "fluid": {"name": "nitrogen", "gamma": 1.4}
}

try:
    metadata = validate_test_metadata(metadata_dict, require_complete=False)
    print("Metadata valid!")
except ValueError as e:
    print(f"Validation error: {e}")
```

---

## Additional Resources

- **Active Configuration Guide**: See `CLAUDE.md` for testbench configuration
- **Migration Guide**: `WHATS_NEW.md` v2.3.0 section
- **API Documentation**: `core/README.md`
- **Examples**: `example_metadata/` folder (after running migration script)

---

## Summary

**Key Takeaways**:

1. **Two-Part System**: Active Configuration (testbench) + Test Metadata (test article)
2. **Auto-Loading**: Put `metadata.json` in test folder → auto-loads
3. **Flexible**: Enter via UI if no file, or use file + UI override
4. **Campaign Requirements**: Need part_number, geometry, fluid for campaign save
5. **Backward Compatible**: Old configs still work via auto-migration

**Recommended Workflow**:
1. Create `metadata.json` template for your test type
2. Copy to each test folder
3. Customize for each test (part number, serial, notes)
4. Upload test data → metadata auto-loads → analyze!

---

**Version**: 2.3.0
**Last Updated**: 2025-12-31
