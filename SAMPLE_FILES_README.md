# Sample Files for Plugin Architecture Testing

This directory contains sample configuration and test files to help you quickly test the new plugin architecture.

---

## Files Included

### 1. **sample_cold_flow_config.json**
Active configuration file representing your testbench hardware setup.

**Contains:**
- Sensor channel mappings (which DAQ channels correspond to which sensors)
- Calibration uncertainties for each sensor
- Hardware geometry (orifice area, throat area, etc.)
- Fluid properties (for Cd calculations)
- Processing settings (resample frequency, steady-state thresholds)

**When to modify:**
- When your testbench hardware changes
- After sensor recalibration
- When testing with different fluids

### 2. **sample_test_metadata.json**
Test article properties and test conditions for a specific test.

**Contains:**
- Test identification (test ID, date, operator, facility)
- Test article properties (part number, serial number, geometry)
- Test conditions (target pressures, flows, ambient conditions)
- Acceptance criteria (min/max Cd, uncertainty limits)
- References (test procedures, drawings, specs)

**When to modify:**
- For every test (different test articles, different conditions)
- Store with test data for traceability

### 3. **sample_plugin_test.py**
Demonstration script showing how to use the plugin architecture.

**Features:**
- Lists available plugins
- Loads configuration and metadata
- Generates synthetic test data
- Runs analysis using `analyze_test()`
- Displays results with uncertainties
- Shows traceability information
- Demonstrates database record format

---

## Quick Start

### Option 1: Run the Demo Script

```bash
python sample_plugin_test.py
```

This will:
1. Load the sample config and metadata
2. Generate synthetic test data
3. Run analysis with the ColdFlowPlugin
4. Display results with full traceability

**Expected output:**
```
======================================================================
HDA Plugin Architecture - Sample Test
======================================================================

1. Available Plugins:
----------------------------------------------------------------------
   • Cold Flow Injector Analysis
     Slug: cold_flow
     Type: cold_flow
     Version: 1.0.0
     Description: Analyze cold flow tests...

...

5. Analysis Results:
----------------------------------------------------------------------
   pressure_upstream    :  10.0125 ± 0.0501 bar
                           (±0.50%)
   delta_p              :   9.0125 ± 0.0501 bar
                           (±0.56%)
   mass_flow            : 124.8935 ± 0.5000 g/s
                           (±0.40%)
   Cd                   :   0.6532 ± 0.0120 -
                           (±1.84%)
```

### Option 2: Use in Your Own Code

```python
import json
from core.integrated_analysis import analyze_test

# Load config and metadata
with open('sample_cold_flow_config.json', 'r') as f:
    config = json.load(f)

with open('sample_test_metadata.json', 'r') as f:
    metadata_file = json.load(f)
    metadata = {
        'part': metadata_file['test_article']['part_number'],
        'serial_num': metadata_file['test_article']['serial_number'],
    }

# Load your actual test data
import pandas as pd
df = pd.read_csv('your_test_data.csv')

# Run analysis
result = analyze_test(
    df=df,
    config=config,
    steady_window=(2000, 8000),  # Adjust to your steady state window
    test_id="YOUR-TEST-ID",
    plugin_slug="cold_flow",
    metadata=metadata
)

# Access results
print(f"Cd = {result.measurements['Cd'].value:.4f} ± {result.measurements['Cd'].uncertainty:.4f}")
print(f"QC Passed: {result.passed_qc}")
```

---

## Customizing for Your Hardware

### Step 1: Update Configuration

Edit `sample_cold_flow_config.json`:

```json
{
  "columns": {
    "upstream_pressure": "YOUR-PT-01",  // Your actual channel name
    "mass_flow": "YOUR-FM-01"           // Your actual channel name
  },

  "uncertainties": {
    "YOUR-PT-01": {
      "type": "relative",
      "value": 0.005  // Your sensor's actual uncertainty
    }
  },

  "geometry": {
    "orifice_area_mm2": 3.14159  // Your actual orifice area
  }
}
```

### Step 2: Update Metadata

Edit `sample_test_metadata.json`:

```json
{
  "test_article": {
    "part_number": "YOUR-PART-NUMBER",
    "serial_number": "YOUR-SERIAL-NUMBER"
  },

  "test_conditions": {
    "target_p_upstream_bar": 10.0  // Your test conditions
  }
}
```

### Step 3: Run Your Test

```bash
python sample_plugin_test.py
```

---

## Configuration vs Metadata

Understanding the difference:

### **Configuration (testbench hardware)**
- Stored in: `saved_configs/` or `sample_cold_flow_config.json`
- Changes: Rarely (only when hardware is modified/recalibrated)
- Scope: Applies to all tests on that testbench
- Contents: Sensor mappings, calibrations, processing settings

### **Metadata (test article properties)**
- Stored in: Test data folders or `sample_test_metadata.json`
- Changes: Every test (different parts, different conditions)
- Scope: Specific to one test or test article
- Contents: Part numbers, geometry, test conditions, acceptance criteria

---

## Data Format Requirements

Your test data CSV should have these columns (names defined in config):

```csv
timestamp,IG-PT-01,IG-PT-02,FM-01,TC-01
0.0,0.523,0.124,5.234,298.5
10.0,1.234,0.234,15.432,299.1
20.0,5.678,0.567,65.432,299.8
...
10000.0,10.123,1.012,125.234,300.2
```

**Requirements:**
- `timestamp` column in milliseconds
- Sensor columns match names in config `"columns"` section
- No missing column names
- Numeric data types

---

## Troubleshooting

### "Plugin 'cold_flow' not found"

**Solution:** Ensure plugins are loaded:
```python
from core.plugins import PluginRegistry
plugins = PluginRegistry.get_plugins()  # Triggers discovery
print([p.metadata.slug for p in plugins])
```

### "Configuration validation failed"

**Solution:** Check that your config has all required fields:
- `columns.upstream_pressure` or `columns.inlet_pressure`
- `columns.mass_flow` or `columns.mf`
- `geometry.orifice_area_mm2`
- `fluid.density_kg_m3`

### "QC checks failed"

**Solution:** Use `skip_qc=True` for testing, or fix data issues:
```python
result = analyze_test(..., skip_qc=True)  # Skip QC for testing
```

### "Column 'IG-PT-01' not found in DataFrame"

**Solution:** Ensure column names in your CSV match the config:
```json
{
  "columns": {
    "upstream_pressure": "IG-PT-01"  // Must match CSV column name exactly
  }
}
```

---

## Next Steps

1. **Test with sample data:**
   ```bash
   python sample_plugin_test.py
   ```

2. **Customize for your hardware:**
   - Edit `sample_cold_flow_config.json`
   - Edit `sample_test_metadata.json`

3. **Integrate into your workflow:**
   - Use `analyze_test()` in your analysis scripts
   - Save configs to `saved_configs/` directory
   - Store metadata with test data

4. **Create custom plugins:**
   - See `PLUGIN_ARCHITECTURE.md` for tutorial
   - Copy `core/plugin_modules/cold_flow.py` as template
   - Add your plugin to `core/plugin_modules/`

---

## Additional Resources

- **Complete Plugin Guide:** `PLUGIN_ARCHITECTURE.md`
- **Example Plugin:** `core/plugin_modules/cold_flow.py`
- **Plugin Tests:** `tests/test_plugins.py`
- **Main HDA Guide:** `CLAUDE.md`

---

**Questions?** Review the documentation or examine the test files for working examples.
