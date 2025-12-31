# v2.3.0 - Configuration Architecture Release

## Overview

This release introduces a fundamental architecture improvement: **separation of testbench hardware configuration from test article properties**.

### The Problem (v2.0-v2.2)

Previously, users had to create a new "config" for every combination of:
- Testbench hardware (sensors, calibrations)
- Test article (injector element, nozzle geometry)
- Fluid type (nitrogen, water, oxygen, etc.)

This led to config proliferation and repetitive data entry.

### The Solution (v2.3.0)

**Active Configuration** (testbench hardware):
- Sensor mappings, channel IDs, calibration uncertainties
- Changes only when testbench is modified/recalibrated
- Stored in `saved_configs/` folder
- **Load once, reuse for all tests**

**Test Metadata** (test article properties):
- Geometry, fluid, part numbers, serial numbers
- Changes every test
- Auto-loads from `metadata.json` in test folder
- Or enter via UI (optional for analysis, required for campaign)

## Key Features

### 1. Config/Metadata Separation
- **Active Configuration**: Testbench-specific settings
- **Test Metadata**: Per-test properties
- Clear separation enforced by Pydantic schema validation
- Automatic migration from v2.0-v2.2 format

### 2. Channel Mapping
- New `channel_mapping` field: DAQ channel IDs → P&ID sensor names
- Clear traceability: `"10001"` → `"FU-PT-01"` → `"pressure_upstream"`

### 3. Metadata Auto-Loading
- Place `metadata.json` in test data folder → auto-loads
- Fallback to UI entry if no file found
- Merge file + UI with configurable override behavior

### 4. Migration Tool
- `scripts/migrate_configs_v2.3.py` for automated migration
- Splits old configs into Active Config + Metadata
- Dry-run mode for safe preview
- Detailed migration reports

### 5. Terminology Cleanup
- "Templates" → "Saved Configs" (clearer purpose)
- "Config Templates" page → "Saved Configurations"
- All UI updated consistently

### 6. Improved UX
- Consolidated configuration source selector
- Single section with 4 clear options: Recent, Saved, Upload, Manual
- Eliminated confusing dual-section layout

## What's Changed

### Core Modules
- **`core/config_validation.py`**: Added `ActiveConfiguration` and `TestMetadata` schemas
- **`core/metadata_manager.py`**: New module for metadata loading/management
- **`core/templates.py` → `core/saved_configs.py`**: Renamed with backward compatibility

### Tools
- **`scripts/migrate_configs_v2.3.py`**: Automated config migration tool

### Documentation
- **`METADATA_GUIDE.md`**: Comprehensive guide for using test metadata
- **`WHATS_NEW.md`**: Updated with v2.3.0 section
- **`README.md`**: Added v2.3.0 architecture overview

### UI Updates
- **`pages/1_Single_Test_Analysis.py`**: Consolidated config source UI
- **`pages/8_Config_Templates.py` → `pages/8_Saved_Configurations.py`**: Renamed
- All pages updated with new terminology

### Testing
- All P0 tests passing (21 tests)
- All P2 tests passing (32 tests)
- Pydantic v2 compatibility fixed
- Migration script tested on existing configs

## Backward Compatibility

✅ **Fully backward compatible**:
- Old configs (v2.0-v2.2) auto-migrate transparently
- All aliases in place (`ConfigTemplate`, `TemplateManager`, `BUILTIN_TEMPLATES`)
- Existing code continues to work
- No breaking changes to APIs

## Migration Guide

### For Existing Users

**Option 1: Automatic (No Action Required)**
- Old configs continue to work via transparent auto-migration
- System automatically detects and splits geometry/fluid into metadata

**Option 2: Permanent Migration**
```bash
# Preview migration
python scripts/migrate_configs_v2.3.py --dry-run

# Execute migration
python scripts/migrate_configs_v2.3.py
```

**Output**:
- `saved_configs/[name]_active_config.json` - Testbench configuration
- `example_metadata/[name]_metadata_example.json` - Example metadata

### New Workflow

1. **Select Active Configuration** (once per test series)
   - Choose saved config for your testbench

2. **Provide Test Metadata** (per test)
   - Place `metadata.json` in test folder (auto-loads), OR
   - Enter via UI fields

3. **Analyze**
   - System merges config + metadata
   - Full analysis with traceability

## Example: Before vs After

### Before (v2.2.0)
```
configs/
├── INJ_B1_nitrogen.json      # Testbench + Element B1 + Nitrogen
├── INJ_B1_water.json          # Testbench + Element B1 + Water
├── INJ_B2_nitrogen.json       # Testbench + Element B2 + Nitrogen
├── INJ_B2_water.json          # Testbench + Element B2 + Water
└── ... (config proliferation)
```

### After (v2.3.0)
```
saved_configs/
└── lcsc_testbench.json        # Testbench hardware (reused)

test_data/
├── TEST-001-B1-nitrogen/
│   ├── data.csv
│   └── metadata.json          # Element B1, nitrogen properties
├── TEST-002-B1-water/
│   ├── data.csv
│   └── metadata.json          # Element B1, water properties
├── TEST-003-B2-nitrogen/
│   ├── data.csv
│   └── metadata.json          # Element B2, nitrogen properties
└── ...
```

## Files Changed

**Core**: 3 files modified, 1 added
**Scripts**: 1 added
**Documentation**: 3 modified, 1 added
**UI**: 9 pages modified, 1 renamed
**Tests**: All passing (53 tests)

## Commits

1. `feat: Add Active Configuration and Test Metadata separation (Phase 1-2)`
2. `feat: Rename templates to saved configs (Phase 3)`
3. `feat: Update UI terminology from Templates to Saved Configs (Phase 4)`
4. `feat: Add migration script and update WHATS_NEW for v2.3.0 (Phase 5-6)`
5. `docs: Complete Phase 6 documentation for v2.3.0`
6. `fix: Pydantic v2 compatibility and backward compatibility`
7. `feat: Consolidate configuration source UI for better UX`

## Version Numbers

- Application Version: **2.3.0** (Configuration Architecture Release)
- Core Version: **2.3.0** (Config/Metadata Separation)
- Schema Version: 3 (no database changes)
- Processing Version: 2.0.0+integrity

## Testing Checklist

- [x] All P0 core tests pass (21 tests)
- [x] All P2 advanced tests pass (32 tests)
- [x] Pydantic v1 and v2 compatibility
- [x] Migration script tested on existing configs
- [x] Backward compatibility verified
- [x] Documentation complete

## Next Steps After Merge

1. Review `METADATA_GUIDE.md` for metadata usage
2. Optionally run migration script: `python scripts/migrate_configs_v2.3.py`
3. Create `metadata.json` files for test series
4. Use new unified Configuration Source UI

## Related Issues

Addresses user feedback about:
- Config proliferation for different test articles
- Unclear difference between "config" and "template"
- Confusing dual-section sidebar layout
- Missing channel ID mapping

---

**Ready to merge** ✅

All phases complete, tests passing, documentation updated, backward compatible.
