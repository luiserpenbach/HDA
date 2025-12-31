# Config/Metadata Separation - Implementation Plan

**Version**: 2.3.0
**Date**: 2025-12-31
**Status**: Implementation in progress

---

## Overview

Separate configuration (testbench hardware) from metadata (test article properties) for clearer organization and less repetition.

## Problem Statement

Current system mixes two types of information in "config":
1. **Testbench hardware** (rarely changes): sensors, uncertainties, channel mappings
2. **Test article properties** (changes every test): geometry, fluid, part numbers

This forces users to create a new "config" for every injector element or fluid type, even though the testbench hasn't changed.

## Proposed Solution

### Active Configuration (Testbench Hardware)
**Purpose**: Describes the test hardware setup
**Frequency**: Changes only when testbench is modified/recalibrated
**Location**: UI selection or `saved_configs/` folder

**Contents**:
```json
{
  "config_name": "LCSC Testbench Standard",
  "config_version": "2.3.0",
  "test_type": "cold_flow",

  "channel_mapping": {
    "10001": "FU-PT-01",
    "10002": "FU-PT-02",
    "10003": "FU-TT-01",
    "10004": "FU-FM-01"
  },

  "sensor_mapping": {
    "timestamp": "Time",
    "pressure_upstream": "FU-PT-01",
    "pressure_downstream": "FU-PT-02",
    "temperature": "FU-TT-01",
    "mass_flow": "FU-FM-01"
  },

  "sensor_uncertainties": {
    "pressure_upstream": {"value": 0.05, "unit": "psi", "type": "absolute"},
    "pressure_downstream": {"value": 0.05, "unit": "psi", "type": "absolute"},
    "temperature": {"value": 1.0, "unit": "K", "type": "absolute"},
    "mass_flow": {"value": 0.1, "unit": "g/s", "type": "absolute"}
  },

  "processing": {
    "steady_state_method": "cv",
    "cv_threshold": 0.02,
    "window_size": 50
  }
}
```

### Test Metadata (Per Test Article)
**Purpose**: Describes the test article and conditions
**Frequency**: Changes every test
**Location**: `metadata.json` in test folder, or UI entry

**Contents**:
```json
{
  "part_number": "INJ-B1-03",
  "serial_number": "SN-2024-045",
  "test_datetime": "2025-01-15T14:30:00",
  "analyst": "John Doe",
  "test_type": "cold_flow",

  "geometry": {
    "orifice_area_mm2": 3.14159,
    "throat_area_mm2": 12.566,
    "num_orifices": 12,
    "length_mm": 25.4
  },

  "fluid": {
    "name": "nitrogen",
    "gamma": 1.4,
    "molecular_weight": 28.014,
    "temperature_K": 293.15
  },

  "test_conditions": {
    "ambient_pressure_psi": 14.7,
    "target_pressure_psi": 100.0,
    "notes": "Nominal pressure test"
  }
}
```

---

## Implementation Steps

### Phase 1: Schema and Validation ✅
- [ ] Create new schema for Active Configuration (no geometry/fluid)
- [ ] Create new schema for Test Metadata
- [ ] Update `core/config_validation.py` with dual validation
- [ ] Add backward compatibility mode

### Phase 2: Core Module Updates
- [ ] Update `core/config_manager.py`:
  - Rename to handle "Active Configuration" and "Saved Configs"
  - Add metadata loading from folder
  - Add UI metadata entry fallback
- [ ] Create `core/metadata_manager.py`:
  - Load metadata.json from test folder
  - Validate metadata schema
  - Merge with UI-provided metadata
- [ ] Update `core/integrated_analysis.py`:
  - Accept separate config + metadata parameters
  - Merge for analysis (backward compatible)
- [ ] Update traceability:
  - Separate hashes for config vs metadata
  - Track both in audit trail

### Phase 3: Saved Configs (Template Replacement)
- [ ] Rename `config_templates/` → `saved_configs/`
- [ ] Rename `core/templates.py` → `core/saved_configs.py`
- [ ] Update SavedConfigManager (was TemplateManager)
- [ ] Add migration utility for existing templates

### Phase 4: UI Updates
- [ ] Page 1 (Single Test Analysis):
  - "Quick Config" section becomes "Active Configuration"
  - "Templates" becomes "Saved Configs"
  - Add "Test Metadata" section (optional input)
  - Auto-load metadata.json if present in folder
- [ ] Page 4 (Batch Processing):
  - Update to use Active Configuration
  - Auto-detect metadata.json in each test folder
- [ ] Page 5 (Reports & Export):
  - Update terminology
- [ ] Page 8 (Config Templates):
  - Rename to "Saved Configurations"
  - Update all text and labels

### Phase 5: Migration Tools
- [ ] Create `scripts/migrate_configs.py`:
  - Auto-split old configs into config + metadata
  - Save to new locations
  - Generate migration report
- [ ] Update existing test folders with metadata.json

### Phase 6: Documentation Updates
- [ ] Update WHATS_NEW.md for v2.3.0
- [ ] Update README.md with new concepts
- [ ] Update CLAUDE.md with new structure
- [ ] Create METADATA_GUIDE.md

### Phase 7: Testing
- [ ] Test config validation with new schema
- [ ] Test metadata loading from folder
- [ ] Test UI metadata entry fallback
- [ ] Test campaign requirement enforcement
- [ ] Test backward compatibility
- [ ] Test migration script

---

## Data Flow

### New Flow (v2.3.0):
```
1. User selects Active Configuration (testbench setup)
   ├─ From Saved Configs library
   └─ Or upload custom config

2. User provides test data folder
   └─ Auto-load metadata.json if present

3. If no metadata.json:
   ├─ UI entry (optional for analysis)
   └─ Required for campaign save

4. Analysis runs with:
   ├─ config (testbench)
   └─ metadata (test article)

5. Campaign save:
   ├─ Validates metadata present
   ├─ Separate hashes for config + metadata
   └─ Full traceability
```

### Backward Compatibility:
```
Old config (has geometry + fluid)
    ↓
Auto-detect old format
    ↓
Split into config + metadata
    ↓
Proceed with new flow
```

---

## Naming Changes

| Old Name | New Name | Location |
|----------|----------|----------|
| Config | Active Configuration | UI, docs |
| Template | Saved Config | UI, docs |
| Template Integration | Saved Config Library | Feature name |
| `config_templates/` | `saved_configs/` | Folder |
| `core/templates.py` | `core/saved_configs.py` | Module |
| `TemplateManager` | SavedConfigManager | Class |
| `create_config_from_template()` | `load_saved_config()` | Function |

---

## Schema Definitions

### Active Configuration Schema
```python
ACTIVE_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["config_name", "config_version", "test_type", "sensor_mapping", "sensor_uncertainties"],
    "properties": {
        "config_name": {"type": "string"},
        "config_version": {"type": "string"},
        "test_type": {"type": "string", "enum": ["cold_flow", "hot_fire"]},

        "channel_mapping": {
            "type": "object",
            "description": "Maps channel IDs to P&ID sensor names",
            "patternProperties": {"^[0-9]+$": {"type": "string"}}
        },

        "sensor_mapping": {
            "type": "object",
            "description": "Maps sensor roles to CSV column names or P&ID names",
            "required": ["timestamp"],
            "properties": {...}
        },

        "sensor_uncertainties": {
            "type": "object",
            "description": "Uncertainty values for each sensor"
        },

        "processing": {
            "type": "object",
            "description": "Processing parameters (optional)",
            "properties": {
                "steady_state_method": {"type": "string"},
                "cv_threshold": {"type": "number"},
                "window_size": {"type": "integer"}
            }
        }
    },
    "additionalProperties": false
}
```

### Test Metadata Schema
```python
TEST_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "part_number": {"type": "string"},
        "serial_number": {"type": "string"},
        "test_datetime": {"type": "string", "format": "date-time"},
        "analyst": {"type": "string"},
        "test_type": {"type": "string", "enum": ["cold_flow", "hot_fire"]},

        "geometry": {
            "type": "object",
            "description": "Test article geometry",
            "properties": {
                "orifice_area_mm2": {"type": "number"},
                "throat_area_mm2": {"type": "number"},
                "num_orifices": {"type": "integer"},
                "length_mm": {"type": "number"}
            }
        },

        "fluid": {
            "type": "object",
            "description": "Working fluid properties",
            "properties": {
                "name": {"type": "string"},
                "gamma": {"type": "number"},
                "molecular_weight": {"type": "number"},
                "temperature_K": {"type": "number"}
            }
        },

        "test_conditions": {
            "type": "object",
            "description": "Test conditions and notes",
            "properties": {
                "ambient_pressure_psi": {"type": "number"},
                "target_pressure_psi": {"type": "number"},
                "notes": {"type": "string"}
            }
        }
    },
    "additionalProperties": true
}
```

---

## Benefits

1. **Clearer separation**: Testbench vs test article
2. **Less repetition**: Don't recreate config for every test
3. **Better traceability**: Separate hashes for hardware vs test
4. **Realistic workflow**: Load testbench config once, provide metadata per test
5. **Channel mapping**: Clear path from DAQ IDs → P&ID names → Analysis roles

---

## Migration Strategy

### Existing Configs
```python
# Old format (has geometry + fluid)
old_config = {
    "sensor_mapping": {...},
    "sensor_uncertainties": {...},
    "geometry": {...},  # ← Move to metadata
    "fluid": {...}      # ← Move to metadata
}

# Auto-split into:
new_config = {
    "config_name": "Migrated from old config",
    "config_version": "2.3.0",
    "sensor_mapping": {...},
    "sensor_uncertainties": {...}
}

new_metadata = {
    "geometry": {...},
    "fluid": {...},
    "source": "migrated_from_config"
}
```

### Existing Templates
```bash
config_templates/ → saved_configs/
  ├── lcsc_b1_injectors___cold_flow_standard.json
  └── ...
```

Each template splits into:
- **Saved Config**: Hardware/sensor info only
- **Example Metadata**: Geometry/fluid as reference (optional)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | Backward compatibility mode |
| Migration errors | Comprehensive testing + dry-run mode |
| User confusion | Clear documentation + UI guidance |
| Missing metadata for campaigns | Validation enforcement |

---

## Testing Checklist

- [ ] Load old-format config → auto-migration works
- [ ] Load new-format config + metadata.json → works
- [ ] Load new-format config, no metadata → analysis works
- [ ] Try to save to campaign without metadata → error shown
- [ ] Saved Config library shows correct configs
- [ ] Channel mapping displayed in UI
- [ ] Traceability has separate config/metadata hashes

---

## Timeline

- **Phase 1-2**: 2-3 hours (schema + core updates)
- **Phase 3**: 1 hour (saved configs rename)
- **Phase 4**: 2-3 hours (UI updates)
- **Phase 5**: 1 hour (migration tools)
- **Phase 6**: 1 hour (documentation)
- **Phase 7**: 1-2 hours (testing)

**Total**: 8-11 hours

---

## Success Criteria

- ✅ All existing configs/templates migrate successfully
- ✅ Backward compatibility maintained
- ✅ UI clearly separates config from metadata
- ✅ metadata.json auto-loads from test folder
- ✅ Campaign save requires metadata
- ✅ Channel mapping visible and functional
- ✅ All tests pass
- ✅ Documentation updated

---

**Status**: Planning complete, ready to implement
**Next Step**: Begin Phase 1 (Schema and Validation)
