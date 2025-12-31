# Template Integration Feature - Quick Reference

**Added**: 2025-12-30
**Location**: Page 1 (Single Test Analysis) → Quick Config section
**Version**: v2.2.0

---

## What's New

Templates are now accessible directly from the analysis workflow! No more switching to Page 8 to browse templates.

## How It Works

### Quick Config Modes

The Quick Config section now has **two modes**:

1. **Recent Configs** - Your last 5 used configurations (existing feature)
2. **Templates** - Browse all available templates filtered by test type (NEW!)

### Using Templates in Quick Config

1. Go to **Page 1: Single Test Analysis**
2. Select your test type (Cold Flow or Hot Fire)
3. In the Quick Config section, click **"Templates"** radio button
4. Browse available templates for your selected test type
5. View template description and tags
6. Click **"Load"** button to apply the template
7. Template automatically saved to Recent Configs for future use

### Visual Guide

```
┌─────────────────────────────────────┐
│  Test Type: [Cold Flow ▼]          │
├─────────────────────────────────────┤
│  ⚡ Quick Config                     │
│  ○ Recent Configs  ● Templates      │  ← Toggle between modes
├─────────────────────────────────────┤
│  Available Cold Flow Templates      │
│  [-- Select Template -- ▼]          │
│  [lcsc_b1_injectors_cf - LCSC...▼] │  ← Select template
├─────────────────────────────────────┤
│  📝 Standard config for swirl...    │  ← Description
│  🏷️ Tags: injector, cold-flow       │  ← Tags
│  [ Load ]                           │  ← Load button
└─────────────────────────────────────┘
```

## Benefits

### Before
```
1. Go to Page 8 (Config Templates)
2. Browse templates
3. Load template to session
4. Go back to Page 1
5. Hope template is still in session
```
**Time**: ~2-3 minutes per template load

### After
```
1. Click "Templates" in Quick Config
2. Select template
3. Click "Load"
```
**Time**: ~10 seconds per template load

**Savings**: **90% faster** template access!

---

## Features

### Smart Filtering
- Templates automatically filtered by test type
- Only see Cold Flow templates when analyzing cold flow tests
- Only see Hot Fire templates when analyzing hot fire tests

### Template Information
- **Description**: See what the template is for
- **Tags**: Quick identification (e.g., "injector", "igniter", "nitrogen")
- **Template ID**: Unique identifier for reference

### Integration with Recent Configs
- Loaded templates automatically added to Recent Configs
- Switch back to "Recent Configs" mode to see recently loaded templates
- Templates marked with source: "template" in recent list

### Graceful Handling
- If no templates exist: Clear message with link to create them
- If no templates for test type: Suggests creating one
- If template system unavailable: Falls back to other config sources

---

## Use Cases

### Use Case 1: Testing Multiple Injector Elements
Engineer testing 10 injector elements with same fluid/geometry:

**Workflow**:
1. Day 1: Create template for nitrogen cold flow injector testing (Page 8)
2. Day 2: Quick Config → Templates → Select "nitrogen_cf_injector" → Load
3. Analyze all 10 tests using the same template
4. Template appears in Recent Configs for even faster access next time

**Time savings**: 27 minutes (no config re-entry) + 1-2 minutes (no Page 8 navigation)

---

### Use Case 2: Different Test Campaigns
Engineer switching between multiple test campaigns daily:

**Before**:
- Go to Page 8
- Browse through all templates
- Load template
- Go back to Page 1
- Hope it's still loaded

**After**:
- Quick Config → Templates
- See only relevant templates for current test type
- Load with one click
- Start analyzing immediately

---

### Use Case 3: Onboarding New Team Members
New engineer needs to run standard tests:

**Before**:
- Learn config file structure
- Copy from existing configs
- Manual editing
- High error risk

**After**:
- Quick Config → Templates
- Select "standard_nitrogen_coldflow"
- Load and analyze
- Configs are pre-validated and tested

---

## Technical Details

### Implementation
- Uses existing `TemplateManager` from `core/templates.py`
- Integrates with `ConfigManager` for recent config tracking
- Radio button toggle between "Recent Configs" and "Templates"
- Real-time filtering by test_type
- Automatic session refresh after template load

### Code Location
File: `pages/1_Single_Test_Analysis.py`
Lines: ~387-477 (Quick Config section)

### Dependencies
- `core.templates.TemplateManager`
- `core.templates.create_config_from_template`
- `core.config_manager.ConfigManager`

### Session State
Templates loaded via this feature are saved to:
- `st.session_state['recent_configs']` (managed by ConfigManager)
- Source marked as: `'template'`

---

## Tips & Tricks

### Tip 1: Create Reusable Templates
Before your first test of the day:
1. Create templates for your common test scenarios
2. Name them descriptively (e.g., "N2_Injector_A_ColdFlow")
3. Add tags for quick identification

### Tip 2: Templates → Recent Workflow
- Load a template once from Templates mode
- Switch to Recent Configs mode
- Template now appears in recent list for one-click access
- Best of both worlds!

### Tip 3: Test Type Switching
- Change test type → templates automatically re-filter
- Only see relevant templates for current test type
- No clutter from irrelevant configurations

### Tip 4: Template Descriptions
- Add clear descriptions when creating templates
- Helps you (and team members) choose the right template quickly
- Shows directly in Quick Config without additional clicks

---

## Comparison: All Config Sources

| Source | Speed | Use Case | Persistence |
|--------|-------|----------|-------------|
| **Recent Configs** | ⚡⚡⚡ Instant | Repeat recent test | Session only |
| **Templates** | ⚡⚡ Very Fast | Standard test setup | Permanent (stored) |
| **Upload JSON** | ⚡ Moderate | One-time config | Saved to recent |
| **Default** | ⚡⚡⚡ Instant | Quick start/demo | Not saved |
| **Manual Edit** | 🐢 Slow | Custom tweaks | Saved when applied |

---

## Known Limitations

1. **Template filtering**: Only filters by test_type, not by tags (future enhancement)
2. **Template creation**: Still requires Page 8 to create new templates
3. **No preview**: Can't see full config before loading (use Recent Configs after loading to edit)
4. **Session-based recent list**: Recent configs clear on page reload

---

## Future Enhancements

Potential improvements for future versions:

1. **Tag-based filtering**: Filter templates by tags (e.g., show only "injector" templates)
2. **Template preview**: Show config JSON before loading
3. **Create from current**: Save current config as template directly from Page 1
4. **Template search**: Search templates by name or description
5. **Favorites**: Mark favorite templates for quick access
6. **Template editing**: Edit templates directly from Quick Config

---

## Migration from Page 8 Workflow

If you currently use Page 8 (Config Templates) for loading templates:

**No change required!** Both workflows still work:

**Option A: Quick Config (Recommended)**
- Faster (no page navigation)
- Auto-filtered by test type
- Integrated with Recent Configs

**Option B: Page 8 (For template management)**
- Better for creating new templates
- Better for browsing all templates
- Better for managing template library

Use Page 8 for **template management**, use Quick Config for **template usage**.

---

## Version History

- **v2.2.0** (2025-12-30): Template integration added to Quick Config
- **v2.1.0** (2025-12-30): Quick Config with Recent Configs introduced
- **v2.0.0**: Template system created (Page 8 only)

---

**Related Features**:
- Quick Config (v2.1.0)
- Config Templates (v2.0.0)
- Persistent Detection (v2.1.0)
