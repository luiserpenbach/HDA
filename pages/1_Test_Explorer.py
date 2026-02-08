"""
Test Explorer Page
==================
Browse test data structure, ingest new tests, and edit metadata.

Features:
- View test data file structure (programs, systems, campaigns, tests)
- Ingest new test with metadata file or manual input
- Create standard test subfolder structure
- Suggest next available Test ID
- Edit existing test metadata via structured form or raw JSON

Folder Hierarchy:
    TEST_ROOT / TEST_PROGRAM / SYSTEM / SYSTEM-CAMPAIGN / SYSTEM-CAMPAIGN-TESTTYPE / TEST_ID

Test ID Naming Scheme: SYSTEM-CAMPAIGN-TESTTYPE-#
Example: RCS-C01-CF-001
Note: Test Program is NOT included in Test ID to keep it concise.
Note: Campaign comes before Test Type since one campaign can have multiple test types.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Import core modules
from core.test_metadata import (
    TestMetadata, TEST_SUBFOLDERS, load_test_metadata,
    load_raw_metadata, save_raw_metadata, validate_raw_metadata,
)
from core.metadata_manager import create_metadata_template
from core.config_manager import ConfigManager
from pages._shared_sidebar import render_global_context
from pages._shared_styles import apply_custom_styles, render_page_header
from pages._shared_widgets import metadata_editor_widget

st.set_page_config(page_title="Test Explorer", page_icon="🔍", layout="wide")

# Apply custom styles
apply_custom_styles()

# =============================================================================
# SIDEBAR - Global Context
# =============================================================================

with st.sidebar:
    context = render_global_context()
    st.divider()
    st.caption("Use the browser below for detailed navigation and test creation.")

# =============================================================================
# MAIN CONTENT
# =============================================================================

render_page_header(
    title="Test Explorer",
    description="Browse test data structure and ingest new tests with metadata",
    badge_text="P1",
    badge_type="info"
)

# =============================================================================
# SESSION STATE INITIALIZATION (page-specific)
# =============================================================================

if 'selected_system' not in st.session_state:
    st.session_state.selected_system = None
if 'selected_campaign' not in st.session_state:
    st.session_state.selected_campaign = None
if 'editor_test_path' not in st.session_state:
    st.session_state.editor_test_path = None
if 'editor_raw_metadata' not in st.session_state:
    st.session_state.editor_raw_metadata = None
if 'editor_original_metadata' not in st.session_state:
    st.session_state.editor_original_metadata = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def scan_test_root(root_path: Path) -> Dict[str, Any]:
    """
    Scan test root directory and return structure.

    Returns dict with:
        - programs: List of test program names
        - program_info: Dict with info per program
        - total_tests: Total test count
    """
    if not root_path.exists():
        return {'programs': [], 'program_info': {}, 'total_tests': 0}

    programs = []
    program_info = {}
    total_tests = 0

    for item in sorted(root_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            program_name = item.name
            programs.append(program_name)

            # Count systems, campaigns, and tests within this program
            systems = []
            campaigns = 0
            tests = 0

            for system_dir in item.iterdir():
                if system_dir.is_dir() and not system_dir.name.startswith('.'):
                    systems.append(system_dir.name)

                    # New structure: SYSTEM-CAMPAIGN / SYSTEM-CAMPAIGN-TESTTYPE
                    for campaign_dir in system_dir.iterdir():
                        if campaign_dir.is_dir() and '-' in campaign_dir.name:
                            campaigns += 1
                            for test_type_dir in campaign_dir.iterdir():
                                if test_type_dir.is_dir() and '-' in test_type_dir.name:
                                    # Count tests in test type folder
                                    for test_dir in test_type_dir.iterdir():
                                        if test_dir.is_dir() and not test_dir.name.startswith('.'):
                                            tests += 1

            program_info[program_name] = {
                'path': str(item),
                'systems': systems,
                'system_count': len(systems),
                'campaign_count': campaigns,
                'test_count': tests
            }
            total_tests += tests

    return {
        'programs': programs,
        'program_info': program_info,
        'total_tests': total_tests
    }


def get_systems_for_program(root_path: Path, program: str) -> List[Dict[str, Any]]:
    """Get all systems for a test program."""
    systems = []
    program_dir = root_path / program

    if not program_dir.exists():
        return systems

    for system_dir in sorted(program_dir.iterdir()):
        if system_dir.is_dir() and not system_dir.name.startswith('.'):
            # Count campaigns and tests
            campaigns = 0
            tests = 0
            test_types = set()

            for campaign_dir in system_dir.iterdir():
                if campaign_dir.is_dir() and '-' in campaign_dir.name:
                    campaigns += 1
                    for test_type_dir in campaign_dir.iterdir():
                        if test_type_dir.is_dir() and '-' in test_type_dir.name:
                            # Extract test type from folder name (last part)
                            test_types.add(test_type_dir.name.split('-')[-1])
                            tests += sum(1 for t in test_type_dir.iterdir()
                                        if t.is_dir() and not t.name.startswith('.'))

            systems.append({
                'name': system_dir.name,
                'path': str(system_dir),
                'test_types': sorted(test_types),
                'campaign_count': campaigns,
                'test_count': tests
            })

    return systems


def get_campaigns_for_system(root_path: Path, program: str, system: str) -> List[Dict[str, Any]]:
    """Get all campaigns for a system within a program."""
    campaigns = []
    system_dir = root_path / program / system

    if not system_dir.exists():
        return campaigns

    for campaign_dir in sorted(system_dir.iterdir()):
        if campaign_dir.is_dir() and '-' in campaign_dir.name:
            # Extract campaign ID from folder name (e.g., RCS-C01 -> C01)
            campaign_id = campaign_dir.name.split('-')[-1]

            # Count test types and tests
            test_types = []
            test_count = 0

            for test_type_dir in campaign_dir.iterdir():
                if test_type_dir.is_dir() and '-' in test_type_dir.name:
                    test_type = test_type_dir.name.split('-')[-1]
                    test_types.append(test_type)
                    test_count += sum(1 for t in test_type_dir.iterdir()
                                     if t.is_dir() and not t.name.startswith('.'))

            campaigns.append({
                'name': campaign_dir.name,
                'campaign_id': campaign_id,
                'path': str(campaign_dir),
                'test_types': test_types,
                'test_count': test_count
            })

    return campaigns


def get_tests_for_campaign(campaign_path: Path) -> List[Dict[str, Any]]:
    """Get all tests in a campaign folder (across all test types)."""
    tests = []

    if not campaign_path.exists():
        return tests

    # Iterate through test type folders
    for test_type_dir in sorted(campaign_path.iterdir()):
        if test_type_dir.is_dir() and '-' in test_type_dir.name:
            test_type = test_type_dir.name.split('-')[-1]

            for test_dir in sorted(test_type_dir.iterdir()):
                if test_dir.is_dir() and not test_dir.name.startswith('.'):
                    test_info = {
                        'test_id': test_dir.name,
                        'test_type': test_type,
                        'path': str(test_dir),
                        'has_metadata': (test_dir / 'metadata.json').exists(),
                        'has_raw_data': (test_dir / 'raw_data').exists() and any((test_dir / 'raw_data').glob('*.csv')),
                        'has_config': (test_dir / 'config').exists() and any((test_dir / 'config').glob('*.json')),
                    }

                    # Try to load metadata
                    if test_info['has_metadata']:
                        try:
                            metadata = load_test_metadata(test_dir)
                            if metadata:
                                test_info['status'] = metadata.status
                                test_info['test_date'] = metadata.test_date
                                test_info['operator'] = metadata.operator
                        except Exception:
                            test_info['status'] = 'error'
                    else:
                        test_info['status'] = 'no_metadata'

                    tests.append(test_info)

    return tests


def get_next_test_id(test_type_path: Path, system: str, campaign_id: str, test_type: str) -> str:
    """
    Get the next available Test ID for a test type within a campaign.

    Format: SYSTEM-CAMPAIGN-TESTTYPE-NNN
    Example: RCS-C01-CF-001
    """
    if not test_type_path.exists():
        return f"{system}-{campaign_id}-{test_type}-001"

    # Get existing test numbers
    existing_numbers = []
    for test_dir in test_type_path.iterdir():
        if test_dir.is_dir():
            parts = test_dir.name.split('-')
            if len(parts) >= 4:
                try:
                    num = int(parts[-1])
                    existing_numbers.append(num)
                except ValueError:
                    pass

    # Get next number
    next_num = max(existing_numbers, default=0) + 1

    return f"{system}-{campaign_id}-{test_type}-{next_num:03d}"


def create_new_test(
    root_path: Path,
    program: str,
    system: str,
    campaign_id: str,
    test_type: str,
    test_id: str,
    metadata: Optional[TestMetadata] = None
) -> Tuple[bool, str, Optional[Path]]:
    """
    Create a new test folder with standard structure.

    Args:
        root_path: Test root directory
        program: Test program name (e.g., "Engine-A", "Hopper-Dev")
        system: System name (e.g., "RCS", "MAIN")
        campaign_id: Campaign ID (e.g., "C01")
        test_type: Test type code (e.g., "CF", "HF")
        test_id: Full test ID (e.g., "RCS-C01-CF-001")
        metadata: Optional TestMetadata object

    Returns: (success, message, test_folder_path)
    """
    try:
        # Build path: root / program / system / system-campaign / system-campaign-testtype / test_id
        campaign_folder = f"{system}-{campaign_id}"
        test_type_folder = f"{system}-{campaign_id}-{test_type}"
        test_folder = root_path / program / system / campaign_folder / test_type_folder / test_id

        if test_folder.exists():
            return False, f"Test folder already exists: {test_id}", None

        # Create folder structure
        test_folder.mkdir(parents=True, exist_ok=True)

        for subfolder in TEST_SUBFOLDERS:
            (test_folder / subfolder).mkdir(exist_ok=True)

        # Save metadata if provided
        if metadata:
            metadata.test_id = test_id
            metadata.program = program
            metadata.system = system
            metadata.campaign_id = campaign_id
            metadata.test_type = test_type
            metadata.save(test_folder)
        else:
            # Create minimal metadata
            min_metadata = TestMetadata(
                test_id=test_id,
                program=program,
                system=system,
                campaign_id=campaign_id,
                test_type=test_type
            )
            min_metadata.save(test_folder)

        return True, f"Created test folder: {test_id}", test_folder

    except Exception as e:
        return False, f"Error creating test: {str(e)}", None


# =============================================================================
# MAIN PAGE LAYOUT
# =============================================================================

# Show expected structure
with st.expander("Expected Folder Structure"):
    st.code("""
TEST_ROOT/
    TEST_PROGRAM/                        # e.g., Engine-A, Hopper-Dev
        SYSTEM/                          # e.g., RCS, MAIN
            SYSTEM-CAMPAIGN/             # e.g., RCS-C01
                SYSTEM-CAMPAIGN-TESTTYPE/   # e.g., RCS-C01-CF
                    TEST_ID/                # e.g., RCS-C01-CF-001
                        config/
                        logs/
                        media/
                        plots/
                        processed_data/
                        raw_data/
                        reports/
                        metadata.json

Note: Test Program is for organization only - NOT included in Test IDs.
Test ID format: SYSTEM-CAMPAIGN-TESTTYPE-NNN (e.g., RCS-C01-CF-001)
    """, language="text")

# =============================================================================
# TABBED LAYOUT (uses global context)
# =============================================================================

if context['root_path']:
    root_path = context['root_path']
    selected_program = context['program']

    # Scan root
    structure = scan_test_root(root_path)

    # Summary metrics (above tabs — applies globally)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Programs", len(structure['programs']))
    with col2:
        total_systems = sum(info['system_count'] for info in structure['program_info'].values())
        st.metric("Systems", total_systems)
    with col3:
        total_campaigns = sum(info['campaign_count'] for info in structure['program_info'].values())
        st.metric("Campaigns", total_campaigns)
    with col4:
        st.metric("Total Tests", structure['total_tests'])

    # Reset downstream selections when program changes
    if st.session_state.get('_last_program') != selected_program:
        st.session_state.selected_system = None
        st.session_state.selected_campaign = None
        st.session_state._last_program = selected_program

    # =========================================================================
    # TOP-LEVEL TABS
    # =========================================================================

    tab_browse, tab_ingest, tab_edit = st.tabs([
        "Browse Tests", "Ingest New Test", "Edit Metadata"
    ])

    # =========================================================================
    # TAB 1: BROWSE TESTS
    # =========================================================================

    with tab_browse:
        # Three-panel browser (Program is now in sidebar)
        browse_col1, browse_col2, browse_col3 = st.columns(3)

        # Panel 1: Systems
        with browse_col1:
            st.subheader("Systems")

            if selected_program:
                systems = get_systems_for_program(root_path, selected_program)

                if systems:
                    systems_df = pd.DataFrame([
                        {
                            'System': s['name'],
                            'Types': ', '.join(s['test_types']),
                            'Tests': s['test_count']
                        }
                        for s in systems
                    ])

                    selected_system = st.selectbox(
                        "Select System",
                        [""] + [s['name'] for s in systems],
                        format_func=lambda x: x if x else "-- Select --",
                        key="system_selector"
                    )

                    if selected_system:
                        st.session_state.selected_system = selected_system
                        # Reset downstream selections when system changes
                        if st.session_state.get('_last_system') != selected_system:
                            st.session_state.selected_campaign = None
                            st.session_state._last_system = selected_system

                    st.dataframe(systems_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No systems in {selected_program}")

                # Create New System button
                with st.popover("+ New System", use_container_width=True):
                    new_sys = st.text_input("System Name", key="create_system_name", placeholder="e.g., RCS")
                    if st.button("Create", key="btn_create_system", disabled=not new_sys):
                        (root_path / selected_program / new_sys).mkdir(parents=True, exist_ok=True)
                        st.success(f"Created: {new_sys}")
                        st.rerun()
            else:
                st.info("Select a program in the sidebar")

        # Panel 2: Campaigns
        with browse_col2:
            st.subheader("Campaigns")

            if selected_program and st.session_state.selected_system:
                campaigns = get_campaigns_for_system(
                    root_path,
                    selected_program,
                    st.session_state.selected_system
                )

                if campaigns:
                    campaigns_df = pd.DataFrame([
                        {
                            'Campaign': c['campaign_id'],
                            'Types': ', '.join(c['test_types']),
                            'Tests': c['test_count']
                        }
                        for c in campaigns
                    ])

                    selected_campaign_name = st.selectbox(
                        "Select Campaign",
                        [""] + [c['name'] for c in campaigns],
                        format_func=lambda x: x.split('-')[-1] if x and '-' in x else (x if x else "-- Select --"),
                        key="campaign_selector"
                    )

                    if selected_campaign_name:
                        st.session_state.selected_campaign = next(
                            c for c in campaigns if c['name'] == selected_campaign_name
                        )

                    st.dataframe(campaigns_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No campaigns in {st.session_state.selected_system}")

                # Create New Campaign button
                with st.popover("+ New Campaign", use_container_width=True):
                    new_camp = st.text_input("Campaign ID", key="create_campaign_id", placeholder="e.g., C01")
                    if st.button("Create", key="btn_create_campaign", disabled=not new_camp):
                        sys = st.session_state.selected_system
                        campaign_folder = f"{sys}-{new_camp}"
                        (root_path / selected_program / sys / campaign_folder).mkdir(parents=True, exist_ok=True)
                        st.success(f"Created: {campaign_folder}")
                        st.rerun()
            else:
                st.info("Select a system first")

        # Panel 3: Tests
        with browse_col3:
            st.subheader("Tests")

            if st.session_state.selected_campaign:
                campaign = st.session_state.selected_campaign
                campaign_path = Path(campaign['path'])
                tests = get_tests_for_campaign(campaign_path)

                if tests:
                    tests_df = pd.DataFrame([
                        {
                            'Test ID': t['test_id'],
                            'Type': t['test_type'],
                            'Data': 'Y' if t['has_raw_data'] else 'N'
                        }
                        for t in tests
                    ])

                    st.dataframe(tests_df, use_container_width=True, hide_index=True)

                    # Quick action: select test
                    selected_test = st.selectbox(
                        "Select Test",
                        [""] + [t['test_id'] for t in tests],
                        format_func=lambda x: x if x else "-- Select test --"
                    )

                    if selected_test:
                        test_info = next(t for t in tests if t['test_id'] == selected_test)
                        st.caption(f"Path: {test_info['path']}")

                        btn_col_a, btn_col_b = st.columns(2)
                        with btn_col_a:
                            if st.button("Open in Analysis", use_container_width=True):
                                st.session_state.test_folder_path = test_info['path']
                                st.info("Test path saved. Navigate to 'Single Test Analysis' page.")
                        with btn_col_b:
                            if st.button("Edit Metadata", use_container_width=True):
                                st.session_state.editor_test_path = test_info['path']
                                # Clear any stale editor state
                                st.session_state.editor_raw_metadata = None
                                st.session_state.editor_original_metadata = None
                                st.info("Test selected for editing. Switch to the 'Edit Metadata' tab.")
                else:
                    st.info(f"No tests in {campaign['campaign_id']}")
            else:
                st.info("Select a campaign first")

    # =========================================================================
    # TAB 2: INGEST NEW TEST
    # =========================================================================

    with tab_ingest:
        # Use selections from browser or allow override
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Test Location")

            # Program (from global context)
            ing_program = selected_program or ""
            st.text_input("Program", value=ing_program, disabled=True, key="ing_program_display")
            if not ing_program:
                st.caption("Select a program in the sidebar")

            # System (from browser selection)
            ing_system = st.session_state.selected_system or ""
            st.text_input("System", value=ing_system, disabled=True, key="ing_system_display")
            if not ing_system:
                st.caption("Select a system in Browse tab")

            # Campaign (from browser selection)
            ing_campaign_id = ""
            if st.session_state.selected_campaign:
                ing_campaign_id = st.session_state.selected_campaign.get('campaign_id', '')
            st.text_input("Campaign", value=ing_campaign_id, disabled=True, key="ing_campaign_display")
            if not ing_campaign_id:
                st.caption("Select a campaign in Browse tab")

            # Test Type (user selects)
            test_type_options = ["CF", "HF", "LK", "PR"]
            test_type_labels = {
                "CF": "CF - Cold Flow",
                "HF": "HF - Hot Fire",
                "LK": "LK - Leak Test",
                "PR": "PR - Pressure Test"
            }
            new_test_type = st.selectbox(
                "Test Type",
                test_type_options,
                format_func=lambda x: test_type_labels.get(x, x)
            )

            # Suggested Test ID
            if ing_program and ing_system and ing_campaign_id and new_test_type:
                campaign_folder = f"{ing_system}-{ing_campaign_id}"
                test_type_folder = f"{ing_system}-{ing_campaign_id}-{new_test_type}"
                test_type_path = root_path / ing_program / ing_system / campaign_folder / test_type_folder

                suggested_id = get_next_test_id(test_type_path, ing_system, ing_campaign_id, new_test_type)

                st.info(f"Suggested Test ID: **{suggested_id}**")

                use_suggested = st.checkbox("Use suggested ID", value=True)
                if use_suggested:
                    new_test_id = suggested_id
                else:
                    new_test_id = st.text_input("Custom Test ID", value=suggested_id)
            else:
                new_test_id = ""
                st.warning("Select Program, System, and Campaign to generate Test ID")

        with col2:
            st.subheader("Metadata")

            metadata_source = st.radio(
                "Metadata Source",
                ["Manual Input", "Upload JSON"],
                horizontal=True
            )

            if metadata_source == "Upload JSON":
                metadata_file = st.file_uploader("Upload metadata.json", type=['json'])

                if metadata_file:
                    try:
                        metadata_dict = json.load(metadata_file)
                        st.success("Metadata loaded")
                        with st.expander("View Metadata"):
                            st.json(metadata_dict)

                        # Create TestMetadata from dict
                        new_metadata = TestMetadata.from_dict(metadata_dict)
                    except Exception as e:
                        st.error(f"Error loading metadata: {e}")
                        new_metadata = None
                else:
                    new_metadata = None
            else:
                # Manual input - minimal fields
                st.caption("Basic metadata (can be edited later)")

                meta_part = st.text_input("Part Name", placeholder="e.g., Injector Assembly")
                meta_pn = st.text_input("Part Number", placeholder="e.g., INJ-001-A")
                meta_sn = st.text_input("Serial Number", placeholder="e.g., SN-0042")
                meta_operator = st.text_input("Operator", placeholder="e.g., J. Smith")
                meta_date = st.date_input("Test Date", value=datetime.now())

                # Test type specific
                if new_test_type in ["CF", "LK", "PR"]:
                    meta_fluid = st.selectbox(
                        "Test Fluid",
                        ["", "Water", "Nitrogen", "Nitrous Oxide", "Helium", "Air", "Ethanol", "IPA"]
                    )
                else:
                    meta_fluid = ""

                meta_notes = st.text_area("Notes", height=68, placeholder="Any relevant notes...")

                # Build metadata
                new_metadata = TestMetadata(
                    test_id=new_test_id,
                    program=ing_program,
                    system=ing_system,
                    campaign_id=ing_campaign_id,
                    test_type=new_test_type,
                    part_name=meta_part,
                    part_number=meta_pn,
                    serial_number=meta_sn,
                    operator=meta_operator,
                    test_date=meta_date.isoformat() if meta_date else "",
                    test_fluid=meta_fluid,
                    notes=meta_notes,
                    status="pending"
                )

        # Raw data upload (optional)
        st.subheader("Raw Data (Optional)")
        raw_data_file = st.file_uploader(
            "Upload raw data CSV",
            type=['csv'],
            help="Optional: Upload raw data file to be placed in raw_data/ folder"
        )

        # Create Test Button
        st.markdown("---")

        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            create_enabled = bool(ing_program and ing_system and ing_campaign_id and new_test_type and new_test_id)

            if st.button(
                "Create New Test",
                type="primary",
                use_container_width=True,
                disabled=not create_enabled
            ):
                success, message, test_folder = create_new_test(
                    root_path,
                    ing_program,
                    ing_system,
                    ing_campaign_id,
                    new_test_type,
                    new_test_id,
                    new_metadata
                )

                if success and test_folder:
                    st.success(message)

                    # Save raw data if uploaded
                    if raw_data_file:
                        raw_data_path = test_folder / "raw_data" / raw_data_file.name
                        with open(raw_data_path, 'wb') as f:
                            f.write(raw_data_file.getbuffer())
                        st.info(f"Saved raw data: {raw_data_file.name}")

                    # Show created structure
                    with st.expander("Created Folder Structure", expanded=True):
                        st.code(f"""
{ing_program}/
  {ing_system}/
    {ing_system}-{ing_campaign_id}/
      {ing_system}-{ing_campaign_id}-{new_test_type}/
        {new_test_id}/
          config/
          logs/
          media/
          plots/
          processed_data/
          raw_data/{'  <- ' + raw_data_file.name if raw_data_file else ''}
          reports/
          metadata.json
                        """)

                    # Save to session for easy navigation
                    st.session_state.test_folder_path = str(test_folder)
                    # Save root path
                    ConfigManager.save_recent_folder(str(root_path))

                    # Option to continue
                    st.info("Test created. You can now:")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        if st.button("Create Another Test", use_container_width=True):
                            st.rerun()
                    with res_col2:
                        st.caption("Or navigate to 'Single Test Analysis' to analyze this test")
                else:
                    st.error(message)

    # =========================================================================
    # TAB 3: EDIT METADATA
    # =========================================================================

    with tab_edit:
        st.subheader("Edit Test Metadata")
        st.caption("Edit existing metadata.json files or create new ones for tests without metadata.")

        # ── Test selector ─────────────────────────────────────────────
        editor_path = st.text_input(
            "Test folder path",
            value=st.session_state.editor_test_path or "",
            placeholder="Select a test in Browse tab or enter path here",
            key="editor_path_input",
        )

        # Sync back to session state
        if editor_path:
            st.session_state.editor_test_path = editor_path

        if not editor_path:
            st.info("Select a test in the **Browse Tests** tab and click **Edit Metadata**, "
                    "or enter a test folder path above.")

        else:
            test_folder = Path(editor_path)
            if not test_folder.exists():
                st.error(f"Folder does not exist: {editor_path}")
            else:
                metadata_file = test_folder / "metadata.json"
                has_existing = metadata_file.exists()

                # ── Load / Create ─────────────────────────────────────
                if has_existing:
                    # Load from file if not already in session
                    if st.session_state.editor_raw_metadata is None:
                        try:
                            loaded = load_raw_metadata(test_folder)
                            st.session_state.editor_raw_metadata = loaded
                            st.session_state.editor_original_metadata = json.loads(
                                json.dumps(loaded, default=str))
                        except ValueError as e:
                            st.error(f"Failed to load metadata: {e}")
                            st.session_state.editor_raw_metadata = None
                else:
                    # No metadata.json — offer to create one
                    st.warning(f"No metadata.json found in `{test_folder.name}`")

                    if st.session_state.editor_raw_metadata is None:
                        create_col1, create_col2 = st.columns([2, 1])
                        with create_col1:
                            template_type = st.selectbox(
                                "Template type",
                                ["cold_flow", "hot_fire"],
                                key="editor_template_type",
                            )
                        with create_col2:
                            if st.button("Create from Template", use_container_width=True):
                                template = create_metadata_template(
                                    template_type, include_examples=True)
                                # Pre-fill identity from folder name
                                folder_name = test_folder.name
                                template['test_id'] = folder_name
                                parsed = TestMetadata.from_test_id(folder_name)
                                template['system'] = parsed.system
                                template['campaign_id'] = parsed.campaign_id
                                template['test_type'] = parsed.test_type
                                template['status'] = 'pending'
                                st.session_state.editor_raw_metadata = template
                                st.session_state.editor_original_metadata = None
                                st.rerun()

                # ── Editor ────────────────────────────────────────────
                if st.session_state.editor_raw_metadata is not None:
                    # Detect test type for form rendering
                    detected_type = (st.session_state.editor_raw_metadata.get('test_type', '')
                                     or test_type if 'test_type' in dir() else '')

                    # Render the editor widget
                    edited_data = metadata_editor_widget(
                        st.session_state.editor_raw_metadata,
                        test_type=detected_type,
                        key_suffix="editor",
                    )

                    # ── Action buttons ────────────────────────────────
                    st.markdown("---")
                    act_col1, act_col2, act_col3 = st.columns([1, 1, 1])

                    with act_col1:
                        if st.button("Validate", use_container_width=True):
                            is_valid, warnings = validate_raw_metadata(edited_data)
                            if is_valid:
                                st.success("Metadata is valid")
                            else:
                                for w in warnings:
                                    st.warning(w)

                    with act_col2:
                        if st.button("Save Changes", type="primary", use_container_width=True):
                            # Validate first
                            is_valid, warnings = validate_raw_metadata(edited_data)
                            for w in warnings:
                                st.warning(w)

                            # Check if file was modified on disk since load
                            if has_existing and st.session_state.editor_original_metadata is not None:
                                disk_data = load_raw_metadata(test_folder)
                                disk_json = json.dumps(disk_data, sort_keys=True, default=str)
                                orig_json = json.dumps(
                                    st.session_state.editor_original_metadata,
                                    sort_keys=True, default=str)
                                if disk_json != orig_json:
                                    st.warning("File was modified on disk since you loaded it. "
                                               "Saving will overwrite those changes.")

                            # Save
                            try:
                                save_raw_metadata(test_folder, edited_data)
                                st.success(f"Saved metadata.json to {test_folder.name}/")
                                # Update original snapshot
                                st.session_state.editor_original_metadata = json.loads(
                                    json.dumps(edited_data, default=str))
                                st.session_state.editor_raw_metadata = edited_data
                            except Exception as e:
                                st.error(f"Failed to save: {e}")

                    with act_col3:
                        if st.button("Revert to Saved", use_container_width=True):
                            if has_existing:
                                try:
                                    loaded = load_raw_metadata(test_folder)
                                    st.session_state.editor_raw_metadata = loaded
                                    st.session_state.editor_original_metadata = json.loads(
                                        json.dumps(loaded, default=str))
                                    st.rerun()
                                except ValueError as e:
                                    st.error(f"Failed to reload: {e}")
                            else:
                                st.session_state.editor_raw_metadata = None
                                st.session_state.editor_original_metadata = None
                                st.rerun()

else:
    st.info("Select a **Test Root** folder and **Program** in the sidebar to get started.")

    # Show expected structure
    st.markdown("---")
    st.subheader("Getting Started")
    st.markdown("""
    1. Select or enter a **Test Root** folder path in the sidebar
    2. Select a **Test Program** from the dropdown
    3. Browse Systems, Campaigns, and Tests
    4. Create new tests with the ingestion form
    """)

    # Quick create option
    st.subheader("Create New Test Root")
    st.markdown("Don't have a test data folder yet?")

    quick_path = st.text_input(
        "New Test Root Path",
        placeholder="/path/to/create/test_data",
        key="quick_create_path"
    )

    if quick_path and st.button("Create Test Root", type="primary"):
        try:
            Path(quick_path).mkdir(parents=True, exist_ok=True)
            st.session_state.global_test_root = quick_path
            ConfigManager.save_recent_folder(quick_path)
            st.success(f"Created: {quick_path}")
            st.rerun()
        except Exception as e:
            st.error(f"Could not create folder: {e}")
