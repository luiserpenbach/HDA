"""
New Test Page
=============
Browse test data structure and ingest new tests with metadata.

Features:
- View test data file structure (programs, systems, campaigns, tests)
- Ingest new test with metadata file or manual input
- Create standard test subfolder structure
- Suggest next available Test ID

Folder Hierarchy:
    TEST_ROOT / TEST_PROGRAM / SYSTEM / SYSTEM-TESTTYPE / SYSTEM-TESTTYPE-CAMPAIGN / TEST_ID

Test ID Naming Scheme: SYSTEM-TESTTYPE-CAMPAIGN-#
Example: RCS-CF-C01-001
Note: Test Program is NOT included in Test ID to keep it concise.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import os
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Import core modules
from core.test_metadata import (
    TestMetadata, TEST_SUBFOLDERS, create_test_folder,
    list_campaigns, list_tests_in_campaign, load_test_metadata
)
from core.config_manager import ConfigManager

st.set_page_config(page_title="New Test", page_icon="NT", layout="wide")

st.title("New Test")
st.markdown("Browse test data structure and ingest new tests")

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'test_root_path' not in st.session_state:
    st.session_state.test_root_path = ""
if 'selected_program' not in st.session_state:
    st.session_state.selected_program = None
if 'selected_system' not in st.session_state:
    st.session_state.selected_system = None
if 'selected_campaign' not in st.session_state:
    st.session_state.selected_campaign = None


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

                    for test_type_dir in system_dir.iterdir():
                        if test_type_dir.is_dir() and '-' in test_type_dir.name:
                            for campaign_dir in test_type_dir.iterdir():
                                if campaign_dir.is_dir():
                                    campaigns += 1
                                    # Count tests in campaign
                                    for test_dir in campaign_dir.iterdir():
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
            # Count test types and campaigns
            test_types = []
            campaigns = 0
            tests = 0

            for test_type_dir in system_dir.iterdir():
                if test_type_dir.is_dir() and '-' in test_type_dir.name:
                    test_types.append(test_type_dir.name.split('-')[-1])

                    for campaign_dir in test_type_dir.iterdir():
                        if campaign_dir.is_dir():
                            campaigns += 1
                            tests += sum(1 for t in campaign_dir.iterdir()
                                        if t.is_dir() and not t.name.startswith('.'))

            systems.append({
                'name': system_dir.name,
                'path': str(system_dir),
                'test_types': list(set(test_types)),
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

    for test_type_dir in sorted(system_dir.iterdir()):
        if test_type_dir.is_dir() and '-' in test_type_dir.name:
            test_type = test_type_dir.name.split('-')[-1]

            for campaign_dir in sorted(test_type_dir.iterdir()):
                if campaign_dir.is_dir():
                    # Count tests
                    test_count = sum(1 for t in campaign_dir.iterdir()
                                    if t.is_dir() and not t.name.startswith('.'))

                    campaigns.append({
                        'name': campaign_dir.name,
                        'path': str(campaign_dir),
                        'test_type': test_type,
                        'test_count': test_count
                    })

    return campaigns


def get_tests_for_campaign(campaign_path: Path) -> List[Dict[str, Any]]:
    """Get all tests in a campaign folder."""
    tests = []

    if not campaign_path.exists():
        return tests

    for test_dir in sorted(campaign_path.iterdir()):
        if test_dir.is_dir() and not test_dir.name.startswith('.'):
            test_info = {
                'test_id': test_dir.name,
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


def get_next_test_id(campaign_path: Path, system: str, test_type: str, campaign_id: str) -> str:
    """
    Get the next available Test ID for a campaign.

    Format: SYSTEM-TESTTYPE-CAMPAIGN-NNN
    Example: RCS-CF-C01-001
    """
    if not campaign_path.exists():
        return f"{system}-{test_type}-{campaign_id}-001"

    # Get existing test numbers
    existing_numbers = []
    for test_dir in campaign_path.iterdir():
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

    return f"{system}-{test_type}-{campaign_id}-{next_num:03d}"


def create_new_test(
    root_path: Path,
    program: str,
    system: str,
    test_type: str,
    campaign_id: str,
    test_id: str,
    metadata: Optional[TestMetadata] = None
) -> Tuple[bool, str, Optional[Path]]:
    """
    Create a new test folder with standard structure.

    Args:
        root_path: Test root directory
        program: Test program name (e.g., "Engine-A", "Hopper-Dev")
        system: System name (e.g., "RCS", "MAIN")
        test_type: Test type code (e.g., "CF", "HF")
        campaign_id: Campaign ID (e.g., "C01")
        test_id: Full test ID (e.g., "RCS-CF-C01-001")
        metadata: Optional TestMetadata object

    Returns: (success, message, test_folder_path)
    """
    try:
        # Build path: root / program / system / system-testtype / system-testtype-campaign / test_id
        test_type_folder = f"{system}-{test_type}"
        campaign_folder = f"{system}-{test_type}-{campaign_id}"
        test_folder = root_path / program / system / test_type_folder / campaign_folder / test_id

        if test_folder.exists():
            return False, f"Test folder already exists: {test_id}", None

        # Create folder structure
        test_folder.mkdir(parents=True, exist_ok=True)

        for subfolder in TEST_SUBFOLDERS:
            (test_folder / subfolder).mkdir(exist_ok=True)

        # Save metadata if provided
        if metadata:
            metadata.test_id = test_id
            metadata.system = system
            metadata.test_type = test_type
            metadata.campaign_id = campaign_id
            metadata.save(test_folder)
        else:
            # Create minimal metadata
            min_metadata = TestMetadata.from_test_id(test_id)
            min_metadata.save(test_folder)

        return True, f"Created test folder: {test_id}", test_folder

    except Exception as e:
        return False, f"Error creating test: {str(e)}", None


# =============================================================================
# MAIN PAGE LAYOUT
# =============================================================================

# Section 1: Test Root Path
st.header("1. Test Data Location")

col1, col2 = st.columns([3, 1])

with col1:
    # Recent folders
    recent_folders = ConfigManager.get_recent_folders(limit=5)

    if recent_folders:
        folder_source = st.radio(
            "Select folder",
            ["Enter Path", "Recent Locations"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if folder_source == "Recent Locations":
            test_root_path = st.selectbox(
                "Recent Test Roots",
                [""] + recent_folders,
                format_func=lambda x: x if x else "-- Select --"
            )
        else:
            test_root_path = st.text_input(
                "Test Root Folder",
                value=st.session_state.test_root_path,
                placeholder="/path/to/test_data",
                help="Root folder containing test data (e.g., /data/tests)"
            )
    else:
        test_root_path = st.text_input(
            "Test Root Folder",
            value=st.session_state.test_root_path,
            placeholder="/path/to/test_data",
            help="Root folder containing test data (e.g., /data/tests)"
        )

with col2:
    if st.button("Load", type="primary", use_container_width=True):
        if test_root_path:
            st.session_state.test_root_path = test_root_path
            if Path(test_root_path).exists():
                ConfigManager.save_recent_folder(test_root_path)
            st.rerun()

# Show expected structure
with st.expander("Expected Folder Structure"):
    st.code("""
TEST_ROOT/
    TEST_PROGRAM/                        # e.g., Engine-A, Hopper-Dev
        SYSTEM/                          # e.g., RCS, MAIN
            SYSTEM-TESTTYPE/             # e.g., RCS-CF, RCS-HF
                SYSTEM-TESTTYPE-CAMPAIGN/   # e.g., RCS-CF-C01
                    TEST_ID/                # e.g., RCS-CF-C01-001
                        config/
                        logs/
                        media/
                        plots/
                        processed_data/
                        raw_data/
                        reports/
                        metadata.json

Note: Test Program is for organization only - NOT included in Test IDs.
    """, language="text")

st.divider()

# =============================================================================
# SECTION 2: BROWSE STRUCTURE
# =============================================================================

if st.session_state.test_root_path:
    root_path = Path(st.session_state.test_root_path)

    if not root_path.exists():
        st.warning(f"Path does not exist: {root_path}")
        st.info("Would you like to create this folder as a new test root?")
        if st.button("Create Test Root Folder"):
            root_path.mkdir(parents=True, exist_ok=True)
            st.success(f"Created: {root_path}")
            st.rerun()
    else:
        st.header("2. Browse Test Structure")

        # Scan root
        structure = scan_test_root(root_path)

        # Summary metrics
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

        # Four-panel browser
        browse_col1, browse_col2, browse_col3, browse_col4 = st.columns(4)

        # Panel 1: Programs
        with browse_col1:
            st.subheader("Programs")

            if structure['programs']:
                programs_df = pd.DataFrame([
                    {
                        'Program': prog,
                        'Systems': structure['program_info'][prog]['system_count'],
                        'Tests': structure['program_info'][prog]['test_count']
                    }
                    for prog in structure['programs']
                ])

                # Make selectable
                selected_program = st.selectbox(
                    "Select Program",
                    [""] + structure['programs'],
                    format_func=lambda x: x if x else "-- Select --",
                    key="program_selector"
                )

                if selected_program:
                    st.session_state.selected_program = selected_program
                    # Reset downstream selections when program changes
                    if st.session_state.get('_last_program') != selected_program:
                        st.session_state.selected_system = None
                        st.session_state.selected_campaign = None
                        st.session_state._last_program = selected_program

                st.dataframe(programs_df, use_container_width=True, hide_index=True)
            else:
                st.info("No programs found")
                st.caption("Create a new test to get started")

        # Panel 2: Systems
        with browse_col2:
            st.subheader("Systems")

            if st.session_state.selected_program:
                systems = get_systems_for_program(root_path, st.session_state.selected_program)

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
                    st.info(f"No systems in {st.session_state.selected_program}")
            else:
                st.info("Select a program first")

        # Panel 3: Campaigns
        with browse_col3:
            st.subheader("Campaigns")

            if st.session_state.selected_program and st.session_state.selected_system:
                campaigns = get_campaigns_for_system(
                    root_path,
                    st.session_state.selected_program,
                    st.session_state.selected_system
                )

                if campaigns:
                    campaigns_df = pd.DataFrame([
                        {
                            'Campaign': c['name'],
                            'Type': c['test_type'],
                            'Tests': c['test_count']
                        }
                        for c in campaigns
                    ])

                    selected_campaign = st.selectbox(
                        "Select Campaign",
                        [""] + [c['name'] for c in campaigns],
                        format_func=lambda x: x if x else "-- Select --",
                        key="campaign_selector"
                    )

                    if selected_campaign:
                        st.session_state.selected_campaign = next(
                            c for c in campaigns if c['name'] == selected_campaign
                        )

                    st.dataframe(campaigns_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No campaigns in {st.session_state.selected_system}")
            else:
                st.info("Select a system first")

        # Panel 4: Tests
        with browse_col4:
            st.subheader("Tests")

            if st.session_state.selected_campaign:
                campaign = st.session_state.selected_campaign
                campaign_path = Path(campaign['path'])
                tests = get_tests_for_campaign(campaign_path)

                if tests:
                    tests_df = pd.DataFrame([
                        {
                            'Test ID': t['test_id'],
                            'Status': t.get('status', 'unknown'),
                            'Data': 'Y' if t['has_raw_data'] else 'N'
                        }
                        for t in tests
                    ])

                    st.dataframe(tests_df, use_container_width=True, hide_index=True)

                    # Quick action: Open test in Single Test Analysis
                    selected_test = st.selectbox(
                        "Quick Open",
                        [""] + [t['test_id'] for t in tests],
                        format_func=lambda x: x if x else "-- Select test --"
                    )

                    if selected_test:
                        test_info = next(t for t in tests if t['test_id'] == selected_test)
                        st.caption(f"Path: {test_info['path']}")

                        if st.button("Open in Analysis", use_container_width=True):
                            # Save to session state for Single Test Analysis page
                            st.session_state.test_folder_path = test_info['path']
                            st.info("Test path saved. Navigate to 'Single Test Analysis' page.")
                else:
                    st.info(f"No tests in {campaign['name']}")
            else:
                st.info("Select a campaign first")

        st.divider()

        # =============================================================================
        # SECTION 3: INGEST NEW TEST
        # =============================================================================

        st.header("3. Ingest New Test")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Test Location")

            # Program selection (from existing or new)
            existing_programs = structure['programs']
            program_option = st.radio(
                "Test Program",
                ["Use Existing", "Create New"] if existing_programs else ["Create New"],
                horizontal=True,
                key="program_option"
            )

            if program_option == "Use Existing" and existing_programs:
                new_program = st.selectbox("Select Program", existing_programs, key="new_program_select")
            else:
                new_program = st.text_input("New Program Name", placeholder="e.g., Engine-A, Hopper-Dev")

            # System selection (from existing within program or new)
            if new_program and st.session_state.selected_program == new_program:
                existing_systems = [s['name'] for s in get_systems_for_program(root_path, new_program)]
            else:
                existing_systems = []

            system_option = st.radio(
                "System",
                ["Use Existing", "Create New"] if existing_systems else ["Create New"],
                horizontal=True,
                key="system_option"
            )

            if system_option == "Use Existing" and existing_systems:
                new_system = st.selectbox("Select System", existing_systems, key="new_system_select")
            else:
                new_system = st.text_input("New System Name", placeholder="e.g., RCS, MAIN")

            # Test Type
            test_type_options = ["CF", "HF", "LK", "PR"]  # Cold Flow, Hot Fire, Leak, Pressure
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

            # Campaign (from existing or new)
            if new_program and new_system and st.session_state.selected_program == new_program:
                campaigns = get_campaigns_for_system(root_path, new_program, new_system)
                matching_campaigns = [c for c in campaigns if c['test_type'] == new_test_type]

                campaign_option = st.radio(
                    "Campaign",
                    ["Use Existing", "Create New"] if matching_campaigns else ["Create New"],
                    horizontal=True,
                    key="campaign_option"
                )

                if campaign_option == "Use Existing" and matching_campaigns:
                    selected = st.selectbox(
                        "Select Campaign",
                        [c['name'] for c in matching_campaigns]
                    )
                    new_campaign_id = selected.split('-')[-1] if selected else "C01"
                else:
                    new_campaign_id = st.text_input("New Campaign ID", value="C01", placeholder="e.g., C01, C02")
            else:
                new_campaign_id = st.text_input("Campaign ID", value="C01", placeholder="e.g., C01, C02")

            # Suggested Test ID (does NOT include program)
            if new_program and new_system and new_test_type and new_campaign_id:
                campaign_folder = f"{new_system}-{new_test_type}-{new_campaign_id}"
                campaign_path = root_path / new_program / new_system / f"{new_system}-{new_test_type}" / campaign_folder

                suggested_id = get_next_test_id(campaign_path, new_system, new_test_type, new_campaign_id)

                st.info(f"Suggested Test ID: **{suggested_id}**")
                st.caption("(Test ID does not include Program name)")

                use_suggested = st.checkbox("Use suggested ID", value=True)
                if use_suggested:
                    new_test_id = suggested_id
                else:
                    new_test_id = st.text_input("Custom Test ID", value=suggested_id)
            else:
                new_test_id = ""
                st.warning("Fill in Program, System, Test Type, and Campaign to generate Test ID")

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
                        ["", "Water", "Nitrogen", "Helium", "Air", "Ethanol", "IPA"]
                    )
                else:
                    meta_fluid = ""

                meta_notes = st.text_area("Notes", height=68, placeholder="Any relevant notes...")

                # Build metadata
                new_metadata = TestMetadata(
                    test_id=new_test_id,
                    system=new_system if new_system else "",
                    test_type=new_test_type,
                    campaign_id=new_campaign_id,
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

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            create_enabled = bool(new_program and new_system and new_test_type and new_campaign_id and new_test_id)

            if st.button(
                "Create New Test",
                type="primary",
                use_container_width=True,
                disabled=not create_enabled
            ):
                success, message, test_folder = create_new_test(
                    root_path,
                    new_program,
                    new_system,
                    new_test_type,
                    new_campaign_id,
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
{new_program}/
  {new_system}/
    {new_system}-{new_test_type}/
      {new_system}-{new_test_type}-{new_campaign_id}/
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
                    # Save root path (go up 5 levels: test -> campaign -> type -> system -> program -> root)
                    ConfigManager.save_recent_folder(str(test_folder.parent.parent.parent.parent.parent))

                    # Option to continue
                    st.info("Test created. You can now:")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Create Another Test", use_container_width=True):
                            st.rerun()
                    with col2:
                        st.caption("Or navigate to 'Single Test Analysis' to analyze this test")
                else:
                    st.error(message)

else:
    st.info("Enter a test root folder path to browse and manage tests")

    # Quick create option
    st.markdown("---")
    st.subheader("Quick Start")
    st.markdown("Don't have a test data folder yet? Create one:")

    quick_path = st.text_input(
        "New Test Root Path",
        placeholder="/path/to/create/test_data",
        key="quick_create_path"
    )

    if quick_path and st.button("Create Test Root", type="primary"):
        try:
            Path(quick_path).mkdir(parents=True, exist_ok=True)
            st.session_state.test_root_path = quick_path
            ConfigManager.save_recent_folder(quick_path)
            st.success(f"Created: {quick_path}")
            st.rerun()
        except Exception as e:
            st.error(f"Could not create folder: {e}")
