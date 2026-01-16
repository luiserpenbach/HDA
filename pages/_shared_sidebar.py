"""
Shared Sidebar Component
========================
Global context selector for Test Root and Test Program.
Used across all pages for consistent navigation.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.config_manager import ConfigManager
from pages._shared_styles import apply_custom_styles

# Apply custom styles whenever sidebar is rendered
apply_custom_styles()


def get_programs_for_root(root_path: Path) -> List[str]:
    """Get list of test programs in the root folder."""
    if not root_path.exists():
        return []

    programs = []
    for item in sorted(root_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            programs.append(item.name)

    return programs


def get_systems_for_program(root_path: Path, program: str) -> List[str]:
    """Get list of systems for a test program."""
    program_dir = root_path / program
    if not program_dir.exists():
        return []

    systems = []
    for item in sorted(program_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            systems.append(item.name)

    return systems


def get_campaigns_for_system(root_path: Path, program: str, system: str) -> List[Dict[str, Any]]:
    """Get list of campaigns for a system."""
    system_dir = root_path / program / system
    if not system_dir.exists():
        return []

    campaigns = []
    for item in sorted(system_dir.iterdir()):
        if item.is_dir() and '-' in item.name:
            # Extract campaign ID from folder name (e.g., RCS-C01 -> C01)
            campaign_id = item.name.split('-')[-1]
            campaigns.append({
                'name': item.name,
                'campaign_id': campaign_id,
                'path': str(item)
            })

    return campaigns


def render_global_context(show_recent: bool = True) -> Dict[str, Any]:
    """
    Render the global context selector in the sidebar.

    Returns a dict with:
        - root_path: Path object or None
        - program: str or None
        - is_configured: bool
    """
    # Modern styled header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 0.875rem; font-weight: 600; text-transform: uppercase;
                   letter-spacing: 0.05em; color: #52525b; margin: 0;">
            📁 Test Data
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'global_test_root' not in st.session_state:
        st.session_state.global_test_root = ""
    if 'global_program' not in st.session_state:
        st.session_state.global_program = None

    # Test Root selection
    recent_folders = ConfigManager.get_recent_folders(limit=5) if show_recent else []

    if recent_folders:
        # Show dropdown of recent folders + option to enter new
        folder_options = [""] + recent_folders + ["📝 Enter new path..."]

        selected_option = st.selectbox(
            "Test Root",
            folder_options,
            format_func=lambda x: "-- Select --" if x == "" else ("Enter new path..." if x == "📝 Enter new path..." else x),
            key="global_root_selector",
            index=folder_options.index(st.session_state.global_test_root) if st.session_state.global_test_root in folder_options else 0
        )

        if selected_option == "📝 Enter new path...":
            new_path = st.text_input(
                "Path",
                placeholder="/path/to/test_data",
                key="global_root_input",
                label_visibility="collapsed"
            )
            if new_path:
                st.session_state.global_test_root = new_path
                if Path(new_path).exists():
                    ConfigManager.save_recent_folder(new_path)
        elif selected_option:
            st.session_state.global_test_root = selected_option
    else:
        # No recent folders, just show text input
        root_path = st.text_input(
            "Test Root",
            value=st.session_state.global_test_root,
            placeholder="/path/to/test_data",
            key="global_root_input_only"
        )
        if root_path != st.session_state.global_test_root:
            st.session_state.global_test_root = root_path
            if root_path and Path(root_path).exists():
                ConfigManager.save_recent_folder(root_path)

    # Program selection (only if root is set and valid)
    root_path = None
    programs = []

    if st.session_state.global_test_root:
        root_path = Path(st.session_state.global_test_root)

        if root_path.exists():
            programs = get_programs_for_root(root_path)

            if programs:
                # Reset program if it no longer exists
                if st.session_state.global_program and st.session_state.global_program not in programs:
                    st.session_state.global_program = None

                selected_program = st.selectbox(
                    "Program",
                    [""] + programs,
                    format_func=lambda x: "-- Select --" if x == "" else x,
                    index=(programs.index(st.session_state.global_program) + 1) if st.session_state.global_program in programs else 0,
                    key="global_program_selector"
                )

                if selected_program:
                    st.session_state.global_program = selected_program
                else:
                    st.session_state.global_program = None
            else:
                st.caption("No programs found")
                st.session_state.global_program = None
        else:
            st.warning("Path not found", icon="⚠️")
            st.session_state.global_program = None

    # Return context
    return {
        'root_path': root_path if root_path and root_path.exists() else None,
        'program': st.session_state.global_program,
        'programs': programs,
        'is_configured': bool(root_path and root_path.exists() and st.session_state.global_program)
    }


def render_system_selector(context: Dict[str, Any], key_prefix: str = "") -> Optional[str]:
    """
    Render a system selector based on the global context.

    Args:
        context: Dict from render_global_context()
        key_prefix: Prefix for widget keys to avoid conflicts

    Returns:
        Selected system name or None
    """
    if not context['is_configured']:
        st.info("Select Test Root and Program above")
        return None

    systems = get_systems_for_program(context['root_path'], context['program'])

    if not systems:
        st.info(f"No systems in {context['program']}")
        return None

    selected_system = st.selectbox(
        "System",
        [""] + systems,
        format_func=lambda x: "-- Select --" if x == "" else x,
        key=f"{key_prefix}system_selector"
    )

    return selected_system if selected_system else None


def render_campaign_selector(
    context: Dict[str, Any],
    system: str,
    key_prefix: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Render a campaign selector based on the global context and system.

    Args:
        context: Dict from render_global_context()
        system: Selected system name
        key_prefix: Prefix for widget keys to avoid conflicts

    Returns:
        Selected campaign dict or None
    """
    if not context['is_configured'] or not system:
        return None

    campaigns = get_campaigns_for_system(context['root_path'], context['program'], system)

    if not campaigns:
        st.info(f"No campaigns in {system}")
        return None

    campaign_names = [c['name'] for c in campaigns]

    selected_campaign_name = st.selectbox(
        "Campaign",
        [""] + campaign_names,
        format_func=lambda x: x.split('-')[-1] if x and '-' in x else ("-- Select --" if x == "" else x),
        key=f"{key_prefix}campaign_selector"
    )

    if selected_campaign_name:
        return next((c for c in campaigns if c['name'] == selected_campaign_name), None)

    return None


def render_not_configured_message():
    """Display a message when global context is not configured."""
    st.markdown("""
    <div class="card" style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
        <h3 style="margin: 0 0 0.5rem 0;">No Test Data Selected</h3>
        <p style="margin: 0; color: #71717a;">
            Select a <strong>Test Root</strong> folder and <strong>Program</strong> in the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)


def get_current_context() -> Dict[str, Any]:
    """
    Get the current global context without rendering UI.
    Useful for pages that need to check context state.

    Returns:
        Dict with root_path, program, is_configured
    """
    root_path = None
    if 'global_test_root' in st.session_state and st.session_state.global_test_root:
        root_path = Path(st.session_state.global_test_root)
        if not root_path.exists():
            root_path = None

    program = st.session_state.get('global_program', None)

    return {
        'root_path': root_path,
        'program': program,
        'is_configured': bool(root_path and program)
    }
