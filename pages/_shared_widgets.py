"""
Shared Widgets for Streamlit Pages
===================================
Reusable UI components to reduce duplication across pages.

This module provides common widgets used across multiple pages:
- Campaign selection and info display
- Configuration management (upload, edit, template selection)
- Export options (CSV, Excel, JSON, HTML reports)
- Steady-state detection method selection
"""

import streamlit as st
import json
import pandas as pd
import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from pages._shared_styles import apply_custom_styles, render_status_badge

# Apply custom styles
apply_custom_styles()


# =============================================================================
# Campaign Selection Widgets
# =============================================================================

def campaign_selector_widget(
    show_info: bool = True,
    show_test_count: bool = True,
    key_suffix: str = ""
) -> Optional[str]:
    """
    Reusable campaign selection widget with optional info display.

    Args:
        show_info: Whether to display campaign type and metadata
        show_test_count: Whether to display test count
        key_suffix: Suffix for widget keys to avoid collisions

    Returns:
        Selected campaign name, or None if no campaigns available

    Usage:
        campaign = campaign_selector_widget()
        if campaign:
            df = get_campaign_data(campaign)
    """
    from core.campaign_manager_v2 import get_available_campaigns

    campaigns = get_available_campaigns()

    if not campaigns:
        st.warning("No campaigns found. Create a campaign in Campaign Management.")
        return None

    campaign_names = [c['name'] for c in campaigns]
    selected_campaign = st.selectbox(
        "Select Campaign",
        campaign_names,
        key=f"campaign_selector_{key_suffix}"
    )

    if show_info and selected_campaign:
        for c in campaigns:
            if c['name'] == selected_campaign:
                st.caption(f"Type: {c.get('type', 'unknown')}")
                if show_test_count:
                    st.caption(f"Tests: {c.get('test_count', 0)}")
                if c.get('created_date'):
                    st.caption(f"Created: {c['created_date'][:10]}")
                break

    return selected_campaign


def campaign_info_display(campaign_name: str, compact: bool = False):
    """
    Display campaign information in a formatted way.

    Args:
        campaign_name: Name of the campaign
        compact: If True, show compact view; otherwise show detailed metrics
    """
    from core.campaign_manager_v2 import get_campaign_info, get_campaign_data

    try:
        info = get_campaign_info(campaign_name)
        df = get_campaign_data(campaign_name)

        if compact:
            st.markdown(f"""
            <div class="card" style="padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="margin: 0; font-size: 0.75rem; color: #71717a;">Campaign</p>
                        <p style="margin: 0; font-size: 1rem; font-weight: 600;">{campaign_name}</p>
                    </div>
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 0.75rem; color: #71717a;">Type</p>
                        <p style="margin: 0; font-size: 1rem; font-weight: 600;">{info.get('campaign_type', 'Unknown')}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; font-size: 0.75rem; color: #71717a;">Tests</p>
                        <p style="margin: 0; font-size: 1rem; font-weight: 600;">{len(df)}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Detailed view
            st.markdown(f"""
            <div class="card-elevated" style="margin-bottom: 1.5rem;">
                <h3 style="margin: 0 0 1rem 0;">Campaign: {campaign_name}</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                    <div>
                        <p style="margin: 0; font-size: 0.75rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Test Type</p>
                        <p style="margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: 700;">{info.get('campaign_type', 'Unknown')}</p>
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.75rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Total Tests</p>
                        <p style="margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: 700;">{len(df)}</p>
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.75rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Created</p>
                        <p style="margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: 700;">{info.get('created_date', 'Unknown')[:10] if info.get('created_date') else 'Unknown'}</p>
                    </div>
                </div>
                {f'<p style="margin: 1rem 0 0 0; padding-top: 1rem; border-top: 1px solid #e4e4e7; font-size: 0.875rem; color: #71717a;"><strong>Description:</strong> {info["description"]}</p>' if info.get('description') else ''}
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading campaign info: {e}")


# =============================================================================
# Configuration Management Widgets
# =============================================================================

def config_source_selector(
    test_type: str = "cold_flow",
    key_suffix: str = ""
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Widget for selecting configuration source and loading config.

    Args:
        test_type: Type of test (cold_flow or hot_fire)
        key_suffix: Suffix for widget keys to avoid collisions

    Returns:
        Tuple of (config_source, config_dict)

    Usage:
        source, config = config_source_selector("cold_flow")
        if config:
            # Use config for analysis
    """
    from core.config_manager import ConfigManager

    config_source = st.radio(
        "Configuration Source",
        ["Use Template", "Upload JSON", "Use Default", "Manual Edit"],
        key=f"config_source_{key_suffix}"
    )

    config = None

    if config_source == "Use Template":
        config = saved_config_selector_widget(test_type, key_suffix)

    elif config_source == "Upload JSON":
        config = config_file_uploader_widget(key_suffix)

    elif config_source == "Use Default":
        config = ConfigManager.get_default_config(test_type)
        st.success(f"Loaded default {test_type} configuration")

    elif config_source == "Manual Edit":
        default_config = ConfigManager.get_default_config(test_type)
        config = config_text_editor_widget(default_config, key_suffix)

    return config_source, config


def saved_config_selector_widget(
    test_type: Optional[str] = None,
    key_suffix: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Widget for selecting and loading a configuration template.

    Args:
        test_type: Filter templates by test type (optional)
        key_suffix: Suffix for widget keys

    Returns:
        Configuration dict from selected template, or None
    """
    from core.saved_configs import TemplateManager, create_config_from_template

    try:
        manager = TemplateManager()
        templates = manager.list_templates()

        if not templates:
            st.warning("No templates available. Create templates in Config Templates page.")
            return None

        # Filter by test type if specified
        if test_type:
            templates = [t for t in templates if t.get('test_type') == test_type]
            if not templates:
                st.warning(f"No templates found for test type: {test_type}")
                return None

        # Create template selection options
        template_options = [
            f"{t['id']} - {t.get('config_name', 'Unnamed')}"
            for t in templates
        ]

        selected = st.selectbox(
            "Select Template",
            template_options,
            key=f"template_selector_{key_suffix}"
        )

        if selected:
            # Extract template ID from selection
            template_id = selected.split(" - ")[0]

            # Show template info
            template_info = next(t for t in templates if t['id'] == template_id)
            with st.expander("Template Info"):
                st.caption(f"**Description:** {template_info.get('description', 'N/A')}")
                st.caption(f"**Version:** {template_info.get('version', 'N/A')}")
                if 'tags' in template_info and template_info['tags']:
                    st.caption(f"**Tags:** {', '.join(template_info['tags'])}")

            # Load template
            if st.button("Load Template", key=f"load_template_{key_suffix}"):
                config = create_config_from_template(template_id)
                st.success(f"Template '{template_id}' loaded successfully")
                return config

        return None

    except Exception as e:
        st.error(f"Error loading templates: {e}")
        return None


def config_file_uploader_widget(key_suffix: str = "") -> Optional[Dict[str, Any]]:
    """
    Widget for uploading configuration JSON file.

    Args:
        key_suffix: Suffix for widget keys

    Returns:
        Configuration dict from uploaded file, or None
    """
    config_file = st.file_uploader(
        "Upload Configuration JSON",
        type=['json'],
        key=f"config_uploader_{key_suffix}"
    )

    if config_file:
        try:
            config = json.load(config_file)
            st.success("Configuration loaded successfully")

            # Display basic info
            with st.expander("Configuration Summary"):
                st.json({
                    'config_name': config.get('config_name', 'N/A'),
                    'test_type': config.get('test_type', 'N/A'),
                    'version': config.get('version', 'N/A')
                })

            return config

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")
            return None
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return None

    return None


def config_text_editor_widget(
    initial_config: Dict[str, Any],
    key_suffix: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Widget for manual JSON configuration editing.

    Args:
        initial_config: Initial configuration to display
        key_suffix: Suffix for widget keys

    Returns:
        Edited configuration dict, or None if invalid
    """
    with st.expander("Edit Configuration JSON", expanded=True):
        config_str = st.text_area(
            "Configuration JSON",
            value=json.dumps(initial_config, indent=2),
            height=400,
            key=f"config_editor_{key_suffix}"
        )

        if st.button("Validate & Apply", key=f"apply_config_{key_suffix}"):
            try:
                config = json.loads(config_str)
                st.success("Configuration is valid and applied")
                return config
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return None

    # Return initial config if not modified
    return initial_config


def config_quick_info(config: Dict[str, Any], expanded: bool = False):
    """
    Display quick summary of configuration.

    Args:
        config: Configuration dictionary
        expanded: Whether to show expanded view
    """
    with st.expander("Configuration Info", expanded=expanded):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**Name:** {config.get('config_name', 'N/A')}")
        with col2:
            st.caption(f"**Type:** {config.get('test_type', 'N/A')}")
        with col3:
            st.caption(f"**Version:** {config.get('version', 'N/A')}")

        if expanded:
            st.caption(f"**Fluid:** {config.get('fluid', {}).get('name', 'N/A')}")
            st.caption(f"**Sensors:** {len(config.get('columns', {}))}")
            st.caption(f"**Uncertainties:** {len(config.get('uncertainties', {}))}")


# =============================================================================
# Export Widgets
# =============================================================================

def export_panel_widget(
    data: pd.DataFrame,
    base_filename: str,
    export_formats: List[str] = ["CSV", "Excel", "JSON"],
    metadata: Optional[Dict[str, Any]] = None,
    key_suffix: str = ""
):
    """
    Unified export panel with multiple format options.

    Args:
        data: DataFrame to export
        base_filename: Base filename (without extension)
        export_formats: List of formats to offer (CSV, Excel, JSON)
        metadata: Optional metadata to include in exports
        key_suffix: Suffix for widget keys

    Usage:
        export_panel_widget(df, "test_results_001", ["CSV", "Excel", "JSON"])
    """
    st.subheader("Export Options")

    export_format = st.selectbox(
        "Select Export Format",
        export_formats,
        key=f"export_format_{key_suffix}"
    )

    # CSV Export
    if export_format == "CSV":
        csv_data = data.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_data,
            file_name=f"{base_filename}.csv",
            mime="text/csv",
            key=f"download_csv_{key_suffix}"
        )
        st.caption(f"Size: {len(csv_data)} bytes | Rows: {len(data)}")

    # Excel Export
    elif export_format == "Excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Data', index=False)

            # Add metadata sheet if provided
            if metadata:
                meta_df = pd.DataFrame([metadata])
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)

        st.download_button(
            "Download Excel",
            buffer.getvalue(),
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_excel_{key_suffix}"
        )
        st.caption(f"Sheets: Data{' + Metadata' if metadata else ''} | Rows: {len(data)}")

    # JSON Export
    elif export_format == "JSON":
        export_data = {
            'export_date': datetime.now().isoformat(),
            'row_count': len(data),
            'columns': list(data.columns),
            'data': data.to_dict(orient='records')
        }

        if metadata:
            export_data['metadata'] = metadata

        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            "Download JSON",
            json_str,
            file_name=f"{base_filename}.json",
            mime="application/json",
            key=f"download_json_{key_suffix}"
        )
        st.caption(f"Size: {len(json_str)} bytes | Rows: {len(data)}")


def html_report_button(
    test_id: str,
    test_type: str,
    measurements: Dict[str, Any],
    traceability: Dict[str, Any],
    qc_report: Dict[str, Any],
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    key_suffix: str = ""
):
    """
    Button to generate and download HTML report.

    Args:
        test_id: Test identifier
        test_type: Type of test (cold_flow, hot_fire)
        measurements: Dictionary of measurements with uncertainties
        traceability: Traceability record
        qc_report: QC check results
        config: Test configuration
        metadata: Optional metadata (part numbers, etc.)
        key_suffix: Suffix for widget key
    """
    from core.reporting import generate_test_report

    if st.button("Generate HTML Report", key=f"gen_html_report_{key_suffix}"):
        try:
            with st.spinner("Generating report..."):
                html = generate_test_report(
                    test_id=test_id,
                    test_type=test_type,
                    measurements=measurements,
                    traceability=traceability,
                    qc_report=qc_report,
                    metadata=metadata or {},
                    config=config,
                    include_config_snapshot=True,
                )

                st.download_button(
                    "Download HTML Report",
                    html,
                    file_name=f"{test_id}_report.html",
                    mime="text/html",
                    key=f"download_html_{key_suffix}"
                )
                st.success("Report generated successfully")

        except Exception as e:
            st.error(f"Error generating report: {e}")


# =============================================================================
# Utility Widgets
# =============================================================================

def test_id_input_widget(
    default_prefix: str = "TEST",
    auto_generate: bool = True,
    key_suffix: str = ""
) -> str:
    """
    Widget for entering or auto-generating test ID.

    Args:
        default_prefix: Prefix for auto-generated IDs
        auto_generate: Whether to show auto-generate option
        key_suffix: Suffix for widget keys

    Returns:
        Test ID string
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        test_id = st.text_input(
            "Test ID",
            value="",
            placeholder=f"e.g., {default_prefix}_001",
            key=f"test_id_input_{key_suffix}"
        )

    with col2:
        if auto_generate and st.button("Auto-Generate", key=f"auto_gen_{key_suffix}"):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_id = f"{default_prefix}_{timestamp}"
            st.session_state[f"test_id_input_{key_suffix}"] = test_id
            st.rerun()

    return test_id or f"{default_prefix}_UNNAMED"


def metadata_input_widget(key_suffix: str = "") -> Dict[str, str]:
    """
    Widget for entering test metadata (part numbers, serial numbers, etc.).

    Args:
        key_suffix: Suffix for widget keys

    Returns:
        Dictionary of metadata fields
    """
    with st.expander("Test Metadata (Optional)"):
        col1, col2 = st.columns(2)

        with col1:
            part_number = st.text_input(
                "Part Number",
                key=f"part_number_{key_suffix}"
            )
            serial_number = st.text_input(
                "Serial Number",
                key=f"serial_number_{key_suffix}"
            )

        with col2:
            analyst = st.text_input(
                "Analyst",
                value="",
                key=f"analyst_{key_suffix}"
            )
            notes = st.text_area(
                "Notes",
                height=100,
                key=f"notes_{key_suffix}"
            )

    return {
        'part_number': part_number,
        'serial_number': serial_number,
        'analyst': analyst,
        'notes': notes
    }


def success_with_next_steps(message: str, next_steps: List[str]):
    """
    Display success message with recommended next steps.

    Args:
        message: Success message to display
        next_steps: List of recommended next actions
    """
    st.markdown(f"""
    <div style="background: #dcfce7; border-left: 4px solid #16a34a; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <p style="margin: 0; color: #16a34a; font-weight: 600;">✓ {message}</p>
    </div>
    """, unsafe_allow_html=True)

    if next_steps:
        st.markdown("""
        <div style="background: #dbeafe; border-left: 4px solid #2563eb; border-radius: 0.5rem; padding: 1rem;">
            <p style="margin: 0 0 0.75rem 0; color: #2563eb; font-weight: 600;">💡 Recommended next steps:</p>
        """, unsafe_allow_html=True)

        for i, step in enumerate(next_steps, 1):
            st.markdown(f"""
            <div style="display: flex; align-items: start; margin-bottom: 0.5rem;">
                <span style="background: #2563eb; color: white; width: 20px; height: 20px; border-radius: 50%;
                            display: flex; align-items: center; justify-content: center; font-size: 0.75rem;
                            font-weight: 600; margin-right: 0.5rem; flex-shrink: 0;">{i}</span>
                <p style="margin: 0; font-size: 0.875rem; color: #1e40af;">{step}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Phase 2 Feature Widgets
# =============================================================================

def data_upload_widget(
    label: str = "Upload Test Data",
    accept_types: List[str] = ['csv'],
    key_suffix: str = "",
    show_preview: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Reusable data upload widget with preview and column info.

    Args:
        label: Upload button label
        accept_types: Accepted file types
        key_suffix: Suffix for widget keys
        show_preview: Whether to show data preview

    Returns:
        DataFrame or None
    """
    uploaded = st.file_uploader(label, type=accept_types,
                                key=f"data_upload_{key_suffix}")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if show_preview:
                with st.expander(f"Data Preview ({len(df)} rows, {len(df.columns)} cols)", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None


def channel_selector_widget(
    df: pd.DataFrame,
    label: str = "Select Channel",
    numeric_only: bool = True,
    exclude_time: bool = True,
    key_suffix: str = "",
) -> Optional[str]:
    """
    Select a data channel/column from a DataFrame.

    Args:
        df: DataFrame with columns to select from
        label: Widget label
        numeric_only: Only show numeric columns
        exclude_time: Exclude common time columns
        key_suffix: Unique key suffix

    Returns:
        Selected column name or None
    """
    cols = list(df.columns)
    if numeric_only:
        cols = [c for c in cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    if exclude_time:
        time_names = {'time', 'time_s', 'time_ms', 'timestamp', 'Time', 'TIME'}
        cols = [c for c in cols if c not in time_names]

    if not cols:
        st.warning("No suitable columns found in data.")
        return None

    return st.selectbox(label, cols, key=f"channel_{key_suffix}")


def time_column_selector_widget(
    df: pd.DataFrame,
    key_suffix: str = "",
) -> str:
    """
    Select the time column from a DataFrame, with auto-detection.

    Returns:
        Selected time column name
    """
    time_candidates = [c for c in df.columns if c.lower() in
                       {'time', 'time_s', 'time_ms', 'timestamp', 't'}]

    if time_candidates:
        default_idx = 0
    else:
        time_candidates = list(df.columns)
        default_idx = 0

    return st.selectbox("Time Column", time_candidates, index=default_idx,
                        key=f"time_col_{key_suffix}")


def render_phase_timeline(phases: list):
    """
    Render a visual phase timeline using HTML/CSS.

    Args:
        phases: List of PhaseResult objects
    """
    if not phases:
        return

    total_duration = sum(getattr(p, 'duration_s', p.get('duration_s', 0))
                         if isinstance(p, dict) else p.duration_s for p in phases)
    if total_duration <= 0:
        return

    phase_colors = {
        'pretest': '#94a3b8', 'startup': '#f59e0b', 'transient': '#f97316',
        'steady_state': '#22c55e', 'shutdown': '#ef4444', 'cooldown': '#6366f1',
    }

    bars = []
    for p in phases:
        if hasattr(p, 'phase'):
            name = p.phase.value if hasattr(p.phase, 'value') else str(p.phase)
            dur = p.duration_s
        else:
            name = p.get('phase', 'unknown')
            dur = p.get('duration_s', 0)

        pct = (dur / total_duration * 100) if total_duration > 0 else 0
        color = phase_colors.get(name, '#71717a')
        bars.append(
            f'<div style="width:{pct:.1f}%; background:{color}; height:36px; '
            f'display:flex; align-items:center; justify-content:center; '
            f'color:white; font-size:0.7rem; font-weight:600; '
            f'border-radius:3px; min-width:40px;" '
            f'title="{name}: {dur:.3f}s">{name.replace("_"," ").title()}</div>'
        )

    st.markdown(
        '<div style="display:flex; gap:2px; margin:0.5rem 0 1rem 0; border-radius:6px; overflow:hidden;">'
        + ''.join(bars) + '</div>',
        unsafe_allow_html=True
    )


def metadata_editor_widget(
    raw_metadata: Dict[str, Any],
    test_type: str = "",
    key_suffix: str = "",
    readonly: bool = False,
) -> Dict[str, Any]:
    """
    Structured metadata editor with form and raw JSON views.

    Operates on raw metadata dicts (not the TestMetadata dataclass) to
    preserve nested sections and unknown fields during editing.

    Args:
        raw_metadata: The raw metadata dict loaded from JSON
        test_type: "CF", "HF", etc. (controls which fluid fields to show)
        key_suffix: Suffix for widget keys to avoid collisions
        readonly: If True, display only (no editing)

    Returns:
        Modified metadata dict with edits applied
    """
    import copy
    data = copy.deepcopy(raw_metadata)

    # Track which top-level keys are handled by form groups
    handled_keys = set()

    # Determine edit mode
    edit_mode = st.radio(
        "Edit Mode",
        ["Form", "JSON"],
        horizontal=True,
        key=f"meta_edit_mode_{key_suffix}",
        disabled=readonly,
    )

    if edit_mode == "JSON":
        return _metadata_json_editor(data, key_suffix, readonly)

    # ── Form View ────────────────────────────────────────────────────

    # Group 1: Test Identity
    with st.expander("Test Identity", expanded=True):
        id_cols = st.columns(3)
        with id_cols[0]:
            data['test_id'] = st.text_input(
                "Test ID", value=data.get('test_id', ''),
                key=f"me_test_id_{key_suffix}", disabled=True)
        with id_cols[1]:
            data['system'] = st.text_input(
                "System", value=data.get('system', ''),
                key=f"me_system_{key_suffix}", disabled=True)
        with id_cols[2]:
            status_options = ['pending', 'analyzed', 'approved', 'rejected']
            current_status = data.get('status', 'pending')
            idx = status_options.index(current_status) if current_status in status_options else 0
            data['status'] = st.selectbox(
                "Status", status_options, index=idx,
                key=f"me_status_{key_suffix}", disabled=readonly)

        id_cols2 = st.columns(3)
        with id_cols2[0]:
            data['program'] = st.text_input(
                "Program", value=data.get('program', ''),
                key=f"me_program_{key_suffix}", disabled=True)
        with id_cols2[1]:
            data['campaign_id'] = st.text_input(
                "Campaign ID", value=data.get('campaign_id', ''),
                key=f"me_campaign_id_{key_suffix}", disabled=True)
        with id_cols2[2]:
            data['test_type'] = st.text_input(
                "Test Type", value=data.get('test_type', ''),
                key=f"me_test_type_{key_suffix}", disabled=True)

    handled_keys.update(['test_id', 'system', 'status', 'program',
                         'campaign_id', 'test_type', 'run_id'])

    # Group 2: Test Article
    with st.expander("Test Article"):
        # Handle both flat fields and nested test_article section
        test_article = data.get('test_article', {})
        has_nested_article = isinstance(test_article, dict) and test_article

        art_cols = st.columns(3)
        with art_cols[0]:
            data['part_name'] = st.text_input(
                "Part Name", value=data.get('part_name', ''),
                key=f"me_part_name_{key_suffix}", disabled=readonly)
        with art_cols[1]:
            pn_val = data.get('part_number', '') or test_article.get('part_number', '')
            data['part_number'] = st.text_input(
                "Part Number", value=pn_val,
                key=f"me_part_number_{key_suffix}", disabled=readonly)
        with art_cols[2]:
            sn_val = data.get('serial_number', '') or test_article.get('serial_number', '')
            data['serial_number'] = st.text_input(
                "Serial Number", value=sn_val,
                key=f"me_serial_number_{key_suffix}", disabled=readonly)

        if has_nested_article:
            st.caption("Additional test article fields")
            nested_art_cols = st.columns(2)
            extra_article_keys = [k for k in test_article if k not in ('part_number', 'serial_number')]
            for i, key in enumerate(extra_article_keys):
                with nested_art_cols[i % 2]:
                    label = key.replace('_', ' ').title()
                    if isinstance(test_article[key], (int, float)):
                        test_article[key] = st.number_input(
                            label, value=float(test_article[key]),
                            key=f"me_art_{key}_{key_suffix}", disabled=readonly)
                    else:
                        test_article[key] = st.text_input(
                            label, value=str(test_article[key] or ''),
                            key=f"me_art_{key}_{key_suffix}", disabled=readonly)
            # Write back nested updates
            if has_nested_article:
                test_article['part_number'] = data['part_number']
                test_article['serial_number'] = data['serial_number']
                data['test_article'] = test_article

    handled_keys.update(['part_name', 'part_number', 'serial_number', 'test_article'])

    # Group 3: Test Info
    with st.expander("Test Info"):
        info_cols = st.columns(3)
        with info_cols[0]:
            data['test_date'] = st.text_input(
                "Test Date (ISO)", value=data.get('test_date', ''),
                placeholder="2026-01-15",
                key=f"me_test_date_{key_suffix}", disabled=readonly)
        with info_cols[1]:
            data['operator'] = st.text_input(
                "Operator", value=data.get('operator', ''),
                key=f"me_operator_{key_suffix}", disabled=readonly)
        with info_cols[2]:
            data['facility'] = st.text_input(
                "Facility", value=data.get('facility', ''),
                key=f"me_facility_{key_suffix}", disabled=readonly)

        info_cols2 = st.columns(2)
        with info_cols2[0]:
            data['test_time'] = st.text_input(
                "Test Time", value=data.get('test_time', ''),
                key=f"me_test_time_{key_suffix}", disabled=readonly)
        with info_cols2[1]:
            data['test_stand'] = st.text_input(
                "Test Stand", value=data.get('test_stand', ''),
                key=f"me_test_stand_{key_suffix}", disabled=readonly)

    handled_keys.update(['test_date', 'test_time', 'operator', 'facility', 'test_stand'])

    # Group 4: Fluid Properties
    detected_type = test_type or data.get('test_type', '')
    is_hot_fire = detected_type.upper() in ('HF', 'HOT_FIRE', 'HOTFIRE')

    with st.expander("Fluid Properties"):
        if is_hot_fire:
            fl_cols = st.columns(2)
            with fl_cols[0]:
                data['oxidizer'] = st.text_input(
                    "Oxidizer", value=data.get('oxidizer', ''),
                    placeholder="e.g., Oxygen, NitrousOxide",
                    key=f"me_oxidizer_{key_suffix}", disabled=readonly)
                data['ox_temperature_K'] = st.number_input(
                    "Ox Temperature [K]", value=float(data.get('ox_temperature_K', 90.0)),
                    format="%.2f", key=f"me_ox_temp_{key_suffix}", disabled=readonly)
                data['ox_pressure_Pa'] = st.number_input(
                    "Ox Pressure [Pa]", value=float(data.get('ox_pressure_Pa', 101325.0)),
                    format="%.1f", key=f"me_ox_press_{key_suffix}", disabled=readonly)
            with fl_cols[1]:
                data['fuel'] = st.text_input(
                    "Fuel", value=data.get('fuel', ''),
                    placeholder="e.g., n-Dodecane, Ethanol, Methane",
                    key=f"me_fuel_{key_suffix}", disabled=readonly)
                data['fuel_temperature_K'] = st.number_input(
                    "Fuel Temperature [K]", value=float(data.get('fuel_temperature_K', 293.15)),
                    format="%.2f", key=f"me_fuel_temp_{key_suffix}", disabled=readonly)
                data['fuel_pressure_Pa'] = st.number_input(
                    "Fuel Pressure [Pa]", value=float(data.get('fuel_pressure_Pa', 101325.0)),
                    format="%.1f", key=f"me_fuel_press_{key_suffix}", disabled=readonly)
        else:
            fl_cols = st.columns(3)
            with fl_cols[0]:
                data['test_fluid'] = st.text_input(
                    "Test Fluid", value=data.get('test_fluid', ''),
                    placeholder="e.g., Water, Nitrogen, Ethanol",
                    key=f"me_test_fluid_{key_suffix}", disabled=readonly)
            with fl_cols[1]:
                data['fluid_temperature_K'] = st.number_input(
                    "Fluid Temperature [K]", value=float(data.get('fluid_temperature_K', 293.15)),
                    format="%.2f", key=f"me_fluid_temp_{key_suffix}", disabled=readonly)
            with fl_cols[2]:
                data['fluid_pressure_Pa'] = st.number_input(
                    "Fluid Pressure [Pa]", value=float(data.get('fluid_pressure_Pa', 101325.0)),
                    format="%.1f", key=f"me_fluid_press_{key_suffix}", disabled=readonly)

        # Handle nested fluid_properties section
        fluid_props = data.get('fluid_properties')
        if isinstance(fluid_props, dict) and fluid_props:
            st.caption("Additional fluid properties (from file)")
            st.json(fluid_props)

    handled_keys.update(['test_fluid', 'fluid_temperature_K', 'fluid_pressure_Pa',
                         'oxidizer', 'fuel', 'ox_temperature_K', 'fuel_temperature_K',
                         'ox_pressure_Pa', 'fuel_pressure_Pa', 'fluid_properties'])

    # Group 5: Geometry (only if nested section exists or user wants to add one)
    geometry = data.get('geometry')
    if isinstance(geometry, dict) and geometry:
        with st.expander("Geometry"):
            geo_keys = sorted(geometry.keys())
            geo_cols = st.columns(2)
            for i, key in enumerate(geo_keys):
                with geo_cols[i % 2]:
                    label = key.replace('_', ' ').title()
                    val = geometry[key]
                    if isinstance(val, (int, float)):
                        geometry[key] = st.number_input(
                            label, value=float(val), format="%.6g",
                            key=f"me_geo_{key}_{key_suffix}", disabled=readonly)
                    elif isinstance(val, str):
                        geometry[key] = st.text_input(
                            label, value=val,
                            key=f"me_geo_{key}_{key_suffix}", disabled=readonly)
                    else:
                        st.text_input(label, value=str(val),
                                      key=f"me_geo_{key}_{key_suffix}", disabled=True)
            data['geometry'] = geometry
    handled_keys.add('geometry')

    # Group 6: Test Conditions & Environment
    with st.expander("Environment & Conditions"):
        env_cols = st.columns(2)
        with env_cols[0]:
            data['ambient_temperature_K'] = st.number_input(
                "Ambient Temperature [K]",
                value=float(data.get('ambient_temperature_K', 293.15)),
                format="%.2f", key=f"me_amb_temp_{key_suffix}", disabled=readonly)
        with env_cols[1]:
            data['ambient_pressure_Pa'] = st.number_input(
                "Ambient Pressure [Pa]",
                value=float(data.get('ambient_pressure_Pa', 101325.0)),
                format="%.1f", key=f"me_amb_press_{key_suffix}", disabled=readonly)

        # Handle nested test_conditions section
        test_conditions = data.get('test_conditions')
        if isinstance(test_conditions, dict) and test_conditions:
            st.caption("Test conditions")
            tc_keys = sorted(test_conditions.keys())
            tc_cols = st.columns(2)
            for i, key in enumerate(tc_keys):
                with tc_cols[i % 2]:
                    label = key.replace('_', ' ').title()
                    val = test_conditions[key]
                    if isinstance(val, (int, float)):
                        test_conditions[key] = st.number_input(
                            label, value=float(val), format="%.6g",
                            key=f"me_tc_{key}_{key_suffix}", disabled=readonly)
                    elif isinstance(val, str):
                        test_conditions[key] = st.text_input(
                            label, value=val,
                            key=f"me_tc_{key}_{key_suffix}", disabled=readonly)
                    else:
                        st.text_input(label, value=str(val),
                                      key=f"me_tc_{key}_{key_suffix}", disabled=True)
            data['test_conditions'] = test_conditions

        # Handle nested nominal_conditions dict
        nom = data.get('nominal_conditions')
        if isinstance(nom, dict) and nom:
            st.caption("Nominal conditions")
            st.json(nom)

    handled_keys.update(['ambient_temperature_K', 'ambient_pressure_Pa',
                         'test_conditions', 'nominal_conditions'])

    # Group 7: Configuration Reference
    config_file = data.get('config_file', '')
    config_name = data.get('config_name', '')
    if config_file or config_name:
        with st.expander("Configuration Reference"):
            cfg_cols = st.columns(2)
            with cfg_cols[0]:
                data['config_file'] = st.text_input(
                    "Config File", value=config_file,
                    key=f"me_config_file_{key_suffix}", disabled=readonly)
            with cfg_cols[1]:
                data['config_name'] = st.text_input(
                    "Config Name", value=config_name,
                    key=f"me_config_name_{key_suffix}", disabled=readonly)
    handled_keys.update(['config_file', 'config_name', 'raw_data_files'])

    # Group 8: Notes
    with st.expander("Notes & Observations"):
        data['notes'] = st.text_area(
            "Notes", value=data.get('notes', ''), height=100,
            key=f"me_notes_{key_suffix}", disabled=readonly)
        data['anomalies'] = st.text_area(
            "Anomalies", value=data.get('anomalies', ''), height=80,
            key=f"me_anomalies_{key_suffix}", disabled=readonly)

    handled_keys.update(['notes', 'anomalies'])

    # Group 9: Analysis info (read-only display)
    analysis_date = data.get('analysis_date', '')
    analysis_version = data.get('analysis_version', '')
    if analysis_date or analysis_version:
        with st.expander("Analysis Info"):
            a_cols = st.columns(2)
            with a_cols[0]:
                st.text_input("Analysis Date", value=analysis_date,
                              key=f"me_analysis_date_{key_suffix}", disabled=True)
            with a_cols[1]:
                st.text_input("Analysis Version", value=analysis_version,
                              key=f"me_analysis_version_{key_suffix}", disabled=True)
    handled_keys.update(['analysis_date', 'analysis_version'])

    # Group 10: Additional / unhandled fields
    extra_keys = [k for k in data.keys() if k not in handled_keys]
    if extra_keys:
        with st.expander(f"Additional Fields ({len(extra_keys)})"):
            for key in sorted(extra_keys):
                val = data[key]
                label = key.replace('_', ' ').title()
                if isinstance(val, dict):
                    st.caption(f"**{label}**")
                    if readonly:
                        st.json(val)
                    else:
                        edited = st.text_area(
                            f"{label} (JSON)", value=json.dumps(val, indent=2),
                            height=150, key=f"me_extra_{key}_{key_suffix}",
                            disabled=readonly)
                        try:
                            data[key] = json.loads(edited)
                        except json.JSONDecodeError:
                            st.warning(f"Invalid JSON for '{key}' - keeping original")
                elif isinstance(val, list):
                    st.caption(f"**{label}**")
                    if readonly:
                        st.json(val)
                    else:
                        edited = st.text_area(
                            f"{label} (JSON)", value=json.dumps(val, indent=2),
                            height=100, key=f"me_extra_{key}_{key_suffix}",
                            disabled=readonly)
                        try:
                            data[key] = json.loads(edited)
                        except json.JSONDecodeError:
                            st.warning(f"Invalid JSON for '{key}' - keeping original")
                elif isinstance(val, (int, float)):
                    data[key] = st.number_input(
                        label, value=float(val), format="%.6g",
                        key=f"me_extra_{key}_{key_suffix}", disabled=readonly)
                else:
                    data[key] = st.text_input(
                        label, value=str(val or ''),
                        key=f"me_extra_{key}_{key_suffix}", disabled=readonly)

    return data


def _metadata_json_editor(
    data: Dict[str, Any],
    key_suffix: str = "",
    readonly: bool = False,
) -> Dict[str, Any]:
    """
    Raw JSON editor for metadata. Internal helper for metadata_editor_widget.

    Args:
        data: Current metadata dict
        key_suffix: Suffix for widget keys
        readonly: If True, display only

    Returns:
        Parsed metadata dict (or original if parse fails)
    """
    json_str = st.text_area(
        "Metadata JSON",
        value=json.dumps(data, indent=2, default=str),
        height=500,
        key=f"me_json_editor_{key_suffix}",
        disabled=readonly,
    )

    if readonly:
        return data

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            st.error("JSON must be an object (dict)")
            return data
        return parsed
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return data


def error_with_troubleshooting(error_message: str, suggestions: List[str]):
    """
    Display error message with troubleshooting suggestions.

    Args:
        error_message: Error message to display
        suggestions: List of troubleshooting suggestions
    """
    st.markdown(f"""
    <div style="background: #fee2e2; border-left: 4px solid #dc2626; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
        <p style="margin: 0; color: #dc2626; font-weight: 600;">✗ {error_message}</p>
    </div>
    """, unsafe_allow_html=True)

    if suggestions:
        with st.expander("🔧 Troubleshooting Suggestions", expanded=True):
            for suggestion in suggestions:
                st.markdown(f"""
                <div style="display: flex; align-items: start; margin-bottom: 0.5rem;">
                    <span style="color: #ca8a04; margin-right: 0.5rem;">•</span>
                    <p style="margin: 0; font-size: 0.875rem;">{suggestion}</p>
                </div>
                """, unsafe_allow_html=True)
