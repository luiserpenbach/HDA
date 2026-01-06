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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Campaign", campaign_name)
            with col2:
                st.metric("Type", info.get('campaign_type', 'Unknown'))
            with col3:
                st.metric("Tests", len(df))
        else:
            st.subheader(f"Campaign: {campaign_name}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Type", info.get('campaign_type', 'Unknown'))
            with col2:
                st.metric("Total Tests", len(df))
            with col3:
                created = info.get('created_date', 'Unknown')
                st.metric("Created", created[:10] if created and created != 'Unknown' else 'Unknown')

            if 'description' in info and info['description']:
                st.caption(f"**Description:** {info['description']}")

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
    st.success(message)

    if next_steps:
        st.info("**Recommended next steps:**")
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")


def error_with_troubleshooting(error_message: str, suggestions: List[str]):
    """
    Display error message with troubleshooting suggestions.

    Args:
        error_message: Error message to display
        suggestions: List of troubleshooting suggestions
    """
    st.error(error_message)

    if suggestions:
        with st.expander("Troubleshooting Suggestions"):
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
