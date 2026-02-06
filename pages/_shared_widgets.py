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
