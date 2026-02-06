"""
Single Test Analysis Page
=========================
Complete analysis pipeline for individual cold flow or hot fire tests.

Pipeline: Setup → Steady State → Analyze → Results → Export

Design: Tabbed workflow with settings/main column layout.
Each tab follows the pattern from Analysis Tools (Operating Envelope).

Version: 2.5.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

# Core imports
from core.integrated_analysis import analyze_cold_flow_test, analyze_hot_fire_test
from core.qc_checks import run_qc_checks
from core.traceability import compute_file_hash
from core.reporting import generate_test_report
from core.campaign_manager_v2 import get_available_campaigns, save_to_campaign
from core.config_manager import ConfigManager
from core.steady_state_detection import (
    detect_steady_state_cv,
    detect_steady_state_ml,
    detect_steady_state_derivative,
)

# UI imports
from pages._shared_styles import (
    apply_custom_styles, render_page_header, render_metric_card, render_status_badge,
)
from pages._shared_sidebar import render_global_context

# Optional imports
try:
    from core.test_metadata import (
        load_test_from_folder, find_raw_data_file,
    )
    TEST_FOLDER_SUPPORT = True
except ImportError:
    TEST_FOLDER_SUPPORT = False

try:
    from core.fluid_properties import get_fluid_properties, COOLPROP_AVAILABLE
    FLUID_PROPS_SUPPORT = True
except ImportError:
    COOLPROP_AVAILABLE = False
    FLUID_PROPS_SUPPORT = False

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Single Test Analysis", page_icon="STA", layout="wide")
apply_custom_styles()
render_page_header(
    title="Single Test Analysis",
    description="Full pipeline: data ingestion, preprocessing, steady-state detection, QC, uncertainty-quantified analysis",
    badge_text="P0",
    badge_type="error",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS = {
    'df': None,
    'df_processed': None,
    'file_hash': None,
    'steady_window': None,
    'detection_sensor': None,
    'active_config': None,
    'active_config_name': None,
    'loaded_metadata': None,
    'qc_report': None,
    'analysis_result': None,
    'test_folder_path': None,
    'iteration_results': [],
    'preprocess_stats': None,
    'detection_preferences': {
        'method': 'CV-based',
        'cv_threshold': 0.02,
        'cv_window_size': 50,
        'ml_contamination': 0.3,
        'deriv_threshold': 0.1,
        'window_adjust_step': 0.1,
    },
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Plotly defaults
_PLOTLY_LAYOUT = dict(template="plotly_white", margin=dict(t=40, b=40, l=50, r=20))
_ZINC = {
    'primary': '#18181b', 'line': '#18181b', 'accent': '#2563eb',
    'success': '#16a34a', 'warning': '#ca8a04', 'error': '#dc2626',
    'muted': '#71717a', 'light': '#f4f4f5', 'border': '#e4e4e7',
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect time column from common names."""
    for name in ['time_s', 'time_ms', 'timestamp', 'Time', 'TIME', 't']:
        if name in df.columns:
            return name
    return None


def _numeric_columns(df: pd.DataFrame, exclude_time: bool = True) -> List[str]:
    """Get numeric column names, optionally excluding time columns."""
    time_names = {'time', 'time_s', 'time_ms', 'timestamp', 'Time', 'TIME', 't'}
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_time:
        cols = [c for c in cols if c not in time_names]
    return cols


def preprocess_data(df: pd.DataFrame, config: dict, time_unit: str = 'ms',
                    shift_to_zero: bool = True) -> pd.DataFrame:
    """Preprocess raw test data: convert timestamps, sort, deduplicate."""
    df_proc = df.copy()
    timestamp_col = config.get('columns', {}).get('timestamp', 'timestamp')

    if timestamp_col in df_proc.columns:
        df_proc = df_proc.sort_values(timestamp_col).reset_index(drop=True)
        df_proc = df_proc.drop_duplicates(subset=[timestamp_col], keep='first')

        if time_unit == 'ms':
            df_proc['time_s'] = df_proc[timestamp_col] / 1000.0
        elif time_unit == 'us':
            df_proc['time_s'] = df_proc[timestamp_col] / 1_000_000.0
        else:
            df_proc['time_s'] = df_proc[timestamp_col].astype(float)

        if shift_to_zero:
            df_proc['time_s'] = df_proc['time_s'] - df_proc['time_s'].iloc[0]

        df_proc['time_ms'] = df_proc['time_s'] * 1000.0

    return df_proc


def resample_data(df: pd.DataFrame, target_rate_hz: float,
                  time_col: str = 'time_s') -> Tuple[pd.DataFrame, dict]:
    """Resample data to a uniform sample rate via linear interpolation."""
    if time_col not in df.columns:
        return df, {'error': f'Time column {time_col} not found'}

    stats = {
        'original_rows': len(df),
        'original_duration_s': float(df[time_col].max() - df[time_col].min()),
        'target_rate_hz': target_rate_hz,
    }

    dt = np.diff(df[time_col].values)
    if len(dt) > 0 and np.mean(dt) > 0:
        stats['original_mean_rate_hz'] = float(1.0 / np.mean(dt))

    new_time = np.arange(df[time_col].min(), df[time_col].max(), 1.0 / target_rate_hz)
    stats['resampled_rows'] = len(new_time)

    df_resampled = pd.DataFrame({time_col: new_time})
    if time_col == 'time_s':
        df_resampled['time_ms'] = new_time * 1000.0

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in [time_col, 'time_s', 'time_ms']:
            continue
        df_resampled[col] = np.interp(new_time, df[time_col].values, df[col].values)

    return df_resampled, stats


def handle_nan_values(df: pd.DataFrame, method: str = 'interpolate',
                      max_gap: int = 5) -> Tuple[pd.DataFrame, dict]:
    """Handle NaN values with interpolation, drop, or forward-fill."""
    stats = {
        'original_rows': len(df),
        'nan_counts': {},
        'method': method,
    }
    df_clean = df.copy()

    for col in df_clean.columns:
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            stats['nan_counts'][col] = int(nan_count)

    stats['rows_affected'] = int(df_clean.isna().any(axis=1).sum())

    if method == 'interpolate':
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            mask = df_clean[col].isna()
            if mask.any():
                groups = (mask != mask.shift()).cumsum()
                gap_sizes = mask.groupby(groups).transform('sum')
                small_gaps = mask & (gap_sizes <= max_gap)
                if small_gaps.any():
                    df_clean[col] = df_clean[col].interpolate(method='linear', limit=max_gap)
    elif method == 'drop':
        df_clean = df_clean.dropna()
    elif method == 'ffill':
        df_clean = df_clean.ffill().bfill()

    stats['final_rows'] = len(df_clean)
    return df_clean, stats


def apply_channel_mapping(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """Rename DAQ channel IDs to sensor names using channel_config."""
    channel_config = config.get('channel_config') or config.get('columns', {})
    if not channel_config:
        return df, {'applied': False, 'mappings_found': 0}

    stats = {'applied': True, 'mappings_found': 0, 'mappings_applied': []}
    df_mapped = df.copy()
    rename_map = {}

    for raw_id, sensor_name in channel_config.items():
        for candidate in [raw_id, str(raw_id)]:
            if candidate in df_mapped.columns:
                rename_map[candidate] = sensor_name
                stats['mappings_applied'].append(f"{raw_id} -> {sensor_name}")
                stats['mappings_found'] += 1
                break
        else:
            try:
                int_id = int(raw_id)
                if int_id in df_mapped.columns:
                    rename_map[int_id] = sensor_name
                    stats['mappings_applied'].append(f"{raw_id} -> {sensor_name}")
                    stats['mappings_found'] += 1
            except (ValueError, TypeError):
                pass

    if rename_map:
        df_mapped = df_mapped.rename(columns=rename_map)

    return df_mapped, stats


def _build_geometry_dict(test_type: str, form_values: dict) -> dict:
    """Build geometry dict from form input values."""
    geom = {}
    if test_type == 'cold_flow':
        for key in ['orifice_area_mm2', 'orifice_diameter_mm', 'downstream_area_mm2']:
            if form_values.get(key, 0) > 0:
                geom[key] = form_values[key]
    elif test_type == 'hot_fire':
        for key in ['throat_area_mm2', 'throat_diameter_mm', 'expansion_ratio', 'chamber_volume_cc']:
            if form_values.get(key, 0) > 0:
                geom[key] = form_values[key]
    return geom


def _render_workflow_status():
    """Render pipeline status in the sidebar."""
    steps = [
        ("Data", st.session_state.df is not None),
        ("Preprocessed", st.session_state.df_processed is not None),
        ("Steady State", st.session_state.steady_window is not None),
        ("QC", st.session_state.qc_report is not None),
        ("Analysis", st.session_state.analysis_result is not None),
    ]
    html_parts = []
    for label, done in steps:
        color = _ZINC['success'] if done else _ZINC['border']
        icon = "&#10003;" if done else "&#9675;"
        text_color = _ZINC['success'] if done else _ZINC['muted']
        html_parts.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
            f'<span style="color:{color};font-size:14px;font-weight:700;">{icon}</span>'
            f'<span style="color:{text_color};font-size:0.8rem;font-weight:500;">{label}</span>'
            f'</div>'
        )
    st.markdown(
        '<div style="background:#fafafa;border:1px solid #e4e4e7;border-radius:8px;padding:12px 14px;">'
        + ''.join(html_parts) + '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    context = render_global_context()
    st.divider()

    # Test type
    test_type = st.selectbox(
        "Test Type",
        ["cold_flow", "hot_fire"],
        format_func=lambda x: "Cold Flow" if x == "cold_flow" else "Hot Fire",
    )

    st.divider()

    # Pipeline status
    st.markdown(
        '<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.05em;color:#52525b;margin-bottom:8px;">Pipeline Status</p>',
        unsafe_allow_html=True,
    )
    _render_workflow_status()

    # Active config summary
    if st.session_state.active_config is not None:
        st.divider()
        cfg = st.session_state.active_config
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.05em;color:#52525b;margin-bottom:8px;">Active Config</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#fafafa;border:1px solid #e4e4e7;border-radius:8px;padding:10px 14px;">'
            f'<p style="margin:0;font-size:0.85rem;font-weight:600;color:#18181b;">'
            f'{st.session_state.active_config_name or "Custom"}</p>'
            f'<p style="margin:2px 0 0;font-size:0.75rem;color:#71717a;">'
            f'{cfg.get("test_type","unknown")} &middot; '
            f'{len(cfg.get("channel_config", cfg.get("columns", {})))} channels</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Config", key="sidebar_clear_config", use_container_width=True):
            st.session_state.active_config = None
            st.session_state.active_config_name = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT - TABBED WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

tab_setup, tab_steady, tab_analyze, tab_results, tab_export = st.tabs([
    "Setup", "Steady State", "Analyze", "Results", "Export",
])


# #############################################################################
# TAB 1: SETUP (Data + Config + Preprocessing)
# #############################################################################
with tab_setup:
    try:
        st.subheader("Data & Configuration")
        st.caption("Upload test data, load configuration, and preprocess for analysis.")

        setup_settings, setup_main = st.columns([1, 3])

        # ── Settings column ──────────────────────────────────────────────
        with setup_settings:
            # --- Data Source ---
            st.markdown("**1. Data Source**")
            data_source = st.radio(
                "Source",
                ["Upload CSV", "Test Folder"],
                horizontal=True,
                key="setup_data_source",
                label_visibility="collapsed",
            )

            df_raw = st.session_state.df

            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "CSV file", type=['csv'], key="setup_csv_upload",
                    help="CSV with timestamp column and sensor data",
                )
                if uploaded_file:
                    try:
                        df_raw = pd.read_csv(uploaded_file)
                        st.session_state.df = df_raw
                        st.session_state.test_folder_path = None
                        uploaded_file.seek(0)
                        file_content = uploaded_file.read()
                        st.session_state.file_hash = (
                            f"sha256:{__import__('hashlib').sha256(file_content).hexdigest()[:16]}"
                        )
                        st.success(f"{len(df_raw):,} rows x {len(df_raw.columns)} cols")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        df_raw = None

                # Optional metadata upload
                meta_file = st.file_uploader(
                    "Metadata JSON (optional)", type=['json'], key="setup_meta_upload",
                )
                if meta_file:
                    try:
                        st.session_state.loaded_metadata = json.load(meta_file)
                        st.success("Metadata loaded")
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")

            else:  # Test Folder
                if not TEST_FOLDER_SUPPORT:
                    st.warning("Test folder support not available.")
                else:
                    recent_folders = ConfigManager.get_recent_folders(limit=5)
                    if recent_folders:
                        folder_options = [""] + recent_folders
                        test_folder_path = st.selectbox(
                            "Recent folders", folder_options,
                            format_func=lambda x: "-- Select --" if x == "" else x,
                            key="setup_recent_folder",
                        )
                    else:
                        test_folder_path = ""

                    test_folder_path = st.text_input(
                        "Folder path",
                        value=test_folder_path or st.session_state.get('test_folder_path', ''),
                        key="setup_folder_path",
                    )

                    if st.button("Load Folder", type="primary", key="setup_load_folder"):
                        if test_folder_path and Path(test_folder_path).exists():
                            try:
                                test_data = load_test_from_folder(test_folder_path)
                                if test_data['raw_data_file']:
                                    df_raw = pd.read_csv(test_data['raw_data_file'])
                                    st.session_state.df = df_raw
                                    st.session_state.test_folder_path = test_folder_path
                                    ConfigManager.save_recent_folder(test_folder_path)
                                    with open(test_data['raw_data_file'], 'rb') as f:
                                        st.session_state.file_hash = (
                                            f"sha256:{__import__('hashlib').sha256(f.read()).hexdigest()[:16]}"
                                        )
                                    st.success(f"Loaded {len(df_raw):,} rows")
                                if test_data.get('metadata'):
                                    st.session_state.loaded_metadata = test_data['metadata']
                                if test_data.get('config'):
                                    st.session_state.active_config = test_data['config']
                                    st.session_state.active_config_name = test_data['config'].get('config_name', 'From folder')
                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning("Enter a valid folder path.")

            st.divider()

            # --- Configuration ---
            st.markdown("**2. Configuration**")

            config_source = st.radio(
                "Config source",
                ["Recent", "Saved Configs", "Upload JSON", "Default"],
                key="setup_config_source",
                label_visibility="collapsed",
            )

            if config_source == "Recent":
                recent_configs = ConfigManager.get_recent_configs(limit=5)
                if recent_configs:
                    recent_options = ["-- Select --"] + [
                        f"{r['info']['config_name']} ({r['info']['timestamp'][:10]})"
                        for r in recent_configs
                    ]
                    selected = st.selectbox("Recent configs", recent_options, key="setup_recent_cfg",
                                            label_visibility="collapsed")
                    if selected != "-- Select --":
                        if st.button("Load", key="setup_load_recent", type="primary"):
                            idx = recent_options.index(selected) - 1
                            st.session_state.active_config = recent_configs[idx]['config']
                            st.session_state.active_config_name = recent_configs[idx]['info']['config_name']
                            st.rerun()
                else:
                    st.caption("No recent configs.")

            elif config_source == "Saved Configs":
                try:
                    from core.saved_configs import SavedConfigManager, load_saved_config
                    manager = SavedConfigManager()
                    templates = manager.list_templates()
                    filtered = [t for t in templates if t.get('test_type') == test_type]
                    if filtered:
                        t_options = ["-- Select --"] + [
                            f"{t['id']} - {t.get('name', t.get('config_name', 'Unnamed'))}"
                            for t in filtered
                        ]
                        sel_t = st.selectbox("Saved configs", t_options, key="setup_saved_cfg",
                                             label_visibility="collapsed")
                        if sel_t != "-- Select --":
                            t_id = sel_t.split(" - ")[0]
                            t_info = next(t for t in filtered if t['id'] == t_id)
                            if t_info.get('description'):
                                st.caption(t_info['description'])
                            if st.button("Load", key="setup_load_saved", type="primary"):
                                loaded = load_saved_config(t_id)
                                st.session_state.active_config = loaded
                                st.session_state.active_config_name = loaded.get('config_name', t_id)
                                ConfigManager.save_to_recent(loaded, 'saved_config', st.session_state.active_config_name)
                                st.rerun()
                    else:
                        st.caption(f"No {test_type} saved configs.")
                except ImportError:
                    st.caption("Saved config system unavailable.")

            elif config_source == "Upload JSON":
                config_file = st.file_uploader("Config JSON", type=['json'], key="setup_cfg_upload")
                if config_file:
                    try:
                        uploaded_cfg = json.load(config_file)
                        if st.button("Load", key="setup_load_upload", type="primary"):
                            st.session_state.active_config = uploaded_cfg
                            st.session_state.active_config_name = uploaded_cfg.get('config_name', 'Uploaded')
                            ConfigManager.save_to_recent(uploaded_cfg, 'uploaded', st.session_state.active_config_name)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")

            else:  # Default
                if st.button("Load Default", key="setup_load_default", type="primary"):
                    default_cfg = ConfigManager.get_default_config(test_type)
                    st.session_state.active_config = default_cfg
                    st.session_state.active_config_name = f"Default ({test_type})"
                    st.rerun()

            # Config viewer
            if st.session_state.active_config:
                with st.expander("View Config"):
                    st.json(st.session_state.active_config)

            st.divider()

            # --- Preprocessing ---
            st.markdown("**3. Preprocessing**")

            config = st.session_state.active_config or ConfigManager.get_default_config(test_type)

            if df_raw is not None:
                channel_config = config.get('channel_config') or config.get('columns', {})
                apply_mapping = False
                if channel_config:
                    apply_mapping = st.checkbox(
                        f"Channel mapping ({len(channel_config)})",
                        value=True, key="setup_apply_mapping",
                    )

                time_unit = st.selectbox("Time unit", ["ms", "s", "us"], key="setup_time_unit")
                shift_to_zero = st.checkbox("Shift to t=0", value=True, key="setup_shift_zero")

                nan_method = st.selectbox(
                    "NaN handling", ["interpolate", "drop", "ffill", "none"],
                    key="setup_nan_method",
                )

                do_resample = st.checkbox("Resample", value=False, key="setup_resample")
                target_rate = 100
                if do_resample:
                    default_rate = config.get('settings', {}).get('sample_rate_hz', 100)
                    target_rate = st.number_input(
                        "Rate (Hz)", min_value=1, max_value=100000,
                        value=default_rate, key="setup_target_rate",
                    )

                if st.button("Preprocess", type="primary", use_container_width=True, key="setup_preprocess"):
                    with st.spinner("Preprocessing..."):
                        df_proc = df_raw.copy()
                        stats = {}

                        if apply_mapping and channel_config:
                            df_proc, m_stats = apply_channel_mapping(df_proc, config)
                            stats['mappings'] = m_stats['mappings_found']

                        df_proc = preprocess_data(df_proc, config, time_unit, shift_to_zero)

                        if nan_method != "none":
                            df_proc, nan_stats = handle_nan_values(df_proc, nan_method)
                            stats['nans_fixed'] = nan_stats.get('rows_affected', 0)

                        if do_resample:
                            df_proc, rs_stats = resample_data(df_proc, target_rate)
                            stats['resampled'] = rs_stats.get('resampled_rows', len(df_proc))

                        st.session_state.df_processed = df_proc
                        st.session_state.preprocess_stats = stats
                        # Clear downstream state
                        st.session_state.steady_window = None
                        st.session_state.qc_report = None
                        st.session_state.analysis_result = None
                        st.rerun()
            else:
                st.info("Load data first.")

        # ── Main column ──────────────────────────────────────────────────
        with setup_main:
            df_display = st.session_state.df_processed or st.session_state.df

            if df_display is not None:
                # Summary cards
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    render_metric_card("Rows", f"{len(df_display):,}")
                with mc2:
                    render_metric_card("Columns", str(len(df_display.columns)))
                with mc3:
                    duration = "N/A"
                    if 'time_s' in df_display.columns:
                        dur_val = df_display['time_s'].max() - df_display['time_s'].min()
                        duration = f"{dur_val:.2f} s"
                    render_metric_card("Duration", duration)
                with mc4:
                    status = "Preprocessed" if st.session_state.df_processed is not None else "Raw"
                    render_metric_card("Status", status)

                st.markdown("")

                # Data preview
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(df_display.head(100), use_container_width=True, hide_index=True)

                # Data exploration plot
                with st.expander("Data Exploration", expanded=True):
                    time_col = _detect_time_column(df_display)
                    num_cols = _numeric_columns(df_display)

                    if num_cols and time_col:
                        default_sel = num_cols[:min(3, len(num_cols))]
                        selected_sensors = st.multiselect(
                            "Sensors to plot", num_cols, default=default_sel,
                            key="setup_plot_sensors",
                        )

                        if selected_sensors:
                            n = len(selected_sensors)
                            fig = make_subplots(
                                rows=n, cols=1, shared_xaxes=True,
                                vertical_spacing=0.04,
                                subplot_titles=selected_sensors,
                            )
                            for i, sensor in enumerate(selected_sensors):
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_display[time_col], y=df_display[sensor],
                                        mode='lines', name=sensor,
                                        line=dict(width=1, color=_ZINC['primary'] if i == 0 else None),
                                    ),
                                    row=i + 1, col=1,
                                )
                            time_label = "Time (s)" if time_col == 'time_s' else "Time (ms)"
                            fig.update_layout(
                                height=220 * n, showlegend=False,
                                **_PLOTLY_LAYOUT,
                            )
                            fig.update_xaxes(title_text=time_label, row=n, col=1)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Select sensors to visualize.")
                    else:
                        st.info("No plottable columns found.")
            else:
                st.markdown(
                    '<div style="text-align:center;padding:4rem 2rem;color:#71717a;">'
                    '<div style="font-size:3rem;margin-bottom:1rem;">&#128203;</div>'
                    '<h3 style="color:#18181b;">No Data Loaded</h3>'
                    '<p>Upload a CSV file or select a test folder in the settings panel.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error(f"Setup error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 2: STEADY STATE DETECTION
# #############################################################################
with tab_steady:
    try:
        st.subheader("Steady-State Detection")
        st.caption("Identify the steady-state region for analysis using automatic or manual methods.")

        df = st.session_state.df_processed or st.session_state.df
        config = st.session_state.active_config or ConfigManager.get_default_config(test_type)

        if df is None:
            st.info("Load and preprocess data in the **Setup** tab first.")
        else:
            ss_settings, ss_main = st.columns([1, 3])

            time_col = _detect_time_column(df) or 'time_s'
            num_cols = _numeric_columns(df)
            prefs = st.session_state.detection_preferences

            with ss_settings:
                st.markdown("**Detection Method**")

                methods = ["CV-based", "ML-based", "Derivative-based", "Manual"]
                method_idx = methods.index(prefs['method']) if prefs['method'] in methods else 0
                detection_method = st.selectbox(
                    "Method", methods, index=method_idx, key="ss_method",
                    label_visibility="collapsed",
                )
                prefs['method'] = detection_method

                st.divider()

                # Default sensor (prefer pressure)
                default_sensor_idx = 0
                for key in ['upstream_pressure', 'chamber_pressure', 'mass_flow']:
                    col_name = config.get('columns', {}).get(key)
                    if col_name and col_name in num_cols:
                        default_sensor_idx = num_cols.index(col_name)
                        break

                if detection_method == "CV-based":
                    st.markdown("**Parameters**")
                    sensor = st.selectbox(
                        "Detection sensor", num_cols, index=default_sensor_idx,
                        key="ss_cv_sensor",
                    )
                    cv_threshold = st.slider(
                        "CV threshold", 0.005, 0.10, prefs['cv_threshold'], 0.005,
                        key="ss_cv_thresh",
                    )
                    window_size = st.slider(
                        "Window size", 10, 200, prefs['cv_window_size'],
                        key="ss_cv_window",
                    )
                    prefs['cv_threshold'] = cv_threshold
                    prefs['cv_window_size'] = window_size

                    if st.button("Detect", type="primary", use_container_width=True, key="ss_detect_cv"):
                        start, end = detect_steady_state_cv(df, sensor, window_size, cv_threshold, time_col)
                        if start is not None:
                            st.session_state.steady_window = (start, end)
                            st.session_state.detection_sensor = sensor
                            st.session_state.qc_report = None
                            st.session_state.analysis_result = None
                            st.success(f"{start:.3f}s - {end:.3f}s ({end - start:.3f}s)")
                        else:
                            st.warning("No steady state found. Adjust parameters.")

                elif detection_method == "ML-based":
                    st.markdown("**Parameters**")
                    default_ml = num_cols[:min(4, len(num_cols))]
                    ml_sensors = st.multiselect(
                        "Sensors", num_cols, default=default_ml,
                        key="ss_ml_sensors",
                    )
                    contamination = st.slider(
                        "Contamination", 0.1, 0.5, prefs['ml_contamination'], 0.05,
                        key="ss_ml_contam",
                    )
                    prefs['ml_contamination'] = contamination

                    if ml_sensors and st.button("Detect", type="primary", use_container_width=True, key="ss_detect_ml"):
                        with st.spinner("Running ML detection..."):
                            start, end = detect_steady_state_ml(df, ml_sensors, time_col, contamination)
                        if start is not None:
                            st.session_state.steady_window = (start, end)
                            st.session_state.detection_sensor = ml_sensors[0]
                            st.session_state.qc_report = None
                            st.session_state.analysis_result = None
                            st.success(f"{start:.3f}s - {end:.3f}s ({end - start:.3f}s)")
                        else:
                            st.warning("ML detection failed.")

                elif detection_method == "Derivative-based":
                    st.markdown("**Parameters**")
                    sensor = st.selectbox(
                        "Detection sensor", num_cols, index=default_sensor_idx,
                        key="ss_deriv_sensor",
                    )
                    deriv_threshold = st.slider(
                        "Derivative threshold", 0.01, 0.5, prefs['deriv_threshold'], 0.01,
                        key="ss_deriv_thresh",
                    )
                    prefs['deriv_threshold'] = deriv_threshold

                    if st.button("Detect", type="primary", use_container_width=True, key="ss_detect_deriv"):
                        start, end = detect_steady_state_derivative(df, sensor, time_col, deriv_threshold)
                        if start is not None:
                            st.session_state.steady_window = (start, end)
                            st.session_state.detection_sensor = sensor
                            st.session_state.qc_report = None
                            st.session_state.analysis_result = None
                            st.success(f"{start:.3f}s - {end:.3f}s ({end - start:.3f}s)")
                        else:
                            st.warning("No steady state found.")

                else:  # Manual
                    if time_col in df.columns:
                        t_min = float(df[time_col].min())
                        t_max = float(df[time_col].max())
                        unit = "s" if time_col == 'time_s' else "ms"
                        window = st.slider(
                            f"Window ({unit})", t_min, t_max,
                            (t_min + (t_max - t_min) * 0.25, t_min + (t_max - t_min) * 0.75),
                            format="%.3f", key="ss_manual_slider",
                        )
                        st.session_state.steady_window = window

                # Manual adjustment (shown if window exists)
                if st.session_state.steady_window:
                    st.divider()
                    st.markdown("**Fine Adjustment**")

                    start_w, end_w = st.session_state.steady_window
                    step = st.number_input(
                        "Step (s)", 0.001, 1.0, prefs['window_adjust_step'],
                        step=0.01, format="%.3f", key="ss_step",
                    )
                    prefs['window_adjust_step'] = step

                    # Start adjustment
                    st.caption("Start")
                    sc1, sc2, sc3 = st.columns([2, 1, 1])
                    with sc1:
                        new_start = st.number_input(
                            "Start", value=float(start_w), format="%.3f",
                            key="ss_adj_start", label_visibility="collapsed",
                        )
                    with sc2:
                        if st.button("-", key="ss_start_minus"):
                            t_min_val = float(df[time_col].min()) if time_col in df.columns else 0
                            new_start = max(t_min_val, start_w - step)
                            st.session_state.steady_window = (new_start, end_w)
                            st.rerun()
                    with sc3:
                        if st.button("+", key="ss_start_plus"):
                            new_start = min(end_w - 0.01, start_w + step)
                            st.session_state.steady_window = (new_start, end_w)
                            st.rerun()

                    # End adjustment
                    st.caption("End")
                    ec1, ec2, ec3 = st.columns([2, 1, 1])
                    with ec1:
                        new_end = st.number_input(
                            "End", value=float(end_w), format="%.3f",
                            key="ss_adj_end", label_visibility="collapsed",
                        )
                    with ec2:
                        if st.button("-", key="ss_end_minus"):
                            new_end = max(start_w + 0.01, end_w - step)
                            st.session_state.steady_window = (start_w, new_end)
                            st.rerun()
                    with ec3:
                        if st.button("+", key="ss_end_plus"):
                            t_max_val = float(df[time_col].max()) if time_col in df.columns else 999
                            new_end = min(t_max_val, end_w + step)
                            st.session_state.steady_window = (start_w, new_end)
                            st.rerun()

                    # Apply typed changes
                    if new_start != start_w or new_end != end_w:
                        if new_start < new_end:
                            st.session_state.steady_window = (new_start, new_end)
                            st.rerun()

                    duration_w = end_w - start_w
                    st.caption(f"Duration: {duration_w:.3f}s")

            # ── Main column: plot ────────────────────────────────────────
            with ss_main:
                if st.session_state.steady_window:
                    sw_start, sw_end = st.session_state.steady_window
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        render_metric_card("Window Start", f"{sw_start:.3f} s")
                    with mc2:
                        render_metric_card("Window End", f"{sw_end:.3f} s")
                    with mc3:
                        render_metric_card("Duration", f"{sw_end - sw_start:.3f} s")
                    st.markdown("")

                # Plot with window overlay
                det_sensor = st.session_state.get('detection_sensor')
                plot_cols = []
                if det_sensor and det_sensor in num_cols:
                    plot_cols.append(det_sensor)
                for c in num_cols:
                    if c not in plot_cols:
                        plot_cols.append(c)
                        if len(plot_cols) >= 4:
                            break

                if plot_cols and time_col in df.columns:
                    n_p = len(plot_cols)
                    fig = make_subplots(
                        rows=n_p, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=[
                            f"{c} {'(detection)' if c == det_sensor else ''}"
                            for c in plot_cols
                        ],
                    )
                    for i, col_name in enumerate(plot_cols):
                        is_det = col_name == det_sensor
                        fig.add_trace(
                            go.Scatter(
                                x=df[time_col], y=df[col_name],
                                mode='lines', name=col_name,
                                line=dict(
                                    width=1.5 if is_det else 1,
                                    color=_ZINC['primary'] if is_det else _ZINC['muted'],
                                ),
                            ),
                            row=i + 1, col=1,
                        )
                        if st.session_state.steady_window:
                            fig.add_vrect(
                                x0=st.session_state.steady_window[0],
                                x1=st.session_state.steady_window[1],
                                fillcolor=_ZINC['success'], opacity=0.12,
                                line_width=0, row=i + 1, col=1,
                            )

                    fig.update_layout(
                        height=180 * n_p, showlegend=False,
                        **_PLOTLY_LAYOUT,
                    )
                    fig.update_xaxes(
                        title_text="Time (s)" if time_col == 'time_s' else "Time (ms)",
                        row=n_p, col=1,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data to plot.")

    except Exception as exc:
        st.error(f"Steady-state detection error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 3: ANALYZE (Metadata + QC + Run)
# #############################################################################
with tab_analyze:
    try:
        st.subheader("Analysis & Quality Control")
        st.caption("Configure metadata, run QC checks, and execute the analysis pipeline.")

        df = st.session_state.df_processed or st.session_state.df
        config = st.session_state.active_config or ConfigManager.get_default_config(test_type)

        if df is None:
            st.info("Load data in the **Setup** tab first.")
        elif st.session_state.steady_window is None:
            st.info("Detect a steady-state window in the **Steady State** tab first.")
        else:
            an_settings, an_main = st.columns([1, 3])

            loaded_meta = st.session_state.get('loaded_metadata') or {}
            geometry = loaded_meta.get('geometry', {})
            sensor_roles = loaded_meta.get('sensor_roles', {})

            # ── Settings column: metadata form ───────────────────────────
            with an_settings:
                st.markdown("**Test Identity**")
                default_tid = loaded_meta.get('test_id', f"TEST-{datetime.now().strftime('%Y%m%d-%H%M')}")
                test_id = st.text_input("Test ID", value=default_tid, key="an_test_id")

                fluid_options = [
                    "", "Water", "Nitrogen", "Air", "Helium", "Oxygen",
                    "Ethanol", "Isopropanol", "NitrousOxide", "CarbonDioxide",
                    "Hydrogen", "Methane",
                ]
                default_fluid = loaded_meta.get('test_fluid', '')
                if default_fluid and default_fluid not in fluid_options:
                    fluid_options.insert(1, default_fluid)
                fluid_idx = fluid_options.index(default_fluid) if default_fluid in fluid_options else 0
                test_fluid = st.selectbox("Test Fluid", fluid_options, index=fluid_idx, key="an_fluid")

                st.divider()
                st.markdown("**Part Info**")
                part_number = st.text_input("Part #", value=loaded_meta.get('part_number', ''), key="an_part")
                serial_number = st.text_input("Serial #", value=loaded_meta.get('serial_number', ''), key="an_serial")
                operator = st.text_input("Operator", value=loaded_meta.get('operator', ''), key="an_operator")

                st.divider()
                st.markdown("**Geometry**")

                form_geom = {}
                if test_type == 'cold_flow':
                    form_geom['orifice_area_mm2'] = st.number_input(
                        "Orifice area (mm2)",
                        value=float(geometry.get('orifice_area_mm2', 0.0)),
                        min_value=0.0, format="%.4f", key="an_orifice_area",
                    )
                    form_geom['orifice_diameter_mm'] = st.number_input(
                        "Orifice dia. (mm)",
                        value=float(geometry.get('orifice_diameter_mm', 0.0)),
                        min_value=0.0, format="%.3f", key="an_orifice_dia",
                    )
                    form_geom['downstream_area_mm2'] = st.number_input(
                        "Downstream area (mm2)",
                        value=float(geometry.get('downstream_area_mm2', 0.0)),
                        min_value=0.0, format="%.4f", key="an_ds_area",
                    )
                else:  # hot_fire
                    form_geom['throat_area_mm2'] = st.number_input(
                        "Throat area (mm2)",
                        value=float(geometry.get('throat_area_mm2', 0.0)),
                        min_value=0.0, format="%.4f", key="an_throat_area",
                    )
                    form_geom['throat_diameter_mm'] = st.number_input(
                        "Throat dia. (mm)",
                        value=float(geometry.get('throat_diameter_mm', 0.0)),
                        min_value=0.0, format="%.3f", key="an_throat_dia",
                    )
                    form_geom['expansion_ratio'] = st.number_input(
                        "Expansion ratio",
                        value=float(geometry.get('expansion_ratio', 0.0)),
                        min_value=0.0, format="%.2f", key="an_exp_ratio",
                    )
                    form_geom['chamber_volume_cc'] = st.number_input(
                        "Chamber vol. (cc)",
                        value=float(geometry.get('chamber_volume_cc', 0.0)),
                        min_value=0.0, format="%.1f", key="an_chamber_vol",
                    )

                st.divider()
                notes = st.text_area("Notes", value=loaded_meta.get('notes', ''), height=68, key="an_notes")

            # ── Main column: QC + requirements + run ─────────────────────
            with an_main:
                # --- QC Checks ---
                st.markdown("#### Quality Control")

                with st.spinner("Running QC checks..."):
                    try:
                        qc_report = run_qc_checks(df, config)
                        st.session_state.qc_report = qc_report

                        qc1, qc2, qc3, qc4 = st.columns(4)
                        with qc1:
                            qc_color = _ZINC['success'] if qc_report.passed else _ZINC['error']
                            qc_label = "PASSED" if qc_report.passed else "FAILED"
                            st.markdown(
                                f'<div style="background:{"#dcfce7" if qc_report.passed else "#fee2e2"};'
                                f'border-radius:8px;padding:12px;text-align:center;">'
                                f'<span style="color:{qc_color};font-weight:700;font-size:1.1rem;">{qc_label}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with qc2:
                            render_metric_card("Passed", str(qc_report.summary.get('passed', 0)))
                        with qc3:
                            render_metric_card("Warnings", str(qc_report.summary.get('warnings', 0)))
                        with qc4:
                            render_metric_card("Failed", str(qc_report.summary.get('failed', 0)))

                        st.markdown("")

                        with st.expander("QC Check Details"):
                            for check in qc_report.checks:
                                if check.status.name == "PASS":
                                    icon, color = "&#10003;", _ZINC['success']
                                elif check.status.name == "WARN":
                                    icon, color = "&#9888;", _ZINC['warning']
                                else:
                                    icon, color = "&#10007;", _ZINC['error']
                                st.markdown(
                                    f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
                                    f'border-bottom:1px solid #f4f4f5;">'
                                    f'<span style="color:{color};font-weight:700;">{icon}</span>'
                                    f'<span style="font-weight:500;">{check.name}</span>'
                                    f'<span style="color:#71717a;font-size:0.85rem;margin-left:auto;">{check.message}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                    except Exception as e:
                        st.error(f"QC error: {e}")
                        qc_report = None

                st.divider()

                # --- Parameter Requirements ---
                st.markdown("#### Parameter Requirements")

                available_columns = list(df.columns)

                if test_type == 'cold_flow':
                    requirements = [
                        ("Upstream Pressure", "sensor", "upstream_pressure", True),
                        ("Mass Flow", "sensor", "mass_flow", True),
                        ("Downstream Pressure", "sensor", "downstream_pressure", False),
                        ("Fluid Temperature", "sensor_role", "fluid_temperature", False),
                        ("Orifice Area", "geometry", "orifice_area_mm2", True),
                        ("Test Fluid", "metadata", "test_fluid", False),
                    ]
                else:
                    requirements = [
                        ("Chamber Pressure", "sensor", "chamber_pressure", True),
                        ("Thrust", "sensor", "thrust", True),
                        ("Mass Flow (Ox)", "sensor", "mass_flow_ox", True),
                        ("Mass Flow (Fuel)", "sensor", "mass_flow_fuel", True),
                        ("Throat Area", "geometry", "throat_area_mm2", True),
                    ]

                req_rows = []
                all_required_ok = True
                for name, src_type, src_key, required in requirements:
                    if src_type in ('sensor', 'sensor_role'):
                        # Check sensor_roles first, then config columns
                        sensor_name = sensor_roles.get(src_key) or config.get('columns', {}).get(src_key)
                        if sensor_name and sensor_name in available_columns:
                            req_rows.append({"Status": "&#10003;", "Parameter": name, "Source": sensor_name})
                        elif sensor_name:
                            req_rows.append({"Status": "&#10007;", "Parameter": name, "Source": f"{sensor_name} (not in data)"})
                            if required:
                                all_required_ok = False
                        else:
                            req_rows.append({"Status": "&#10007;" if required else "&#9675;", "Parameter": name, "Source": "(not configured)"})
                            if required:
                                all_required_ok = False
                    elif src_type == 'geometry':
                        val = form_geom.get(src_key, 0)
                        if val > 0:
                            req_rows.append({"Status": "&#10003;", "Parameter": name, "Source": f"{val}"})
                        else:
                            req_rows.append({"Status": "&#10007;" if required else "&#9675;", "Parameter": name, "Source": "(not set)"})
                            if required:
                                all_required_ok = False
                    elif src_type == 'metadata':
                        val = test_fluid if src_key == 'test_fluid' else loaded_meta.get(src_key)
                        if val:
                            req_rows.append({"Status": "&#10003;", "Parameter": name, "Source": str(val)})
                        else:
                            req_rows.append({"Status": "&#9675;", "Parameter": name, "Source": "(not set)"})

                # Render requirements as styled HTML table
                table_html = (
                    '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
                    '<thead><tr style="border-bottom:2px solid #e4e4e7;">'
                    '<th style="text-align:left;padding:8px;">Status</th>'
                    '<th style="text-align:left;padding:8px;">Parameter</th>'
                    '<th style="text-align:left;padding:8px;">Source</th>'
                    '</tr></thead><tbody>'
                )
                for row in req_rows:
                    color = _ZINC['success'] if '10003' in row['Status'] else (
                        _ZINC['error'] if '10007' in row['Status'] else _ZINC['muted']
                    )
                    table_html += (
                        f'<tr style="border-bottom:1px solid #f4f4f5;">'
                        f'<td style="padding:6px 8px;color:{color};font-weight:700;">{row["Status"]}</td>'
                        f'<td style="padding:6px 8px;font-weight:500;">{row["Parameter"]}</td>'
                        f'<td style="padding:6px 8px;color:#71717a;">{row["Source"]}</td>'
                        f'</tr>'
                    )
                table_html += '</tbody></table>'
                st.markdown(table_html, unsafe_allow_html=True)

                st.markdown("")

                # --- Run Analysis ---
                st.divider()
                st.markdown("#### Run Analysis")

                run_col1, run_col2 = st.columns([3, 1])
                with run_col2:
                    skip_qc = st.checkbox("Skip QC gate", value=False, key="an_skip_qc",
                                          help="Allow analysis even if QC fails")

                with run_col1:
                    run_btn = st.button(
                        "Run Full Analysis", type="primary",
                        use_container_width=True, key="an_run",
                    )

                if run_btn:
                    with st.spinner("Running analysis pipeline..."):
                        try:
                            # Extract fluid conditions from sensor data
                            fluid_temperature_K = 293.15
                            fluid_pressure_Pa = 101325.0
                            sw_start, sw_end = st.session_state.steady_window
                            tc = _detect_time_column(df) or 'time_s'

                            if tc in df.columns:
                                steady_df = df[(df[tc] >= sw_start) & (df[tc] <= sw_end)]
                                temp_sensor = sensor_roles.get('fluid_temperature')
                                if temp_sensor and temp_sensor in df.columns:
                                    fluid_temperature_K = steady_df[temp_sensor].mean()
                                    if fluid_temperature_K < 200:
                                        fluid_temperature_K += 273.15
                                press_sensor = sensor_roles.get('fluid_pressure')
                                if press_sensor and press_sensor in df.columns:
                                    fluid_pressure_Pa = steady_df[press_sensor].mean() * 1e5

                            geometry_dict = _build_geometry_dict(test_type, form_geom)

                            metadata = {
                                'part_number': part_number,
                                'serial_number': serial_number,
                                'operator': operator,
                                'notes': notes,
                                'test_folder': st.session_state.get('test_folder_path'),
                                'test_fluid': test_fluid,
                                'fluid_temperature_K': fluid_temperature_K,
                                'fluid_pressure_Pa': fluid_pressure_Pa,
                                'sensor_roles': sensor_roles,
                                'geometry': geometry_dict,
                            }

                            # Add fluid properties
                            if test_fluid and FLUID_PROPS_SUPPORT:
                                try:
                                    fp = get_fluid_properties(test_fluid, fluid_temperature_K, fluid_pressure_Pa)
                                    if fp:
                                        metadata['fluid_density_kg_m3'] = fp.density_kg_m3
                                        metadata['fluid_density_uncertainty_kg_m3'] = fp.density_kg_m3 * fp.density_uncertainty
                                        metadata['fluid_viscosity_Pa_s'] = fp.viscosity_Pa_s
                                        metadata['fluid_phase'] = fp.phase
                                        metadata['fluid_source'] = fp.source
                                        metadata['fluid'] = {
                                            'name': test_fluid,
                                            'temperature_k': fluid_temperature_K,
                                            'pressure_pa': fluid_pressure_Pa,
                                            'density_kg_m3': fp.density_kg_m3,
                                            'density_uncertainty_kg_m3': fp.density_kg_m3 * fp.density_uncertainty,
                                            'viscosity_pa_s': fp.viscosity_Pa_s,
                                            'phase': fp.phase,
                                        }
                                except Exception:
                                    pass

                            if test_fluid and 'fluid' not in metadata:
                                metadata['fluid'] = {
                                    'name': test_fluid, 'density_kg_m3': 1.0,
                                    'density_uncertainty_kg_m3': 0.1,
                                }

                            # Determine file path for traceability
                            file_path = None
                            tfp = st.session_state.get('test_folder_path')
                            if tfp and TEST_FOLDER_SUPPORT:
                                raw_file = find_raw_data_file(tfp)
                                if raw_file:
                                    file_path = str(raw_file)

                            if test_type == "cold_flow":
                                result = analyze_cold_flow_test(
                                    df=df, config=config,
                                    steady_window=st.session_state.steady_window,
                                    test_id=test_id, file_path=file_path,
                                    metadata=metadata, skip_qc=skip_qc,
                                )
                            else:
                                result = analyze_hot_fire_test(
                                    df=df, config=config,
                                    steady_window=st.session_state.steady_window,
                                    test_id=test_id, file_path=file_path,
                                    metadata=metadata, skip_qc=skip_qc,
                                )

                            st.session_state.analysis_result = result

                            st.success("Analysis complete! View results in the **Results** tab.")

                            # Quick preview of key metrics
                            metrics = list(result.measurements.items())[:4]
                            if metrics:
                                prev_cols = st.columns(len(metrics))
                                for i, (name, meas) in enumerate(metrics):
                                    with prev_cols[i]:
                                        if hasattr(meas, 'value'):
                                            render_metric_card(
                                                name,
                                                f"{meas.value:.4g}",
                                                f"+/-{meas.uncertainty:.4g} ({meas.relative_uncertainty_percent:.1f}%)",
                                            )
                                        else:
                                            render_metric_card(name, f"{meas:.4g}")

                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            with st.expander("Traceback"):
                                st.code(traceback.format_exc())

    except Exception as exc:
        st.error(f"Analysis tab error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 4: RESULTS
# #############################################################################
with tab_results:
    try:
        st.subheader("Analysis Results")

        result = st.session_state.analysis_result
        df = st.session_state.df_processed or st.session_state.df
        config = st.session_state.active_config or ConfigManager.get_default_config(test_type)

        if result is None:
            st.markdown(
                '<div style="text-align:center;padding:4rem 2rem;color:#71717a;">'
                '<div style="font-size:3rem;margin-bottom:1rem;">&#128202;</div>'
                '<h3 style="color:#18181b;">No Results Yet</h3>'
                '<p>Run an analysis in the <strong>Analyze</strong> tab to see results here.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            # ── Key Metric Cards ─────────────────────────────────────────
            metrics = list(result.measurements.items())
            top_metrics = metrics[:4]
            if top_metrics:
                cols = st.columns(len(top_metrics))
                for i, (name, meas) in enumerate(top_metrics):
                    with cols[i]:
                        if hasattr(meas, 'value'):
                            render_metric_card(
                                name,
                                f"{meas.value:.4g}",
                                f"+/-{meas.uncertainty:.4g} ({meas.relative_uncertainty_percent:.1f}%)",
                            )
                        else:
                            render_metric_card(name, f"{meas:.4g}" if isinstance(meas, (int, float)) else str(meas))

            st.markdown("")

            # ── Measurements Table ───────────────────────────────────────
            with st.expander("All Measurements with Uncertainties", expanded=True):
                table_data = []
                for name, meas in result.measurements.items():
                    if hasattr(meas, 'value'):
                        table_data.append({
                            'Parameter': name,
                            'Value': f"{meas.value:.4g}",
                            'Uncertainty (1 sigma)': f"+/-{meas.uncertainty:.4g}",
                            'Relative (%)': f"{meas.relative_uncertainty_percent:.2f}",
                            'Unit': getattr(meas, 'unit', '-'),
                        })
                    else:
                        table_data.append({
                            'Parameter': name,
                            'Value': f"{meas:.4g}" if isinstance(meas, (int, float)) else str(meas),
                            'Uncertainty (1 sigma)': '-',
                            'Relative (%)': '-',
                            'Unit': '-',
                        })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            # ── Traceability ─────────────────────────────────────────────
            with st.expander("Traceability Record"):
                trace = result.traceability
                tr1, tr2 = st.columns(2)
                with tr1:
                    st.markdown("**Data Provenance**")
                    trace_items = [
                        ("Raw Data Hash", trace.get('raw_data_hash', 'N/A')),
                        ("Config Hash", trace.get('config_hash', 'N/A')),
                        ("Processing Version", trace.get('processing_version', 'N/A')),
                    ]
                    for label, val in trace_items:
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                            f'border-bottom:1px solid #f4f4f5;">'
                            f'<span style="font-weight:500;font-size:0.85rem;">{label}</span>'
                            f'<code style="font-size:0.8rem;color:#71717a;">{val}</code>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                with tr2:
                    st.markdown("**Analysis Context**")
                    ctx_items = [
                        ("Analyst", trace.get('analyst_username', 'N/A')),
                        ("Timestamp", trace.get('analysis_timestamp_utc', 'N/A')),
                        ("Steady Window",
                         f"{trace.get('steady_window_start_ms', 'N/A')} - "
                         f"{trace.get('steady_window_end_ms', 'N/A')} ms"),
                    ]
                    for label, val in ctx_items:
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                            f'border-bottom:1px solid #f4f4f5;">'
                            f'<span style="font-weight:500;font-size:0.85rem;">{label}</span>'
                            f'<span style="font-size:0.8rem;color:#71717a;">{val}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # ── QC Summary ───────────────────────────────────────────────
            qc_report = st.session_state.qc_report
            if qc_report:
                with st.expander("QC Summary"):
                    qc_status = "PASSED" if qc_report.passed else "FAILED"
                    qc_color = _ZINC['success'] if qc_report.passed else _ZINC['error']
                    st.markdown(
                        f'<span style="color:{qc_color};font-weight:700;">{qc_status}</span> '
                        f'&mdash; {qc_report.summary.get("passed", 0)} passed, '
                        f'{qc_report.summary.get("warnings", 0)} warnings, '
                        f'{qc_report.summary.get("failed", 0)} failed',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── Quick Iteration Mode ─────────────────────────────────────
            st.markdown("#### Quick Iteration")
            st.caption("Test different steady-state windows and compare results.")

            enable_iter = st.checkbox("Enable iteration mode", key="res_enable_iter")

            if enable_iter and df is not None:
                time_col = _detect_time_column(df) or 'time_s'
                if time_col in df.columns:
                    t_min = float(df[time_col].min())
                    t_max = float(df[time_col].max())
                    sw = st.session_state.steady_window or (t_min, t_max)

                    iter_s, iter_m = st.columns([1, 3])

                    with iter_s:
                        new_start = st.slider(
                            "Start (s)", t_min, t_max, float(sw[0]),
                            step=0.1, format="%.2f", key="iter_start",
                        )
                        new_end = st.slider(
                            "End (s)", t_min, t_max, float(sw[1]),
                            step=0.1, format="%.2f", key="iter_end",
                        )

                    with iter_m:
                        if new_start >= new_end:
                            st.warning("Start must be less than end.")
                        else:
                            changed = (new_start != sw[0]) or (new_end != sw[1])
                            if changed:
                                try:
                                    # Build minimal metadata for iteration
                                    loaded_meta_iter = st.session_state.get('loaded_metadata') or {}
                                    iter_metadata = {
                                        'part_number': st.session_state.get('an_part', ''),
                                        'serial_number': st.session_state.get('an_serial', ''),
                                        'test_fluid': st.session_state.get('an_fluid', ''),
                                        'geometry': loaded_meta_iter.get('geometry', {}),
                                        'sensor_roles': loaded_meta_iter.get('sensor_roles', {}),
                                    }
                                    if iter_metadata.get('test_fluid'):
                                        iter_metadata['fluid'] = {
                                            'name': iter_metadata['test_fluid'],
                                            'density_kg_m3': 1.0,
                                            'density_uncertainty_kg_m3': 0.1,
                                        }

                                    tid = st.session_state.get('an_test_id', 'TEST') + "_iter"

                                    if test_type == "cold_flow":
                                        new_result = analyze_cold_flow_test(
                                            df=df, config=config,
                                            steady_window=(new_start, new_end),
                                            test_id=tid, metadata=iter_metadata, skip_qc=True,
                                        )
                                    else:
                                        new_result = analyze_hot_fire_test(
                                            df=df, config=config,
                                            steady_window=(new_start, new_end),
                                            test_id=tid, metadata=iter_metadata, skip_qc=True,
                                        )

                                    st.success(f"Window: {new_start:.2f}s - {new_end:.2f}s")
                                    iter_metrics = list(new_result.measurements.items())[:4]
                                    if iter_metrics:
                                        ic = st.columns(len(iter_metrics))
                                        for i, (name, meas) in enumerate(iter_metrics):
                                            with ic[i]:
                                                if hasattr(meas, 'value'):
                                                    orig = result.measurements.get(name)
                                                    delta = ""
                                                    if orig and hasattr(orig, 'value') and orig.value != 0:
                                                        d = (meas.value - orig.value) / orig.value * 100
                                                        delta = f"{d:+.2f}% vs original"
                                                    render_metric_card(name, f"{meas.value:.4g}", delta)
                                                else:
                                                    render_metric_card(name, f"{meas:.4g}")

                                    # Save to comparison
                                    if st.button("Save to comparison", key="iter_save"):
                                        st.session_state.iteration_results.append({
                                            'window': (new_start, new_end),
                                            'duration': new_end - new_start,
                                            'measurements': {
                                                k: getattr(v, 'value', v)
                                                for k, v in new_result.measurements.items()
                                            },
                                        })
                                        st.rerun()

                                except Exception as e:
                                    st.error(f"Iteration error: {e}")

                    # Comparison table
                    if st.session_state.iteration_results:
                        st.divider()
                        st.markdown("**Parameter Sweep**")
                        comp_data = []
                        for idx, ir in enumerate(st.session_state.iteration_results):
                            row = {
                                '#': idx + 1,
                                'Start (s)': f"{ir['window'][0]:.2f}",
                                'End (s)': f"{ir['window'][1]:.2f}",
                                'Duration (s)': f"{ir['duration']:.2f}",
                            }
                            for j, (name, val) in enumerate(list(ir['measurements'].items())[:6]):
                                row[name] = f"{val:.4g}" if isinstance(val, (int, float)) else str(val)
                            comp_data.append(row)

                        comp_df = pd.DataFrame(comp_data)
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)

                        bc1, bc2 = st.columns([1, 3])
                        with bc1:
                            if st.button("Clear", key="iter_clear"):
                                st.session_state.iteration_results = []
                                st.rerun()
                        with bc2:
                            st.download_button(
                                "Download sweep (CSV)",
                                comp_df.to_csv(index=False),
                                file_name=f"{st.session_state.get('an_test_id', 'test')}_sweep.csv",
                                mime="text/csv", key="iter_download",
                            )

    except Exception as exc:
        st.error(f"Results tab error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# #############################################################################
# TAB 5: SAVE & EXPORT
# #############################################################################
with tab_export:
    try:
        st.subheader("Save & Export")
        st.caption("Save results to a campaign database, generate reports, and download data.")

        result = st.session_state.analysis_result
        config = st.session_state.active_config or ConfigManager.get_default_config(test_type)
        qc_report = st.session_state.qc_report

        if result is None:
            st.markdown(
                '<div style="text-align:center;padding:4rem 2rem;color:#71717a;">'
                '<div style="font-size:3rem;margin-bottom:1rem;">&#128230;</div>'
                '<h3 style="color:#18181b;">No Results to Export</h3>'
                '<p>Complete an analysis first, then return here to save and export.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            ex_col1, ex_col2, ex_col3 = st.columns(3)

            # ── Save to Campaign ─────────────────────────────────────────
            with ex_col1:
                st.markdown("#### Save to Campaign")

                test_id = st.session_state.get('an_test_id', 'TEST')
                part_number = st.session_state.get('an_part', '')
                serial_number = st.session_state.get('an_serial', '')
                operator = st.session_state.get('an_operator', '')
                notes = st.session_state.get('an_notes', '')

                try:
                    from core.campaign_manager_v2 import suggest_campaign_name, check_campaign_exists, create_campaign

                    suggested = suggest_campaign_name(test_id, test_type)
                    if suggested:
                        exists = check_campaign_exists(suggested)
                        if exists:
                            st.info(f"Detected: **{suggested}**")
                        else:
                            st.info(f"Suggested: **{suggested}**")
                            if st.button(f"Create '{suggested}'", key="ex_create_campaign"):
                                create_campaign(suggested, test_type, "Auto-created")
                                st.rerun()
                except ImportError:
                    pass

                campaigns = get_available_campaigns()
                camp_options = [c['name'] for c in campaigns if c.get('type') == test_type]

                if camp_options:
                    default_idx = 0
                    try:
                        suggested = suggest_campaign_name(
                            st.session_state.get('an_test_id', ''), test_type,
                        )
                        if suggested and suggested in camp_options:
                            default_idx = camp_options.index(suggested)
                    except Exception:
                        pass

                    selected_campaign = st.selectbox(
                        "Campaign", camp_options, index=default_idx, key="ex_campaign",
                    )

                    if st.button("Save to Campaign", type="primary", use_container_width=True, key="ex_save"):
                        try:
                            record = result.to_database_record(test_type)
                            record['part'] = part_number
                            record['serial_num'] = serial_number
                            record['operator'] = operator
                            record['notes'] = notes
                            save_to_campaign(selected_campaign, record)
                            st.success(f"Saved to **{selected_campaign}**")
                        except Exception as e:
                            st.error(f"Save error: {e}")
                            with st.expander("Details"):
                                st.code(traceback.format_exc())
                else:
                    st.info(f"No {test_type} campaigns found.")

            # ── Generate Report ──────────────────────────────────────────
            with ex_col2:
                st.markdown("#### HTML Report")

                if st.button("Generate Report", type="primary", use_container_width=True, key="ex_report"):
                    try:
                        with st.spinner("Generating..."):
                            measurements_dict = {k: v for k, v in result.measurements.items()}
                            qc_dict = {
                                'passed': result.passed_qc,
                                'summary': qc_report.summary if qc_report else {},
                                'checks': [
                                    {'name': c.name, 'status': c.status.name, 'message': c.message}
                                    for c in (qc_report.checks if qc_report else [])
                                ],
                            }
                            html = generate_test_report(
                                test_id=test_id,
                                test_type=test_type,
                                measurements=measurements_dict,
                                traceability=result.traceability,
                                qc_report=qc_dict,
                                metadata={'part': part_number, 'serial': serial_number},
                                config=config,
                                include_config_snapshot=True,
                            )
                            st.download_button(
                                "Download Report",
                                html,
                                file_name=f"{test_id}_report.html",
                                mime="text/html",
                                key="ex_dl_report",
                            )
                            st.success("Report ready.")
                    except Exception as e:
                        st.error(f"Report error: {e}")
                        with st.expander("Details"):
                            st.code(traceback.format_exc())

            # ── Export Data ──────────────────────────────────────────────
            with ex_col3:
                st.markdown("#### Export Data")

                export_format = st.selectbox("Format", ["JSON", "CSV"], key="ex_format")

                if st.button("Export", type="primary", use_container_width=True, key="ex_export"):
                    try:
                        if export_format == "JSON":
                            export_data = {
                                'test_id': test_id,
                                'test_type': test_type,
                                'measurements': {
                                    k: {
                                        'value': v.value if hasattr(v, 'value') else v,
                                        'uncertainty': v.uncertainty if hasattr(v, 'uncertainty') else None,
                                        'unit': getattr(v, 'unit', None),
                                    }
                                    for k, v in result.measurements.items()
                                },
                                'traceability': result.traceability,
                                'qc_passed': result.passed_qc,
                            }
                            st.download_button(
                                "Download JSON",
                                json.dumps(export_data, indent=2, default=str),
                                file_name=f"{test_id}_results.json",
                                mime="application/json",
                                key="ex_dl_json",
                            )
                        else:
                            csv_data = []
                            for k, v in result.measurements.items():
                                csv_data.append({
                                    'parameter': k,
                                    'value': v.value if hasattr(v, 'value') else v,
                                    'uncertainty': v.uncertainty if hasattr(v, 'uncertainty') else None,
                                    'unit': getattr(v, 'unit', '-'),
                                })
                            csv_df = pd.DataFrame(csv_data)
                            st.download_button(
                                "Download CSV",
                                csv_df.to_csv(index=False),
                                file_name=f"{test_id}_results.csv",
                                mime="text/csv",
                                key="ex_dl_csv",
                            )
                        st.success("Export ready.")
                    except Exception as e:
                        st.error(f"Export error: {e}")

    except Exception as exc:
        st.error(f"Export tab error: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
