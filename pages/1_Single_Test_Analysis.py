"""
Single Test Analysis Page
=========================
Analyze individual cold flow or hot fire tests with:
- Data preprocessing (time conversion, NaN handling)
- CV-based and ML-based steady-state detection
- Full uncertainty propagation
- Pre-analysis QC checks
- Traceability records
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import tempfile
from datetime import datetime

# Import core modules
from core.integrated_analysis import analyze_cold_flow_test, analyze_hot_fire_test, format_measurement_table
from core.qc_checks import run_qc_checks, format_qc_for_display
from core.config_validation import validate_config_simple
from core.traceability import compute_file_hash
from core.reporting import generate_test_report, save_report
from core.campaign_manager_v2 import get_available_campaigns, save_to_campaign
from core.config_manager import ConfigManager
from core.steady_state_detection import (
    detect_steady_state_cv,
    detect_steady_state_ml,
    detect_steady_state_derivative,
    validate_steady_window
)

st.set_page_config(page_title="Single Test Analysis", page_icon="STA", layout="wide")

st.title("Single Test Analysis")
st.markdown("Analyze individual cold flow or hot fire tests with full engineering integrity.")

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'steady_window' not in st.session_state:
    st.session_state.steady_window = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'detection_sensor' not in st.session_state:
    st.session_state.detection_sensor = None  # Track which sensor was used for detection

# Initialize detection preferences (persistent across tests)
if 'detection_preferences' not in st.session_state:
    st.session_state.detection_preferences = {
        'method': 'CV-based (Simple)',
        'cv_threshold': 0.02,
        'cv_window_size': 50,
        'ml_contamination': 0.3,
        'deriv_threshold': 0.1,
        'window_adjust_step': 0.1,  # Step size for manual adjustment (seconds)
    }


# =============================================================================
# DATA PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_data(df: pd.DataFrame, config: dict, time_unit: str = 'ms',
                    shift_to_zero: bool = True) -> pd.DataFrame:
    """
    Preprocess raw test data:
    - Convert timestamp to seconds (time_s column)
    - Optionally shift time to start at 0
    - Sort by timestamp
    - Remove duplicates

    Args:
        df: Raw DataFrame
        config: Configuration dict
        time_unit: Original time unit ('ms', 's', 'us')
        shift_to_zero: If True, shift time so it starts at 0

    Returns:
        Preprocessed DataFrame with time_s column
    """
    df_proc = df.copy()

    # Get timestamp column
    timestamp_col = config.get('columns', {}).get('timestamp', 'timestamp')

    if timestamp_col in df_proc.columns:
        # Sort by timestamp
        df_proc = df_proc.sort_values(timestamp_col).reset_index(drop=True)

        # Remove duplicate timestamps
        df_proc = df_proc.drop_duplicates(subset=[timestamp_col], keep='first')

        # Convert to seconds
        if time_unit == 'ms':
            df_proc['time_s'] = df_proc[timestamp_col] / 1000.0
        elif time_unit == 'us':
            df_proc['time_s'] = df_proc[timestamp_col] / 1_000_000.0
        else:  # Already seconds
            df_proc['time_s'] = df_proc[timestamp_col].astype(float)

        # Shift to zero
        if shift_to_zero:
            t_start = df_proc['time_s'].iloc[0]
            df_proc['time_s'] = df_proc['time_s'] - t_start

        # Also create time_ms for compatibility
        df_proc['time_ms'] = df_proc['time_s'] * 1000.0

    return df_proc


def resample_data(df: pd.DataFrame, target_rate_hz: float,
                  time_col: str = 'time_s') -> tuple[pd.DataFrame, dict]:
    """
    Resample data to a uniform sample rate.

    Uses linear interpolation to resample irregular data to a fixed rate.

    Args:
        df: DataFrame with time_s column
        target_rate_hz: Target sample rate in Hz
        time_col: Time column to use

    Returns:
        Tuple of (resampled DataFrame, statistics dict)
    """
    if time_col not in df.columns:
        return df, {'error': f'Time column {time_col} not found'}

    stats = {
        'original_rows': len(df),
        'original_duration_s': float(df[time_col].max() - df[time_col].min()),
        'target_rate_hz': target_rate_hz,
    }

    # Calculate original effective sample rate
    dt = np.diff(df[time_col].values)
    if len(dt) > 0:
        stats['original_mean_rate_hz'] = float(1.0 / np.mean(dt)) if np.mean(dt) > 0 else 0
        stats['original_min_rate_hz'] = float(1.0 / np.max(dt)) if np.max(dt) > 0 else 0
        stats['original_max_rate_hz'] = float(1.0 / np.min(dt)) if np.min(dt) > 0 else 0

    # Create new uniform time vector
    t_start = df[time_col].min()
    t_end = df[time_col].max()
    dt_target = 1.0 / target_rate_hz

    new_time = np.arange(t_start, t_end, dt_target)
    stats['resampled_rows'] = len(new_time)

    # Interpolate each numeric column
    df_resampled = pd.DataFrame({time_col: new_time})

    # Also add time_ms if we're using time_s
    if time_col == 'time_s':
        df_resampled['time_ms'] = new_time * 1000.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in [time_col, 'time_s', 'time_ms']:
            continue

        # Use numpy interpolation for speed
        df_resampled[col] = np.interp(
            new_time,
            df[time_col].values,
            df[col].values
        )

    stats['columns_resampled'] = [c for c in df_resampled.columns if c not in [time_col, 'time_s', 'time_ms']]

    return df_resampled, stats


def handle_nan_values(df: pd.DataFrame, method: str = 'interpolate',
                      max_gap: int = 5) -> tuple[pd.DataFrame, dict]:
    """
    Handle NaN values in the data.

    Args:
        df: DataFrame with potential NaN values
        method: 'interpolate', 'drop', or 'ffill'
        max_gap: Maximum consecutive NaNs to interpolate

    Returns:
        Tuple of (processed DataFrame, statistics dict)
    """
    stats = {
        'original_rows': len(df),
        'nan_counts': {},
        'rows_affected': 0,
        'method': method,
    }

    df_clean = df.copy()

    # Count NaNs per column
    for col in df_clean.columns:
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            stats['nan_counts'][col] = int(nan_count)

    stats['rows_affected'] = int(df_clean.isna().any(axis=1).sum())

    if method == 'interpolate':
        # Interpolate small gaps, leave large gaps as NaN
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            # Find gap sizes
            mask = df_clean[col].isna()
            if mask.any():
                # Group consecutive NaNs
                groups = (mask != mask.shift()).cumsum()
                gap_sizes = mask.groupby(groups).transform('sum')

                # Only interpolate small gaps
                small_gaps = mask & (gap_sizes <= max_gap)
                if small_gaps.any():
                    df_clean[col] = df_clean[col].interpolate(method='linear', limit=max_gap)

    elif method == 'drop':
        df_clean = df_clean.dropna()

    elif method == 'ffill':
        df_clean = df_clean.ffill().bfill()

    stats['final_rows'] = len(df_clean)
    stats['rows_removed'] = stats['original_rows'] - stats['final_rows']

    return df_clean, stats


# =============================================================================
# STEADY-STATE DETECTION FUNCTIONS
# =============================================================================

# Steady-state detection functions now imported from core.steady_state_detection
# (detect_steady_state_cv, detect_steady_state_ml, detect_steady_state_derivative)


def apply_channel_mapping(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Apply channel mapping from config to rename DAQ channel IDs to sensor names.

    Args:
        df: DataFrame with raw channel names (e.g., "10001", "10002")
        config: Configuration dict with 'channel_config' mapping

    Returns:
        Tuple of (DataFrame with renamed columns, mapping stats)
    """
    channel_config = config.get('channel_config', {})

    if not channel_config:
        return df, {'applied': False, 'reason': 'No channel_config in config'}

    stats = {
        'applied': True,
        'mappings_found': 0,
        'mappings_applied': [],
        'unmapped_columns': [],
    }

    df_mapped = df.copy()
    rename_map = {}

    for raw_id, sensor_name in channel_config.items():
        # Handle both string and int column names
        if raw_id in df_mapped.columns:
            rename_map[raw_id] = sensor_name
            stats['mappings_applied'].append(f"{raw_id} -> {sensor_name}")
            stats['mappings_found'] += 1
        elif str(raw_id) in df_mapped.columns:
            rename_map[str(raw_id)] = sensor_name
            stats['mappings_applied'].append(f"{raw_id} -> {sensor_name}")
            stats['mappings_found'] += 1
        # Try as integer if columns are numeric
        try:
            int_id = int(raw_id)
            if int_id in df_mapped.columns:
                rename_map[int_id] = sensor_name
                stats['mappings_applied'].append(f"{raw_id} -> {sensor_name}")
                stats['mappings_found'] += 1
        except (ValueError, TypeError):
            pass

    # Find unmapped columns (excluding known system columns)
    system_cols = {'timestamp', 'time_s', 'time_ms', 'index'}
    mapped_cols = set(rename_map.keys()) | set(rename_map.values()) | system_cols
    for col in df_mapped.columns:
        if col not in mapped_cols and str(col) not in mapped_cols:
            stats['unmapped_columns'].append(str(col))

    if rename_map:
        df_mapped = df_mapped.rename(columns=rename_map)

    return df_mapped, stats


# Default configuration functions now provided by ConfigManager.get_default_config()
# (replaced create_default_cold_flow_config and create_default_hot_fire_config)


def plot_data_with_window(df, config, steady_window, detection_sensor=None):
    """
    Create interactive plot with steady-state window.

    Args:
        df: DataFrame with test data
        config: Configuration dict
        steady_window: (start, end) tuple for steady-state window
        detection_sensor: Sensor used for detection (shown first, highlighted)
    """
    # Prefer time_s, fall back to timestamp column
    if 'time_s' in df.columns:
        time_col = 'time_s'
        time_label = "Time (s)"
    else:
        time_col = config.get('columns', {}).get('timestamp', 'timestamp')
        time_label = "Time (ms)"

    if time_col not in df.columns:
        st.warning("No timestamp column found")
        return None

    # Identify data columns (exclude time columns)
    exclude_cols = {time_col, 'time_s', 'time_ms', 'timestamp'}
    all_data_cols = [c for c in df.columns
                     if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    if not all_data_cols:
        st.warning("No numeric data columns found")
        return None

    # Determine which sensors to plot
    plot_cols = []

    # Add detection sensor first if specified
    if detection_sensor and detection_sensor in all_data_cols:
        plot_cols.append(detection_sensor)

    # Add other sensors (up to 4 total)
    for col in all_data_cols:
        if col not in plot_cols:
            plot_cols.append(col)
            if len(plot_cols) >= 4:
                break

    # Create subplots
    n_cols = len(plot_cols)
    fig = make_subplots(
        rows=n_cols, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{c}{' (detection)' if c == detection_sensor else ''}" for c in plot_cols]
    )

    for i, col in enumerate(plot_cols):
        # Highlight detection sensor
        line_color = 'blue' if col == detection_sensor else None
        line_width = 2 if col == detection_sensor else 1

        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(width=line_width, color=line_color),
            ),
            row=i+1, col=1
        )

        # Add steady-state window
        if steady_window:
            fig.add_vrect(
                x0=steady_window[0],
                x1=steady_window[1],
                fillcolor="green",
                opacity=0.15,
                line_width=0,
                row=i+1, col=1
            )

    fig.update_layout(
        height=150 * n_cols,
        showlegend=False,
        title="Test Data with Steady-State Window (green)",
        margin=dict(t=50, b=40)
    )
    fig.update_xaxes(title_text=time_label, row=n_cols, col=1)

    return fig


# =============================================================================
# SIDEBAR - Configuration
# =============================================================================

with st.sidebar:
    st.header("Configuration")

    test_type = st.selectbox(
        "Test Type",
        ["cold_flow", "hot_fire"],
        format_func=lambda x: "Cold Flow" if x == "cold_flow" else "Hot Fire"
    )

    # Configuration Source - Unified selector
    st.subheader("Configuration Source")

    config_source = st.radio(
        "Select configuration source",
        ["Recent Configs", "Saved Configs", "Upload JSON", "Manual"],
        label_visibility="collapsed",
        help="Choose where to load your configuration from"
    )

    config = None  # Will be set based on user selection

    # ===== RECENT CONFIGS =====
    if config_source == "Recent Configs":
        recent_configs = ConfigManager.get_recent_configs(limit=5)

        if recent_configs:
            # Create options for recent configs
            recent_options = ["-- Select Recent Config --"] + [
                f"{r['info']['config_name']} ({r['info']['source']}, {r['info']['timestamp'][:10]})"
                for r in recent_configs
            ]

            selected_recent = st.selectbox(
                "Recent Configurations",
                recent_options,
                help="Select from your 5 most recently used configurations"
            )

            if selected_recent != "-- Select Recent Config --":
                # Find the selected config
                idx = recent_options.index(selected_recent) - 1  # -1 for placeholder
                config = recent_configs[idx]['config']
                st.success(f"✓ Loaded: {recent_configs[idx]['info']['config_name']}")
            else:
                # Fallback to default if nothing selected
                config = ConfigManager.get_default_config(test_type)
        else:
            st.info("No recent configs yet. Use Saved Configs or upload a config file.")
            config = ConfigManager.get_default_config(test_type)

    # ===== SAVED CONFIGS =====
    elif config_source == "Saved Configs":
        try:
            from core.saved_configs import SavedConfigManager, load_saved_config

            manager = SavedConfigManager()
            templates = manager.list_templates()

            if templates:
                # Filter by test type
                filtered_templates = [t for t in templates if t.get('test_type') == test_type]

                if filtered_templates:
                    # Create template options
                    template_options = ["-- Select Saved Config --"] + [
                        f"{t['id']} - {t.get('config_name', 'Unnamed')}"
                        for t in filtered_templates
                    ]

                    selected_template = st.selectbox(
                        f"Available {test_type.replace('_', ' ').title()} Saved Configs",
                        template_options,
                        help="Select a pre-configured testbench configuration"
                    )

                    if selected_template != "-- Select Saved Config --":
                        # Extract template ID
                        template_id = selected_template.split(" - ")[0]

                        # Show template info
                        template_info = next(t for t in filtered_templates if t['id'] == template_id)

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if template_info.get('description'):
                                st.caption(f"📝 {template_info['description']}")
                            if template_info.get('tags'):
                                st.caption(f"🏷️ Tags: {', '.join(template_info['tags'])}")

                        with col2:
                            if st.button("Load", key="load_saved_config_btn", use_container_width=True):
                                try:
                                    config = load_saved_config(template_id)
                                    ConfigManager.save_to_recent(config, 'saved_config', config.get('config_name', template_id))
                                    st.success(f"✓ Loaded: {template_id}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error loading saved config: {e}")

                    # Fallback if nothing selected
                    if config is None:
                        config = ConfigManager.get_default_config(test_type)
                else:
                    st.info(f"No {test_type} saved configs available. Create one in Saved Configurations page (Page 8).")
                    config = ConfigManager.get_default_config(test_type)
            else:
                st.info("No saved configs available. Create saved configs in Page 8 (Saved Configurations).")
                config = ConfigManager.get_default_config(test_type)

        except ImportError:
            st.warning("Saved config system not available. Use Upload JSON or Manual.")
            config = ConfigManager.get_default_config(test_type)

    # ===== UPLOAD JSON =====
    elif config_source == "Upload JSON":
        config_file = st.file_uploader("Upload config JSON", type=['json'])
        if config_file:
            try:
                config = json.load(config_file)
                st.success("✓ Configuration loaded from file")
                # Save to recent configs
                ConfigManager.save_to_recent(config, 'uploaded', config.get('config_name', 'Uploaded Config'))
            except Exception as e:
                st.error(f"Error loading config: {e}")
                config = ConfigManager.get_default_config(test_type)
        else:
            # No file uploaded yet - use default
            config = ConfigManager.get_default_config(test_type)
            st.info("Upload a JSON config file or use Manual to edit the default config below.")

    # ===== MANUAL =====
    else:  # Manual
        st.info("Manual configuration - edit the JSON in the expander below")
        config = ConfigManager.get_default_config(test_type)

    st.divider()

    # Show config expander for all cases
    with st.expander("📝 View/Edit Configuration"):
        config_str = st.text_area(
            "Config JSON",
            value=json.dumps(config, indent=2),
            height=300,
            key="config_editor"
        )
        if st.button("Apply Changes"):
            try:
                config = json.loads(config_str)
                st.success("Configuration updated")
                # Save to recent configs
                ConfigManager.save_to_recent(config, 'custom', config.get('config_name', 'Custom Config'))
                st.rerun()  # Refresh to update recent configs list
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Import test folder utilities
try:
    from core.test_metadata import (
        load_test_from_folder, load_test_metadata, find_raw_data_file,
        find_config_file, TestMetadata
    )
    TEST_FOLDER_SUPPORT = True
except ImportError:
    TEST_FOLDER_SUPPORT = False

# Import fluid properties module
try:
    from core.fluid_properties import (
        get_fluid_properties, FluidProperties, FluidState,
        COOLPROP_AVAILABLE, list_available_fluids
    )
    FLUID_PROPS_SUPPORT = True
except ImportError:
    COOLPROP_AVAILABLE = False
    FLUID_PROPS_SUPPORT = False

# Data source selection
st.header("1. Load Test Data")

data_source = st.radio(
    "Data Source",
    ["Upload CSV", "Load from Test Folder"],
    horizontal=True,
    help="Upload a single CSV file or load from a structured test folder with metadata"
)

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with test data. Must include timestamp column."
    )

    if uploaded_file:
        # Load data
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.session_state.df = df_raw
            st.session_state.test_folder_path = None
            st.session_state.loaded_metadata = None

            # Compute file hash for traceability
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            file_hash = f"sha256:{__import__('hashlib').sha256(file_content).hexdigest()[:16]}"
            st.session_state.file_hash = file_hash

            # Compact success message
            st.success(f"✓ Loaded: {len(df_raw)} rows × {len(df_raw.columns)} columns")

            # Details in expander
            with st.expander("Data Info"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df_raw))
                with col2:
                    st.metric("Columns", len(df_raw.columns))
                with col3:
                    st.metric("File Hash", file_hash[:12] + "...")

        except Exception as e:
            st.error(f"Error loading file: {e}")
            df_raw = None
    else:
        df_raw = st.session_state.df

else:  # Load from Test Folder
    if not TEST_FOLDER_SUPPORT:
        st.warning("Test folder support not available. Please use CSV upload.")
        df_raw = st.session_state.df
    else:
        # Recent folders selection
        recent_folders = ConfigManager.get_recent_folders(limit=5)

        if recent_folders:
            folder_source = st.radio(
                "Folder Selection",
                ["Recent Locations", "Enter Path"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if folder_source == "Recent Locations":
                folder_options = ["-- Select Recent Folder --"] + recent_folders
                selected_folder = st.selectbox(
                    "Recent Folder Locations",
                    folder_options,
                    help="Select from recently used folder paths"
                )

                if selected_folder != "-- Select Recent Folder --":
                    test_folder_path = selected_folder
                else:
                    test_folder_path = st.session_state.get('test_folder_path', '')
            else:
                test_folder_path = st.text_input(
                    "Test Folder Path",
                    value=st.session_state.get('test_folder_path', ''),
                    help="Path to test folder or S3 path (future support)"
                )
        else:
            test_folder_path = st.text_input(
                "Test Folder Path",
                value=st.session_state.get('test_folder_path', ''),
                help="Path to test folder or S3 path (future support)"
            )

        if st.button("Load Test Folder", type="primary"):
            if test_folder_path and Path(test_folder_path).exists():
                try:
                    test_data = load_test_from_folder(test_folder_path)

                    # Load raw data
                    if test_data['raw_data_file']:
                        df_raw = pd.read_csv(test_data['raw_data_file'])
                        st.session_state.df = df_raw
                        st.session_state.test_folder_path = test_folder_path

                        # Save to recent folders
                        ConfigManager.save_recent_folder(test_folder_path)

                        # Compute file hash
                        with open(test_data['raw_data_file'], 'rb') as f:
                            file_hash = f"sha256:{__import__('hashlib').sha256(f.read()).hexdigest()[:16]}"
                        st.session_state.file_hash = file_hash

                        st.success(f"✓ Loaded {len(df_raw)} rows from {Path(test_data['raw_data_file']).name}")
                    else:
                        st.error("No raw data file found in test folder")
                        df_raw = None

                    # Load metadata
                    if test_data['metadata']:
                        st.session_state.loaded_metadata = test_data['metadata']
                        with st.expander("Loaded Metadata"):
                            st.json(test_data['metadata'])

                    # Load config if found
                    if test_data['config']:
                        config = test_data['config']
                        st.info(f"Loaded configuration from {Path(test_data['config_file']).name}")

                except Exception as e:
                    st.error(f"Error loading test folder: {e}")
                    df_raw = st.session_state.df
            else:
                st.warning("Please enter a valid test folder path")
                df_raw = st.session_state.df
        else:
            df_raw = st.session_state.df

        # Show expected folder structure
        with st.expander("Expected Folder Structure"):
            st.markdown("""
```
TEST_ID/
    config/           - Configuration files (JSON)
    logs/             - DAQ logs, event logs
    media/            - Photos, videos
    plots/            - Generated plots
    processed_data/   - Resampled, filtered data
    raw_data/         - Original sensor data (CSV)
    reports/          - Analysis reports
    metadata.json     - Test metadata
```

**metadata.json example:**
```json
{
    "test_id": "RCS-CF-C01-RUN01",
    "system": "RCS",
    "test_type": "CF",
    "campaign_id": "C01",
    "run_id": "RUN01",
    "part_name": "Injector Assembly",
    "part_number": "INJ-001-A",
    "serial_number": "SN-0042",
    "test_date": "2024-01-15",
    "operator": "J. Smith"
}
```
            """)

if df_raw is not None:
    # Preprocessing section - compact layout
    st.subheader("Data Preprocessing")

    # Quick preprocess with defaults button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("⚡ Quick Preprocess", type="primary", help="Preprocess with default settings"):
            with st.spinner("Preprocessing..."):
                df_proc = df_raw.copy()

                # Apply channel mapping if available
                channel_config = config.get('channel_config', {})
                if channel_config:
                    df_proc, _ = apply_channel_mapping(df_proc, config)

                # Convert time and add time_s column
                df_proc = preprocess_data(df_proc, config, time_unit='ms', shift_to_zero=True)

                # Handle NaN values
                df_proc, _ = handle_nan_values(df_proc, method='interpolate', max_gap=5)

                st.session_state.df_processed = df_proc
                st.success(f"✓ Preprocessed: {len(df_proc)} rows, {df_proc['time_s'].min():.2f}s - {df_proc['time_s'].max():.2f}s")
                st.rerun()

    with col_btn2:
        st.caption("Use Quick Preprocess for standard settings, or expand below for custom options")

    # Advanced preprocessing options in expander
    with st.expander("⚙️ Advanced Preprocessing Options"):
        # Channel mapping option
        channel_config = config.get('channel_config', {})
        if channel_config:
            apply_mapping = st.checkbox(
                f"Apply channel mapping ({len(channel_config)} channels defined)",
                value=True,
                help="Rename raw DAQ channel IDs to sensor names using channel_config"
            )
            with st.expander("View Channel Mapping"):
                st.json(channel_config)
        else:
            apply_mapping = False
            st.info("No channel_config defined in config. Raw column names will be used.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            time_unit = st.selectbox(
                "Original time unit",
                ["ms", "s", "us"],
                index=0,
                help="Unit of the timestamp column in raw data"
            )

        with col2:
            shift_to_zero = st.checkbox(
                "Shift time to 0",
                value=True,
                help="Shift timestamps so data starts at t=0"
            )

        with col3:
            nan_method = st.selectbox(
                "NaN handling",
                ["interpolate", "drop", "ffill", "none"],
                index=0,
                help="How to handle missing values"
            )

        with col4:
            max_interp_gap = st.number_input(
                "Max interp. gap",
                min_value=1,
                max_value=50,
                value=5,
                help="Maximum consecutive NaNs to interpolate"
            )

        # Resampling options
        col1, col2, col3 = st.columns(3)

        with col1:
            do_resample = st.checkbox(
                "Resample to uniform rate",
                value=False,
                help="Resample data to a fixed sample rate using interpolation"
            )

        with col2:
            # Get default from config
            default_rate = config.get('settings', {}).get('sample_rate_hz', 100)
            target_rate = st.number_input(
                "Target sample rate (Hz)",
                min_value=1,
                max_value=100000,
                value=default_rate,
                disabled=not do_resample,
                help="Target uniform sample rate"
            )

        with col3:
            if do_resample and df_raw is not None:
                # Show current rate estimate
                timestamp_col = config.get('columns', {}).get('timestamp', 'timestamp')
                if timestamp_col in df_raw.columns:
                    dt = np.diff(df_raw[timestamp_col].values)
                    if len(dt) > 0 and np.mean(dt) > 0:
                        # Convert based on time unit
                        if time_unit == 'ms':
                            current_rate = 1000.0 / np.mean(dt)
                        elif time_unit == 'us':
                            current_rate = 1_000_000.0 / np.mean(dt)
                        else:
                            current_rate = 1.0 / np.mean(dt)
                        st.metric("Current avg rate", f"{current_rate:.1f} Hz")

        # Process data
        if st.button("Preprocess Data (Custom)", type="secondary"):
            with st.spinner("Preprocessing..."):
                df_proc = df_raw.copy()

                # Step 1: Apply channel mapping (before time processing)
                channel_config = config.get('channel_config', {})
                if apply_mapping and channel_config:
                    df_proc, mapping_stats = apply_channel_mapping(df_proc, config)
                    if mapping_stats['mappings_found'] > 0:
                        st.caption(f"✓ Mapped {mapping_stats['mappings_found']} channels")

                # Step 2: Convert time and add time_s column
                df_proc = preprocess_data(df_proc, config, time_unit, shift_to_zero)

                # Step 3: Handle NaN values
                if nan_method != "none":
                    df_proc, nan_stats = handle_nan_values(df_proc, nan_method, max_interp_gap)

                # Step 4: Resample if requested
                if do_resample:
                    df_proc, resample_stats = resample_data(df_proc, target_rate, 'time_s')

                st.session_state.df_processed = df_proc
                st.success(f"✓ Preprocessed: {len(df_proc)} rows, {df_proc['time_s'].min():.2f}s - {df_proc['time_s'].max():.2f}s")

    # Use processed data if available, otherwise raw
    df = st.session_state.df_processed if st.session_state.df_processed is not None else df_raw

    # Show preview
    with st.expander("Data Preview"):
        display_cols = ['time_s', 'time_ms'] if 'time_s' in df.columns else []
        display_cols += [c for c in df.columns if c not in display_cols]
        st.dataframe(df[display_cols].head(100), use_container_width=True)

        # Show column info
        st.markdown("**Available columns:**")
        col_info = {col: str(df[col].dtype) for col in df.columns}
        st.json(col_info)

if df_raw is not None:
    df = st.session_state.df_processed if st.session_state.df_processed is not None else df_raw

    st.divider()

    # =============================================================================
    # DATA EXPLORATION
    # =============================================================================

    st.header("1.5 Data Exploration")
    st.caption("Visualize your data before steady-state detection")

    # Get all numeric columns
    time_col = 'time_s' if 'time_s' in df.columns else config.get('columns', {}).get('timestamp', 'timestamp')
    exclude_time_cols = {'time_s', 'time_ms', 'timestamp', config.get('columns', {}).get('timestamp', '')}
    numeric_cols = [c for c in df.columns
                   if c not in exclude_time_cols
                   and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    if numeric_cols and time_col in df.columns:
        # Sensor selection for plotting
        default_sensors = numeric_cols[:min(4, len(numeric_cols))]  # Default to first 4
        selected_sensors = st.multiselect(
            "Select sensors to plot",
            numeric_cols,
            default=default_sensors,
            help="Choose which sensors to visualize"
        )

        if selected_sensors:
            # Create plot
            n_sensors = len(selected_sensors)
            fig = make_subplots(
                rows=n_sensors, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=selected_sensors
            )

            for i, sensor in enumerate(selected_sensors):
                fig.add_trace(
                    go.Scatter(
                        x=df[time_col],
                        y=df[sensor],
                        mode='lines',
                        name=sensor,
                        line=dict(width=1),
                    ),
                    row=i+1, col=1
                )

            time_label = "Time (s)" if time_col == 'time_s' else "Time (ms)"
            fig.update_layout(
                height=150 * n_sensors,
                showlegend=False,
                title="Data Overview",
                margin=dict(t=40, b=40)
            )
            fig.update_xaxes(title_text=time_label, row=n_sensors, col=1)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one sensor to plot")
    else:
        st.warning("No numeric columns found for plotting")

    st.divider()

    # =============================================================================
    # STEADY-STATE DETECTION
    # =============================================================================

    st.header("2. Steady-State Detection")

    col1, col2 = st.columns([2, 1])

    with col2:
        # Pre-select last-used detection method
        methods = ["CV-based (Simple)", "ML-based (Isolation Forest)", "Derivative-based", "Manual Selection"]
        default_method_idx = methods.index(st.session_state.detection_preferences['method']) \
            if st.session_state.detection_preferences['method'] in methods else 0

        detection_method = st.selectbox(
            "Detection Method",
            methods,
            index=default_method_idx,
            help="CV-based: Uses coefficient of variation threshold\n"
                 "ML-based: Uses Isolation Forest anomaly detection\n"
                 "Derivative-based: Finds regions with near-zero slope\n"
                 "Manual: Select window manually"
        )

        # Save selected method
        if detection_method != st.session_state.detection_preferences['method']:
            st.session_state.detection_preferences['method'] = detection_method

        # Determine time column
        time_col = 'time_s' if 'time_s' in df.columns else config.get('columns', {}).get('timestamp', 'timestamp')

        # Get all numeric columns as potential sensors for detection
        exclude_time_cols = {'time_s', 'time_ms', 'timestamp', config.get('columns', {}).get('timestamp', '')}
        numeric_cols = [c for c in df.columns
                       if c not in exclude_time_cols
                       and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        # Determine default sensor (prefer pressure columns)
        default_sensor = None
        for key in ['upstream_pressure', 'chamber_pressure', 'downstream_pressure', 'mass_flow']:
            col_name = config.get('columns', {}).get(key)
            if col_name and col_name in numeric_cols:
                default_sensor = col_name
                break
        if default_sensor is None and numeric_cols:
            default_sensor = numeric_cols[0]

        default_idx = numeric_cols.index(default_sensor) if default_sensor in numeric_cols else 0

        if detection_method == "CV-based (Simple)":
            if numeric_cols:
                # Sensor selection
                selected_sensor = st.selectbox(
                    "Detection Sensor",
                    numeric_cols,
                    index=default_idx,
                    help="Select which sensor/channel to use for steady-state detection"
                )

                # Use saved preferences for default values
                cv_threshold = st.slider("CV Threshold", 0.005, 0.10,
                                        st.session_state.detection_preferences['cv_threshold'], 0.005,
                                        help="Maximum coefficient of variation for steady state")
                window_size = st.slider("Window Size (samples)", 10, 200,
                                       st.session_state.detection_preferences['cv_window_size'])

                # Save preferences when changed
                st.session_state.detection_preferences['cv_threshold'] = cv_threshold
                st.session_state.detection_preferences['cv_window_size'] = window_size

                if st.button("Detect Steady State"):
                    start, end = detect_steady_state_cv(df, selected_sensor, window_size, cv_threshold, time_col)

                    if start is not None:
                        st.session_state.steady_window = (start, end)
                        st.session_state.detection_sensor = selected_sensor  # Save which sensor was used
                        duration = end - start
                        st.success(f"✓ Detected: {start:.3f}s - {end:.3f}s ({duration:.3f}s) using {selected_sensor}")
                    else:
                        st.warning("Could not detect steady state - try adjusting parameters or use Manual Selection")
            else:
                st.warning("No numeric columns found in data")

        elif detection_method == "ML-based (Isolation Forest)":
            if numeric_cols:
                # Multi-select for ML method
                default_ml_sensors = [c for c in numeric_cols[:4]]  # Default to first 4
                selected_sensors = st.multiselect(
                    "Detection Sensors",
                    numeric_cols,
                    default=default_ml_sensors,
                    help="Select sensors to use for ML-based detection (multiple allowed)"
                )

                if selected_sensors:
                    # Use saved preference for contamination
                    contamination = st.slider("Contamination", 0.1, 0.5,
                                             st.session_state.detection_preferences['ml_contamination'], 0.05,
                                             help="Expected fraction of non-steady-state data")

                    # Save preference
                    st.session_state.detection_preferences['ml_contamination'] = contamination

                    if st.button("Detect Steady State (ML)"):
                        with st.spinner("Running ML detection..."):
                            start, end = detect_steady_state_ml(df, selected_sensors, time_col, contamination)

                        if start is not None:
                            st.session_state.steady_window = (start, end)
                            st.session_state.detection_sensor = selected_sensors[0] if selected_sensors else None
                            duration = end - start
                            st.success(f"✓ Detected: {start:.3f}s - {end:.3f}s ({duration:.3f}s)")
                        else:
                            st.warning("ML detection failed - try CV-based method or Manual Selection")
                else:
                    st.info("Select at least one sensor for ML detection")
            else:
                st.warning("No numeric columns found for ML detection")

        elif detection_method == "Derivative-based":
            if numeric_cols:
                # Sensor selection
                selected_sensor = st.selectbox(
                    "Detection Sensor",
                    numeric_cols,
                    index=default_idx,
                    help="Select which sensor/channel to use for steady-state detection"
                )

                # Use saved preference for derivative threshold
                deriv_threshold = st.slider("Derivative Threshold", 0.01, 0.5,
                                           st.session_state.detection_preferences['deriv_threshold'], 0.01,
                                           help="Maximum normalized derivative for steady state")

                # Save preference
                st.session_state.detection_preferences['deriv_threshold'] = deriv_threshold

                if st.button("Detect Steady State"):
                    start, end = detect_steady_state_derivative(df, selected_sensor, time_col, deriv_threshold)

                    if start is not None:
                        st.session_state.steady_window = (start, end)
                        st.session_state.detection_sensor = selected_sensor
                        duration = end - start
                        st.success(f"✓ Detected: {start:.3f}s - {end:.3f}s ({duration:.3f}s) using {selected_sensor}")
                    else:
                        st.warning("Could not detect steady state - use Manual Selection")
            else:
                st.warning("No numeric columns found")

        else:  # Manual Selection
            if time_col in df.columns:
                t_min = float(df[time_col].min())
                t_max = float(df[time_col].max())

                # Determine unit label
                unit_label = "s" if time_col == 'time_s' else "ms"

                window = st.slider(
                    f"Select Window ({unit_label})",
                    min_value=t_min,
                    max_value=t_max,
                    value=(t_min + (t_max-t_min)*0.25, t_min + (t_max-t_min)*0.75),
                    format=f"%.3f"
                )
                st.session_state.steady_window = window

    # Manual window adjustment controls (shown if window detected/selected)
    if st.session_state.steady_window:
        with col2:
            st.markdown("**Manual Adjustment**")

            # Step size control
            step_size = st.number_input(
                "Step size (s)",
                min_value=0.001,
                max_value=1.0,
                value=st.session_state.detection_preferences['window_adjust_step'],
                step=0.01,
                format="%.3f",
                help="Step size for +/- buttons"
            )
            st.session_state.detection_preferences['window_adjust_step'] = step_size

            # Current window display and adjustment
            start, end = st.session_state.steady_window
            time_label = "s" if time_col == 'time_s' else "ms"

            col_start1, col_start2, col_start3 = st.columns([2, 1, 1])
            with col_start1:
                new_start = st.number_input(f"Start ({time_label})", value=float(start), format="%.3f", key="manual_start")
            with col_start2:
                if st.button("-", key="start_minus"):
                    new_start = max(float(df[time_col].min()), start - step_size)
                    st.session_state.steady_window = (new_start, end)
                    st.rerun()
            with col_start3:
                if st.button("+", key="start_plus"):
                    new_start = min(end - 0.01, start + step_size)
                    st.session_state.steady_window = (new_start, end)
                    st.rerun()

            col_end1, col_end2, col_end3 = st.columns([2, 1, 1])
            with col_end1:
                new_end = st.number_input(f"End ({time_label})", value=float(end), format="%.3f", key="manual_end")
            with col_end2:
                if st.button("-", key="end_minus"):
                    new_end = max(start + 0.01, end - step_size)
                    st.session_state.steady_window = (start, new_end)
                    st.rerun()
            with col_end3:
                if st.button("+", key="end_plus"):
                    new_end = min(float(df[time_col].max()), end + step_size)
                    st.session_state.steady_window = (start, new_end)
                    st.rerun()

            # Apply manual changes
            if new_start != start or new_end != end:
                if new_start < new_end:
                    st.session_state.steady_window = (new_start, new_end)
                    st.rerun()

            duration = new_end - new_start
            st.caption(f"Duration: {duration:.3f}{time_label}")

    with col1:
        if st.session_state.steady_window:
            # Pass detection sensor to plot function
            fig = plot_data_with_window(
                df,
                config,
                st.session_state.steady_window,
                detection_sensor=st.session_state.get('detection_sensor')
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =============================================================================
    # QUALITY CONTROL
    # =============================================================================

    st.header("3. Quality Control Checks")

    if st.session_state.steady_window:
        with st.spinner("Running QC checks..."):
            try:
                qc_report = run_qc_checks(df, config)

                # Display QC results
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    status_color = "green" if qc_report.passed else "red"
                    st.markdown(f"### :{status_color}[{'PASSED' if qc_report.passed else 'FAILED'}]")

                with col2:
                    st.metric("Checks Passed", qc_report.summary.get('passed', 0))
                with col3:
                    st.metric("Warnings", qc_report.summary.get('warnings', 0))
                with col4:
                    st.metric("Failed", qc_report.summary.get('failed', 0))

                # Show details
                with st.expander("QC Check Details"):
                    for check in qc_report.checks:
                        icon = "[PASS]" if check.status.name == "PASS" else ("[WARN]" if check.status.name == "WARN" else "[FAIL]")
                        st.markdown(f"{icon} **{check.name}**: {check.message}")

            except Exception as e:
                st.error(f"QC check error: {e}")
                qc_report = None
    else:
        st.info("Select a steady-state window to run QC checks")
        qc_report = None

    st.divider()

    # =============================================================================
    # ANALYSIS
    # =============================================================================

    st.header("4. Run Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Pre-populate from loaded metadata if available
        loaded_meta = st.session_state.get('loaded_metadata', {})

        default_test_id = loaded_meta.get('test_id', f"TEST-{datetime.now().strftime('%Y%m%d-%H%M')}")
        test_id = st.text_input("Test ID", value=default_test_id)

        with st.expander("Metadata", expanded=bool(loaded_meta)):
            part_number = st.text_input(
                "Part Number",
                value=loaded_meta.get('part_number', '')
            )
            serial_number = st.text_input(
                "Serial Number",
                value=loaded_meta.get('serial_number', '')
            )
            operator = st.text_input(
                "Operator",
                value=loaded_meta.get('operator', '')
            )
            part_name = st.text_input(
                "Part Name",
                value=loaded_meta.get('part_name', '')
            )
            facility = st.text_input(
                "Facility",
                value=loaded_meta.get('facility', '')
            )
            notes = st.text_area(
                "Notes",
                value=loaded_meta.get('notes', '')
            )

        # Test Conditions (fluid properties via CoolProp)
        with st.expander("Test Conditions", expanded=True):
            st.caption("Fluid properties calculated via CoolProp from these inputs")

            # Common fluid options
            fluid_options = [
                "", "Water", "Nitrogen", "Air", "Helium", "Oxygen",
                "Ethanol", "Isopropanol", "Methanol",
                "NitrousOxide", "CarbonDioxide", "Hydrogen", "Methane"
            ]

            # Get default from loaded metadata
            default_fluid = loaded_meta.get('test_fluid', '')
            if default_fluid and default_fluid not in fluid_options:
                fluid_options.insert(1, default_fluid)

            default_idx = fluid_options.index(default_fluid) if default_fluid in fluid_options else 0

            test_fluid = st.selectbox(
                "Test Fluid",
                options=fluid_options,
                index=default_idx,
                help="Select fluid for property calculation via CoolProp"
            )

            col_temp, col_press = st.columns(2)
            with col_temp:
                fluid_temp_C = st.number_input(
                    "Fluid Temperature [C]",
                    value=loaded_meta.get('fluid_temperature_K', 293.15) - 273.15,
                    format="%.1f",
                    help="Fluid temperature for property lookup"
                )
            with col_press:
                fluid_press_bar = st.number_input(
                    "Fluid Pressure [bar]",
                    value=loaded_meta.get('fluid_pressure_Pa', 101325.0) / 1e5,
                    format="%.2f",
                    help="Fluid pressure for property lookup"
                )

            # Convert to SI units
            fluid_temperature_K = fluid_temp_C + 273.15
            fluid_pressure_Pa = fluid_press_bar * 1e5

            # Show calculated properties if fluid module available
            if test_fluid and FLUID_PROPS_SUPPORT:
                try:
                    props = get_fluid_properties(test_fluid, fluid_temperature_K, fluid_pressure_Pa)
                    if props:
                        st.success(f"Density: {props.density_kg_m3:.2f} kg/m3 | "
                                  f"Viscosity: {props.viscosity_Pa_s*1000:.4f} mPa.s | "
                                  f"Phase: {props.phase}")
                        if not COOLPROP_AVAILABLE:
                            st.info("Using fallback values. Install CoolProp for accurate properties: pip install CoolProp")
                except Exception as e:
                    st.warning(f"Could not calculate properties: {e}")

        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

    with col2:
        if run_analysis and st.session_state.steady_window:
            with st.spinner("Analyzing..."):
                try:
                    metadata = {
                        'part_number': part_number,
                        'part_name': part_name,
                        'serial_number': serial_number,
                        'operator': operator,
                        'facility': facility,
                        'notes': notes,
                        'test_folder': st.session_state.get('test_folder_path'),
                        # Test conditions
                        'test_fluid': test_fluid,
                        'fluid_temperature_K': fluid_temperature_K,
                        'fluid_pressure_Pa': fluid_pressure_Pa,
                    }

                    # Get fluid properties and add to metadata
                    if test_fluid and FLUID_PROPS_SUPPORT:
                        try:
                            fluid_props = get_fluid_properties(test_fluid, fluid_temperature_K, fluid_pressure_Pa)
                            if fluid_props:
                                metadata['fluid_density_kg_m3'] = fluid_props.density_kg_m3
                                metadata['fluid_density_uncertainty_kg_m3'] = fluid_props.density_kg_m3 * fluid_props.density_uncertainty
                                metadata['fluid_viscosity_Pa_s'] = fluid_props.viscosity_Pa_s
                                metadata['fluid_phase'] = fluid_props.phase
                                metadata['fluid_source'] = fluid_props.source
                        except Exception:
                            pass

                    # Determine file path for traceability
                    # If loaded from test folder with a real file, use that path
                    test_folder_path = st.session_state.get('test_folder_path')
                    file_path = None
                    if test_folder_path and TEST_FOLDER_SUPPORT:
                        raw_file = find_raw_data_file(test_folder_path)
                        if raw_file:
                            file_path = str(raw_file)

                    if test_type == "cold_flow":
                        result = analyze_cold_flow_test(
                            df=df,
                            config=config,
                            steady_window=st.session_state.steady_window,
                            test_id=test_id,
                            file_path=file_path,
                            metadata=metadata,
                            skip_qc=False,
                        )
                    else:
                        result = analyze_hot_fire_test(
                            df=df,
                            config=config,
                            steady_window=st.session_state.steady_window,
                            test_id=test_id,
                            file_path=file_path,
                            metadata=metadata,
                            skip_qc=False,
                        )

                    st.session_state.analysis_result = result
                    st.success("Analysis complete!")

                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        elif run_analysis:
            st.warning("Please select a steady-state window first")

    # =============================================================================
    # RESULTS
    # =============================================================================

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result

        st.divider()
        st.header("5. Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        metrics = list(result.measurements.items())
        for i, (name, meas) in enumerate(metrics[:4]):
            with [col1, col2, col3, col4][i]:
                if hasattr(meas, 'value'):
                    st.metric(
                        name,
                        f"{meas.value:.4g}",
                        delta=f"±{meas.uncertainty:.4g} ({meas.relative_uncertainty_percent:.1f}%)"
                    )
                else:
                    st.metric(name, f"{meas:.4g}")

        # Full results table
        with st.expander("All Measurements with Uncertainties", expanded=True):
            table_data = []
            for name, meas in result.measurements.items():
                if hasattr(meas, 'value'):
                    table_data.append({
                        'Parameter': name,
                        'Value': f"{meas.value:.4g}",
                        'Uncertainty (1σ)': f"±{meas.uncertainty:.4g}",
                        'Relative (%)': f"{meas.relative_uncertainty_percent:.2f}",
                        'Unit': getattr(meas, 'unit', '-'),
                    })
                else:
                    table_data.append({
                        'Parameter': name,
                        'Value': f"{meas:.4g}" if isinstance(meas, (int, float)) else str(meas),
                        'Uncertainty (1σ)': '-',
                        'Relative (%)': '-',
                        'Unit': '-',
                    })

            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Traceability
        with st.expander("Traceability Record"):
            trace_data = result.traceability
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Data Provenance**")
                st.text(f"Raw Data Hash: {trace_data.get('raw_data_hash', 'N/A')}")
                st.text(f"Config Hash: {trace_data.get('config_hash', 'N/A')}")
                st.text(f"Processing Version: {trace_data.get('processing_version', 'N/A')}")

            with col2:
                st.markdown("**Analysis Context**")
                st.text(f"Analyst: {trace_data.get('analyst_username', 'N/A')}")
                st.text(f"Timestamp: {trace_data.get('analysis_timestamp_utc', 'N/A')}")
                st.text(f"Steady Window: {trace_data.get('steady_window_start_ms', 'N/A')} - {trace_data.get('steady_window_end_ms', 'N/A')} ms")

        st.divider()

        # =============================================================================
        # QUICK ITERATION MODE
        # =============================================================================

        st.header("Quick Iteration Mode")
        st.markdown("Rapidly test different steady-state windows and see instant results without re-running full analysis.")

        enable_quick_iteration = st.checkbox(
            "Enable Quick Iteration Mode",
            value=False,
            help="Adjust steady-state window with sliders and see results update in real-time"
        )

        if enable_quick_iteration:
            # Initialize comparison results storage
            if 'iteration_results' not in st.session_state:
                st.session_state.iteration_results = []

            st.info("Adjust the steady-state window below. Results will update automatically.")

            # Get time bounds from data
            time_col = 'time_s' if 'time_s' in df.columns else config.get('columns', {}).get('timestamp', 'timestamp')
            if time_col in df.columns:
                t_min = float(df[time_col].min())
                t_max = float(df[time_col].max())

                # Current window from previous analysis
                current_start, current_end = st.session_state.steady_window

                col1, col2 = st.columns(2)

                with col1:
                    new_start = st.slider(
                        "Window Start (s)",
                        min_value=t_min,
                        max_value=t_max,
                        value=float(current_start),
                        step=0.1,
                        format="%.2f",
                        help="Adjust the start time of the steady-state window"
                    )

                with col2:
                    new_end = st.slider(
                        "Window End (s)",
                        min_value=t_min,
                        max_value=t_max,
                        value=float(current_end),
                        step=0.1,
                        format="%.2f",
                        help="Adjust the end time of the steady-state window"
                    )

                # Validate window
                if new_start >= new_end:
                    st.warning("Start time must be less than end time")
                else:
                    # Check if window changed
                    window_changed = (new_start != current_start) or (new_end != current_end)

                    if window_changed:
                        # Use cached analysis function
                        @st.cache_data(show_spinner=False)
                        def run_quick_analysis(_df, _config, window, _test_id, _test_type, _metadata):
                            """Cached analysis function for quick iteration."""
                            if _test_type == "cold_flow":
                                return analyze_cold_flow_test(
                                    df=_df,
                                    config=_config,
                                    steady_window=window,
                                    test_id=_test_id,
                                    file_path=None,
                                    metadata=_metadata,
                                    skip_qc=True,  # Skip QC for speed
                                )
                            else:
                                return analyze_hot_fire_test(
                                    df=_df,
                                    config=_config,
                                    steady_window=window,
                                    test_id=_test_id,
                                    file_path=None,
                                    metadata=_metadata,
                                    skip_qc=True,
                                )

                        # Run analysis with new window
                        try:
                            with st.spinner("Updating results..."):
                                # Prepare metadata (simplified for iteration)
                                iter_metadata = {
                                    'part_number': metadata.get('part_number', ''),
                                    'serial_number': metadata.get('serial_number', ''),
                                }

                                new_result = run_quick_analysis(
                                    df,
                                    config,
                                    (new_start, new_end),
                                    test_id + "_iter",
                                    test_type,
                                    iter_metadata
                                )

                                # Display updated metrics
                                st.success(f"Results updated for window: {new_start:.2f}s - {new_end:.2f}s")

                                # Key metrics comparison
                                st.subheader("Updated Metrics")
                                col1, col2, col3, col4 = st.columns(4)

                                metrics = list(new_result.measurements.items())
                                for i, (name, meas) in enumerate(metrics[:4]):
                                    with [col1, col2, col3, col4][i]:
                                        if hasattr(meas, 'value'):
                                            # Calculate delta from original
                                            original_meas = result.measurements.get(name)
                                            if original_meas and hasattr(original_meas, 'value'):
                                                delta = ((meas.value - original_meas.value) / original_meas.value * 100)
                                                delta_str = f"{delta:+.2f}% vs original"
                                            else:
                                                delta_str = None

                                            st.metric(
                                                name,
                                                f"{meas.value:.4g}",
                                                delta=delta_str
                                            )
                                        else:
                                            st.metric(name, f"{meas:.4g}")

                                # Save to comparison
                                if st.button("Save to Comparison", key="save_iter"):
                                    st.session_state.iteration_results.append({
                                        'window': (new_start, new_end),
                                        'duration': new_end - new_start,
                                        'measurements': {k: getattr(v, 'value', v) for k, v in new_result.measurements.items()}
                                    })
                                    st.success("Added to comparison table")
                                    st.rerun()

                        except Exception as e:
                            st.error(f"Quick analysis error: {e}")

                # Comparison table
                if st.session_state.iteration_results:
                    st.divider()
                    st.subheader("Parameter Sweep Comparison")

                    # Build comparison DataFrame
                    comparison_data = []
                    for idx, iter_result in enumerate(st.session_state.iteration_results):
                        row = {
                            '#': idx + 1,
                            'Window Start (s)': f"{iter_result['window'][0]:.2f}",
                            'Window End (s)': f"{iter_result['window'][1]:.2f}",
                            'Duration (s)': f"{iter_result['duration']:.2f}",
                        }
                        # Add first 6 measurements
                        for i, (name, value) in enumerate(list(iter_result['measurements'].items())[:6]):
                            if isinstance(value, (int, float)):
                                row[name] = f"{value:.4g}"
                            else:
                                row[name] = str(value)
                        comparison_data.append(row)

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("Clear Comparison", key="clear_comparison"):
                            st.session_state.iteration_results = []
                            st.rerun()

                    with col2:
                        # Export comparison
                        csv_data = comparison_df.to_csv(index=False)
                        st.download_button(
                            "Download Comparison (CSV)",
                            csv_data,
                            file_name=f"{test_id}_parameter_sweep.csv",
                            mime="text/csv"
                        )

            else:
                st.warning("Time column not found in data")

        st.divider()

        # =============================================================================
        # SAVE & EXPORT
        # =============================================================================

        st.header("6. Save & Export")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Save to Campaign")

            campaigns = get_available_campaigns()
            campaign_options = [c['name'] for c in campaigns if c.get('type') == test_type]

            if campaign_options:
                selected_campaign = st.selectbox("Select Campaign", campaign_options)

                if st.button("Save to Campaign"):
                    try:
                        record = result.to_database_record(test_type)
                        record['part'] = part_number
                        record['serial_num'] = serial_number
                        record['operator'] = operator
                        record['notes'] = notes

                        save_to_campaign(selected_campaign, record)
                        st.success(f"Saved to {selected_campaign}")
                    except Exception as e:
                        st.error(f"Save error: {e}")
            else:
                st.info(f"No {test_type} campaigns found. Create one in Campaign Management.")

        with col2:
            st.subheader("Generate Report")

            if st.button("Generate HTML Report"):
                try:
                    # Convert measurements for report
                    measurements_dict = {}
                    for name, meas in result.measurements.items():
                        measurements_dict[name] = meas

                    qc_dict = {
                        'passed': result.passed_qc,
                        'summary': qc_report.summary if qc_report else {},
                        'checks': [
                            {'name': c.name, 'status': c.status.name, 'message': c.message}
                            for c in (qc_report.checks if qc_report else [])
                        ]
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
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Report error: {e}")

        with col3:
            st.subheader("Export Data")

            export_format = st.selectbox("Format", ["JSON", "CSV"])

            if st.button("Export Results"):
                try:
                    if export_format == "JSON":
                        export_data = {
                            'test_id': test_id,
                            'test_type': test_type,
                            'measurements': {
                                k: {
                                    'value': v.value if hasattr(v, 'value') else v,
                                    'uncertainty': v.uncertainty if hasattr(v, 'uncertainty') else None,
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
                            mime="application/json"
                        )
                    else:
                        # CSV export
                        csv_data = []
                        for k, v in result.measurements.items():
                            csv_data.append({
                                'parameter': k,
                                'value': v.value if hasattr(v, 'value') else v,
                                'uncertainty': v.uncertainty if hasattr(v, 'uncertainty') else None,
                            })

                        csv_df = pd.DataFrame(csv_data)
                        st.download_button(
                            "Download CSV",
                            csv_df.to_csv(index=False),
                            file_name=f"{test_id}_results.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Export error: {e}")

else:
    st.info("Upload a CSV file to begin analysis")