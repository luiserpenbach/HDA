

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path

st.set_page_config(page_title="Data Inspector", page_icon="🔍", layout="wide")

st.title("🔍 Data Inspector")
st.markdown("General-purpose data viewer for exploring raw test files")

# --- FILE BROWSER ---
with st.sidebar:
    st.header("📂 File Browser")

    # Root directory selection
    if 'data_root' not in st.session_state:
        st.session_state['data_root'] = os.path.join(os.getcwd(), "test_data")

    browse_root = st.text_input("Browse Root", value=st.session_state['data_root'])

    if os.path.exists(browse_root):
        st.success("✓ Directory found")

        # Recursive file finder
        data_files = []
        for root, dirs, files in os.walk(browse_root):
            for file in files:
                if any(file.endswith(ext) for ext in ['.csv', '.parquet', '.txt', '.tdms']):
                    rel_path = os.path.relpath(os.path.join(root, file), browse_root)
                    data_files.append(rel_path)

        if data_files:
            st.info(f"Found {len(data_files)} data files")

            # Filter
            filter_text = st.text_input("🔎 Filter files", placeholder="Type to filter...")

            if filter_text:
                data_files = [f for f in data_files if filter_text.lower() in f.lower()]

            # Sort options
            sort_by = st.radio("Sort by", ["Name", "Path depth", "Extension"])

            if sort_by == "Name":
                data_files.sort(key=lambda x: os.path.basename(x))
            elif sort_by == "Path depth":
                data_files.sort(key=lambda x: x.count(os.sep))
            else:
                data_files.sort(key=lambda x: os.path.splitext(x)[1])

            # File selection
            selected_file_rel = st.selectbox(
                "Select File",
                data_files,
                format_func=lambda x: f"{'  ' * x.count(os.sep)}📄 {os.path.basename(x)}"
            )

            selected_file = os.path.join(browse_root, selected_file_rel)

        else:
            st.warning("No data files found")
            selected_file = None
    else:
        st.error("Directory not found")
        selected_file = None

    st.markdown("---")

    # Quick load from last ingested
    if 'last_ingested_test' in st.session_state:
        st.subheader("⚡ Quick Load")
        if st.button("Load Last Ingested"):
            selected_file = st.session_state['last_ingested_test']['data_file']
            st.rerun()

# --- FILE UPLOAD (Alternative) ---
st.subheader("📂 Load Data")

col_method1, col_method2 = st.columns(2)

with col_method1:
    st.markdown("**Method 1: Browse File System**")
    if selected_file:
        st.success(f"Selected: `{os.path.basename(selected_file)}`")
        use_browsed = True
    else:
        st.info("No file selected from browser")
        use_browsed = False

with col_method2:
    st.markdown("**Method 2: Upload File**")
    uploaded_file = st.file_uploader("Upload CSV/Parquet", type=['csv', 'parquet', 'txt'])
    use_uploaded = uploaded_file is not None

# Determine which file to use
if use_uploaded:
    data_source = uploaded_file
    source_name = uploaded_file.name
    st.info(f"Using uploaded file: {source_name}")
elif use_browsed:
    data_source = selected_file
    source_name = os.path.basename(selected_file)
else:
    st.warning("⚠️ Please select or upload a file to inspect")
    st.stop()

# --- LOAD DATA ---
try:
    # Load based on file type
    if isinstance(data_source, str):
        # File path
        if data_source.endswith('.parquet'):
            df_raw = pd.read_parquet(data_source)
        elif data_source.endswith('.csv'):
            df_raw = pd.read_csv(data_source)
        elif data_source.endswith('.txt'):
            # Try different delimiters
            try:
                df_raw = pd.read_csv(data_source, delimiter='\t')
            except:
                df_raw = pd.read_csv(data_source, delimiter=',')
        else:
            st.error(f"Unsupported file type: {data_source}")
            st.stop()
    else:
        # Uploaded file
        if source_name.endswith('.parquet'):
            df_raw = pd.read_parquet(data_source)
        else:
            df_raw = pd.read_csv(data_source)

    st.success(f"✓ Loaded {len(df_raw)} rows × {len(df_raw.columns)} columns")

except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# --- METADATA INSPECTION ---
if isinstance(data_source, str):
    # Check for metadata.json in parent directory
    parent_dir = os.path.dirname(data_source)
    metadata_path = os.path.join(parent_dir, 'metadata.json')

    if os.path.exists(metadata_path):
        with st.expander("📋 Test Metadata", expanded=True):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                col_m1, col_m2, col_m3 = st.columns(3)

                col_m1.markdown(f"**Test ID:** {metadata.get('test_id', 'N/A')}")
                col_m1.markdown(f"**System:** {metadata.get('system', 'N/A')}")
                col_m1.markdown(f"**Type:** {metadata.get('type', 'N/A')}")

                col_m2.markdown(f"**Campaign:** {metadata.get('campaign', 'N/A')}")
                col_m2.markdown(f"**Run:** {metadata.get('run', 'N/A')}")
                col_m2.markdown(f"**Operator:** {metadata.get('operator', 'N/A')}")

                col_m3.markdown(f"**Serial #:** {metadata.get('serial_num', 'N/A')}")
                col_m3.markdown(f"**Config:** {metadata.get('config_used', 'N/A')}")
                col_m3.markdown(f"**Timestamp:** {metadata.get('timestamp_utc', 'N/A')[:19]}")

                if metadata.get('notes'):
                    st.markdown(f"**Notes:** {metadata['notes']}")

            except Exception as e:
                st.warning(f"Could not parse metadata: {e}")

# --- DATA OVERVIEW ---
st.markdown("---")
st.subheader("📊 Data Overview")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)

col_info1.metric("Rows", f"{len(df_raw):,}")
col_info2.metric("Columns", f"{len(df_raw.columns)}")
col_info3.metric("Memory", f"{df_raw.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# Detect time column
time_cols = [c for c in df_raw.columns if any(x in c.lower() for x in ['time', 'timestamp', 't_'])]
if time_cols:
    time_col = time_cols[0]
    try:
        if df_raw[time_col].dtype == 'object':
            df_raw[time_col] = pd.to_datetime(df_raw[time_col])

        duration = (df_raw[time_col].max() - df_raw[time_col].min())
        if hasattr(duration, 'total_seconds'):
            duration_s = duration.total_seconds()
        else:
            duration_s = duration / 1000.0 if df_raw[time_col].max() > 1e9 else duration

        col_info4.metric("Duration", f"{duration_s:.2f} s")
    except:
        col_info4.metric("Duration", "N/A")
else:
    time_col = None
    col_info4.metric("Duration", "N/A")

# --- COLUMN ANALYSIS ---
with st.expander("🔢 Column Statistics", expanded=False):
    # Data types summary
    st.markdown("**Data Types:**")
    dtype_counts = df_raw.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        st.caption(f"• {dtype}: {count} columns")

    # Numeric columns
    numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_cols:
        st.markdown("---")
        st.markdown("**Numeric Column Statistics:**")

        stats_df = df_raw[numeric_cols].describe().T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['cv_%'] = (stats_df['std'] / stats_df['mean']) * 100

        st.dataframe(stats_df, use_container_width=True)

# --- INTERACTIVE VISUALIZATION ---
st.markdown("---")
st.subheader("📈 Interactive Visualization")

# Channel selection
numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()

if not numeric_cols:
    st.warning("No numeric columns found for plotting")
else:
    # Time axis setup
    if time_col:
        # Create time_s column
        if df_raw[time_col].dtype == 'datetime64[ns]':
            df_raw['time_s'] = (df_raw[time_col] - df_raw[time_col].min()).dt.total_seconds()
        else:
            df_raw['time_s'] = (df_raw[time_col] - df_raw[time_col].min()) / 1000.0

        x_axis_options = [time_col, 'time_s'] + [c for c in numeric_cols if c != time_col]
        default_x = 'time_s'
    else:
        x_axis_options = numeric_cols
        default_x = numeric_cols[0] if numeric_cols else None

    col_plot1, col_plot2 = st.columns([2, 1])

    with col_plot1:
        x_axis = st.selectbox("X-Axis", x_axis_options,
                              index=x_axis_options.index(default_x) if default_x in x_axis_options else 0)

    with col_plot2:
        plot_type = st.radio("Plot Type", ["Line", "Scatter", "Both"], horizontal=True)

    # Channel selection for Y-axes
    col_y1, col_y2 = st.columns(2)

    with col_y1:
        # Auto-suggest pressure channels for Y1
        pressure_cols = [c for c in numeric_cols if
                         any(x in c.lower() for x in ['pressure', 'p_', 'pc', 'pup', 'pdown', 'bar'])]
        default_y1 = pressure_cols if pressure_cols else numeric_cols[:3]

        y1_channels = st.multiselect(
            "Y1 Axis (Left)",
            numeric_cols,
            default=default_y1[:3]
        )

    with col_y2:
        # Auto-suggest flow/temp channels for Y2
        other_cols = [c for c in numeric_cols if
                      any(x in c.lower() for x in ['flow', 'mf', 'mass', 'temp', 'tc', 'thrust', 'force'])]
        default_y2 = other_cols if other_cols else []

        y2_channels = st.multiselect(
            "Y2 Axis (Right) - Optional",
            numeric_cols,
            default=default_y2[:2]
        )

    # Plotting
    if y1_channels or y2_channels:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Y1 traces
        for i, channel in enumerate(y1_channels):
            mode = 'lines' if plot_type == 'Line' else 'markers' if plot_type == 'Scatter' else 'lines+markers'

            fig.add_trace(
                go.Scatter(
                    x=df_raw[x_axis],
                    y=df_raw[channel],
                    mode=mode,
                    name=channel,
                    line=dict(width=2) if 'lines' in mode else None,
                    marker=dict(size=3) if 'markers' in mode else None
                ),
                secondary_y=False
            )

        # Y2 traces
        for i, channel in enumerate(y2_channels):
            mode = 'lines' if plot_type == 'Line' else 'markers' if plot_type == 'Scatter' else 'lines+markers'

            fig.add_trace(
                go.Scatter(
                    x=df_raw[x_axis],
                    y=df_raw[channel],
                    mode=mode,
                    name=channel,
                    line=dict(width=2, dash='dot') if 'lines' in mode else None,
                    marker=dict(size=3, symbol='diamond') if 'markers' in mode else None
                ),
                secondary_y=True
            )

        # Layout
        fig.update_xaxes(title_text=x_axis)
        fig.update_yaxes(title_text="Y1 Axis", secondary_y=False)
        if y2_channels:
            fig.update_yaxes(title_text="Y2 Axis", secondary_y=True)

        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add zoom/selection tools
        with st.expander("🔧 Plot Tools"):
            col_tool1, col_tool2 = st.columns(2)

            with col_tool1:
                if time_col and 'time_s' in df_raw.columns:
                    time_range = st.slider(
                        "Time Range (s)",
                        float(df_raw['time_s'].min()),
                        float(df_raw['time_s'].max()),
                        (float(df_raw['time_s'].min()), float(df_raw['time_s'].max()))
                    )

                    if st.button("Apply Time Filter"):
                        df_filtered = df_raw[(df_raw['time_s'] >= time_range[0]) & (df_raw['time_s'] <= time_range[1])]
                        st.session_state['inspector_filtered'] = df_filtered
                        st.success(f"Filtered to {len(df_filtered)} rows")

            with col_tool2:
                downsample = st.checkbox("Downsample for Performance")
                if downsample:
                    n_points = st.number_input("Max Points", value=10000, min_value=100, step=1000)
                    if len(df_raw) > n_points:
                        st.caption(f"Plotting every {len(df_raw) // n_points}th point")
    else:
        st.info("Select channels to plot")

# --- DISTRIBUTION ANALYSIS ---
st.markdown("---")
st.subheader("📊 Distribution Analysis")

if numeric_cols:
    col_dist1, col_dist2 = st.columns([1, 3])

    with col_dist1:
        dist_channel = st.selectbox("Select Channel", numeric_cols)
        show_stats = st.checkbox("Show Statistics", value=True)

    with col_dist2:
        # Histogram
        fig_hist = go.Figure()

        fig_hist.add_trace(go.Histogram(
            x=df_raw[dist_channel],
            nbinsx=50,
            name=dist_channel,
            marker_color='#3498DB'
        ))

        if show_stats:
            mean_val = df_raw[dist_channel].mean()
            std_val = df_raw[dist_channel].std()

            fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="red",
                               annotation_text=f"Mean: {mean_val:.3f}")
            fig_hist.add_vline(x=mean_val + std_val, line_dash="dot", line_color="orange",
                               annotation_text=f"+1σ: {mean_val + std_val:.3f}")
            fig_hist.add_vline(x=mean_val - std_val, line_dash="dot", line_color="orange",
                               annotation_text=f"-1σ: {mean_val - std_val:.3f}")

        fig_hist.update_layout(
            title=f"Distribution: {dist_channel}",
            xaxis_title=dist_channel,
            yaxis_title="Count",
            height=300
        )

        st.plotly_chart(fig_hist, use_container_width=True)

# --- RAW DATA TABLE ---
st.markdown("---")
st.subheader("📋 Raw Data Table")

col_table1, col_table2, col_table3 = st.columns([2, 1, 1])

with col_table1:
    cols_to_show = st.multiselect(
        "Columns to Display",
        df_raw.columns.tolist(),
        default=df_raw.columns.tolist()[:10]
    )

with col_table2:
    n_rows = st.number_input("Rows to Display", value=100, min_value=10, max_value=len(df_raw), step=50)

with col_table3:
    table_mode = st.radio("Mode", ["Head", "Tail", "Sample"])

if cols_to_show:
    if table_mode == "Head":
        df_display = df_raw[cols_to_show].head(n_rows)
    elif table_mode == "Tail":
        df_display = df_raw[cols_to_show].tail(n_rows)
    else:
        df_display = df_raw[cols_to_show].sample(min(n_rows, len(df_raw)))

    st.dataframe(df_display, use_container_width=True, height=400)

# --- EXPORT OPTIONS ---
st.markdown("---")
st.subheader("💾 Export Options")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    # Export full data
    if st.button("📥 Export Full Dataset"):
        csv = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            f"{source_name}_full.csv",
            "text/csv"
        )

with col_exp2:
    # Export filtered data
    if 'inspector_filtered' in st.session_state:
        if st.button("📥 Export Filtered"):
            csv = st.session_state['inspector_filtered'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Filtered CSV",
                csv,
                f"{source_name}_filtered.csv",
                "text/csv"
            )

with col_exp3:
    # Export statistics
    if st.button("📥 Export Statistics"):
        if numeric_cols:
            stats = df_raw[numeric_cols].describe()
            csv = stats.to_csv().encode('utf-8')
            st.download_button(
                "Download Stats CSV",
                csv,
                f"{source_name}_stats.csv",
                "text/csv"
            )

# --- QUICK ANALYSIS TOOLS ---
st.markdown("---")
st.subheader("🔬 Quick Analysis")

col_qa1, col_qa2 = st.columns(2)

with col_qa1:
    st.markdown("**Signal Quality Check**")

    if numeric_cols:
        check_channel = st.selectbox("Channel to Check", numeric_cols, key='qa_channel')

        if st.button("Run Quality Check"):
            # Check for NaNs
            nan_count = df_raw[check_channel].isna().sum()
            nan_pct = (nan_count / len(df_raw)) * 100

            # Check for constant values
            unique_vals = df_raw[check_channel].nunique()

            # Check for outliers (3-sigma)
            mean = df_raw[check_channel].mean()
            std = df_raw[check_channel].std()
            outliers = ((df_raw[check_channel] - mean).abs() > 3 * std).sum()
            outlier_pct = (outliers / len(df_raw)) * 100

            # Display results
            st.markdown(f"""
            **Quality Report: {check_channel}**
            - Missing values: {nan_count} ({nan_pct:.2f}%)
            - Unique values: {unique_vals}
            - Outliers (3σ): {outliers} ({outlier_pct:.2f}%)
            - Range: {df_raw[check_channel].min():.3f} to {df_raw[check_channel].max():.3f}
            """)

            if nan_pct > 5:
                st.warning("⚠️ High percentage of missing values")
            if unique_vals < 10:
                st.warning("⚠️ Very few unique values - may be constant or quantized")
            if outlier_pct > 1:
                st.warning("⚠️ Significant outliers detected")

with col_qa2:
    st.markdown("**Correlation Analysis**")

    if len(numeric_cols) >= 2:
        corr_cols = st.multiselect("Select Channels (2+)", numeric_cols,
                                   default=numeric_cols[:min(5, len(numeric_cols))], key='corr_channels')

        if len(corr_cols) >= 2 and st.button("Calculate Correlation"):
            corr_matrix = df_raw[corr_cols].corr()

            # Heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))

            fig_corr.update_layout(
                title="Correlation Matrix",
                height=400
            )

            st.plotly_chart(fig_corr, use_container_width=True)