"""
Analysis by System Page
=======================
System-level analysis aggregating data across multiple campaigns
for a single test system (testbench).

Features:
- System Overview: View all campaigns and tests for a selected system
- Cross-Campaign Trends: Track metrics across campaigns over time
- Campaign Comparison: Compare performance between two campaigns
- System SPC: Statistical process control across the full system history
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import io
from typing import Dict, Any, List, Optional, Tuple

from core.campaign_manager_v2 import (
    get_available_campaigns,
    get_campaign_info,
    get_campaign_data,
)
from core.spc import (
    create_imr_chart,
    format_spc_summary,
)
from core.comparison import (
    compare_campaigns,
    format_campaign_comparison,
    calculate_correlation_matrix,
    linear_regression,
)
from pages._shared_sidebar import render_global_context, render_system_selector
from pages._shared_styles import apply_custom_styles, render_page_header

st.set_page_config(page_title="Analysis by System", page_icon="AS", layout="wide")

apply_custom_styles()


# =============================================================================
# PAGE-LOCAL HELPER FUNCTIONS
# =============================================================================

def get_system_campaigns(system_name: str) -> List[Dict[str, Any]]:
    """
    Get all campaign databases belonging to a system.

    Filters get_available_campaigns() by matching the system prefix
    (first segment before the first hyphen in the campaign name).

    Args:
        system_name: System identifier (e.g., "INJ", "RCS", "IGN")

    Returns:
        List of campaign info dicts with keys: name, type, test_count, created_date
    """
    all_campaigns = get_available_campaigns()
    return [
        c for c in all_campaigns
        if c['name'].split('-')[0].upper() == system_name.upper()
    ]


def load_system_data(
    system_name: str,
    type_filter: Optional[str] = None,
    qc_filter: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and concatenate data from all campaigns for a system.

    Adds 'campaign_name' and 'campaign_type' columns to each DataFrame
    before concatenation.

    Args:
        system_name: System identifier
        type_filter: Optional filter for campaign type ('cold_flow' / 'hot_fire')
        qc_filter: If True, only include tests that passed QC

    Returns:
        Tuple of (combined DataFrame, list of loaded campaign names).
        Returns (empty DataFrame, []) if no data found.
    """
    campaigns = get_system_campaigns(system_name)

    if type_filter and type_filter != "All":
        campaigns = [c for c in campaigns if c.get('type') == type_filter]

    dfs = []
    loaded_names = []

    for campaign in campaigns:
        try:
            df = get_campaign_data(campaign['name'])
            if df is not None and len(df) > 0:
                df = df.copy()
                df['campaign_name'] = campaign['name']
                df['campaign_type'] = campaign.get('type', 'unknown')
                dfs.append(df)
                loaded_names.append(campaign['name'])
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(), []

    # Drop all-NA columns from each df before concat to avoid FutureWarning
    dfs = [df.dropna(axis=1, how='all') for df in dfs]
    system_df = pd.concat(dfs, ignore_index=True, sort=False)

    if qc_filter and 'qc_passed' in system_df.columns:
        system_df = system_df[system_df['qc_passed'] == True].reset_index(drop=True)  # noqa: E712

    return system_df, loaded_names


def get_metric_columns(df: pd.DataFrame) -> List[str]:
    """Get measurement metric columns (avg_* columns with numeric type)."""
    numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    return [c for c in numeric_cols if c.startswith('avg_')]


def get_campaign_boundaries(df: pd.DataFrame) -> List[Tuple[int, str]]:
    """
    Find index positions where campaign transitions occur.

    Used to draw vertical boundary lines on cross-campaign SPC charts.

    Returns:
        List of (index, campaign_name) tuples marking start of each campaign.
    """
    if 'campaign_name' not in df.columns:
        return []

    boundaries = []
    current_campaign = None
    for i in range(len(df)):
        campaign = df.iloc[i].get('campaign_name', '')
        if campaign != current_campaign:
            boundaries.append((i, campaign))
            current_campaign = campaign
    return boundaries


def get_available_systems_from_db() -> List[str]:
    """Derive unique system prefixes from campaign database names."""
    all_campaigns = get_available_campaigns()
    prefixes = set()
    for c in all_campaigns:
        name = c.get('name', '')
        if '-' in name:
            prefixes.add(name.split('-')[0])
    return sorted(prefixes)


# =============================================================================
# SIDEBAR - Global Context & System Selection
# =============================================================================

# Initialize filter defaults (set in sidebar, used in main content)
selected_system = None
type_filter = "All"
qc_filter = False
part_filter = []
serial_filter = []

with st.sidebar:
    context = render_global_context()
    st.divider()

    st.header("System Selection")

    if context['is_configured']:
        # Path A: folder-based system selection
        selected_system = render_system_selector(context, key_prefix="sys_")
    else:
        # Path B: derive from campaign DB names
        system_prefixes = get_available_systems_from_db()
        if system_prefixes:
            sel = st.selectbox(
                "System",
                [""] + system_prefixes,
                format_func=lambda x: "-- Select --" if x == "" else x,
                key="sys_system_selector_db"
            )
            selected_system = sel if sel else None
        else:
            st.info("No campaigns found")

    # Filters
    if selected_system:
        st.divider()
        st.subheader("Filters")

        # Campaign type filter
        sys_campaigns = get_system_campaigns(selected_system)
        available_types = sorted(set(c.get('type', 'unknown') for c in sys_campaigns))

        if len(available_types) > 1:
            type_options = ["All"] + available_types
            type_filter = st.selectbox(
                "Campaign Type",
                type_options,
                key="sys_type_filter"
            )
        else:
            type_filter = "All"

        # QC filter
        qc_filter = st.checkbox("Passed QC only", value=False, key="sys_qc_filter")


# =============================================================================
# MAIN CONTENT
# =============================================================================

render_page_header(
    title="Analysis by System",
    description="System-level analysis and cross-campaign comparison",
    badge_text="P1",
    badge_type="info"
)

if not selected_system:
    st.info("Select a **System** in the sidebar to view system-level analysis.")

    # Show available systems summary
    system_prefixes = get_available_systems_from_db()
    if system_prefixes:
        st.subheader("Available Systems")
        for prefix in system_prefixes:
            campaigns = get_system_campaigns(prefix)
            total_tests = sum(c.get('test_count', 0) for c in campaigns)
            types = sorted(set(c.get('type', 'unknown') for c in campaigns))
            st.markdown(
                f"- **{prefix}**: {len(campaigns)} campaign(s), "
                f"{total_tests} test(s), type(s): {', '.join(types)}"
            )
    else:
        st.markdown(
            "No campaign databases found. Analyze tests in the "
            "**Single Test Analysis** page first to create campaign data."
        )
    st.stop()

# Load system data
system_df, loaded_campaigns = load_system_data(
    selected_system,
    type_filter=type_filter,
    qc_filter=qc_filter,
)

if system_df.empty or len(loaded_campaigns) == 0:
    st.warning(f"No campaign data found for system **{selected_system}**.")
    st.markdown(
        "Ensure tests have been analyzed and saved to campaigns. "
        "Campaign database names must start with the system prefix "
        f"(e.g., `{selected_system}-CF-C1`)."
    )
    st.stop()

# Part/Serial filters (in sidebar, populated after data load)
with st.sidebar:
    st.divider()
    st.subheader("Part/Serial Filter")

    # Part filter
    if 'part' in system_df.columns:
        available_parts = sorted([p for p in system_df['part'].dropna().unique().tolist() if p])
        if available_parts:
            part_filter = st.multiselect(
                "Part",
                available_parts,
                default=[],
                key="sys_part_filter"
            )

    # Serial number filter
    if 'serial_num' in system_df.columns:
        available_serials = sorted([s for s in system_df['serial_num'].dropna().unique().tolist() if s])
        if available_serials:
            serial_filter = st.multiselect(
                "Serial Number",
                available_serials,
                default=[],
                key="sys_serial_filter"
            )

# Apply part/serial filters
if part_filter:
    system_df = system_df[system_df['part'].isin(part_filter)].reset_index(drop=True)
if serial_filter:
    system_df = system_df[system_df['serial_num'].isin(serial_filter)].reset_index(drop=True)

# Check if filters resulted in empty data
if system_df.empty:
    st.warning("No tests match the selected filters. Adjust Part/Serial filters in the sidebar.")
    st.stop()

# Gather campaign info
campaign_infos = {}
for cname in loaded_campaigns:
    try:
        campaign_infos[cname] = get_campaign_info(cname)
    except Exception:
        campaign_infos[cname] = {}

# =============================================================================
# HEADER METRICS
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tests", len(system_df))

with col2:
    st.metric("Campaigns", len(loaded_campaigns))

with col3:
    if 'qc_passed' in system_df.columns:
        qc_pass = int(system_df['qc_passed'].sum())
        pct = (qc_pass / len(system_df) * 100) if len(system_df) > 0 else 0
        st.metric("QC Pass Rate", f"{pct:.0f}%")
    else:
        st.metric("QC Pass Rate", "N/A")

with col4:
    types_in_data = system_df['campaign_type'].unique().tolist() if 'campaign_type' in system_df.columns else []
    st.metric("Campaign Types", ", ".join(t.replace('_', ' ').title() for t in types_in_data) or "N/A")

st.divider()

# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "System Overview",
    "Cross-Campaign Trends",
    "Campaign Comparison",
    "System SPC",
])

# =============================================================================
# TAB 1: System Overview
# =============================================================================

with tab1:
    subtab1, subtab2, subtab3 = st.tabs(["Dashboard", "Data Table", "Export"])

    # --- Dashboard ---
    with subtab1:
        st.subheader("Campaign Summary")

        # Campaign cards as a table
        summary_rows = []
        for cname in loaded_campaigns:
            cdf = system_df[system_df['campaign_name'] == cname]
            info = campaign_infos.get(cname, {})
            ctype = info.get('type', 'unknown')

            row = {
                'Campaign': cname,
                'Type': ctype.replace('_', ' ').title(),
                'Tests': len(cdf),
            }

            # QC pass count
            if 'qc_passed' in cdf.columns:
                row['QC Passed'] = int(cdf['qc_passed'].sum())
            else:
                row['QC Passed'] = 'N/A'

            # Primary metric mean
            metric_cols = get_metric_columns(cdf)
            if metric_cols:
                primary = metric_cols[0]
                valid = cdf[primary].dropna()
                if len(valid) > 0:
                    row[f'Mean {primary}'] = f"{valid.mean():.4f}"
                    row[f'Std {primary}'] = f"{valid.std():.4f}"

            summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Primary metric scatter plot colored by campaign
        st.subheader("Primary Metric Across Campaigns")

        metric_cols = get_metric_columns(system_df)

        if metric_cols:
            selected_overview_metric = st.selectbox(
                "Select metric",
                metric_cols,
                key="sys_ovr_metric"
            )

            if selected_overview_metric and selected_overview_metric in system_df.columns:
                plot_df = system_df[['campaign_name', selected_overview_metric]].dropna(subset=[selected_overview_metric]).copy()
                plot_df['test_index'] = range(len(plot_df))

                col_plot, col_stats = st.columns([2, 1])

                with col_plot:
                    fig = px.scatter(
                        plot_df,
                        x='test_index',
                        y=selected_overview_metric,
                        color='campaign_name',
                        title=f"{selected_overview_metric} — All Campaigns",
                        labels={
                            'test_index': 'Test Number',
                            selected_overview_metric: selected_overview_metric,
                            'campaign_name': 'Campaign',
                        },
                    )

                    # Add overall mean line
                    mean_val = plot_df[selected_overview_metric].mean()
                    fig.add_hline(
                        y=mean_val, line_dash="dash", line_color="gray",
                        annotation_text=f"Mean: {mean_val:.4f}"
                    )

                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    st.markdown("**Aggregate Statistics**")
                    valid_data = system_df[selected_overview_metric].dropna()
                    if len(valid_data) > 0:
                        st.metric("Mean", f"{valid_data.mean():.4f}")
                        st.metric("Std Dev", f"{valid_data.std():.4f}")
                        st.metric("Min", f"{valid_data.min():.4f}")
                        st.metric("Max", f"{valid_data.max():.4f}")
                        if valid_data.mean() != 0:
                            st.metric("CV (%)", f"{(valid_data.std() / abs(valid_data.mean()) * 100):.2f}")

                # Distribution by campaign
                st.subheader("Distribution by Campaign")
                fig_dist = px.histogram(
                    plot_df,
                    x=selected_overview_metric,
                    color='campaign_name',
                    barmode='overlay',
                    nbins=20,
                    opacity=0.7,
                    title=f"{selected_overview_metric} Distribution by Campaign"
                )
                fig_dist.update_layout(height=350)
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No metric columns (avg_*) found in system data.")

    # --- Data Table ---
    with subtab2:
        st.subheader("Combined System Data")

        all_columns = list(system_df.columns)
        default_cols = ['campaign_name', 'test_id', 'part', 'serial_num']

        # Add primary metrics to default display
        metric_cols = get_metric_columns(system_df)
        if metric_cols:
            default_cols.extend(metric_cols[:3])
        if 'qc_passed' in all_columns:
            default_cols.append('qc_passed')

        default_cols = [c for c in default_cols if c in all_columns]

        selected_cols = st.multiselect(
            "Columns to display",
            all_columns,
            default=default_cols,
            key="sys_ovr_cols"
        )

        if selected_cols:
            st.dataframe(system_df[selected_cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(system_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV",
            system_df.to_csv(index=False),
            file_name=f"{selected_system}_system_data.csv",
            mime="text/csv",
            key="sys_ovr_dl_csv"
        )

    # --- Export ---
    with subtab3:
        st.subheader("Export System Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Export Options**")
            include_unc = st.checkbox("Include uncertainty columns", value=True, key="sys_exp_unc")
            include_trace = st.checkbox("Include traceability columns", value=True, key="sys_exp_trace")
            export_fmt = st.selectbox("Format", ["CSV", "JSON"], key="sys_exp_fmt")

        with col2:
            st.markdown("**Export**")

            export_df = system_df.copy()
            if not include_unc:
                export_df = export_df[[c for c in export_df.columns if not c.startswith('u_')]]
            if not include_trace:
                trace_cols = ['raw_data_hash', 'config_hash', 'config_snapshot',
                              'analyst_username', 'analysis_timestamp_utc', 'processing_version']
                export_df = export_df[[c for c in export_df.columns if c not in trace_cols]]

            if export_fmt == "CSV":
                st.download_button(
                    "Download CSV",
                    export_df.to_csv(index=False),
                    file_name=f"{selected_system}_export.csv",
                    mime="text/csv",
                    key="sys_exp_dl_csv"
                )
            elif export_fmt == "JSON":
                export_data = {
                    'system': selected_system,
                    'campaigns': loaded_campaigns,
                    'export_date': datetime.now().isoformat(),
                    'test_count': len(export_df),
                    'data': export_df.to_dict(orient='records'),
                }
                st.download_button(
                    "Download JSON",
                    json.dumps(export_data, indent=2, default=str),
                    file_name=f"{selected_system}_export.json",
                    mime="application/json",
                    key="sys_exp_dl_json"
                )


# =============================================================================
# TAB 2: Cross-Campaign Trends
# =============================================================================

with tab2:
    st.subheader("Cross-Campaign Trends")

    metric_cols = get_metric_columns(system_df)

    if not metric_cols:
        st.info("No metric columns (avg_*) found in system data.")
    else:
        col_ctrl, col_chart = st.columns([1, 3])

        with col_ctrl:
            trend_param = st.selectbox(
                "Parameter",
                metric_cols,
                key="sys_trend_param"
            )

            show_trend_line = st.checkbox("Show trend line", value=True, key="sys_trend_line")
            show_error_bars = st.checkbox("Show error bars", value=False, key="sys_trend_errbars")

        with col_chart:
            if trend_param and trend_param in system_df.columns:
                plot_df = system_df[['campaign_name', trend_param]].copy()

                # Add uncertainty column if available and requested
                u_col = trend_param.replace('avg_', 'u_')
                has_uncertainty = u_col in system_df.columns

                if has_uncertainty:
                    plot_df[u_col] = system_df[u_col]

                plot_df = plot_df.dropna(subset=[trend_param]).reset_index(drop=True)
                plot_df['test_index'] = range(len(plot_df))

                fig = go.Figure()

                # Plot each campaign separately for color coding
                for cname in loaded_campaigns:
                    cdf = plot_df[plot_df['campaign_name'] == cname]
                    if cdf.empty:
                        continue

                    error_y = None
                    if show_error_bars and has_uncertainty and u_col in cdf.columns:
                        error_y = dict(type='data', array=cdf[u_col].values, visible=True)

                    fig.add_trace(go.Scatter(
                        x=cdf['test_index'],
                        y=cdf[trend_param],
                        mode='markers+lines',
                        name=cname,
                        marker=dict(size=8),
                        line=dict(width=1),
                        error_y=error_y,
                        hovertemplate=f'{cname}<br>{trend_param}: %{{y:.4f}}<extra></extra>',
                    ))

                # Overall trend line
                if show_trend_line and len(plot_df) >= 3:
                    try:
                        x_vals = plot_df['test_index'].values.astype(float)
                        y_vals = plot_df[trend_param].values.astype(float)
                        reg = linear_regression(x_vals, y_vals, "test_index", trend_param)

                        x_line = np.array([x_vals.min(), x_vals.max()])
                        y_line = reg.predict(x_line)

                        fig.add_trace(go.Scatter(
                            x=x_line, y=y_line,
                            mode='lines',
                            name=f'Trend (R²={reg.r_squared:.3f})',
                            line=dict(color='gray', dash='dash', width=2),
                        ))
                    except (ValueError, Exception):
                        pass

                # Campaign boundary lines
                boundaries = get_campaign_boundaries(plot_df)
                for idx, cname in boundaries[1:]:  # Skip first boundary
                    fig.add_vline(
                        x=idx - 0.5, line_dash="dot", line_color="lightgray",
                        annotation_text=cname, annotation_position="top",
                        annotation_font_size=9,
                    )

                fig.update_layout(
                    title=f"{trend_param} — Cross-Campaign Trend",
                    xaxis_title="Test Number (chronological)",
                    yaxis_title=trend_param,
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Per-campaign statistics table
        st.subheader("Per-Campaign Statistics")
        stats_rows = []
        for cname in loaded_campaigns:
            cdf = system_df[system_df['campaign_name'] == cname]
            if trend_param not in cdf.columns:
                continue
            valid = cdf[trend_param].dropna()
            if len(valid) == 0:
                continue
            stats_rows.append({
                'Campaign': cname,
                'N': len(valid),
                'Mean': f"{valid.mean():.4f}",
                'Std Dev': f"{valid.std():.4f}",
                'Min': f"{valid.min():.4f}",
                'Max': f"{valid.max():.4f}",
                'CV (%)': f"{(valid.std() / abs(valid.mean()) * 100):.2f}" if valid.mean() != 0 else "N/A",
            })
        if stats_rows:
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

        # Correlation matrix
        st.subheader("Metric Correlations")
        corr_params = metric_cols[:10]  # Limit to avoid oversized matrix

        if len(corr_params) >= 2:
            try:
                corr_matrix = calculate_correlation_matrix(system_df, corr_params)
                corr_df = corr_matrix.to_dataframe()

                fig_corr = px.imshow(
                    corr_df,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix",
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)

                # Strong correlations
                strong = corr_matrix.get_strong_correlations(threshold=0.7)
                if strong:
                    with st.expander("Strong Correlations (|r| >= 0.7)"):
                        for p1, p2, r in strong:
                            st.text(f"{p1} <-> {p2}: r = {r:.3f}")
            except (ValueError, Exception) as e:
                st.info(f"Could not compute correlation matrix: {e}")
        else:
            st.info("Need at least 2 metric columns for correlation analysis.")


# =============================================================================
# TAB 3: Campaign Comparison
# =============================================================================

with tab3:
    st.subheader("Campaign Comparison")

    if len(loaded_campaigns) < 2:
        st.info("At least 2 campaigns are required for comparison. "
                f"System **{selected_system}** has {len(loaded_campaigns)} campaign(s).")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            campaign_a = st.selectbox(
                "Campaign A",
                loaded_campaigns,
                index=0,
                key="sys_cmp_campaign_a"
            )
        with col_b:
            # Default to second campaign
            default_b = 1 if len(loaded_campaigns) > 1 else 0
            campaign_b = st.selectbox(
                "Campaign B",
                loaded_campaigns,
                index=default_b,
                key="sys_cmp_campaign_b"
            )

        if campaign_a == campaign_b:
            st.warning("Select two different campaigns to compare.")
        else:
            # Parameter selection
            metric_cols = get_metric_columns(system_df)
            if metric_cols:
                default_params = metric_cols[:4]
                cmp_params = st.multiselect(
                    "Parameters to compare",
                    metric_cols,
                    default=default_params,
                    key="sys_cmp_params"
                )

                cmp_tolerance = st.number_input(
                    "Default tolerance (%)",
                    value=5.0,
                    min_value=0.1,
                    max_value=50.0,
                    step=0.5,
                    key="sys_cmp_tol"
                )

                if cmp_params and st.button("Compare", type="primary", key="sys_cmp_btn"):
                    df_a = system_df[system_df['campaign_name'] == campaign_a]
                    df_b = system_df[system_df['campaign_name'] == campaign_b]

                    try:
                        comparison = compare_campaigns(
                            df_a, df_b,
                            campaign_a, campaign_b,
                            parameters=cmp_params,
                            tolerances={p: cmp_tolerance for p in cmp_params},
                        )

                        # Summary metrics
                        n_params = len(comparison.get('parameters', {}))
                        n_pass = sum(
                            1 for p in comparison.get('parameters', {}).values()
                            if p.get('means_equivalent', False)
                        )

                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.metric(f"{campaign_a} Tests", comparison.get('n_tests_a', 0))
                        with mcol2:
                            st.metric(f"{campaign_b} Tests", comparison.get('n_tests_b', 0))
                        with mcol3:
                            status = "PASS" if n_pass == n_params else "FAIL"
                            st.metric("Result", f"{status} ({n_pass}/{n_params})")

                        # Comparison table
                        st.markdown("**Comparison Results**")
                        cmp_rows = []
                        for param, data in comparison.get('parameters', {}).items():
                            cmp_rows.append({
                                'Parameter': param,
                                f'Mean A ({campaign_a})': f"{data['mean_a']:.4g}",
                                f'Mean B ({campaign_b})': f"{data['mean_b']:.4g}",
                                'Diff (%)': f"{data['mean_diff_pct']:+.2f}%",
                                'Tolerance': f"±{data['tolerance']:.1f}%",
                                'Status': 'PASS' if data['means_equivalent'] else 'FAIL',
                            })
                        if cmp_rows:
                            cmp_df = pd.DataFrame(cmp_rows)
                            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                        # Formatted text summary
                        with st.expander("Full Comparison Summary"):
                            st.code(format_campaign_comparison(comparison))

                        # Side-by-side box plots
                        st.subheader("Distribution Comparison")
                        box_param = st.selectbox(
                            "Parameter for box plot",
                            cmp_params,
                            key="sys_cmp_box_param"
                        )

                        if box_param:
                            box_data = system_df[
                                system_df['campaign_name'].isin([campaign_a, campaign_b])
                            ][[box_param, 'campaign_name']].dropna(subset=[box_param])

                            fig_box = px.box(
                                box_data,
                                x='campaign_name',
                                y=box_param,
                                color='campaign_name',
                                title=f"{box_param} — {campaign_a} vs {campaign_b}",
                                points="all",
                            )
                            fig_box.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig_box, use_container_width=True)

                    except Exception as e:
                        st.error(f"Comparison error: {e}")
            else:
                st.info("No metric columns (avg_*) found for comparison.")


# =============================================================================
# TAB 4: System SPC
# =============================================================================

with tab4:
    st.subheader("System-Wide Statistical Process Control")
    st.caption("SPC analysis across all campaigns in the system.")

    metric_cols = get_metric_columns(system_df)

    if not metric_cols:
        st.info("No metric columns (avg_*) found for SPC analysis.")
    elif len(system_df) < 2:
        st.info("Need at least 2 data points for SPC analysis.")
    else:
        col_ctrl, col_chart = st.columns([1, 2])

        with col_ctrl:
            spc_param = st.selectbox("Parameter", metric_cols, key="sys_spc_param")

            st.divider()
            st.markdown("**Specification Limits**")
            use_specs = st.checkbox("Use specification limits", key="sys_spc_use_specs")

            lsl, usl, target = None, None, None
            if use_specs and spc_param:
                valid = system_df[spc_param].dropna()
                if len(valid) > 0:
                    col_mean = float(valid.mean())
                    col_range = float(valid.max() - valid.min())
                    lsl = st.number_input("LSL", value=col_mean - col_range, key="sys_spc_lsl")
                    usl = st.number_input("USL", value=col_mean + col_range, key="sys_spc_usl")
                    target = st.number_input("Target", value=col_mean, key="sys_spc_target")

        with col_chart:
            if spc_param and spc_param in system_df.columns:
                try:
                    analysis = create_imr_chart(
                        system_df,
                        parameter=spc_param,
                        test_id_col='test_id' if 'test_id' in system_df.columns else system_df.columns[0],
                        usl=usl if use_specs else None,
                        lsl=lsl if use_specs else None,
                        target=target if use_specs else None,
                    )

                    # Header metrics
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                    with mcol1:
                        status_color = "green" if analysis.n_violations == 0 else "red"
                        st.markdown(
                            f"### :{status_color}[{'In Control' if analysis.n_violations == 0 else 'Out of Control'}]"
                        )
                    with mcol2:
                        st.metric("Points", analysis.n_points)
                    with mcol3:
                        st.metric("Violations", analysis.n_violations)
                    with mcol4:
                        if analysis.capability and analysis.capability.cpk is not None:
                            st.metric("Cpk", f"{analysis.capability.cpk:.2f}")
                        else:
                            st.metric("Cpk", "N/A")

                    # I-MR Control Chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=[
                            f"Individual Chart: {analysis.parameter_name}",
                            "Moving Range"
                        ]
                    )

                    x = list(range(len(analysis.points)))
                    y = [p.value for p in analysis.points]
                    test_ids = [p.test_id for p in analysis.points]
                    colors = ['red' if not p.in_control else 'blue' for p in analysis.points]

                    # Individual chart
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=y,
                            mode='markers+lines',
                            marker=dict(color=colors, size=10),
                            line=dict(color='lightblue', width=1),
                            text=test_ids,
                            hovertemplate='%{text}<br>Value: %{y:.4f}<extra></extra>',
                            name='Data',
                        ),
                        row=1, col=1
                    )

                    # Control limits
                    fig.add_hline(
                        y=analysis.limits.center_line, line_dash="solid",
                        line_color="green", row=1, col=1,
                        annotation_text=f"CL: {analysis.limits.center_line:.4f}"
                    )
                    fig.add_hline(
                        y=analysis.limits.ucl, line_dash="dash",
                        line_color="red", row=1, col=1,
                        annotation_text=f"UCL: {analysis.limits.ucl:.4f}"
                    )
                    fig.add_hline(
                        y=analysis.limits.lcl, line_dash="dash",
                        line_color="red", row=1, col=1,
                        annotation_text=f"LCL: {analysis.limits.lcl:.4f}"
                    )

                    # Campaign boundary lines
                    boundaries = get_campaign_boundaries(system_df)
                    for idx, cname in boundaries[1:]:  # Skip first
                        fig.add_vline(
                            x=idx - 0.5, line_dash="dot", line_color="orange",
                            row=1, col=1,
                            annotation_text=cname, annotation_position="top right",
                            annotation_font_size=8, annotation_font_color="orange",
                        )
                        fig.add_vline(
                            x=idx - 0.5, line_dash="dot", line_color="orange",
                            row=2, col=1,
                        )

                    # Moving range chart
                    mr = np.abs(np.diff(y))
                    mr_x = list(range(1, len(y)))

                    fig.add_trace(
                        go.Scatter(
                            x=mr_x, y=mr,
                            mode='markers+lines',
                            marker=dict(color='blue', size=6),
                            line=dict(color='lightblue', width=1),
                            name='MR',
                        ),
                        row=2, col=1
                    )

                    # MR limits
                    if len(mr) > 0:
                        mr_bar = np.mean(mr)
                        mr_ucl = 3.267 * mr_bar
                        fig.add_hline(y=mr_bar, line_dash="solid", line_color="green", row=2, col=1)
                        fig.add_hline(y=mr_ucl, line_dash="dash", line_color="red", row=2, col=1)

                    fig.update_layout(height=550, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # SPC Summary
                    with st.expander("SPC Summary"):
                        st.markdown(format_spc_summary(analysis))

                    # Per-campaign capability comparison
                    if use_specs and usl is not None and lsl is not None:
                        st.subheader("Per-Campaign Capability")
                        cap_rows = []
                        for cname in loaded_campaigns:
                            cdf = system_df[system_df['campaign_name'] == cname]
                            if spc_param not in cdf.columns:
                                continue
                            valid = cdf[spc_param].dropna()
                            if len(valid) < 2:
                                continue

                            mean_val = valid.mean()
                            std_val = valid.std()

                            if std_val > 0:
                                cp = (usl - lsl) / (6 * std_val)
                                cpk = min(
                                    (usl - mean_val) / (3 * std_val),
                                    (mean_val - lsl) / (3 * std_val)
                                )
                            else:
                                cp = float('inf')
                                cpk = float('inf')

                            cap_rows.append({
                                'Campaign': cname,
                                'N': len(valid),
                                'Mean': f"{mean_val:.4f}",
                                'Std': f"{std_val:.4f}",
                                'Cp': f"{cp:.2f}",
                                'Cpk': f"{cpk:.2f}",
                            })

                        if cap_rows:
                            st.dataframe(
                                pd.DataFrame(cap_rows),
                                use_container_width=True,
                                hide_index=True
                            )

                except Exception as e:
                    st.error(f"SPC Analysis error: {e}")
