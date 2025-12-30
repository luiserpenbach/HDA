"""
Campaign Management Page
========================
Create and manage test campaigns with full traceability.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from core.campaign_manager_v2 import (
    get_available_campaigns,
    create_campaign,
    get_campaign_info,
    get_campaign_data,
    verify_test_data_integrity,
    get_test_traceability,
)

st.set_page_config(page_title="Campaign Management", page_icon="CM", layout="wide")

st.title("Campaign Management")
st.markdown("Create and manage test campaigns with full data traceability.")

# =============================================================================
# SIDEBAR - Campaign List
# =============================================================================

with st.sidebar:
    st.header("Campaigns")

    campaigns = get_available_campaigns()

    if campaigns:
        campaign_names = [c['name'] for c in campaigns]
        selected_campaign = st.selectbox("Select Campaign", campaign_names)

        # Show campaign info
        for c in campaigns:
            if c['name'] == selected_campaign:
                st.caption(f"Type: {c.get('type', 'unknown')}")
                st.caption(f"Tests: {c.get('test_count', 0)}")
                created = c.get('created_date')
                st.caption(f"Created: {created[:10] if created else 'N/A'}")
    else:
        selected_campaign = None
        st.info("No campaigns found")

    st.divider()

    # Create new campaign
    st.subheader("Create New Campaign")

    new_name = st.text_input("Campaign Name")
    new_type = st.selectbox("Campaign Type", ["cold_flow", "hot_fire"])
    new_desc = st.text_area("Description", height=100)

    if st.button("Create Campaign", use_container_width=True):
        if new_name:
            try:
                create_campaign(
                    campaign_name=new_name,
                    campaign_type=new_type,
                )
                st.success(f"Created campaign: {new_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Enter a campaign name")

# =============================================================================
# MAIN CONTENT
# =============================================================================

if selected_campaign:
    # Load campaign data
    try:
        info = get_campaign_info(selected_campaign)
        df = get_campaign_data(selected_campaign)
    except Exception as e:
        st.error(f"Error loading campaign: {e}")
        st.stop()

    # Header
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tests", len(df) if df is not None else 0)
    with col2:
        if df is not None and 'qc_passed' in df.columns:
            qc_pass = df['qc_passed'].sum()
            st.metric("QC Passed", f"{qc_pass}/{len(df)}")
        else:
            st.metric("QC Passed", "N/A")
    with col3:
        campaign_type = info.get('type', 'unknown')
        st.metric("Type", campaign_type.replace('_', ' ').title())
    with col4:
        st.metric("Schema Version", info.get('schema_version', 'N/A'))

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Table", "Traceability", "Export"])

    # =============================================================================
    # TAB 1: Overview
    # =============================================================================

    with tab1:
        if df is not None and len(df) > 0:
            st.subheader("Campaign Overview")

            # Identify metric columns
            if campaign_type == 'cold_flow':
                primary_metric = 'avg_cd_CALC'
                secondary_metrics = ['avg_p_up_bar', 'avg_mf_g_s', 'avg_delta_p_bar']
            else:
                primary_metric = 'avg_isp_s'
                secondary_metrics = ['avg_c_star_m_s', 'avg_of_ratio', 'avg_p_c_bar']

            # Time series plot of primary metric
            if primary_metric in df.columns:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = go.Figure()

                    # Add data points
                    x_vals = df['test_id'] if 'test_id' in df.columns else df.index
                    y_vals = df[primary_metric]

                    # Color by QC status if available
                    if 'qc_passed' in df.columns:
                        colors = ['green' if qc else 'red' for qc in df['qc_passed']]
                    else:
                        colors = 'blue'

                    fig.add_trace(go.Scatter(
                        x=list(range(len(x_vals))),
                        y=y_vals,
                        mode='markers+lines',
                        marker=dict(color=colors, size=10),
                        line=dict(color='lightblue', width=1),
                        text=x_vals,
                        hovertemplate='%{text}<br>Value: %{y:.4f}<extra></extra>'
                    ))

                    # Add mean line
                    mean_val = y_vals.mean()
                    fig.add_hline(y=mean_val, line_dash="dash",
                                  annotation_text=f"Mean: {mean_val:.4f}")

                    fig.update_layout(
                        title=f"{primary_metric} Trend",
                        xaxis_title="Test Number",
                        yaxis_title=primary_metric,
                        height=400,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Statistics
                    st.markdown("**Statistics**")

                    valid_data = y_vals.dropna()

                    st.metric("Mean", f"{valid_data.mean():.4f}")
                    st.metric("Std Dev", f"{valid_data.std():.4f}")
                    st.metric("Min", f"{valid_data.min():.4f}")
                    st.metric("Max", f"{valid_data.max():.4f}")
                    st.metric("CV (%)", f"{(valid_data.std() / valid_data.mean() * 100):.2f}")

            # Distribution plot
            st.subheader("Distribution")

            col1, col2 = st.columns(2)

            with col1:
                if primary_metric in df.columns:
                    fig = px.histogram(
                        df, x=primary_metric,
                        nbins=20,
                        title=f"{primary_metric} Distribution"
                    )
                    fig.add_vline(x=df[primary_metric].mean(), line_dash="dash",
                                  annotation_text="Mean")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Show available metrics
                available_metrics = [m for m in secondary_metrics if m in df.columns]
                if available_metrics:
                    selected_metric = st.selectbox("Select metric", available_metrics)

                    fig = px.histogram(
                        df, x=selected_metric,
                        nbins=20,
                        title=f"{selected_metric} Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No test data in this campaign yet.")

    # =============================================================================
    # TAB 2: Data Table
    # =============================================================================

    with tab2:
        if df is not None and len(df) > 0:
            st.subheader("Test Data")

            # Column filter
            all_columns = list(df.columns)
            default_cols = ['test_id', 'test_timestamp', 'part', 'serial_num']

            if campaign_type == 'cold_flow':
                default_cols.extend(['avg_cd_CALC', 'u_cd_CALC', 'avg_p_up_bar', 'qc_passed'])
            else:
                default_cols.extend(['avg_isp_s', 'u_isp_s', 'avg_c_star_m_s', 'qc_passed'])

            default_cols = [c for c in default_cols if c in all_columns]

            selected_cols = st.multiselect(
                "Columns to display",
                all_columns,
                default=default_cols
            )

            if selected_cols:
                st.dataframe(df[selected_cols], use_container_width=True, hide_index=True)
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Download
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name=f"{selected_campaign}_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No test data in this campaign yet.")

    # =============================================================================
    # TAB 3: Traceability
    # =============================================================================

    with tab3:
        st.subheader("Data Traceability")

        if df is not None and len(df) > 0 and 'test_id' in df.columns:
            # Select test to verify
            test_ids = df['test_id'].tolist()
            selected_test = st.selectbox("Select Test", test_ids)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Traceability Record**")

                try:
                    trace = get_test_traceability(selected_campaign, selected_test)

                    if trace:
                        for key, value in trace.items():
                            if value is not None:
                                st.text(f"{key}: {value}")
                    else:
                        st.info("No traceability data found")
                except Exception as e:
                    st.error(f"Error loading traceability: {e}")

            with col2:
                st.markdown("**Integrity Verification**")

                if st.button("Verify Data Integrity"):
                    try:
                        # This would need the original file to verify
                        st.info("To verify integrity, upload the original test data file.")

                        original_file = st.file_uploader("Original file", type=['csv'])

                        if original_file:
                            is_valid = verify_test_data_integrity(
                                selected_campaign,
                                selected_test,
                                original_file.name
                            )

                            if is_valid:
                                st.success("[PASS] Data integrity verified")
                            else:
                                st.error("[FAIL] Data integrity check failed")
                    except Exception as e:
                        st.error(f"Verification error: {e}")

            # Traceability summary
            st.divider()
            st.markdown("**Campaign Traceability Summary**")

            trace_cols = ['test_id', 'raw_data_hash', 'config_hash', 'analyst_username',
                          'analysis_timestamp_utc', 'processing_version']
            available_trace = [c for c in trace_cols if c in df.columns]

            if available_trace:
                st.dataframe(df[available_trace], use_container_width=True, hide_index=True)
            else:
                st.info("No traceability columns in campaign data")
        else:
            st.info("No test data in this campaign yet.")

    # =============================================================================
    # TAB 4: Export
    # =============================================================================

    with tab4:
        st.subheader("Export Campaign Data")

        if df is not None and len(df) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Export Options**")

                include_uncertainties = st.checkbox("Include uncertainty columns", value=True)
                include_traceability = st.checkbox("Include traceability columns", value=True)
                export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])

            with col2:
                st.markdown("**Export**")

                if export_format == "CSV":
                    cols_to_export = list(df.columns)
                    if not include_uncertainties:
                        cols_to_export = [c for c in cols_to_export if not c.startswith('u_')]
                    if not include_traceability:
                        trace_cols = ['raw_data_hash', 'config_hash', 'analyst_username',
                                      'analysis_timestamp_utc', 'processing_version']
                        cols_to_export = [c for c in cols_to_export if c not in trace_cols]

                    export_df = df[cols_to_export]

                    st.download_button(
                        "Download CSV",
                        export_df.to_csv(index=False),
                        file_name=f"{selected_campaign}_export.csv",
                        mime="text/csv"
                    )

                elif export_format == "Excel":
                    from core.export import export_campaign_excel
                    import tempfile

                    if st.button("Generate Excel"):
                        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
                            export_campaign_excel(
                                df, f.name,
                                campaign_info={'name': selected_campaign, 'type': campaign_type}
                            )

                            with open(f.name, 'rb') as excel_file:
                                st.download_button(
                                    "Download Excel",
                                    excel_file.read(),
                                    file_name=f"{selected_campaign}_export.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                elif export_format == "JSON":
                    import json
                    from core.export import export_campaign_json

                    export_data = {
                        'campaign_name': selected_campaign,
                        'campaign_type': campaign_type,
                        'export_date': datetime.now().isoformat(),
                        'test_count': len(df),
                        'data': df.to_dict(orient='records')
                    }

                    st.download_button(
                        "Download JSON",
                        json.dumps(export_data, indent=2, default=str),
                        file_name=f"{selected_campaign}_export.json",
                        mime="application/json"
                    )
        else:
            st.info("No test data to export.")

else:
    st.info("Select or create a campaign in the sidebar")