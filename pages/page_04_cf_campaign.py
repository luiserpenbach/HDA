# pages/04_❄️_CF_Campaign.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

from data_lib.campaign_manager import (
    get_available_campaigns, create_campaign, get_campaign_data, get_campaign_info
)
from data_lib.cf_analytics import (
    analyze_cd_by_part, plot_cd_vs_pressure, plot_cd_by_fluid, generate_cf_summary_stats
)
from data_lib.spc_analysis import (
    calculate_control_limits, detect_control_violations, plot_spc_chart, calculate_process_capability
)
from data_lib.report_generator import generate_cold_flow_campaign_report

st.set_page_config(page_title="CF Campaign", page_icon="❄️", layout="wide")

st.title("❄️ Cold Flow Campaign Analysis")
st.markdown("Multi-test analysis, SPC tracking, and performance trending")

# --- CAMPAIGN SELECTION / CREATION ---
with st.sidebar:
    st.header("📊 Campaign Manager")

    campaigns = get_available_campaigns()

    mode = st.radio("Mode", ["View Existing", "Create New"])

    if mode == "Create New":
        st.subheader("Create Campaign")
        new_campaign_name = st.text_input("Campaign Name", placeholder="INJ_C1_Acceptance")
        new_description = st.text_area("Description", placeholder="Injector acceptance testing")
        new_part_family = st.text_input("Part Family", placeholder="Swirl Injectors")

        if st.button("➕ Create Campaign", type="primary"):
            try:
                create_campaign(new_campaign_name, campaign_type='cold_flow')
                st.success(f"✅ Created: {new_campaign_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        if campaigns:
            selected_campaign = st.selectbox("Select Campaign", campaigns)

            # Load campaign info
            info = get_campaign_info(selected_campaign)
            if info:
                st.markdown("---")
                st.caption(f"**Created:** {info['created_date'][:10]}")
                st.caption(f"**Type:** {info['campaign_type']}")
                if info['description']:
                    st.caption(f"**Description:** {info['description']}")
        else:
            st.warning("No campaigns found. Create one first.")
            st.stop()

# --- LOAD CAMPAIGN DATA ---
try:
    df = get_campaign_data(selected_campaign)

    if df.empty:
        st.info(f"📭 Campaign '{selected_campaign}' is empty. Add tests using Cold Flow Analysis page.")
        st.stop()

    # Ensure timestamp is datetime
    df['test_timestamp'] = pd.to_datetime(df['test_timestamp'])
    df = df.sort_values('test_timestamp')

    st.success(f"✓ Loaded {len(df)} tests from **{selected_campaign}**")

except Exception as e:
    st.error(f"Error loading campaign: {e}")
    st.stop()

# --- SUMMARY STATISTICS ---
st.subheader("📊 Campaign Summary")

summary = generate_cf_summary_stats(df)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Tests", summary['total_tests'])
col2.metric("Unique Parts", summary['unique_parts'])
col3.metric("Unique Fluids", summary['unique_fluids'])
col4.metric("Avg Cd", f"{summary['cd_overall']['mean']:.4f}")
col5.metric("Cd Std Dev", f"{summary['cd_overall']['std']:.4f}")

# Date range
st.caption(f"**Date Range:** {summary['date_range'][0]} to {summary['date_range'][1]}")

# --- FILTERS ---
st.markdown("---")
st.subheader("🔍 Filters & Selection")

col_filter1, col_filter2, col_filter3 = st.columns(3)

with col_filter1:
    all_parts = df['part'].unique().tolist()
    part_filter = st.multiselect("Filter by Part", all_parts, default=all_parts)

with col_filter2:
    all_fluids = df['fluid'].unique().tolist()
    fluid_filter = st.multiselect("Filter by Fluid", all_fluids, default=all_fluids)

with col_filter3:
    date_filter = st.checkbox("Enable Date Filter")
    if date_filter:
        min_date = df['test_timestamp'].min().date()
        max_date = df['test_timestamp'].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date))

# Apply filters
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered['part'].isin(part_filter)]
df_filtered = df_filtered[df_filtered['fluid'].isin(fluid_filter)]

if date_filter and len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered['test_timestamp'].dt.date >= date_range[0]) &
        (df_filtered['test_timestamp'].dt.date <= date_range[1])
        ]

st.info(f"Showing {len(df_filtered)} tests after filtering")

# --- TABS FOR DIFFERENT ANALYSES ---
tab_overview, tab_cd_analysis, tab_spc, tab_data, tab_report = st.tabs([
    "📈 Overview",
    "🎯 Cd Analysis",
    "📊 SPC Charts",
    "📋 Data Table",
    "📄 Report"
])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    st.subheader("Campaign Overview")

    # Timeline plot
    fig_timeline = go.Figure()

    for part in df_filtered['part'].unique():
        part_data = df_filtered[df_filtered['part'] == part]

        fig_timeline.add_trace(go.Scatter(
            x=part_data['test_timestamp'],
            y=part_data['avg_cd_CALC'],
            mode='lines+markers',
            name=part,
            text=part_data['test_id'],
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Cd: %{y:.4f}<extra></extra>'
        ))

    fig_timeline.update_layout(
        title="Cd Timeline by Part",
        xaxis_title="Test Date",
        yaxis_title="Discharge Coefficient (Cd)",
        height=400,
        hovermode='closest'
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

    # Performance distribution
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        # Cd distribution histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_filtered['avg_cd_CALC'],
            nbinsx=20,
            name='Cd Distribution'
        ))
        fig_hist.update_layout(
            title="Cd Distribution",
            xaxis_title="Cd",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_dist2:
        # Cd by fluid box plot
        fig_box = plot_cd_by_fluid(df_filtered)
        st.plotly_chart(fig_box, use_container_width=True)

    # Parts summary table
    st.subheader("Parts Performance Summary")

    parts_stats = analyze_cd_by_part(df_filtered)

    summary_data = []
    for part, stats in parts_stats.items():
        summary_data.append({
            'Part': part,
            'Tests': stats['n_tests'],
            'Mean Cd': f"{stats['cd_mean']:.4f}",
            'Std Dev': f"{stats['cd_std']:.4f}",
            'Min Cd': f"{stats['cd_min']:.4f}",
            'Max Cd': f"{stats['cd_max']:.4f}",
            'Pressure Range': f"{stats['pressure_range'][0]:.1f} - {stats['pressure_range'][1]:.1f} bar"
        })

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

# --- TAB 2: Cd ANALYSIS ---
with tab_cd_analysis:
    st.subheader("🎯 Discharge Coefficient Analysis")

    # Cd vs Pressure with trendlines
    st.markdown("#### Cd vs Pressure (with Linear Regression)")

    fig_cd_p = plot_cd_vs_pressure(df_filtered, part_filter, fluid_filter)
    st.plotly_chart(fig_cd_p, use_container_width=True)

    # Multi-variable analysis
    st.markdown("---")
    st.markdown("#### Multi-Variable Analysis")

    col_mv1, col_mv2 = st.columns(2)

    with col_mv1:
        # Cd vs Flow
        fig_cd_flow = go.Figure()

        for part in df_filtered['part'].unique():
            part_data = df_filtered[df_filtered['part'] == part]

            fig_cd_flow.add_trace(go.Scatter(
                x=part_data['avg_mf_g_s'],
                y=part_data['avg_cd_CALC'],
                mode='markers',
                name=part,
                text=part_data['test_id'],
                marker=dict(size=10)
            ))

        fig_cd_flow.update_layout(
            title="Cd vs Mass Flow",
            xaxis_title="Mass Flow (g/s)",
            yaxis_title="Cd",
            height=400
        )

        st.plotly_chart(fig_cd_flow, use_container_width=True)

    with col_mv2:
        # Cd vs Temperature
        fig_cd_temp = go.Figure()

        for part in df_filtered['part'].unique():
            part_data = df_filtered[df_filtered['part'] == part]

            fig_cd_temp.add_trace(go.Scatter(
                x=part_data['avg_T_up_K'] - 273.15,
                y=part_data['avg_cd_CALC'],
                mode='markers',
                name=part,
                text=part_data['test_id'],
                marker=dict(size=10)
            ))

        fig_cd_temp.update_layout(
            title="Cd vs Temperature",
            xaxis_title="Temperature (°C)",
            yaxis_title="Cd",
            height=400
        )

        st.plotly_chart(fig_cd_temp, use_container_width=True)

    # 3D scatter (Cd vs P vs Flow)
    st.markdown("---")
    st.markdown("#### 3D Operating Space")

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df_filtered['avg_p_up_bar'],
        y=df_filtered['avg_mf_g_s'],
        z=df_filtered['avg_cd_CALC'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_filtered['avg_cd_CALC'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cd")
        ),
        text=df_filtered['test_id'],
        hovertemplate='<b>%{text}</b><br>P: %{x:.2f} bar<br>Flow: %{y:.2f} g/s<br>Cd: %{z:.4f}<extra></extra>'
    )])

    fig_3d.update_layout(
        title="Operating Space: Pressure, Flow, Cd",
        scene=dict(
            xaxis_title='Pressure (bar)',
            yaxis_title='Mass Flow (g/s)',
            zaxis_title='Cd'
        ),
        height=600
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 3: SPC CHARTS ---
with tab_spc:
    st.subheader("📊 Statistical Process Control")

    if len(df_filtered) < 2:
        st.warning("Need at least 2 data points for SPC analysis")
    else:
        # Configuration
        col_spc1, col_spc2, col_spc3 = st.columns(3)

        spc_metric = col_spc1.selectbox(
            "Metric",
            ['avg_cd_CALC', 'avg_p_up_bar', 'avg_mf_g_s'],
            format_func=lambda x: {'avg_cd_CALC': 'Cd', 'avg_p_up_bar': 'Pressure', 'avg_mf_g_s': 'Mass Flow'}[x]
        )

        spc_method = col_spc2.selectbox("Control Limit Method", ['3sigma', 'individuals'])

        # Specification limits
        enable_specs = col_spc3.checkbox("Enable Spec Limits")

        spec_limits = None
        if enable_specs:
            col_spec1, col_spec2 = st.columns(2)
            usl = col_spec1.number_input("USL", value=float(df_filtered[spc_metric].max() * 1.05))
            lsl = col_spec2.number_input("LSL", value=float(df_filtered[spc_metric].min() * 0.95))
            spec_limits = {'usl': usl, 'lsl': lsl}

        # Calculate SPC
        limits = calculate_control_limits(df_filtered[spc_metric].values, method=spc_method)

        violations = detect_control_violations(
            df_filtered[spc_metric].values,
            limits,
            timestamps=df_filtered['test_timestamp']
        )

        # Plot SPC chart
        fig_spc = plot_spc_chart(
            data=df_filtered[spc_metric],
            timestamps=df_filtered['test_timestamp'],
            metric_name=spc_metric,
            limits=limits,
            spec_limits=spec_limits,
            violations=violations,
            highlight_recent=5
        )

        st.plotly_chart(fig_spc, use_container_width=True)

        # Metrics summary
        st.markdown("---")
        st.subheader("Process Metrics")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        col_m1.metric("Process Mean", f"{limits['mean']:.4f}")
        col_m2.metric("Std Deviation", f"{limits['std']:.4f}")
        col_m3.metric("Sample Size", limits['n'])
        col_m4.metric("Violations", len(violations) if not violations.empty else 0)

        # Violations table
        if not violations.empty:
            st.markdown("---")
            st.subheader("⚠️ Control Violations")

            st.dataframe(violations, use_container_width=True, hide_index=True)

            # Download violations
            csv_violations = violations.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Violations",
                csv_violations,
                f"{selected_campaign}_violations.csv",
                "text/csv"
            )

        # Process capability
        if spec_limits:
            st.markdown("---")
            st.subheader("Process Capability")

            capability = calculate_process_capability(df_filtered[spc_metric], spec_limits)

            col_cap1, col_cap2, col_cap3 = st.columns(3)

            if 'Cp' in capability:
                col_cap1.metric("Cp (Potential)", f"{capability['Cp']:.2f}")

            if 'Cpk' in capability:
                cpk = capability['Cpk']
                col_cap2.metric(
                    "Cpk (Actual)",
                    f"{cpk:.2f}",
                    delta="Good" if cpk >= 1.33 else "Poor",
                    delta_color="normal" if cpk >= 1.33 else "inverse"
                )

            if 'interpretation' in capability:
                col_cap3.info(f"**{capability['interpretation']}**")

# --- TAB 4: DATA TABLE ---
with tab_data:
    st.subheader("📋 Test Data Table")

    # Display options
    col_disp1, col_disp2 = st.columns([3, 1])

    with col_disp1:
        columns_to_show = st.multiselect(
            "Columns to Display",
            df_filtered.columns.tolist(),
            default=['test_id', 'test_timestamp', 'part', 'fluid', 'avg_cd_CALC', 'avg_p_up_bar', 'avg_mf_g_s']
        )

    with col_disp2:
        sort_by = st.selectbox("Sort by", columns_to_show)
        sort_order = st.radio("Order", ["Descending", "Ascending"])

    # Apply display settings
    df_display = df_filtered[columns_to_show].sort_values(
        sort_by,
        ascending=(sort_order == "Ascending")
    )

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Download options
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download as CSV",
            csv,
            f"{selected_campaign}_data.csv",
            "text/csv"
        )

    with col_dl2:
        # Excel download
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Campaign Data')

        st.download_button(
            "📥 Download as Excel",
            buffer.getvalue(),
            f"{selected_campaign}_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- TAB 5: REPORT GENERATION ---
with tab_report:
    st.subheader("📄 Campaign Report Generation")

    st.markdown("""
    Generate a comprehensive HTML report including:
    - Campaign summary statistics
    - Performance charts and trends
    - Parts comparison
    - SPC analysis
    """)

    col_rep1, col_rep2 = st.columns([2, 1])

    with col_rep1:
        report_title = st.text_input("Report Title", value=f"{selected_campaign} - Campaign Report")
        include_spc = st.checkbox("Include SPC Charts", value=True)
        include_3d = st.checkbox("Include 3D Plots", value=True)

    with col_rep2:
        st.markdown("**Report Preview**")
        st.info(
            f"Tests: {len(df_filtered)}\nParts: {df_filtered['part'].nunique()}\nDate Range: {df_filtered['test_timestamp'].min().date()} to {df_filtered['test_timestamp'].max().date()}")

    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            # Collect all figures
            figures = []

            # Timeline
            figures.append(fig_timeline)

            # Cd vs Pressure
            figures.append(plot_cd_vs_pressure(df_filtered, part_filter, fluid_filter))

            # Cd by Fluid
            figures.append(plot_cd_by_fluid(df_filtered))

            # SPC if enabled
            if include_spc:
                figures.append(fig_spc)

            # 3D if enabled
            if include_3d:
                figures.append(fig_3d)

            # Generate report
            html_report = generate_cold_flow_campaign_report(
                selected_campaign,
                df_filtered,
                summary,
                figures
            )

            st.download_button(
                "📥 Download Report",
                html_report,
                f"{selected_campaign}_report_{datetime.now().strftime('%Y%m%d')}.html",
                "text/html",
                use_container_width=True
            )

            st.success("✅ Report generated!")
            st.balloons()