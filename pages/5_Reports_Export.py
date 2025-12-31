"""
Reports & Export Page
=====================
Generate professional reports and export data packages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import zipfile
from datetime import datetime
from pathlib import Path

from core.campaign_manager_v2 import get_available_campaigns, get_campaign_data, get_campaign_info
from core.spc import analyze_campaign_spc
from core.reporting import generate_campaign_report, generate_test_report
from core.uncertainty import MeasurementWithUncertainty

st.set_page_config(page_title="Reports & Export", page_icon="RPT", layout="wide")

st.title("Reports & Export")
st.markdown("Generate professional reports and export data packages.")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Settings")
    
    campaigns = get_available_campaigns()
    
    if campaigns:
        campaign_names = [c['name'] for c in campaigns]
        selected_campaign = st.selectbox("Select Campaign", campaign_names)
        
        for c in campaigns:
            if c['name'] == selected_campaign:
                st.caption(f"Type: {c.get('type', 'unknown')}")
                st.caption(f"Tests: {c.get('test_count', 0)}")
    else:
        selected_campaign = None
        st.warning("No campaigns found")

# =============================================================================
# MAIN CONTENT
# =============================================================================

if selected_campaign:
    df = get_campaign_data(selected_campaign)
    info = get_campaign_info(selected_campaign)
    campaign_type = info.get('type', 'cold_flow')
    
    if df is not None and len(df) > 0:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Campaign Report", 
            "Test Reports", 
            "Data Export",
            " Qualification Package"
        ])
        
        # =============================================================================
        # TAB 1: Campaign Report
        # =============================================================================
        
        with tab1:
            st.header("Campaign Summary Report")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Report Options")
                
                numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
                metric_cols = [c for c in numeric_cols if c.startswith('avg_')]
                
                default_params = metric_cols[:4] if metric_cols else numeric_cols[:4]
                
                selected_params = st.multiselect(
                    "Parameters to include",
                    numeric_cols,
                    default=default_params
                )
                
                include_spc = st.checkbox("Include SPC analysis", value=True)
                
                specs = {}
                if include_spc and selected_params:
                    with st.expander("Specification Limits"):
                        for param in selected_params[:2]:
                            st.markdown(f"**{param}**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                lsl = st.number_input(f"LSL", key=f"lsl_{param}", value=0.0)
                            with col_b:
                                usl = st.number_input(f"USL", key=f"usl_{param}", value=1.0)
                            
                            if lsl != 0.0 or usl != 1.0:
                                specs[param] = {'lsl': lsl, 'usl': usl}
                
                generate_report = st.button("Generate Report", type="primary")
            
            with col2:
                if generate_report and selected_params:
                    with st.spinner("Generating report..."):
                        try:
                            spc_analyses = None
                            if include_spc and specs:
                                spc_analyses = analyze_campaign_spc(df, list(specs.keys()), specs)
                            
                            html = generate_campaign_report(
                                campaign_name=selected_campaign,
                                df=df,
                                parameters=selected_params,
                                spc_analyses=spc_analyses,
                            )
                            
                            st.success("Report generated!")
                            
                            with st.expander("Preview Report"):
                                st.components.v1.html(html, height=600, scrolling=True)
                            
                            st.download_button(
                                "Download HTML Report",
                                html,
                                file_name=f"{selected_campaign}_report.html",
                                mime="text/html"
                            )
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.info("Configure report options and click Generate")
        
        # =============================================================================
        # TAB 2: Test Reports
        # =============================================================================
        
        with tab2:
            st.header("Individual Test Reports")

            if 'test_id' in df.columns:
                test_ids = df['test_id'].tolist()

                # Bulk Report Generation Option
                generate_all = st.checkbox(
                    f"📦 Generate Reports for All Tests ({len(test_ids)} tests)",
                    value=False,
                    help="Generate HTML reports for all tests in this campaign and download as ZIP"
                )

                if generate_all:
                    st.info(f"Bulk mode: Will generate {len(test_ids)} reports")
                    include_config_bulk = st.checkbox("Include config snapshots in all reports", value=False)

                    if st.button("🚀 Generate All Reports", type="primary"):
                        import zipfile
                        import tempfile

                        with st.spinner(f"Generating {len(test_ids)} reports..."):
                            # Create progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # Create a temporary directory for reports
                            with tempfile.TemporaryDirectory() as tmpdir:
                                tmpdir_path = Path(tmpdir)
                                reports_generated = 0
                                errors = []

                                for idx, test_id in enumerate(test_ids):
                                    try:
                                        status_text.text(f"Generating report {idx+1}/{len(test_ids)}: {test_id}")

                                        test_row = df[df['test_id'] == test_id].iloc[0]

                                        # Extract measurements
                                        measurements = {}
                                        for col in df.columns:
                                            if col.startswith('avg_'):
                                                val = test_row[col]
                                                if pd.notna(val):
                                                    u_col = col.replace('avg_', 'u_')
                                                    if u_col in df.columns and pd.notna(test_row[u_col]):
                                                        measurements[col] = MeasurementWithUncertainty(
                                                            float(val), float(test_row[u_col]), '', col
                                                        )
                                                    else:
                                                        measurements[col] = float(val)

                                        # Extract traceability
                                        traceability = {}
                                        trace_cols = ['raw_data_hash', 'config_hash', 'analyst_username',
                                                     'analysis_timestamp_utc', 'processing_version']
                                        for col in trace_cols:
                                            if col in df.columns:
                                                traceability[col] = test_row[col]

                                        # QC report
                                        qc_report = {
                                            'passed': bool(test_row.get('qc_passed', True)),
                                            'summary': {},
                                            'checks': []
                                        }

                                        # Generate report
                                        html = generate_test_report(
                                            test_id=test_id,
                                            test_type=campaign_type,
                                            measurements=measurements,
                                            traceability=traceability,
                                            qc_report=qc_report,
                                            include_config_snapshot=include_config_bulk,
                                        )

                                        # Save to temp directory
                                        report_path = tmpdir_path / f"{test_id}_report.html"
                                        with open(report_path, 'w') as f:
                                            f.write(html)

                                        reports_generated += 1

                                    except Exception as e:
                                        errors.append(f"{test_id}: {str(e)}")

                                    # Update progress
                                    progress_bar.progress((idx + 1) / len(test_ids))

                                progress_bar.empty()
                                status_text.empty()

                                # Create ZIP file
                                if reports_generated > 0:
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                        for report_file in tmpdir_path.glob("*.html"):
                                            zf.write(report_file, arcname=report_file.name)

                                        # Add summary file
                                        summary = f"""Bulk Report Generation Summary
Campaign: {selected_campaign}
Date: {datetime.now().isoformat()}
Total Tests: {len(test_ids)}
Reports Generated: {reports_generated}
Errors: {len(errors)}
"""
                                        if errors:
                                            summary += "\n\nErrors:\n" + "\n".join(errors)

                                        zf.writestr("SUMMARY.txt", summary)

                                    zip_buffer.seek(0)

                                    st.success(f"✅ Generated {reports_generated} reports!")

                                    if errors:
                                        with st.expander(f"⚠️ {len(errors)} Errors"):
                                            for error in errors:
                                                st.warning(error)

                                    st.download_button(
                                        "📥 Download All Reports (ZIP)",
                                        zip_buffer,
                                        file_name=f"{selected_campaign}_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                        mime="application/zip"
                                    )
                                else:
                                    st.error("No reports were generated successfully")

                    st.divider()
                    st.caption("Or generate individual reports below:")

                col1, col2 = st.columns([1, 2])

                with col1:
                    selected_test = st.selectbox("Select Test", test_ids, disabled=generate_all)
                    include_config = st.checkbox("Include config snapshot", value=False, disabled=generate_all)
                    generate_test_btn = st.button("Generate Test Report", disabled=generate_all)
                
                with col2:
                    if generate_test_btn and selected_test:
                        test_row = df[df['test_id'] == selected_test].iloc[0]
                        
                        measurements = {}
                        for col in df.columns:
                            if col.startswith('avg_'):
                                val = test_row[col]
                                if pd.notna(val):
                                    u_col = col.replace('avg_', 'u_')
                                    if u_col in df.columns and pd.notna(test_row[u_col]):
                                        measurements[col] = MeasurementWithUncertainty(
                                            float(val), float(test_row[u_col]), '', col
                                        )
                                    else:
                                        measurements[col] = float(val)
                        
                        traceability = {}
                        trace_cols = ['raw_data_hash', 'config_hash', 'analyst_username', 
                                     'analysis_timestamp_utc', 'processing_version']
                        for col in trace_cols:
                            if col in df.columns:
                                traceability[col] = test_row[col]
                        
                        qc_report = {
                            'passed': bool(test_row.get('qc_passed', True)),
                            'summary': {},
                            'checks': []
                        }
                        
                        try:
                            html = generate_test_report(
                                test_id=selected_test,
                                test_type=campaign_type,
                                measurements=measurements,
                                traceability=traceability,
                                qc_report=qc_report,
                                include_config_snapshot=include_config,
                            )
                            
                            st.success("Report generated!")
                            
                            st.download_button(
                                "Download Test Report",
                                html,
                                file_name=f"{selected_test}_report.html",
                                mime="text/html"
                            )
                            
                            with st.expander("Preview"):
                                st.components.v1.html(html, height=500, scrolling=True)
                                
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.warning("No test_id column in campaign data")
        
        # =============================================================================
        # TAB 3: Data Export
        # =============================================================================
        
        with tab3:
            st.header("Export Campaign Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Export Options")
                
                export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])
                include_uncertainties = st.checkbox("Include uncertainty columns", value=True)
                include_traceability = st.checkbox("Include traceability columns", value=True)
                
                with st.expander("Column Selection"):
                    all_cols = list(df.columns)
                    selected_cols = st.multiselect("Columns", all_cols, default=all_cols)
            
            with col2:
                st.subheader("Export")
                
                if st.button("Generate Export", type="primary"):
                    try:
                        export_df = df[selected_cols].copy()
                        
                        if not include_uncertainties:
                            export_df = export_df[[c for c in export_df.columns if not c.startswith('u_')]]
                        
                        if not include_traceability:
                            trace_cols = ['raw_data_hash', 'config_hash', 'analyst_username',
                                         'analysis_timestamp_utc', 'processing_version']
                            export_df = export_df[[c for c in export_df.columns if c not in trace_cols]]
                        
                        if export_format == "CSV":
                            st.download_button(
                                "Download CSV",
                                export_df.to_csv(index=False),
                                file_name=f"{selected_campaign}_export.csv",
                                mime="text/csv"
                            )
                        
                        elif export_format == "Excel":
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                export_df.to_excel(writer, sheet_name='Data', index=False)
                                
                                meta_df = pd.DataFrame([
                                    {'Field': 'Campaign', 'Value': selected_campaign},
                                    {'Field': 'Type', 'Value': campaign_type},
                                    {'Field': 'Export Date', 'Value': datetime.now().isoformat()},
                                    {'Field': 'Test Count', 'Value': len(export_df)},
                                ])
                                meta_df.to_excel(writer, sheet_name='Metadata', index=False)
                            
                            st.download_button(
                                "Download Excel",
                                buffer.getvalue(),
                                file_name=f"{selected_campaign}_export.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        elif export_format == "JSON":
                            export_data = {
                                'campaign': selected_campaign,
                                'type': campaign_type,
                                'export_date': datetime.now().isoformat(),
                                'test_count': len(export_df),
                                'data': export_df.to_dict(orient='records')
                            }
                            st.download_button(
                                "Download JSON",
                                json.dumps(export_data, indent=2, default=str),
                                file_name=f"{selected_campaign}_export.json",
                                mime="application/json"
                            )
                        
                        st.success("Export ready!")
                        
                    except Exception as e:
                        st.error(f"Export error: {e}")
            
            st.divider()
            st.subheader("Traceability Report")
            
            if st.button("Generate Traceability Report"):
                try:
                    lines = [
                        "=" * 70,
                        "DATA TRACEABILITY REPORT",
                        f"Campaign: {selected_campaign}",
                        f"Generated: {datetime.now().isoformat()}",
                        "=" * 70,
                        "",
                        f"Total Tests: {len(df)}",
                    ]
                    
                    if 'analyst_username' in df.columns:
                        analysts = df['analyst_username'].dropna().unique()
                        lines.append(f"Analysts: {', '.join(str(a) for a in analysts)}")
                    
                    lines.extend(["", "-" * 70, "TEST RECORDS", "-" * 70, ""])
                    
                    trace_cols = ['test_id', 'raw_data_hash', 'config_hash', 
                                 'analyst_username', 'analysis_timestamp_utc']
                    available = [c for c in trace_cols if c in df.columns]
                    
                    for _, row in df.iterrows():
                        lines.append(f"Test: {row.get('test_id', 'Unknown')}")
                        for col in available:
                            if col != 'test_id' and pd.notna(row.get(col)):
                                lines.append(f"  {col}: {row[col]}")
                        lines.append("")
                    
                    report_text = "\n".join(lines)
                    
                    st.download_button(
                        "Download Traceability Report",
                        report_text,
                        file_name=f"{selected_campaign}_traceability.txt",
                        mime="text/plain"
                    )
                    
                    with st.expander("Preview"):
                        st.text(report_text[:2000] + "..." if len(report_text) > 2000 else report_text)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # =============================================================================
        # TAB 4: Qualification Package
        # =============================================================================
        
        with tab4:
            st.header("Qualification Documentation Package")
            
            st.markdown("""
            Generate a complete documentation package suitable for flight qualification:
            - **Summary CSV**: Key metrics only
            - **Full Data CSV**: Complete dataset
            - **Traceability Report**: Audit trail
            - **JSON Archive**: Machine-readable data
            - **Manifest**: Package contents
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Package Options")
                
                with st.expander("Additional Metadata"):
                    project_name = st.text_input("Project Name", value="")
                    part_number = st.text_input("Part Number", value="")
                    revision = st.text_input("Revision", value="A")
                    prepared_by = st.text_input("Prepared By", value="")
            
            with col2:
                st.subheader("Generate Package")
                
                if st.button(" Generate Qualification Package", type="primary"):
                    with st.spinner("Generating package..."):
                        try:
                            zip_buffer = io.BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                # Summary CSV
                                summary_cols = ['test_id', 'test_timestamp', 'part', 'serial_num']
                                metric_cols = [c for c in df.columns if c.startswith('avg_')]
                                summary_cols.extend([c for c in metric_cols if c in df.columns])
                                summary_cols.extend(['qc_passed'])
                                available_summary = [c for c in summary_cols if c in df.columns]
                                
                                summary_csv = df[available_summary].to_csv(index=False)
                                zf.writestr(f"{selected_campaign}_summary.csv", summary_csv)
                                
                                # Full data CSV
                                full_csv = df.to_csv(index=False)
                                zf.writestr(f"{selected_campaign}_full_data.csv", full_csv)
                                
                                # JSON archive
                                archive_data = {
                                    'campaign_name': selected_campaign,
                                    'campaign_type': campaign_type,
                                    'export_date': datetime.now().isoformat(),
                                    'package_metadata': {
                                        'project': project_name,
                                        'part_number': part_number,
                                        'revision': revision,
                                        'prepared_by': prepared_by,
                                    },
                                    'summary': {
                                        'total_tests': len(df),
                                        'qc_passed': int(df['qc_passed'].sum()) if 'qc_passed' in df.columns else len(df),
                                    },
                                    'data': df.to_dict(orient='records')
                                }
                                json_content = json.dumps(archive_data, indent=2, default=str)
                                zf.writestr(f"{selected_campaign}_archive.json", json_content)
                                
                                # Traceability report
                                trace_lines = [
                                    "DATA TRACEABILITY REPORT",
                                    f"Campaign: {selected_campaign}",
                                    f"Generated: {datetime.now().isoformat()}",
                                    f"Total Tests: {len(df)}",
                                    "",
                                ]
                                for _, row in df.iterrows():
                                    trace_lines.append(f"Test: {row.get('test_id', 'Unknown')}")
                                    if 'raw_data_hash' in df.columns:
                                        trace_lines.append(f"  Hash: {row.get('raw_data_hash', 'N/A')}")
                                    trace_lines.append("")
                                
                                zf.writestr(f"{selected_campaign}_traceability.txt", "\n".join(trace_lines))
                                
                                # Manifest
                                manifest = {
                                    'package_name': f"{selected_campaign}_qual_package",
                                    'created': datetime.now().isoformat(),
                                    'campaign': selected_campaign,
                                    'test_count': len(df),
                                    'files': [
                                        f"{selected_campaign}_summary.csv",
                                        f"{selected_campaign}_full_data.csv",
                                        f"{selected_campaign}_archive.json",
                                        f"{selected_campaign}_traceability.txt",
                                    ],
                                    'metadata': {
                                        'project': project_name,
                                        'part_number': part_number,
                                        'revision': revision,
                                        'prepared_by': prepared_by,
                                    }
                                }
                                zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
                            
                            zip_buffer.seek(0)
                            
                            st.success("Qualification package generated!")
                            
                            st.download_button(
                                "Download Qualification Package (ZIP)",
                                zip_buffer.getvalue(),
                                file_name=f"{selected_campaign}_qual_package.zip",
                                mime="application/zip"
                            )
                            
                        except Exception as e:
                            st.error(f"Error generating package: {e}")
                            import traceback
                            st.code(traceback.format_exc())
    
    else:
        st.info("No test data in selected campaign.")

else:
    st.info("Select a campaign in the sidebar")
