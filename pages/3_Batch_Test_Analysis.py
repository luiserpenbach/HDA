"""
Batch Test Analysis Page
========================
Process multiple test files efficiently with consistent settings.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import zipfile
import io
from datetime import datetime
import json

from core.batch_analysis import (
    BatchTestResult,
    BatchAnalysisReport,
    discover_test_files,
    run_batch_analysis,
    load_csv_with_timestamp,
    export_batch_results,
)
from core.campaign_manager_v2 import get_available_campaigns, save_to_campaign
from core.integrated_analysis import analyze_cold_flow_test, analyze_hot_fire_test
from core.config_manager import ConfigManager
from core.steady_state_detection import detect_steady_state_simple, detect_steady_state_auto

st.set_page_config(page_title="Batch Test Analysis", page_icon="BT", layout="wide")

st.title("Batch Test Analysis")
st.markdown("Process multiple test files efficiently with consistent configuration.")

# Initialize session state
if 'batch_report' not in st.session_state:
    st.session_state.batch_report = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}


# Configuration management now uses ConfigManager.get_default_config()
# (replaced create_default_config function)

# Steady-state detection now uses core.steady_state_detection functions
# (replaced detect_steady_simple function - use detect_steady_state_auto or detect_steady_state_simple instead)


def analyze_file_wrapper(df, config, test_id, file_path):
    """Wrapper for analysis function."""
    # Detect steady state using core module
    steady_window = detect_steady_state_simple(df, config, time_col='time_s' if 'time_s' in df.columns else 'timestamp')
    
    test_type = config.get('test_type', 'cold_flow')
    
    if test_type == 'cold_flow':
        return analyze_cold_flow_test(
            df=df,
            config=config,
            steady_window=steady_window,
            test_id=test_id,
            file_path=file_path,
            skip_qc=False,
        )
    else:
        return analyze_hot_fire_test(
            df=df,
            config=config,
            steady_window=steady_window,
            test_id=test_id,
            file_path=file_path,
            skip_qc=False,
        )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Configuration")
    
    test_type = st.selectbox(
        "Test Type",
        ["cold_flow", "hot_fire"],
        format_func=lambda x: "Cold Flow" if x == "cold_flow" else "Hot Fire"
    )

    # Get default config using ConfigManager
    config = ConfigManager.get_default_config(test_type)

    # Config editor
    with st.expander("Edit Configuration"):
        config_str = st.text_area(
            "Config JSON",
            value=json.dumps(config, indent=2),
            height=300
        )
        if st.button("Apply"):
            try:
                config = json.loads(config_str)
                st.success("Updated")
                # Save to recent configs
                ConfigManager.save_to_recent(config, 'custom', config.get('config_name', 'Batch Config'))
            except:
                st.error("Invalid JSON")
    
    st.divider()
    
    # Processing options
    st.subheader("Processing Options")
    
    parallel = st.checkbox("Parallel processing", value=False)
    max_workers = st.slider("Max workers", 1, 8, 4) if parallel else 1
    
    continue_on_error = st.checkbox("Continue on error", value=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.header("1. Upload Files")

uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=['csv'],
    accept_multiple_files=True,
    help="Upload multiple CSV files to process as a batch"
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} files")
    
    # Store file data
    for f in uploaded_files:
        if f.name not in st.session_state.uploaded_files_data:
            f.seek(0)
            st.session_state.uploaded_files_data[f.name] = f.read()
    
    # Preview
    with st.expander("File List"):
        file_info = []
        for f in uploaded_files:
            f.seek(0)
            df_preview = pd.read_csv(f)
            file_info.append({
                'File': f.name,
                'Rows': len(df_preview),
                'Columns': len(df_preview.columns)
            })
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# PROCESSING
# =============================================================================

st.header("2. Run Batch Analysis")

col1, col2 = st.columns([1, 2])

with col1:
    batch_id = st.text_input("Batch ID", value=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}")
    
    run_batch = st.button("Process All Files", type="primary", use_container_width=True)

with col2:
    if run_batch and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, f in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {f.name} ({i+1}/{len(uploaded_files)})")
            
            try:
                f.seek(0)
                df = pd.read_csv(f)
                
                # Ensure timestamp column
                if 'timestamp' not in df.columns:
                    timestamp_candidates = ['time', 'Time', 't', 'time_ms']
                    for col in timestamp_candidates:
                        if col in df.columns:
                            df = df.rename(columns={col: 'timestamp'})
                            break
                    else:
                        df['timestamp'] = df.index * 10
                
                # Analyze
                result = analyze_file_wrapper(df, config, f.name.replace('.csv', ''), f.name)
                
                batch_result = BatchTestResult(
                    file_path=f.name,
                    test_id=f.name.replace('.csv', ''),
                    success=True,
                    measurements=result.measurements,
                    traceability=result.traceability,
                    qc_passed=result.passed_qc,
                    steady_window=result.steady_window,
                )
                results.append(batch_result)
                
            except Exception as e:
                if continue_on_error:
                    batch_result = BatchTestResult(
                        file_path=f.name,
                        test_id=f.name.replace('.csv', ''),
                        success=False,
                        error_message=str(e),
                    )
                    results.append(batch_result)
                else:
                    st.error(f"Error processing {f.name}: {e}")
                    break
        
        # Create report
        report = BatchAnalysisReport(
            batch_id=batch_id,
            config_name=config.get('config_name', 'unknown'),
            start_time=datetime.now(),
            results=results,
        )
        report.update_summary()
        report.end_time = datetime.now()
        
        st.session_state.batch_report = report
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        
        st.success(f"Processed {report.successful}/{report.total_files} files successfully")
    
    elif run_batch:
        st.warning("Please upload files first")

# =============================================================================
# RESULTS
# =============================================================================

if st.session_state.batch_report:
    report = st.session_state.batch_report
    
    st.divider()
    st.header("3. Results")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Files", report.total_files)
    with col2:
        st.metric("Successful", report.successful)
    with col3:
        st.metric("Failed", report.failed)
    with col4:
        st.metric("QC Failed", report.qc_failed)
    with col5:
        st.metric("Time (s)", f"{report.total_time_s:.1f}")
    
    # Results table
    tabs = st.tabs(["All Results", "[PASS] Successful", "[FAIL] Failed", "Export"])
    
    with tabs[0]:
        results_df = report.to_dataframe()
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    with tabs[1]:
        successful = [r for r in report.results if r.success]
        if successful:
            # Show key metrics
            success_data = []
            for r in successful:
                row = {
                    'Test ID': r.test_id,
                    'QC Passed': '' if r.qc_passed else '',
                }
                if r.measurements:
                    for k, v in list(r.measurements.items())[:3]:
                        if hasattr(v, 'value'):
                            row[k] = f"{v.value:.4g} ± {v.uncertainty:.4g}"
                success_data.append(row)
            
            st.dataframe(pd.DataFrame(success_data), use_container_width=True, hide_index=True)
        else:
            st.info("No successful results")
    
    with tabs[2]:
        failed = report.get_failed_tests()
        if failed:
            failed_data = []
            for r in failed:
                failed_data.append({
                    'Test ID': r.test_id,
                    'File': r.file_path,
                    'Error': r.error_message
                })
            st.dataframe(pd.DataFrame(failed_data), use_container_width=True, hide_index=True)
        else:
            st.success("No failed tests!")
    
    with tabs[3]:
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv_data = report.to_dataframe().to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"{report.batch_id}_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON export
            json_data = {
                'batch_id': report.batch_id,
                'config_name': report.config_name,
                'start_time': report.start_time.isoformat(),
                'end_time': report.end_time.isoformat() if report.end_time else None,
                'summary': {
                    'total': report.total_files,
                    'successful': report.successful,
                    'failed': report.failed,
                },
                'results': [r.to_dict() for r in report.results],
            }
            st.download_button(
                "Download JSON",
                json.dumps(json_data, indent=2, default=str),
                file_name=f"{report.batch_id}_results.json",
                mime="application/json"
            )
        
        with col3:
            # Summary text
            st.download_button(
                "Download Summary",
                report.summary(),
                file_name=f"{report.batch_id}_summary.txt",
                mime="text/plain"
            )
        
        # Save to campaign
        st.divider()
        st.subheader("Save to Campaign")
        
        campaigns = get_available_campaigns()
        campaign_options = [c['name'] for c in campaigns if c.get('type') == test_type]
        
        if campaign_options:
            selected_campaign = st.selectbox("Target Campaign", campaign_options)
            
            col1, col2 = st.columns(2)
            
            with col1:
                save_qc_passed_only = st.checkbox("Only save QC-passed tests", value=True)
            
            with col2:
                if st.button("Save to Campaign"):
                    saved = 0
                    skipped = 0
                    
                    for r in report.results:
                        if not r.success:
                            skipped += 1
                            continue
                        
                        if save_qc_passed_only and not r.qc_passed:
                            skipped += 1
                            continue
                        
                        try:
                            # Build record
                            record = {
                                'test_id': r.test_id,
                                'test_path': r.file_path,
                                'qc_passed': 1 if r.qc_passed else 0,
                            }
                            
                            if r.traceability:
                                record.update(r.traceability)
                            
                            if r.measurements:
                                if test_type == 'cold_flow':
                                    for key, meas in r.measurements.items():
                                        if hasattr(meas, 'value'):
                                            if key == 'Cd':
                                                record['avg_cd_CALC'] = meas.value
                                                record['u_cd_CALC'] = meas.uncertainty
                                            elif key == 'pressure_upstream':
                                                record['avg_p_up_bar'] = meas.value
                                            elif key == 'mass_flow':
                                                record['avg_mf_g_s'] = meas.value
                                else:
                                    for key, meas in r.measurements.items():
                                        if hasattr(meas, 'value'):
                                            if key == 'Isp':
                                                record['avg_isp_s'] = meas.value
                                                record['u_isp_s'] = meas.uncertainty
                            
                            save_to_campaign(selected_campaign, record)
                            saved += 1
                        except Exception as e:
                            st.warning(f"Could not save {r.test_id}: {e}")
                            skipped += 1
                    
                    st.success(f"Saved {saved} tests to {selected_campaign} ({skipped} skipped)")
        else:
            st.info(f"No {test_type} campaigns found. Create one in Campaign Management.")

else:
    st.info("Upload files and run batch analysis to see results.")
