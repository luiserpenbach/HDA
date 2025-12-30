"""
Enhanced Data Export Module
===========================
Export campaign data with full metadata, uncertainties, and traceability.

Supported Formats:
- CSV with uncertainty columns
- Excel with multiple sheets (data, metadata, QC)
- JSON with full structure
- Parquet for efficient storage
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import json


def export_campaign_csv(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    include_uncertainties: bool = True,
    include_traceability: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Export campaign data to CSV with optional metadata header.
    
    Args:
        df: Campaign DataFrame
        output_path: Output file path
        include_uncertainties: Include uncertainty columns
        include_traceability: Include traceability columns
        metadata: Additional metadata to include in header
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    # Select columns to export
    columns = list(df.columns)
    
    if not include_uncertainties:
        columns = [c for c in columns if not c.startswith('u_')]
    
    if not include_traceability:
        trace_cols = [
            'raw_data_hash', 'raw_data_path', 'config_hash', 'config_snapshot',
            'analyst_username', 'analyst_hostname', 'analysis_timestamp_utc',
            'processing_version', 'detection_parameters', 'stability_channels'
        ]
        columns = [c for c in columns if c not in trace_cols]
    
    export_df = df[columns].copy()
    
    # Write with metadata header if provided
    if metadata:
        with open(output_path, 'w') as f:
            f.write("# Hopper Data Studio Export\n")
            f.write(f"# Export Date: {datetime.now().isoformat()}\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")
        
        export_df.to_csv(output_path, mode='a', index=False)
    else:
        export_df.to_csv(output_path, index=False)
    
    return output_path


def export_campaign_excel(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    campaign_info: Optional[Dict[str, Any]] = None,
    qc_summary: Optional[pd.DataFrame] = None,
    spc_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Export campaign data to Excel with multiple sheets.
    
    Sheets:
    - Data: Main test results
    - Metadata: Campaign information
    - QC_Summary: Quality control statistics
    - SPC: Statistical process control results
    
    Args:
        df: Campaign DataFrame
        output_path: Output file path
        campaign_info: Campaign metadata dictionary
        qc_summary: QC summary DataFrame
        spc_summary: SPC analysis results
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # Metadata sheet
        if campaign_info:
            meta_df = pd.DataFrame([
                {'Field': k, 'Value': str(v)} 
                for k, v in campaign_info.items()
            ])
            meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        # QC summary sheet
        if qc_summary is not None:
            qc_summary.to_excel(writer, sheet_name='QC_Summary', index=False)
        
        # SPC summary sheet
        if spc_summary:
            spc_rows = []
            for param, analysis in spc_summary.items():
                row = {
                    'Parameter': param,
                    'Center Line': analysis.limits.center_line,
                    'UCL': analysis.limits.ucl,
                    'LCL': analysis.limits.lcl,
                    'Points': analysis.n_points,
                    'Violations': analysis.n_violations,
                    'In Control': analysis.n_violations == 0,
                }
                if analysis.capability:
                    row['Cpk'] = analysis.capability.cpk
                    row['Ppk'] = analysis.capability.ppk
                if analysis.has_trend:
                    row['Trend'] = analysis.trend_direction
                spc_rows.append(row)
            
            spc_df = pd.DataFrame(spc_rows)
            spc_df.to_excel(writer, sheet_name='SPC', index=False)
    
    return output_path


def export_campaign_json(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    campaign_info: Optional[Dict[str, Any]] = None,
    include_full_data: bool = True,
    pretty_print: bool = True,
) -> Path:
    """
    Export campaign data to JSON with full structure.
    
    Args:
        df: Campaign DataFrame
        output_path: Output file path
        campaign_info: Campaign metadata
        include_full_data: Include all test data (can be large)
        pretty_print: Format JSON with indentation
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    export_data = {
        'export_info': {
            'format_version': '2.0',
            'export_date': datetime.now().isoformat(),
            'exporter': 'Hopper Data Studio',
        },
        'campaign': campaign_info or {},
        'summary': {
            'total_tests': len(df),
            'columns': list(df.columns),
        },
    }
    
    # Add statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    for col in numeric_cols:
        if df[col].notna().any():
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].notna().sum()),
            }
    export_data['statistics'] = stats
    
    # Add full data if requested
    if include_full_data:
        # Convert DataFrame to list of records
        export_data['tests'] = df.to_dict(orient='records')
    
    indent = 2 if pretty_print else None
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=indent, default=str)
    
    return output_path


def export_campaign_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    compression: str = 'snappy',
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Export campaign data to Parquet format.
    
    Parquet is efficient for large datasets and preserves types.
    
    Args:
        df: Campaign DataFrame
        output_path: Output file path
        compression: Compression codec ('snappy', 'gzip', 'brotli', None)
        metadata: Custom metadata to embed in file
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    # Add metadata if provided
    if metadata:
        import pyarrow as pa
        table = pa.Table.from_pandas(df)
        
        existing_meta = table.schema.metadata or {}
        new_meta = {k.encode(): v.encode() for k, v in metadata.items()}
        new_meta.update(existing_meta)
        
        table = table.replace_schema_metadata(new_meta)
        
        import pyarrow.parquet as pq
        pq.write_table(table, output_path, compression=compression)
    else:
        df.to_parquet(output_path, compression=compression, index=False)
    
    return output_path


def export_test_data_with_context(
    test_id: str,
    measurements: Dict[str, Any],
    traceability: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = 'json',
) -> Path:
    """
    Export a single test's data with full context.
    
    Creates a self-contained export that includes:
    - Measurements with uncertainties
    - Complete traceability record
    - Configuration snapshot
    
    Args:
        test_id: Test identifier
        measurements: Measurements dictionary
        traceability: Traceability record
        config: Test configuration
        output_path: Output file path
        format: 'json' or 'yaml'
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    # Build export structure
    export_data = {
        'test_id': test_id,
        'export_date': datetime.now().isoformat(),
        'format_version': '2.0',
        
        'measurements': {},
        'traceability': traceability,
        'configuration': config,
    }
    
    # Convert measurements
    for name, meas in measurements.items():
        if hasattr(meas, 'value') and hasattr(meas, 'uncertainty'):
            export_data['measurements'][name] = {
                'value': float(meas.value),
                'uncertainty': float(meas.uncertainty),
                'unit': getattr(meas, 'unit', ''),
                'relative_uncertainty_pct': float(meas.relative_uncertainty_percent),
            }
        elif isinstance(meas, (int, float)):
            export_data['measurements'][name] = {'value': float(meas)}
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    elif format == 'yaml':
        try:
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML export: pip install pyyaml")
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return output_path


def create_traceability_report(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Create a traceability report showing data provenance.
    
    Args:
        df: Campaign DataFrame with traceability columns
        output_path: Output file path
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    trace_cols = [
        'test_id', 'raw_data_hash', 'raw_data_filename', 
        'config_name', 'config_hash',
        'analyst_username', 'analysis_timestamp_utc',
        'processing_version', 'qc_passed'
    ]
    
    available_cols = [c for c in trace_cols if c in df.columns]
    
    if not available_cols:
        raise ValueError("No traceability columns found in DataFrame")
    
    trace_df = df[available_cols].copy()
    
    # Generate report
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATA TRACEABILITY REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Tests: {len(df)}\n")
        
        if 'analyst_username' in df.columns:
            analysts = df['analyst_username'].dropna().unique()
            f.write(f"Analysts: {', '.join(str(a) for a in analysts)}\n")
        
        if 'processing_version' in df.columns:
            versions = df['processing_version'].dropna().unique()
            f.write(f"Processing Versions: {', '.join(str(v) for v in versions)}\n")
        
        if 'qc_passed' in df.columns:
            qc_pass = df['qc_passed'].sum()
            qc_fail = len(df) - qc_pass
            f.write(f"QC Status: {qc_pass} passed, {qc_fail} failed\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("TEST RECORDS\n")
        f.write("-" * 70 + "\n\n")
        
        for _, row in trace_df.iterrows():
            f.write(f"Test: {row.get('test_id', 'Unknown')}\n")
            for col in available_cols:
                if col != 'test_id' and pd.notna(row.get(col)):
                    f.write(f"  {col}: {row[col]}\n")
            f.write("\n")
    
    return output_path


def export_for_qualification(
    df: pd.DataFrame,
    campaign_info: Dict[str, Any],
    output_dir: Union[str, Path],
    include_raw_hashes: bool = True,
) -> Dict[str, Path]:
    """
    Create qualification-ready export package.
    
    Creates multiple files suitable for flight qualification documentation:
    - Summary CSV with key metrics
    - Full data Excel workbook
    - Traceability report
    - JSON archive
    
    Args:
        df: Campaign DataFrame
        campaign_info: Campaign metadata
        output_dir: Output directory
        include_raw_hashes: Include file hashes for verification
        
    Returns:
        Dictionary mapping file type to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    campaign_name = campaign_info.get('campaign_name', 'campaign')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    outputs = {}
    
    # Summary CSV (key metrics only)
    summary_cols = ['test_id', 'test_timestamp', 'part', 'serial_num']
    
    # Add primary metrics based on campaign type
    metric_cols = [c for c in df.columns if c.startswith('avg_') and not c.startswith('avg_rho')]
    uncertainty_cols = [c for c in df.columns if c.startswith('u_')]
    
    summary_cols.extend([c for c in metric_cols if c in df.columns])
    summary_cols.extend([c for c in ['qc_passed'] if c in df.columns])
    
    available_summary = [c for c in summary_cols if c in df.columns]
    summary_df = df[available_summary].copy()
    
    summary_path = output_dir / f"{campaign_name}_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    outputs['summary_csv'] = summary_path
    
    # Full Excel workbook
    excel_path = output_dir / f"{campaign_name}_full_{timestamp}.xlsx"
    export_campaign_excel(df, excel_path, campaign_info)
    outputs['full_excel'] = excel_path
    
    # Traceability report
    if any(c in df.columns for c in ['raw_data_hash', 'config_hash']):
        trace_path = output_dir / f"{campaign_name}_traceability_{timestamp}.txt"
        create_traceability_report(df, trace_path)
        outputs['traceability'] = trace_path
    
    # JSON archive
    json_path = output_dir / f"{campaign_name}_archive_{timestamp}.json"
    export_campaign_json(df, json_path, campaign_info)
    outputs['json_archive'] = json_path
    
    # Manifest file
    manifest = {
        'campaign_name': campaign_name,
        'export_timestamp': datetime.now().isoformat(),
        'files': {k: str(v.name) for k, v in outputs.items()},
        'total_tests': len(df),
        'campaign_info': campaign_info,
    }
    
    manifest_path = output_dir / f"{campaign_name}_manifest_{timestamp}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    outputs['manifest'] = manifest_path
    
    return outputs
