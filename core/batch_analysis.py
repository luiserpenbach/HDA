"""
Batch Analysis Module
=====================
Process multiple test files efficiently with consistent settings.

Features:
- Parallel processing of multiple files
- Consistent configuration across batch
- Progress tracking and error handling
- Aggregate statistics and reporting
- Export to campaign database
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


@dataclass
class BatchTestResult:
    """Result from processing a single test in batch mode."""
    file_path: str
    test_id: str
    success: bool
    
    # On success
    measurements: Optional[Dict[str, Any]] = None
    traceability: Optional[Dict[str, Any]] = None
    qc_passed: Optional[bool] = None
    steady_window: Optional[Tuple[float, float]] = None
    
    # On failure
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Timing
    processing_time_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'file_path': self.file_path,
            'test_id': self.test_id,
            'success': self.success,
            'qc_passed': self.qc_passed,
            'processing_time_s': self.processing_time_s,
            'error_message': self.error_message,
        }
        
        if self.steady_window:
            result['steady_window_start_ms'] = self.steady_window[0]
            result['steady_window_end_ms'] = self.steady_window[1]
        
        if self.measurements:
            for key, meas in self.measurements.items():
                if hasattr(meas, 'value'):
                    result[f'{key}_value'] = meas.value
                    result[f'{key}_uncertainty'] = meas.uncertainty
                elif isinstance(meas, (int, float)):
                    result[key] = meas
        
        return result


@dataclass
class BatchAnalysisReport:
    """Summary report for a batch analysis run."""
    batch_id: str
    config_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Results
    results: List[BatchTestResult] = field(default_factory=list)
    
    # Summary stats
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    qc_failed: int = 0
    
    def __post_init__(self):
        self.update_summary()
    
    def update_summary(self):
        """Update summary statistics from results."""
        self.total_files = len(self.results)
        self.successful = sum(1 for r in self.results if r.success)
        self.failed = sum(1 for r in self.results if not r.success)
        self.qc_failed = sum(1 for r in self.results if r.success and not r.qc_passed)
    
    @property
    def success_rate(self) -> float:
        """Percentage of successfully processed files."""
        return (self.successful / self.total_files * 100) if self.total_files > 0 else 0
    
    @property
    def total_time_s(self) -> float:
        """Total processing time in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.processing_time_s for r in self.results)
    
    def get_failed_tests(self) -> List[BatchTestResult]:
        """Get list of failed test results."""
        return [r for r in self.results if not r.success]
    
    def get_qc_failed_tests(self) -> List[BatchTestResult]:
        """Get tests that processed but failed QC."""
        return [r for r in self.results if r.success and not r.qc_passed]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"Batch Analysis Report: {self.batch_id}",
            f"Configuration: {self.config_name}",
            f"",
            f"Results:",
            f"  Total files: {self.total_files}",
            f"  Successful: {self.successful} ({self.success_rate:.1f}%)",
            f"  Failed: {self.failed}",
            f"  QC Failed: {self.qc_failed}",
            f"",
            f"Processing time: {self.total_time_s:.1f}s",
        ]
        
        if self.failed > 0:
            lines.append(f"\nFailed tests:")
            for r in self.get_failed_tests()[:5]:
                lines.append(f"  - {r.test_id}: {r.error_message}")
            if self.failed > 5:
                lines.append(f"  ... and {self.failed - 5} more")
        
        return "\n".join(lines)


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def discover_test_files(
    directory: Union[str, Path],
    pattern: str = "*.csv",
    recursive: bool = False,
) -> List[Path]:
    """
    Discover test files in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for files (default: *.csv)
        recursive: Search subdirectories
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Sort by name for consistent ordering
    files.sort(key=lambda x: x.name)
    
    return files


def extract_test_id_from_path(
    file_path: Path,
    pattern: Optional[str] = None,
) -> str:
    """
    Extract test ID from file path.
    
    Args:
        file_path: Path to file
        pattern: Regex pattern with named group 'test_id' (optional)
        
    Returns:
        Extracted test ID or filename stem
    """
    if pattern:
        import re
        match = re.search(pattern, str(file_path))
        if match and 'test_id' in match.groupdict():
            return match.group('test_id')
    
    # Default: use filename without extension
    return file_path.stem


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_single_file(
    file_path: Path,
    config: Dict[str, Any],
    load_func: Callable[[Path], pd.DataFrame],
    analyze_func: Callable,
    test_id_pattern: Optional[str] = None,
) -> BatchTestResult:
    """
    Process a single test file.
    
    Args:
        file_path: Path to test file
        config: Test configuration
        load_func: Function to load data from file
        analyze_func: Function to analyze loaded data
        test_id_pattern: Pattern to extract test ID
        
    Returns:
        BatchTestResult with processing outcome
    """
    import time
    start_time = time.time()
    
    test_id = extract_test_id_from_path(file_path, test_id_pattern)
    
    try:
        # Load data
        df = load_func(file_path)
        
        # Analyze
        result = analyze_func(df, config, test_id, str(file_path))
        
        processing_time = time.time() - start_time
        
        return BatchTestResult(
            file_path=str(file_path),
            test_id=test_id,
            success=True,
            measurements=result.measurements if hasattr(result, 'measurements') else None,
            traceability=result.traceability if hasattr(result, 'traceability') else None,
            qc_passed=result.passed_qc if hasattr(result, 'passed_qc') else True,
            steady_window=result.steady_window if hasattr(result, 'steady_window') else None,
            processing_time_s=processing_time,
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        return BatchTestResult(
            file_path=str(file_path),
            test_id=test_id,
            success=False,
            error_message=str(e),
            error_traceback=traceback.format_exc(),
            processing_time_s=processing_time,
        )


def run_batch_analysis(
    files: List[Path],
    config: Dict[str, Any],
    load_func: Callable[[Path], pd.DataFrame],
    analyze_func: Callable,
    batch_id: Optional[str] = None,
    test_id_pattern: Optional[str] = None,
    max_workers: int = 1,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BatchAnalysisReport:
    """
    Run batch analysis on multiple files.
    
    Args:
        files: List of file paths to process
        config: Test configuration
        load_func: Function to load data from file
        analyze_func: Function to analyze data (df, config, test_id, file_path) -> result
        batch_id: Unique identifier for this batch
        test_id_pattern: Pattern to extract test IDs
        max_workers: Number of parallel workers (1 = sequential)
        progress_callback: Called with (completed, total, current_file)
        
    Returns:
        BatchAnalysisReport with all results
    """
    if batch_id is None:
        batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    
    config_name = config.get('config_name', 'unnamed')
    
    report = BatchAnalysisReport(
        batch_id=batch_id,
        config_name=config_name,
        start_time=datetime.now(),
    )
    
    if max_workers <= 1:
        # Sequential processing
        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i, len(files), str(file_path))
            
            result = process_single_file(
                file_path, config, load_func, analyze_func, test_id_pattern
            )
            report.results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    f, config, load_func, analyze_func, test_id_pattern
                ): f for f in files
            }
            
            completed = 0
            for future in as_completed(futures):
                file_path = futures[future]
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(files), str(file_path))
                
                try:
                    result = future.result()
                    report.results.append(result)
                except Exception as e:
                    report.results.append(BatchTestResult(
                        file_path=str(file_path),
                        test_id=file_path.stem,
                        success=False,
                        error_message=str(e),
                    ))
    
    report.end_time = datetime.now()
    report.update_summary()
    
    return report


# =============================================================================
# DEFAULT LOADING FUNCTIONS
# =============================================================================

def load_csv_with_timestamp(file_path: Path) -> pd.DataFrame:
    """
    Load CSV file and ensure timestamp column exists.
    
    Handles common timestamp formats and column names.
    """
    df = pd.read_csv(file_path)
    
    # Try to find timestamp column
    timestamp_candidates = ['timestamp', 'time', 'Time', 'TIMESTAMP', 't', 'time_ms']
    
    for col in timestamp_candidates:
        if col in df.columns:
            if col != 'timestamp':
                df = df.rename(columns={col: 'timestamp'})
            break
    else:
        # No timestamp found - create from index
        df['timestamp'] = df.index * 10  # Assume 100 Hz default
    
    return df


def load_tdms_file(file_path: Path) -> pd.DataFrame:
    """
    Load TDMS file (requires nptdms library).
    """
    try:
        from nptdms import TdmsFile
    except ImportError:
        raise ImportError("nptdms library required for TDMS files: pip install nptdms")
    
    tdms_file = TdmsFile.read(file_path)
    
    # Flatten all channels into DataFrame
    data = {}
    for group in tdms_file.groups():
        for channel in group.channels():
            data[channel.name] = channel[:]
    
    return pd.DataFrame(data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def batch_cold_flow_analysis(
    files: List[Path],
    config: Dict[str, Any],
    steady_detector: Callable[[pd.DataFrame, Dict], Tuple[float, float]],
    max_workers: int = 1,
    progress_callback: Optional[Callable] = None,
) -> BatchAnalysisReport:
    """
    Convenience function for batch cold flow analysis.
    
    Args:
        files: List of test files
        config: Cold flow configuration
        steady_detector: Function to detect steady window (df, config) -> (start, end)
        max_workers: Number of parallel workers
        progress_callback: Progress callback
        
    Returns:
        BatchAnalysisReport
    """
    from .integrated_analysis import analyze_cold_flow_test
    
    def analyze_func(df, cfg, test_id, file_path):
        # Detect steady window
        window = steady_detector(df, cfg)
        
        # Run analysis
        return analyze_cold_flow_test(
            df=df,
            config=cfg,
            steady_window=window,
            test_id=test_id,
            file_path=file_path,
            skip_qc=False,
        )
    
    return run_batch_analysis(
        files=files,
        config=config,
        load_func=load_csv_with_timestamp,
        analyze_func=analyze_func,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )


def batch_hot_fire_analysis(
    files: List[Path],
    config: Dict[str, Any],
    steady_detector: Callable[[pd.DataFrame, Dict], Tuple[float, float]],
    max_workers: int = 1,
    progress_callback: Optional[Callable] = None,
) -> BatchAnalysisReport:
    """
    Convenience function for batch hot fire analysis.
    
    Args:
        files: List of test files
        config: Hot fire configuration
        steady_detector: Function to detect steady window
        max_workers: Number of parallel workers
        progress_callback: Progress callback
        
    Returns:
        BatchAnalysisReport
    """
    from .integrated_analysis import analyze_hot_fire_test
    
    def analyze_func(df, cfg, test_id, file_path):
        window = steady_detector(df, cfg)
        
        return analyze_hot_fire_test(
            df=df,
            config=cfg,
            steady_window=window,
            test_id=test_id,
            file_path=file_path,
            skip_qc=False,
        )
    
    return run_batch_analysis(
        files=files,
        config=config,
        load_func=load_csv_with_timestamp,
        analyze_func=analyze_func,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )


def save_batch_to_campaign(
    report: BatchAnalysisReport,
    campaign_name: str,
    campaign_type: str = 'cold_flow',
) -> Tuple[int, int]:
    """
    Save successful batch results to a campaign database.
    
    Args:
        report: BatchAnalysisReport with results
        campaign_name: Target campaign
        campaign_type: 'cold_flow' or 'hot_fire'
        
    Returns:
        Tuple of (saved_count, skipped_count)
    """
    from .campaign_manager_v2 import save_to_campaign
    
    saved = 0
    skipped = 0
    
    for result in report.results:
        if not result.success:
            skipped += 1
            continue
        
        try:
            # Build database record
            record = {
                'test_id': result.test_id,
                'test_path': result.file_path,
                'qc_passed': 1 if result.qc_passed else 0,
            }
            
            if result.traceability:
                record.update(result.traceability)
            
            if result.measurements:
                if campaign_type == 'cold_flow':
                    for key, meas in result.measurements.items():
                        if hasattr(meas, 'value'):
                            if key == 'Cd':
                                record['avg_cd_CALC'] = meas.value
                                record['u_cd_CALC'] = meas.uncertainty
                            elif key == 'pressure_upstream':
                                record['avg_p_up_bar'] = meas.value
                                record['u_p_up_bar'] = meas.uncertainty
                            elif key == 'mass_flow':
                                record['avg_mf_g_s'] = meas.value
                                record['u_mf_g_s'] = meas.uncertainty
                else:
                    for key, meas in result.measurements.items():
                        if hasattr(meas, 'value'):
                            if key == 'Isp':
                                record['avg_isp_s'] = meas.value
                                record['u_isp_s'] = meas.uncertainty
                            elif key == 'c_star':
                                record['avg_c_star_m_s'] = meas.value
                                record['u_c_star_m_s'] = meas.uncertainty
            
            save_to_campaign(campaign_name, record)
            saved += 1
            
        except Exception as e:
            print(f"Warning: Failed to save {result.test_id}: {e}")
            skipped += 1
    
    return saved, skipped


def export_batch_results(
    report: BatchAnalysisReport,
    output_path: Union[str, Path],
    format: str = 'csv',
) -> Path:
    """
    Export batch results to file.
    
    Args:
        report: BatchAnalysisReport
        output_path: Output file path
        format: 'csv', 'json', or 'excel'
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    if format == 'csv':
        df = report.to_dataframe()
        df.to_csv(output_path, index=False)
    
    elif format == 'json':
        data = {
            'batch_id': report.batch_id,
            'config_name': report.config_name,
            'start_time': report.start_time.isoformat(),
            'end_time': report.end_time.isoformat() if report.end_time else None,
            'summary': {
                'total': report.total_files,
                'successful': report.successful,
                'failed': report.failed,
                'qc_failed': report.qc_failed,
            },
            'results': [r.to_dict() for r in report.results],
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    elif format == 'excel':
        df = report.to_dataframe()
        df.to_excel(output_path, index=False)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return output_path
