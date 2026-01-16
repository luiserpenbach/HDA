"""
Data Traceability Module
========================
Provides cryptographic verification and audit trail for test data analysis.

Every analysis result must be traceable back to:
1. The exact raw data file (via SHA-256 hash)
2. The exact configuration used (via config hash + snapshot)
3. The processing parameters (steady window, detection method, etc.)
4. When and by whom the analysis was performed

This is MANDATORY for flight qualification testing.
"""

import hashlib
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
import getpass
import platform
import os


# Module version - increment when calculation methods change
# v2.1.0 - Added plugin architecture (Phase 1), maintained backward compatibility
PROCESSING_VERSION = "2.1.0"


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex string of SHA-256 hash prefixed with 'sha256:'
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return f"sha256:{sha256_hash.hexdigest()}"


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute hash of DataFrame contents.
    
    Useful when data is uploaded directly (not from file).
    Uses pandas' internal hashing for consistency.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        Hex string of SHA-256 hash prefixed with 'sha256:'
    """
    # Convert to bytes in a deterministic way
    # Sort columns to ensure consistency
    df_sorted = df[sorted(df.columns)]
    
    # Use to_json for deterministic serialization
    json_str = df_sorted.to_json(orient='split', date_format='iso')
    
    sha256_hash = hashlib.sha256(json_str.encode('utf-8'))
    return f"sha256:{sha256_hash.hexdigest()}"


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of configuration dictionary.
    
    Sorts keys for deterministic output.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Hex string of SHA-256 hash prefixed with 'sha256:'
    """
    # Sort keys recursively for deterministic output
    config_str = json.dumps(config, sort_keys=True, default=str)
    sha256_hash = hashlib.sha256(config_str.encode('utf-8'))
    return f"sha256:{sha256_hash.hexdigest()}"


def create_config_snapshot(config: Dict[str, Any]) -> str:
    """
    Create a JSON snapshot of the configuration.
    
    This is stored alongside results so the exact config
    can be recovered even if the config file changes.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        JSON string of configuration
    """
    return json.dumps(config, sort_keys=True, indent=2, default=str)


@dataclass
class AnalysisContext:
    """
    Captures the complete context of an analysis run.
    
    This is attached to every saved result to enable full reproducibility.
    """
    # Analyst information
    analyst_username: str = field(default_factory=lambda: getpass.getuser())
    analyst_hostname: str = field(default_factory=lambda: platform.node())
    
    # Timing
    analysis_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    # Software version
    processing_version: str = PROCESSING_VERSION
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class DataTraceability:
    """
    Complete traceability record for a dataset.
    
    Captures everything needed to verify and reproduce the analysis.
    """
    # File identification
    raw_data_path: Optional[str] = None
    raw_data_hash: Optional[str] = None
    raw_data_filename: Optional[str] = None
    
    # Config identification
    config_name: Optional[str] = None
    config_hash: Optional[str] = None
    config_snapshot: Optional[str] = None
    
    # Analysis context
    context: AnalysisContext = field(default_factory=AnalysisContext)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'raw_data_path': self.raw_data_path,
            'raw_data_hash': self.raw_data_hash,
            'raw_data_filename': self.raw_data_filename,
            'config_name': self.config_name,
            'config_hash': self.config_hash,
            'config_snapshot': self.config_snapshot,
        }
        result.update(self.context.to_dict())
        return result
    
    @classmethod
    def from_file(
        cls, 
        file_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None
    ) -> 'DataTraceability':
        """
        Create traceability record from a file.
        
        Args:
            file_path: Path to raw data file
            config: Configuration dictionary used for analysis
            config_name: Name/identifier of the configuration
            
        Returns:
            DataTraceability instance with all fields populated
        """
        file_path = Path(file_path)
        
        trace = cls(
            raw_data_path=str(file_path.absolute()),
            raw_data_hash=compute_file_hash(file_path),
            raw_data_filename=file_path.name,
        )
        
        if config is not None:
            trace.config_name = config_name or config.get('config_name', 'unnamed')
            trace.config_hash = compute_config_hash(config)
            trace.config_snapshot = create_config_snapshot(config)
        
        return trace
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        source_name: str = "uploaded_data",
        config: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None
    ) -> 'DataTraceability':
        """
        Create traceability record from a DataFrame (e.g., uploaded file).
        
        Args:
            df: DataFrame containing raw data
            source_name: Name to identify the data source
            config: Configuration dictionary used for analysis
            config_name: Name/identifier of the configuration
            
        Returns:
            DataTraceability instance with all fields populated
        """
        trace = cls(
            raw_data_path=None,  # No file path for uploads
            raw_data_hash=compute_dataframe_hash(df),
            raw_data_filename=source_name,
        )
        
        if config is not None:
            trace.config_name = config_name or config.get('config_name', 'unnamed')
            trace.config_hash = compute_config_hash(config)
            trace.config_snapshot = create_config_snapshot(config)
        
        return trace


@dataclass
class ProcessingRecord:
    """
    Records the specific processing parameters used in an analysis.
    
    This captures HOW the analysis was done, complementing the
    DataTraceability which captures WHAT data was analyzed.
    """
    # Steady state detection
    steady_window_start_ms: Optional[float] = None
    steady_window_end_ms: Optional[float] = None
    steady_window_duration_ms: Optional[float] = None
    detection_method: Optional[str] = None  # 'CV-based' or 'ML-based'
    detection_parameters: Optional[Dict[str, Any]] = None
    
    # Resampling
    resample_freq_ms: Optional[float] = None
    original_sample_count: Optional[int] = None
    resampled_sample_count: Optional[int] = None
    
    # Channels used
    stability_channels: Optional[list] = None
    primary_pressure_channel: Optional[str] = None
    primary_flow_channel: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert detection_parameters to JSON string for storage
        if result['detection_parameters'] is not None:
            result['detection_parameters'] = json.dumps(result['detection_parameters'])
        if result['stability_channels'] is not None:
            result['stability_channels'] = json.dumps(result['stability_channels'])
        return result
    
    def set_steady_window(
        self, 
        window: Tuple[float, float],
        method: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Set steady window bounds and detection info."""
        self.steady_window_start_ms = window[0]
        self.steady_window_end_ms = window[1]
        self.steady_window_duration_ms = window[1] - window[0]
        self.detection_method = method
        self.detection_parameters = parameters or {}


def verify_data_integrity(
    file_path: Union[str, Path],
    expected_hash: str
) -> Tuple[bool, str]:
    """
    Verify that a file matches its recorded hash.
    
    Args:
        file_path: Path to the file to verify
        expected_hash: Expected hash (with 'sha256:' prefix)
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        current_hash = compute_file_hash(file_path)
        
        if current_hash == expected_hash:
            return True, "Data integrity verified: hash matches"
        else:
            return False, (
                f"Data integrity FAILED: hash mismatch\n"
                f"  Expected: {expected_hash}\n"
                f"  Current:  {current_hash}"
            )
    except FileNotFoundError:
        return False, f"Data integrity FAILED: file not found at {file_path}"
    except Exception as e:
        return False, f"Data integrity check error: {str(e)}"


def create_full_traceability_record(
    df: pd.DataFrame,
    file_path: Optional[Union[str, Path]],
    config: Dict[str, Any],
    config_name: str,
    steady_window: Tuple[float, float],
    detection_method: str,
    detection_params: Dict[str, Any],
    stability_channels: list,
    resample_freq_ms: float
) -> Dict[str, Any]:
    """
    Create a complete traceability record for database storage.
    
    This is the main entry point for creating traceability records.
    Combines DataTraceability and ProcessingRecord into a single dict.
    
    Args:
        df: The raw (or resampled) DataFrame
        file_path: Path to source file (None if uploaded)
        config: Configuration dictionary
        config_name: Name of the configuration
        steady_window: (start_ms, end_ms) of steady window
        detection_method: 'CV-based' or 'ML-based'
        detection_params: Parameters used for detection
        stability_channels: List of channels used for stability detection
        resample_freq_ms: Resampling frequency used
        
    Returns:
        Dictionary containing all traceability fields
    """
    # Create data traceability
    if file_path is not None:
        data_trace = DataTraceability.from_file(file_path, config, config_name)
    else:
        data_trace = DataTraceability.from_dataframe(
            df, 
            source_name="uploaded_data",
            config=config,
            config_name=config_name
        )
    
    # Create processing record
    proc_record = ProcessingRecord(
        resample_freq_ms=resample_freq_ms,
        original_sample_count=len(df),
        stability_channels=stability_channels
    )
    proc_record.set_steady_window(steady_window, detection_method, detection_params)
    
    # Combine into single record
    record = data_trace.to_dict()
    record.update(proc_record.to_dict())
    
    return record


# Additional database schema fields for traceability
TRACEABILITY_SCHEMA_FIELDS = """
    -- Data Traceability
    raw_data_path TEXT,
    raw_data_hash TEXT,
    raw_data_filename TEXT,
    config_name TEXT,
    config_hash TEXT,
    config_snapshot TEXT,
    
    -- Analysis Context
    analyst_username TEXT,
    analyst_hostname TEXT,
    analysis_timestamp_utc TEXT,
    processing_version TEXT,
    
    -- Processing Record
    steady_window_start_ms REAL,
    steady_window_end_ms REAL,
    steady_window_duration_ms REAL,
    detection_method TEXT,
    detection_parameters TEXT,
    resample_freq_ms REAL,
    original_sample_count INTEGER,
    resampled_sample_count INTEGER,
    stability_channels TEXT,
    primary_pressure_channel TEXT,
    primary_flow_channel TEXT
"""
