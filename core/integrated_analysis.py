"""
Integrated Analysis Module
==========================
Combines all P0 components into a cohesive analysis workflow.

This module provides high-level functions that:
1. Validate data quality (QC checks)
2. Process with full traceability
3. Calculate metrics with uncertainties
4. Save results with complete audit trail

Usage:
    from core.integrated_analysis import (
        analyze_cold_flow_test,
        analyze_hot_fire_test,
        prepare_analysis_record,
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
from datetime import datetime
import json

from .traceability import (
    DataTraceability,
    ProcessingRecord,
    create_full_traceability_record,
    compute_file_hash,
    compute_dataframe_hash,
)

from .uncertainty import (
    calculate_cold_flow_uncertainties,
    calculate_hot_fire_uncertainties,
    MeasurementWithUncertainty,
    parse_uncertainty_config,
)

from .qc_checks import (
    run_qc_checks,
    run_quick_qc,
    QCReport,
    assert_qc_passed,
    format_qc_for_display,
)

from .config_validation import (
    validate_config,
    validate_config_simple,
)

# Plugin system (Phase 1 - Plugin Architecture)
from .plugins import PluginRegistry, AnalysisPlugin


class AnalysisResult:
    """
    Complete analysis result with all P0 components.
    
    Contains:
    - QC report
    - Measurements with uncertainties
    - Full traceability record
    - Ready-to-save database record
    """
    
    def __init__(
        self,
        test_id: str,
        qc_report: QCReport,
        measurements: Dict[str, MeasurementWithUncertainty],
        raw_values: Dict[str, float],
        traceability: Dict[str, Any],
        config: Dict[str, Any],
        steady_window: Tuple[float, float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.test_id = test_id
        self.qc_report = qc_report
        self.measurements = measurements
        self.raw_values = raw_values
        self.traceability = traceability
        self.config = config
        self.steady_window = steady_window
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    @property
    def passed_qc(self) -> bool:
        """Whether QC checks passed."""
        return self.qc_report.passed
    
    @property
    def has_warnings(self) -> bool:
        """Whether there are QC warnings."""
        return self.qc_report.has_warnings
    
    def get_measurement(self, name: str) -> Optional[MeasurementWithUncertainty]:
        """Get a specific measurement with uncertainty."""
        return self.measurements.get(name)
    
    def get_value(self, name: str) -> Optional[float]:
        """Get just the value of a measurement."""
        meas = self.measurements.get(name)
        return meas.value if meas else None
    
    def get_uncertainty(self, name: str) -> Optional[float]:
        """Get just the uncertainty of a measurement."""
        meas = self.measurements.get(name)
        return meas.uncertainty if meas else None
    
    def to_database_record(self, campaign_type: str = 'cold_flow') -> Dict[str, Any]:
        """
        Convert to a complete database record.
        
        Args:
            campaign_type: 'cold_flow' or 'hot_fire'
            
        Returns:
            Dictionary ready for save_to_campaign()
        """
        record = {
            'test_id': self.test_id,
            'test_timestamp': self.timestamp,
            'qc_passed': 1 if self.passed_qc else 0,
            'qc_summary': json.dumps(self.qc_report.summary),
        }
        
        # Add metadata
        record.update(self.metadata)
        
        # Add traceability
        record.update(self.traceability)
        
        # Add measurements with uncertainties
        if campaign_type == 'cold_flow':
            record.update(self._cold_flow_record())
        else:
            record.update(self._hot_fire_record())
        
        return record
    
    def _cold_flow_record(self) -> Dict[str, Any]:
        """Extract cold flow specific fields."""
        record = {}
        
        # Pressure
        if 'pressure_upstream' in self.measurements:
            m = self.measurements['pressure_upstream']
            record['avg_p_up_bar'] = m.value
            record['u_p_up_bar'] = m.uncertainty
        
        # Mass flow
        if 'mass_flow' in self.measurements:
            m = self.measurements['mass_flow']
            record['avg_mf_g_s'] = m.value
            record['u_mf_g_s'] = m.uncertainty
        
        # Delta P
        if 'delta_p' in self.measurements:
            m = self.measurements['delta_p']
            record['dp_bar'] = m.value
            record['u_dp_bar'] = m.uncertainty
        
        # Cd
        if 'Cd' in self.measurements:
            m = self.measurements['Cd']
            record['avg_cd_CALC'] = m.value
            record['u_cd_CALC'] = m.uncertainty
            record['cd_rel_uncertainty_pct'] = m.relative_uncertainty_percent
        
        return record
    
    def _hot_fire_record(self) -> Dict[str, Any]:
        """Extract hot fire specific fields."""
        record = {}
        
        # Chamber pressure
        if 'chamber_pressure' in self.measurements:
            m = self.measurements['chamber_pressure']
            record['avg_pc_bar'] = m.value
            record['u_pc_bar'] = m.uncertainty
        
        # Thrust
        if 'thrust' in self.measurements:
            m = self.measurements['thrust']
            record['avg_thrust_n'] = m.value
            record['u_thrust_n'] = m.uncertainty
        
        # Mass flow
        if 'mass_flow_total' in self.measurements:
            m = self.measurements['mass_flow_total']
            record['avg_mf_total_g_s'] = m.value
            record['u_mf_total_g_s'] = m.uncertainty
        
        # O/F ratio
        if 'of_ratio' in self.measurements:
            m = self.measurements['of_ratio']
            record['avg_of_ratio'] = m.value
            record['u_of_ratio'] = m.uncertainty
        
        # Isp
        if 'Isp' in self.measurements:
            m = self.measurements['Isp']
            record['avg_isp_s'] = m.value
            record['u_isp_s'] = m.uncertainty
        
        # C*
        if 'c_star' in self.measurements:
            m = self.measurements['c_star']
            record['avg_c_star_m_s'] = m.value
            record['u_c_star_m_s'] = m.uncertainty
        
        return record
    
    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            f"Analysis Result: {self.test_id}",
            f"QC Status: {'PASSED' if self.passed_qc else 'FAILED'}",
            "",
            "Measurements:",
        ]
        
        for name, meas in self.measurements.items():
            lines.append(f"  {name}: {meas}")
        
        return "\n".join(lines)


# =============================================================================
# PLUGIN-BASED ANALYSIS (Phase 1 - Plugin Architecture)
# =============================================================================

def analyze_test(
    df: pd.DataFrame,
    config: Dict[str, Any],
    steady_window: Tuple[float, float],
    test_id: str,
    plugin_slug: str,
    file_path: Optional[Union[str, Path]] = None,
    detection_method: str = 'CV-based',
    detection_params: Optional[Dict[str, Any]] = None,
    stability_channels: Optional[List[str]] = None,
    resample_freq_ms: float = 10.0,
    metadata: Optional[Dict[str, Any]] = None,
    skip_qc: bool = False,
) -> AnalysisResult:
    """
    Generic test analysis function using plugin system.

    This is the new plugin-based entry point that routes analysis
    to the appropriate plugin based on plugin_slug.

    Core responsibilities (enforced by wrapper):
    - QC validation
    - Traceability creation
    - Result packaging

    Plugin responsibilities (delegated):
    - Config validation
    - Test-specific QC checks
    - Steady-state extraction
    - Metric calculation
    - Uncertainty propagation

    Args:
        df: DataFrame with test data (already resampled)
        config: Test configuration dictionary
        steady_window: (start_ms, end_ms) of steady state window
        test_id: Unique identifier for this test
        plugin_slug: Plugin identifier (e.g., 'cold_flow', 'hot_fire')
        file_path: Path to raw data file (for traceability)
        detection_method: How steady window was detected
        detection_params: Parameters used for detection
        stability_channels: Channels used for stability detection
        resample_freq_ms: Resampling frequency used
        metadata: Additional metadata (part, serial, operator, etc.)
        skip_qc: If True, skip QC checks (not recommended)

    Returns:
        AnalysisResult with all P0 components

    Raises:
        KeyError: If plugin not found
        ValueError: If QC fails and skip_qc is False

    Example:
        >>> result = analyze_test(
        ...     df=df,
        ...     config=config,
        ...     steady_window=(1500, 5000),
        ...     test_id="INJ-CF-001",
        ...     plugin_slug="cold_flow",
        ...     file_path="test_data.csv"
        ... )
    """
    # Get plugin from registry
    plugin = PluginRegistry.get_plugin(plugin_slug)

    # Merge config and metadata (geometry/fluid come from metadata)
    from .config_validation import merge_config_and_metadata
    if metadata:
        merged_config = merge_config_and_metadata(config, metadata)
    else:
        merged_config = config

    # Step 1: Plugin validates merged config (test-specific)
    plugin.validate_config(merged_config)

    # Step 2: Run QC checks
    if not skip_qc:
        # Plugin runs test-specific QC
        qc_report = plugin.run_qc_checks(df, merged_config, quick=False)

        # Check if QC passed
        if not qc_report.passed:
            raise ValueError(
                f"QC checks failed for {test_id}. "
                f"Blocking failures: {[c.name for c in qc_report.blocking_failures]}"
            )
    else:
        # Quick QC (lightweight)
        qc_report = plugin.run_qc_checks(df, merged_config, quick=True)

    # Step 3: Plugin extracts steady state
    steady_df = plugin.extract_steady_state(df, steady_window, merged_config)

    # Step 4: Plugin computes raw metrics
    avg_values = plugin.compute_raw_metrics(steady_df, merged_config, metadata)

    # Step 5: Plugin calculates measurements WITH uncertainties
    # (Phase 1: Delegates to existing uncertainty functions)
    # (Future phases: Core enforces uncertainty wrapper)
    if hasattr(plugin, 'calculate_measurements_with_uncertainties'):
        measurements = plugin.calculate_measurements_with_uncertainties(
            avg_values, merged_config, metadata
        )
    else:
        # Fallback: Just wrap raw values (no uncertainties)
        measurements = {
            name: MeasurementWithUncertainty(value=value, uncertainty=0.0, name=name)
            for name, value in avg_values.items()
        }

    # Step 6: Core creates traceability (P0 - enforced by wrapper)
    traceability = create_full_traceability_record(
        df=df,
        file_path=file_path,
        config=merged_config,
        config_name=merged_config.get('config_name', 'unnamed'),
        steady_window=steady_window,
        detection_method=detection_method,
        detection_params=detection_params or {},
        stability_channels=stability_channels or [],
        resample_freq_ms=resample_freq_ms,
    )

    # Step 7: Core packages result
    return AnalysisResult(
        test_id=test_id,
        qc_report=qc_report,
        measurements=measurements,
        raw_values=avg_values,
        traceability=traceability,
        config=merged_config,
        steady_window=steady_window,
        metadata=metadata,
    )


# =============================================================================
# BACKWARD-COMPATIBLE WRAPPERS
# =============================================================================

def analyze_cold_flow_test(
    df: pd.DataFrame,
    config: Dict[str, Any],
    steady_window: Tuple[float, float],
    test_id: str,
    file_path: Optional[Union[str, Path]] = None,
    detection_method: str = 'CV-based',
    detection_params: Optional[Dict[str, Any]] = None,
    stability_channels: Optional[List[str]] = None,
    resample_freq_ms: float = 10.0,
    metadata: Optional[Dict[str, Any]] = None,
    skip_qc: bool = False,
) -> AnalysisResult:
    """
    Complete cold flow test analysis with all P0 components.

    **Backward-Compatible Wrapper** (Phase 1 - Plugin Architecture)

    This function now routes through the plugin system while maintaining
    100% backward compatibility with existing code.

    New code should use: analyze_test(..., plugin_slug='cold_flow')

    Args:
        df: DataFrame with test data (already resampled)
        config: Test configuration dictionary
        steady_window: (start_ms, end_ms) of steady state window
        test_id: Unique identifier for this test
        file_path: Path to raw data file (for traceability)
        detection_method: How steady window was detected
        detection_params: Parameters used for detection
        stability_channels: Channels used for stability detection
        resample_freq_ms: Resampling frequency used
        metadata: Additional metadata (part, serial, operator, etc.)
        skip_qc: If True, skip QC checks (not recommended)

    Returns:
        AnalysisResult with all components

    Raises:
        ValueError: If QC fails and skip_qc is False
    """
    # Route through plugin system
    return analyze_test(
        df=df,
        config=config,
        steady_window=steady_window,
        test_id=test_id,
        plugin_slug='cold_flow',
        file_path=file_path,
        detection_method=detection_method,
        detection_params=detection_params,
        stability_channels=stability_channels,
        resample_freq_ms=resample_freq_ms,
        metadata=metadata,
        skip_qc=skip_qc,
    )


def analyze_hot_fire_test(
    df: pd.DataFrame,
    config: Dict[str, Any],
    steady_window: Tuple[float, float],
    test_id: str,
    file_path: Optional[Union[str, Path]] = None,
    detection_method: str = 'CV-based',
    detection_params: Optional[Dict[str, Any]] = None,
    stability_channels: Optional[List[str]] = None,
    resample_freq_ms: float = 10.0,
    metadata: Optional[Dict[str, Any]] = None,
    skip_qc: bool = False,
) -> AnalysisResult:
    """
    Complete hot fire test analysis with all P0 components.

    **Backward-Compatible Wrapper** (Phase 1 - Plugin Architecture)

    This function now routes through the plugin system while maintaining
    100% backward compatibility with existing code.

    New code should use: analyze_test(..., plugin_slug='hot_fire')

    Args:
        df: DataFrame with test data (already resampled)
        config: Test configuration dictionary
        steady_window: (start_ms, end_ms) of steady state window
        test_id: Unique identifier for this test
        file_path: Path to raw data file (for traceability)
        detection_method: How steady window was detected
        detection_params: Parameters used for detection
        stability_channels: Channels used for stability detection
        resample_freq_ms: Resampling frequency used
        metadata: Additional metadata (part, serial, operator, etc.)
        skip_qc: If True, skip QC checks (not recommended)

    Returns:
        AnalysisResult with all components

    Raises:
        ValueError: If QC fails and skip_qc is False
    """
    # Route through plugin system
    return analyze_test(
        df=df,
        config=config,
        steady_window=steady_window,
        test_id=test_id,
        plugin_slug='hot_fire',
        file_path=file_path,
        detection_method=detection_method,
        detection_params=detection_params,
        stability_channels=stability_channels,
        resample_freq_ms=resample_freq_ms,
        metadata=metadata,
        skip_qc=skip_qc,
    )


def quick_analyze(
    df: pd.DataFrame,
    config: Dict[str, Any],
    test_type: str = 'cold_flow',
) -> Dict[str, MeasurementWithUncertainty]:
    """
    Quick analysis without full traceability.
    
    Use for interactive exploration. For saved results, use
    analyze_cold_flow_test() or analyze_hot_fire_test().
    
    Args:
        df: DataFrame with steady-state data only
        config: Test configuration
        test_type: 'cold_flow' or 'hot_fire'
        
    Returns:
        Dictionary of measurements with uncertainties
    """
    avg_values = df.mean().to_dict()
    
    if test_type == 'cold_flow':
        return calculate_cold_flow_uncertainties(avg_values, config)
    else:
        return calculate_hot_fire_uncertainties(avg_values, config)


def format_measurement_table(
    measurements: Dict[str, MeasurementWithUncertainty]
) -> str:
    """
    Format measurements as a markdown table.
    
    Args:
        measurements: Dictionary of measurements
        
    Returns:
        Markdown table string
    """
    lines = [
        "| Parameter | Value | Uncertainty | Rel. Uncertainty |",
        "|-----------|-------|-------------|------------------|",
    ]
    
    for name, meas in measurements.items():
        rel_pct = meas.relative_uncertainty_percent
        lines.append(
            f"| {name} | {meas.value:.4g} {meas.unit} | "
            f"±{meas.uncertainty:.4g} | {rel_pct:.1f}% |"
        )
    
    return "\n".join(lines)
