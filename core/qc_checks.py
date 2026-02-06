"""
Pre-Analysis Quality Control Module
====================================
Automated data quality checks that MUST pass before analysis proceeds.

Key Principle: The system will NOT analyze garbage data silently.
If QC fails, analysis is blocked with clear error messages.

QC Check Categories:
1. Timestamp integrity (monotonic, no gaps, consistent rate)
2. Sensor range validation (within physical limits)
3. Signal quality (no flatlines, dropouts, or saturation)
4. Cross-correlation sanity (related sensors make sense together)
5. Data completeness (required channels present, no excess NaN)

Each check returns PASS, WARN, or FAIL with detailed diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class QCStatus(Enum):
    """Quality control check status."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"  # Check not applicable


@dataclass
class QCCheckResult:
    """
    Result of a single QC check.
    
    Attributes:
        name: Check identifier
        status: PASS, WARN, FAIL, or SKIP
        message: Human-readable result description
        details: Additional diagnostic information
        blocking: If True, FAIL status blocks analysis
    """
    name: str
    status: QCStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    blocking: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'blocking': self.blocking
        }
    
    def __str__(self) -> str:
        icon = {'PASS': '', 'WARN': '[WARN]', 'FAIL': '', 'SKIP': ''}[self.status.value]
        return f"[{icon}] {self.name}: {self.message}"


@dataclass
class QCReport:
    """
    Complete QC report for a dataset.
    
    Aggregates all individual check results and determines
    overall pass/fail status.
    """
    checks: List[QCCheckResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    
    @property
    def passed(self) -> bool:
        """True if no blocking checks failed."""
        return not any(
            c.status == QCStatus.FAIL and c.blocking 
            for c in self.checks
        )
    
    @property
    def has_warnings(self) -> bool:
        """True if any checks returned warnings."""
        return any(c.status == QCStatus.WARN for c in self.checks)
    
    @property
    def blocking_failures(self) -> List[QCCheckResult]:
        """List of checks that failed and block analysis."""
        return [c for c in self.checks if c.status == QCStatus.FAIL and c.blocking]
    
    @property
    def warnings(self) -> List[QCCheckResult]:
        """List of checks that returned warnings."""
        return [c for c in self.checks if c.status == QCStatus.WARN]
    
    @property
    def summary(self) -> Dict[str, int]:
        """Count of checks by status."""
        return {
            'total': len(self.checks),
            'passed': sum(1 for c in self.checks if c.status == QCStatus.PASS),
            'warnings': sum(1 for c in self.checks if c.status == QCStatus.WARN),
            'failed': sum(1 for c in self.checks if c.status == QCStatus.FAIL),
            'skipped': sum(1 for c in self.checks if c.status == QCStatus.SKIP),
        }
    
    def add_check(self, check: QCCheckResult):
        """Add a check result to the report."""
        self.checks.append(check)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'has_warnings': self.has_warnings,
            'summary': self.summary,
            'checks': [c.to_dict() for c in self.checks],
            'timestamp': self.timestamp,
            'blocking_failures': [c.to_dict() for c in self.blocking_failures],
        }
    
    def __str__(self) -> str:
        lines = [
            f"QC Report - {'PASSED' if self.passed else 'FAILED'}",
            f"  Checks: {self.summary['passed']} passed, {self.summary['warnings']} warnings, {self.summary['failed']} failed",
            ""
        ]
        for check in self.checks:
            lines.append(f"  {check}")
        return "\n".join(lines)


# =============================================================================
# TIMESTAMP CHECKS
# =============================================================================

def check_timestamp_monotonic(
    df: pd.DataFrame,
    time_col: str = 'timestamp'
) -> QCCheckResult:
    """
    Verify timestamps are strictly monotonically increasing.
    
    Time going backwards indicates:
    - DAQ clock reset
    - Data corruption
    - Improper file concatenation
    """
    if time_col not in df.columns:
        return QCCheckResult(
            name="timestamp_monotonic",
            status=QCStatus.SKIP,
            message=f"Time column '{time_col}' not found",
            blocking=True
        )
    
    timestamps = df[time_col].values
    diffs = np.diff(timestamps)
    
    # Check for any negative or zero differences
    non_positive = diffs <= 0
    n_violations = np.sum(non_positive)
    
    if n_violations == 0:
        return QCCheckResult(
            name="timestamp_monotonic",
            status=QCStatus.PASS,
            message="Timestamps are strictly monotonically increasing",
            details={'n_samples': len(df)}
        )
    
    # Find violation locations
    violation_indices = np.where(non_positive)[0]
    first_violations = violation_indices[:5].tolist()  # First 5
    
    return QCCheckResult(
        name="timestamp_monotonic",
        status=QCStatus.FAIL,
        message=f"Timestamps not monotonic: {n_violations} violations found",
        details={
            'n_violations': int(n_violations),
            'first_violation_indices': first_violations,
            'violation_values': [float(diffs[i]) for i in first_violations]
        },
        blocking=True
    )


def check_timestamp_gaps(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    max_gap_factor: float = 3.0
) -> QCCheckResult:
    """
    Check for excessive gaps in timestamps.
    
    A gap larger than max_gap_factor times the median interval
    indicates dropped samples.
    
    Args:
        df: DataFrame with time column
        time_col: Name of timestamp column
        max_gap_factor: Maximum allowed gap as multiple of median interval
    """
    if time_col not in df.columns:
        return QCCheckResult(
            name="timestamp_gaps",
            status=QCStatus.SKIP,
            message=f"Time column '{time_col}' not found",
            blocking=False
        )
    
    timestamps = df[time_col].values
    diffs = np.diff(timestamps)
    
    if len(diffs) == 0:
        return QCCheckResult(
            name="timestamp_gaps",
            status=QCStatus.SKIP,
            message="Insufficient data for gap analysis",
            blocking=False
        )
    
    median_interval = np.median(diffs)
    max_allowed = median_interval * max_gap_factor
    
    large_gaps = diffs > max_allowed
    n_gaps = np.sum(large_gaps)
    
    if n_gaps == 0:
        return QCCheckResult(
            name="timestamp_gaps",
            status=QCStatus.PASS,
            message=f"No timestamp gaps > {max_gap_factor}x median interval",
            details={
                'median_interval_ms': float(median_interval),
                'max_gap_ms': float(np.max(diffs)),
                'threshold_ms': float(max_allowed)
            }
        )
    
    gap_indices = np.where(large_gaps)[0]
    gap_sizes = diffs[large_gaps]
    
    # WARN for < 1% of data, FAIL otherwise
    gap_percentage = n_gaps / len(diffs) * 100
    status = QCStatus.WARN if gap_percentage < 1.0 else QCStatus.FAIL
    
    return QCCheckResult(
        name="timestamp_gaps",
        status=status,
        message=f"Found {n_gaps} timestamp gaps ({gap_percentage:.2f}% of intervals)",
        details={
            'n_gaps': int(n_gaps),
            'gap_percentage': gap_percentage,
            'median_interval_ms': float(median_interval),
            'max_gap_ms': float(np.max(gap_sizes)),
            'gap_indices': gap_indices[:10].tolist()  # First 10
        },
        blocking=(status == QCStatus.FAIL)
    )


def check_sampling_rate_stability(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    max_cv_percent: float = 10.0
) -> QCCheckResult:
    """
    Check that sampling rate is consistent.
    
    High variation in sample intervals indicates DAQ issues.
    
    Args:
        df: DataFrame with time column
        time_col: Name of timestamp column
        max_cv_percent: Maximum allowed coefficient of variation (%)
    """
    if time_col not in df.columns:
        return QCCheckResult(
            name="sampling_rate_stability",
            status=QCStatus.SKIP,
            message=f"Time column '{time_col}' not found",
            blocking=False
        )
    
    timestamps = df[time_col].values
    diffs = np.diff(timestamps)
    
    if len(diffs) < 10:
        return QCCheckResult(
            name="sampling_rate_stability",
            status=QCStatus.SKIP,
            message="Insufficient data for sampling rate analysis",
            blocking=False
        )
    
    mean_interval = np.mean(diffs)
    std_interval = np.std(diffs)
    cv_percent = (std_interval / mean_interval) * 100 if mean_interval > 0 else float('inf')
    
    estimated_rate_hz = 1000.0 / mean_interval if mean_interval > 0 else 0
    
    if cv_percent <= max_cv_percent:
        return QCCheckResult(
            name="sampling_rate_stability",
            status=QCStatus.PASS,
            message=f"Sampling rate stable: {estimated_rate_hz:.1f} Hz (CV={cv_percent:.1f}%)",
            details={
                'mean_interval_ms': float(mean_interval),
                'std_interval_ms': float(std_interval),
                'cv_percent': cv_percent,
                'estimated_rate_hz': estimated_rate_hz
            }
        )
    
    return QCCheckResult(
        name="sampling_rate_stability",
        status=QCStatus.WARN,
        message=f"Sampling rate unstable: CV={cv_percent:.1f}% (>{max_cv_percent}%)",
        details={
            'mean_interval_ms': float(mean_interval),
            'std_interval_ms': float(std_interval),
            'cv_percent': cv_percent,
            'estimated_rate_hz': estimated_rate_hz
        },
        blocking=False
    )


# =============================================================================
# SENSOR RANGE CHECKS
# =============================================================================

def check_sensor_range(
    df: pd.DataFrame,
    channel: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_negative: bool = True
) -> QCCheckResult:
    """
    Check if sensor values are within expected physical range.
    
    Args:
        df: DataFrame with sensor data
        channel: Column name to check
        min_val: Minimum allowed value (None = no limit)
        max_val: Maximum allowed value (None = no limit)
        allow_negative: If False, negative values cause FAIL
    """
    if channel not in df.columns:
        return QCCheckResult(
            name=f"sensor_range_{channel}",
            status=QCStatus.SKIP,
            message=f"Channel '{channel}' not found",
            blocking=False
        )
    
    data = df[channel].dropna()
    
    if len(data) == 0:
        return QCCheckResult(
            name=f"sensor_range_{channel}",
            status=QCStatus.FAIL,
            message=f"Channel '{channel}' has no valid data",
            blocking=True
        )
    
    actual_min = data.min()
    actual_max = data.max()
    violations = []
    
    if not allow_negative and actual_min < 0:
        n_negative = (data < 0).sum()
        violations.append(f"{n_negative} negative values (min={actual_min:.2f})")
    
    if min_val is not None and actual_min < min_val:
        n_below = (data < min_val).sum()
        violations.append(f"{n_below} values below min={min_val} (actual min={actual_min:.2f})")
    
    if max_val is not None and actual_max > max_val:
        n_above = (data > max_val).sum()
        violations.append(f"{n_above} values above max={max_val} (actual max={actual_max:.2f})")
    
    if not violations:
        return QCCheckResult(
            name=f"sensor_range_{channel}",
            status=QCStatus.PASS,
            message=f"'{channel}' in range: [{actual_min:.2f}, {actual_max:.2f}]",
            details={
                'min': float(actual_min),
                'max': float(actual_max),
                'mean': float(data.mean())
            }
        )
    
    return QCCheckResult(
        name=f"sensor_range_{channel}",
        status=QCStatus.FAIL,
        message=f"'{channel}' out of range: " + "; ".join(violations),
        details={
            'min': float(actual_min),
            'max': float(actual_max),
            'expected_min': min_val,
            'expected_max': max_val,
            'violations': violations
        },
        blocking=True
    )


def check_sensor_ranges_from_config(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> List[QCCheckResult]:
    """
    Check all sensor ranges defined in configuration.
    
    Config format (either key is accepted):
    {
        "sensor_limits": {  // or "sensor_ranges"
            "PT-01": {"min": 0, "max": 100, "unit": "bar"},
            "FM-01": {"min": 0, "max": 500, "unit": "g/s"},
            ...
        }
    }
    """
    results = []
    sensor_limits = config.get('sensor_limits', config.get('sensor_ranges', {}))
    
    for channel, limits in sensor_limits.items():
        if channel not in df.columns:
            continue
        
        result = check_sensor_range(
            df, channel,
            min_val=limits.get('min'),
            max_val=limits.get('max'),
            allow_negative=limits.get('allow_negative', True)
        )
        results.append(result)
    
    return results


# =============================================================================
# SIGNAL QUALITY CHECKS
# =============================================================================

def check_flatline(
    df: pd.DataFrame,
    channel: str,
    window_size: int = 100,
    min_variance: float = 1e-10
) -> QCCheckResult:
    """
    Check for flatline (constant value) segments indicating sensor failure.
    
    A sensor reading exactly the same value for 'window_size' consecutive
    samples is almost certainly failed or disconnected.
    
    Args:
        df: DataFrame with sensor data
        channel: Column name to check
        window_size: Number of consecutive identical values to flag
        min_variance: Minimum variance in window to be considered valid
    """
    if channel not in df.columns:
        return QCCheckResult(
            name=f"flatline_{channel}",
            status=QCStatus.SKIP,
            message=f"Channel '{channel}' not found",
            blocking=False
        )
    
    data = df[channel].values
    
    if len(data) < window_size:
        return QCCheckResult(
            name=f"flatline_{channel}",
            status=QCStatus.SKIP,
            message=f"Insufficient data for flatline detection (n={len(data)} < {window_size})",
            blocking=False
        )
    
    # Rolling variance
    rolling_var = pd.Series(data).rolling(window=window_size, center=True).var()
    
    # Find flatline segments (variance = 0 or very small)
    flatline_mask = rolling_var < min_variance
    n_flatline = flatline_mask.sum()
    
    # Calculate percentage
    flatline_pct = (n_flatline / len(data)) * 100
    
    if n_flatline == 0:
        return QCCheckResult(
            name=f"flatline_{channel}",
            status=QCStatus.PASS,
            message=f"No flatline segments detected in '{channel}'",
            details={
                'window_size': window_size,
                'min_variance_threshold': min_variance
            }
        )
    
    # Find flatline locations
    flatline_indices = np.where(flatline_mask)[0]
    
    # Determine severity
    if flatline_pct < 1.0:
        status = QCStatus.WARN
    elif flatline_pct < 10.0:
        status = QCStatus.WARN
    else:
        status = QCStatus.FAIL
    
    return QCCheckResult(
        name=f"flatline_{channel}",
        status=status,
        message=f"Flatline detected in '{channel}': {flatline_pct:.1f}% of samples",
        details={
            'flatline_percentage': flatline_pct,
            'n_flatline_samples': int(n_flatline),
            'first_flatline_index': int(flatline_indices[0]) if len(flatline_indices) > 0 else None,
            'window_size': window_size
        },
        blocking=(status == QCStatus.FAIL)
    )


def check_saturation(
    df: pd.DataFrame,
    channel: str,
    saturation_threshold: float = 0.01
) -> QCCheckResult:
    """
    Check for sensor saturation (stuck at min or max).
    
    If more than saturation_threshold fraction of samples are at
    the extreme values, sensor may be saturated.
    
    Args:
        df: DataFrame with sensor data
        channel: Column name to check
        saturation_threshold: Fraction of samples at extremes to flag
    """
    if channel not in df.columns:
        return QCCheckResult(
            name=f"saturation_{channel}",
            status=QCStatus.SKIP,
            message=f"Channel '{channel}' not found",
            blocking=False
        )
    
    data = df[channel].dropna()
    
    if len(data) < 10:
        return QCCheckResult(
            name=f"saturation_{channel}",
            status=QCStatus.SKIP,
            message="Insufficient data for saturation check",
            blocking=False
        )
    
    min_val = data.min()
    max_val = data.max()
    
    # Count samples at extremes
    at_min = (data == min_val).sum()
    at_max = (data == max_val).sum()
    
    min_fraction = at_min / len(data)
    max_fraction = at_max / len(data)
    
    issues = []
    if min_fraction > saturation_threshold:
        issues.append(f"{min_fraction*100:.1f}% at min ({min_val:.2f})")
    if max_fraction > saturation_threshold:
        issues.append(f"{max_fraction*100:.1f}% at max ({max_val:.2f})")
    
    if not issues:
        return QCCheckResult(
            name=f"saturation_{channel}",
            status=QCStatus.PASS,
            message=f"No saturation detected in '{channel}'",
            details={
                'min_val': float(min_val),
                'max_val': float(max_val),
                'at_min_pct': min_fraction * 100,
                'at_max_pct': max_fraction * 100
            }
        )
    
    # Determine severity
    max_sat_fraction = max(min_fraction, max_fraction)
    status = QCStatus.FAIL if max_sat_fraction > 0.10 else QCStatus.WARN
    
    return QCCheckResult(
        name=f"saturation_{channel}",
        status=status,
        message=f"Possible saturation in '{channel}': " + "; ".join(issues),
        details={
            'min_val': float(min_val),
            'max_val': float(max_val),
            'at_min_pct': min_fraction * 100,
            'at_max_pct': max_fraction * 100
        },
        blocking=(status == QCStatus.FAIL)
    )


def check_nan_ratio(
    df: pd.DataFrame,
    channel: str,
    max_nan_ratio: float = 0.05
) -> QCCheckResult:
    """
    Check for excessive NaN/missing values.
    
    Args:
        df: DataFrame with sensor data
        channel: Column name to check
        max_nan_ratio: Maximum allowed ratio of NaN values (0-1)
    """
    if channel not in df.columns:
        return QCCheckResult(
            name=f"nan_ratio_{channel}",
            status=QCStatus.SKIP,
            message=f"Channel '{channel}' not found",
            blocking=False
        )
    
    n_total = len(df)
    n_nan = df[channel].isna().sum()
    nan_ratio = n_nan / n_total if n_total > 0 else 0
    
    if nan_ratio <= max_nan_ratio:
        return QCCheckResult(
            name=f"nan_ratio_{channel}",
            status=QCStatus.PASS,
            message=f"'{channel}': {nan_ratio*100:.1f}% NaN (≤{max_nan_ratio*100:.0f}%)",
            details={
                'n_nan': int(n_nan),
                'n_total': n_total,
                'nan_ratio': nan_ratio
            }
        )
    
    # High NaN ratio
    status = QCStatus.FAIL if nan_ratio > 0.20 else QCStatus.WARN
    
    return QCCheckResult(
        name=f"nan_ratio_{channel}",
        status=status,
        message=f"'{channel}': {nan_ratio*100:.1f}% NaN (>{max_nan_ratio*100:.0f}%)",
        details={
            'n_nan': int(n_nan),
            'n_total': n_total,
            'nan_ratio': nan_ratio
        },
        blocking=(status == QCStatus.FAIL)
    )


# =============================================================================
# CROSS-CORRELATION CHECKS
# =============================================================================

def check_pressure_flow_correlation(
    df: pd.DataFrame,
    pressure_col: str,
    flow_col: str,
    min_correlation: float = 0.3
) -> QCCheckResult:
    """
    Check that pressure and flow are positively correlated.
    
    For cold flow tests, higher pressure should mean higher flow.
    Negative or no correlation indicates sensor issues.
    
    Args:
        df: DataFrame with sensor data
        pressure_col: Pressure column name
        flow_col: Flow column name
        min_correlation: Minimum expected correlation
    """
    if pressure_col not in df.columns or flow_col not in df.columns:
        missing = []
        if pressure_col not in df.columns:
            missing.append(pressure_col)
        if flow_col not in df.columns:
            missing.append(flow_col)
        
        return QCCheckResult(
            name="pressure_flow_correlation",
            status=QCStatus.SKIP,
            message=f"Missing columns: {missing}",
            blocking=False
        )
    
    # Get valid data
    valid_mask = df[pressure_col].notna() & df[flow_col].notna()
    p_data = df.loc[valid_mask, pressure_col]
    f_data = df.loc[valid_mask, flow_col]
    
    if len(p_data) < 10:
        return QCCheckResult(
            name="pressure_flow_correlation",
            status=QCStatus.SKIP,
            message="Insufficient valid data for correlation",
            blocking=False
        )
    
    correlation = p_data.corr(f_data)
    
    if np.isnan(correlation):
        return QCCheckResult(
            name="pressure_flow_correlation",
            status=QCStatus.WARN,
            message="Could not compute correlation (constant data?)",
            blocking=False
        )
    
    if correlation >= min_correlation:
        return QCCheckResult(
            name="pressure_flow_correlation",
            status=QCStatus.PASS,
            message=f"Pressure-flow correlation: {correlation:.3f} (≥{min_correlation})",
            details={'correlation': float(correlation)}
        )
    
    if correlation < 0:
        return QCCheckResult(
            name="pressure_flow_correlation",
            status=QCStatus.FAIL,
            message=f"NEGATIVE pressure-flow correlation: {correlation:.3f} - check sensor wiring!",
            details={'correlation': float(correlation)},
            blocking=True
        )
    
    return QCCheckResult(
        name="pressure_flow_correlation",
        status=QCStatus.WARN,
        message=f"Weak pressure-flow correlation: {correlation:.3f} (<{min_correlation})",
        details={'correlation': float(correlation)},
        blocking=False
    )


# =============================================================================
# MAIN QC RUNNER
# =============================================================================

def run_qc_checks(
    df: pd.DataFrame,
    config: Dict[str, Any],
    time_col: str = 'timestamp',
    critical_channels: Optional[List[str]] = None
) -> QCReport:
    """
    Run all QC checks on a dataset.
    
    This is the main entry point for pre-analysis data validation.
    
    Args:
        df: DataFrame containing test data
        config: Test configuration dictionary
        time_col: Name of timestamp column
        critical_channels: List of channels that MUST pass all checks
                          (if None, derived from config)
    
    Returns:
        QCReport with all check results
    """
    report = QCReport()
    
    # Determine critical channels from config if not specified
    if critical_channels is None:
        # Support both sensor_roles (v2.4.0+) and columns (legacy)
        cols = config.get('sensor_roles', config.get('columns', {}))
        critical_channels = [
            v for k, v in cols.items()
            if v and k in ('upstream_pressure', 'mass_flow', 'chamber_pressure',
                          'thrust', 'mass_flow_ox', 'mass_flow_fuel')
        ]
    
    # -------------------------------------------------------------------------
    # 1. TIMESTAMP CHECKS
    # -------------------------------------------------------------------------
    report.add_check(check_timestamp_monotonic(df, time_col))
    report.add_check(check_timestamp_gaps(df, time_col))
    report.add_check(check_sampling_rate_stability(df, time_col))
    
    # -------------------------------------------------------------------------
    # 2. SENSOR RANGE CHECKS (from config)
    # -------------------------------------------------------------------------
    if 'sensor_limits' in config or 'sensor_ranges' in config:
        range_results = check_sensor_ranges_from_config(df, config)
        for result in range_results:
            report.add_check(result)
    
    # -------------------------------------------------------------------------
    # 3. SIGNAL QUALITY CHECKS (on critical channels)
    # -------------------------------------------------------------------------
    for channel in critical_channels:
        if channel not in df.columns:
            continue
        
        # NaN check
        report.add_check(check_nan_ratio(df, channel, max_nan_ratio=0.05))
        
        # Flatline check
        report.add_check(check_flatline(df, channel, window_size=50))
        
        # Saturation check
        report.add_check(check_saturation(df, channel))
    
    # -------------------------------------------------------------------------
    # 4. CROSS-CORRELATION CHECKS
    # -------------------------------------------------------------------------
    # Support both sensor_roles (v2.4.0+) and columns (legacy)
    cols = config.get('sensor_roles', config.get('columns', {}))

    # Cold flow: pressure-flow correlation
    p_col = cols.get('upstream_pressure') or cols.get('inlet_pressure')
    f_col = cols.get('mass_flow') or cols.get('mf')

    if p_col and f_col:
        report.add_check(check_pressure_flow_correlation(df, p_col, f_col))

    # Hot fire: thrust-pressure correlation
    pc_col = cols.get('chamber_pressure')
    thrust_col = cols.get('thrust')
    
    if pc_col and thrust_col:
        report.add_check(check_pressure_flow_correlation(
            df, pc_col, thrust_col, min_correlation=0.5
        ))
    
    return report


def run_quick_qc(
    df: pd.DataFrame,
    time_col: str = 'timestamp'
) -> QCReport:
    """
    Run minimal QC checks (for quick analysis mode).
    
    Only checks critical issues that would make analysis meaningless:
    - Timestamp monotonicity
    - Excessive NaN
    - All-constant data
    
    Args:
        df: DataFrame containing test data
        time_col: Name of timestamp column
        
    Returns:
        QCReport with critical check results only
    """
    report = QCReport()
    
    # Timestamp check
    report.add_check(check_timestamp_monotonic(df, time_col))
    
    # Check each numeric column for basic validity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == time_col:
            continue
        
        # Quick NaN check
        nan_ratio = df[col].isna().mean()
        if nan_ratio > 0.5:
            report.add_check(QCCheckResult(
                name=f"quick_nan_{col}",
                status=QCStatus.FAIL,
                message=f"'{col}': {nan_ratio*100:.0f}% NaN",
                blocking=True
            ))
        
        # Quick constant check
        unique_vals = df[col].nunique()
        if unique_vals <= 1:
            report.add_check(QCCheckResult(
                name=f"quick_constant_{col}",
                status=QCStatus.FAIL,
                message=f"'{col}' is constant (only {unique_vals} unique value)",
                blocking=True
            ))
    
    return report


# =============================================================================
# HOT FIRE SPECIFIC QC CHECKS
# =============================================================================

def check_combustion_stability(
    df: pd.DataFrame,
    pc_col: str,
    max_roughness_pct: float = 10.0,
    window_size: int = 50
) -> QCCheckResult:
    """
    Check combustion stability via chamber pressure roughness.

    Roughness is defined as (peak-to-peak oscillation / mean Pc) over rolling
    windows. High roughness indicates combustion instability.

    Args:
        df: DataFrame with chamber pressure data
        pc_col: Chamber pressure column name
        max_roughness_pct: Maximum allowed roughness (%)
        window_size: Rolling window size for roughness calculation
    """
    if pc_col not in df.columns:
        return QCCheckResult(
            name="combustion_stability",
            status=QCStatus.SKIP,
            message=f"Chamber pressure column '{pc_col}' not found",
            blocking=False
        )

    data = df[pc_col].dropna()
    if len(data) < window_size * 2:
        return QCCheckResult(
            name="combustion_stability",
            status=QCStatus.SKIP,
            message="Insufficient data for combustion stability check",
            blocking=False
        )

    mean_pc = data.mean()
    if mean_pc <= 0:
        return QCCheckResult(
            name="combustion_stability",
            status=QCStatus.FAIL,
            message=f"Chamber pressure non-positive (mean={mean_pc:.2f})",
            blocking=True
        )

    # Rolling peak-to-peak roughness
    rolling_max = data.rolling(window=window_size, center=True).max()
    rolling_min = data.rolling(window=window_size, center=True).min()
    roughness = ((rolling_max - rolling_min) / mean_pc * 100).dropna()

    if len(roughness) == 0:
        return QCCheckResult(
            name="combustion_stability",
            status=QCStatus.SKIP,
            message="Could not compute roughness",
            blocking=False
        )

    max_roughness = roughness.max()
    mean_roughness = roughness.mean()

    if max_roughness <= max_roughness_pct:
        return QCCheckResult(
            name="combustion_stability",
            status=QCStatus.PASS,
            message=f"Combustion stable: roughness={mean_roughness:.1f}% mean, {max_roughness:.1f}% peak",
            details={
                'mean_roughness_pct': float(mean_roughness),
                'max_roughness_pct': float(max_roughness),
                'mean_pc': float(mean_pc),
            }
        )

    status = QCStatus.FAIL if max_roughness > max_roughness_pct * 2 else QCStatus.WARN
    return QCCheckResult(
        name="combustion_stability",
        status=status,
        message=f"Combustion roughness high: {max_roughness:.1f}% peak (limit: {max_roughness_pct}%)",
        details={
            'mean_roughness_pct': float(mean_roughness),
            'max_roughness_pct': float(max_roughness),
            'mean_pc': float(mean_pc),
        },
        blocking=(status == QCStatus.FAIL)
    )


def check_ignition_detection(
    df: pd.DataFrame,
    pc_col: str,
    time_col: str = 'timestamp',
    min_pc_bar: float = 2.0
) -> QCCheckResult:
    """
    Check that ignition was detected (chamber pressure exceeds threshold).

    For hot fire tests, chamber pressure must rise above a minimum threshold
    to confirm successful ignition.

    Args:
        df: DataFrame with test data
        pc_col: Chamber pressure column name
        time_col: Timestamp column name
        min_pc_bar: Minimum Pc (bar) to consider ignition successful
    """
    if pc_col not in df.columns:
        return QCCheckResult(
            name="ignition_detection",
            status=QCStatus.SKIP,
            message=f"Chamber pressure column '{pc_col}' not found",
            blocking=False
        )

    data = df[pc_col].dropna()
    if len(data) == 0:
        return QCCheckResult(
            name="ignition_detection",
            status=QCStatus.FAIL,
            message="No chamber pressure data",
            blocking=True
        )

    max_pc = data.max()
    if max_pc >= min_pc_bar:
        # Find ignition time (first crossing of threshold)
        above_threshold = data >= min_pc_bar
        first_idx = above_threshold.idxmax() if above_threshold.any() else None
        ignition_time = None
        if first_idx is not None and time_col in df.columns:
            ignition_time = float(df.loc[first_idx, time_col])

        return QCCheckResult(
            name="ignition_detection",
            status=QCStatus.PASS,
            message=f"Ignition confirmed: peak Pc={max_pc:.1f} bar",
            details={
                'max_pc_bar': float(max_pc),
                'threshold_bar': min_pc_bar,
                'ignition_time': ignition_time,
            }
        )

    return QCCheckResult(
        name="ignition_detection",
        status=QCStatus.FAIL,
        message=f"Ignition NOT detected: max Pc={max_pc:.2f} bar < threshold {min_pc_bar} bar",
        details={
            'max_pc_bar': float(max_pc),
            'threshold_bar': min_pc_bar,
        },
        blocking=True
    )


def check_propellant_lead_lag(
    df: pd.DataFrame,
    ox_col: str,
    fuel_col: str,
    time_col: str = 'timestamp',
    max_lead_lag_ms: float = 500.0
) -> QCCheckResult:
    """
    Check propellant lead/lag timing.

    Excessive lead/lag between oxidizer and fuel flow onset can indicate
    valve sequencing issues and is a safety concern (hard start risk).

    Args:
        df: DataFrame with mass flow data
        ox_col: Oxidizer mass flow column
        fuel_col: Fuel mass flow column
        time_col: Timestamp column
        max_lead_lag_ms: Maximum allowed lead/lag (ms)
    """
    missing = [c for c in [ox_col, fuel_col, time_col] if c not in df.columns]
    if missing:
        return QCCheckResult(
            name="propellant_lead_lag",
            status=QCStatus.SKIP,
            message=f"Missing columns: {missing}",
            blocking=False
        )

    ox_data = df[ox_col].values
    fuel_data = df[fuel_col].values
    times = df[time_col].values

    # Detect flow onset: first time signal exceeds 10% of max
    ox_threshold = np.max(ox_data) * 0.10
    fuel_threshold = np.max(fuel_data) * 0.10

    ox_onset_idx = np.argmax(ox_data > ox_threshold) if np.any(ox_data > ox_threshold) else None
    fuel_onset_idx = np.argmax(fuel_data > fuel_threshold) if np.any(fuel_data > fuel_threshold) else None

    if ox_onset_idx is None or fuel_onset_idx is None:
        return QCCheckResult(
            name="propellant_lead_lag",
            status=QCStatus.WARN,
            message="Could not detect flow onset for lead/lag analysis",
            blocking=False
        )

    ox_onset_time = times[ox_onset_idx]
    fuel_onset_time = times[fuel_onset_idx]
    lead_lag_ms = float(abs(ox_onset_time - fuel_onset_time))

    # Determine which leads
    if ox_onset_time < fuel_onset_time:
        leader = "oxidizer"
    elif fuel_onset_time < ox_onset_time:
        leader = "fuel"
    else:
        leader = "simultaneous"

    if lead_lag_ms <= max_lead_lag_ms:
        return QCCheckResult(
            name="propellant_lead_lag",
            status=QCStatus.PASS,
            message=f"Propellant lead/lag: {lead_lag_ms:.1f} ms ({leader} leads)",
            details={
                'lead_lag_ms': lead_lag_ms,
                'leader': leader,
                'ox_onset_time': float(ox_onset_time),
                'fuel_onset_time': float(fuel_onset_time),
            }
        )

    status = QCStatus.FAIL if lead_lag_ms > max_lead_lag_ms * 2 else QCStatus.WARN
    return QCCheckResult(
        name="propellant_lead_lag",
        status=status,
        message=f"Excessive propellant lead/lag: {lead_lag_ms:.1f} ms ({leader} leads, limit: {max_lead_lag_ms} ms)",
        details={
            'lead_lag_ms': lead_lag_ms,
            'leader': leader,
            'ox_onset_time': float(ox_onset_time),
            'fuel_onset_time': float(fuel_onset_time),
        },
        blocking=(status == QCStatus.FAIL)
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def assert_qc_passed(report: QCReport, raise_on_fail: bool = True) -> bool:
    """
    Check if QC passed and optionally raise exception on failure.
    
    Args:
        report: QC report to check
        raise_on_fail: If True, raise ValueError on failure
        
    Returns:
        True if passed
        
    Raises:
        ValueError: If QC failed and raise_on_fail is True
    """
    if report.passed:
        return True
    
    if raise_on_fail:
        failures = report.blocking_failures
        failure_msgs = [f"  - {f.name}: {f.message}" for f in failures]
        raise ValueError(
            f"QC FAILED - Analysis blocked:\n" + "\n".join(failure_msgs)
        )
    
    return False


def format_qc_for_display(report: QCReport) -> str:
    """
    Format QC report for display in UI.
    
    Returns:
        Markdown-formatted string
    """
    lines = []
    
    # Header
    if report.passed:
        lines.append("##  Quality Control: PASSED")
    else:
        lines.append("##  Quality Control: FAILED")
    
    lines.append("")
    lines.append(f"**Summary:** {report.summary['passed']} passed, "
                 f"{report.summary['warnings']} warnings, "
                 f"{report.summary['failed']} failed")
    lines.append("")
    
    # Group by status
    if report.blocking_failures:
        lines.append("###  Blocking Failures")
        for check in report.blocking_failures:
            lines.append(f"- **{check.name}**: {check.message}")
        lines.append("")
    
    if report.warnings:
        lines.append("### [WARN] Warnings")
        for check in report.warnings:
            lines.append(f"- **{check.name}**: {check.message}")
        lines.append("")
    
    passed_checks = [c for c in report.checks if c.status == QCStatus.PASS]
    if passed_checks:
        lines.append("###  Passed Checks")
        for check in passed_checks:
            lines.append(f"- {check.name}")
    
    return "\n".join(lines)
