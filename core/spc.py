"""
Statistical Process Control (SPC) Module
=========================================
Control charts, trend detection, and process capability analysis
for propulsion test campaigns.

Key Features:
- X-bar and R control charts
- Individual/Moving Range (I-MR) charts
- Western Electric rules for out-of-control detection
- Process capability indices (Cp, Cpk, Pp, Ppk)
- Trend detection and drift analysis

Use Cases:
- Monitor Cd stability across a production run
- Detect injector degradation over hot fire tests
- Track process capability for flight qualification
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class ControlChartType(Enum):
    """Types of control charts."""
    XBAR_R = "X-bar and R"
    XBAR_S = "X-bar and S"
    I_MR = "Individual and Moving Range"
    P_CHART = "P Chart (proportion)"
    C_CHART = "C Chart (count)"
    CUSUM = "Cumulative Sum"
    EWMA = "Exponentially Weighted Moving Average"


class ViolationType(Enum):
    """Western Electric rule violations."""
    NONE = "No violation"
    BEYOND_3SIGMA = "Point beyond 3σ"
    ZONE_A_2OF3 = "2 of 3 consecutive in Zone A"
    ZONE_B_4OF5 = "4 of 5 consecutive in Zone B"
    RUN_OF_8 = "8 consecutive on same side"
    TREND_6 = "6 consecutive increasing/decreasing"
    ALTERNATING_14 = "14 consecutive alternating"
    BEYOND_2SIGMA_4OF5 = "4 of 5 beyond 2σ same side"


@dataclass
class ControlLimits:
    """Control chart limits."""
    center_line: float
    ucl: float  # Upper Control Limit (3σ)
    lcl: float  # Lower Control Limit (3σ)
    
    # Optional warning limits (2σ)
    uwl: Optional[float] = None  # Upper Warning Limit
    lwl: Optional[float] = None  # Lower Warning Limit
    
    # Zone boundaries for Western Electric rules
    zone_a_upper: Optional[float] = None  # 2σ to 3σ
    zone_a_lower: Optional[float] = None
    zone_b_upper: Optional[float] = None  # 1σ to 2σ
    zone_b_lower: Optional[float] = None
    zone_c_upper: Optional[float] = None  # 0 to 1σ
    zone_c_lower: Optional[float] = None
    
    def calculate_zones(self):
        """Calculate zone boundaries from control limits."""
        sigma = (self.ucl - self.center_line) / 3
        
        self.zone_a_upper = self.center_line + 2 * sigma
        self.zone_a_lower = self.center_line - 2 * sigma
        self.zone_b_upper = self.center_line + sigma
        self.zone_b_lower = self.center_line - sigma
        self.zone_c_upper = self.center_line + sigma
        self.zone_c_lower = self.center_line - sigma
        
        self.uwl = self.zone_a_upper
        self.lwl = self.zone_a_lower
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'center_line': self.center_line,
            'ucl': self.ucl,
            'lcl': self.lcl,
            'uwl': self.uwl,
            'lwl': self.lwl,
        }


@dataclass
class ControlChartPoint:
    """A single point on a control chart."""
    index: int
    value: float
    test_id: str
    timestamp: Optional[str] = None
    in_control: bool = True
    violations: List[ViolationType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'value': self.value,
            'test_id': self.test_id,
            'timestamp': self.timestamp,
            'in_control': self.in_control,
            'violations': [v.value for v in self.violations],
        }


@dataclass
class ProcessCapability:
    """Process capability indices."""
    # Potential capability (uses spec limits and process sigma)
    cp: Optional[float] = None      # (USL - LSL) / 6σ
    cpu: Optional[float] = None     # (USL - μ) / 3σ
    cpl: Optional[float] = None     # (μ - LSL) / 3σ
    cpk: Optional[float] = None     # min(Cpu, Cpl)
    
    # Performance (uses actual data spread)
    pp: Optional[float] = None      # (USL - LSL) / 6s
    ppu: Optional[float] = None     # (USL - μ) / 3s
    ppl: Optional[float] = None     # (μ - LSL) / 3s
    ppk: Optional[float] = None     # min(Ppu, Ppl)
    
    # Additional metrics
    cpm: Optional[float] = None     # Taguchi capability index
    sigma_level: Optional[float] = None  # Process sigma level
    
    # Spec limits used
    usl: Optional[float] = None
    lsl: Optional[float] = None
    target: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Cp': self.cp,
            'Cpk': self.cpk,
            'Cpu': self.cpu,
            'Cpl': self.cpl,
            'Pp': self.pp,
            'Ppk': self.ppk,
            'Ppu': self.ppu,
            'Ppl': self.ppl,
            'Cpm': self.cpm,
            'sigma_level': self.sigma_level,
            'USL': self.usl,
            'LSL': self.lsl,
            'target': self.target,
        }
    
    def summary(self) -> str:
        """Return a summary interpretation."""
        if self.cpk is None:
            return "Insufficient data for capability analysis"
        
        if self.cpk >= 1.67:
            return f"Cpk = {self.cpk:.2f}: Excellent capability (Six Sigma equivalent)"
        elif self.cpk >= 1.33:
            return f"Cpk = {self.cpk:.2f}: Good capability (meets typical requirements)"
        elif self.cpk >= 1.0:
            return f"Cpk = {self.cpk:.2f}: Marginal capability (process improvement recommended)"
        else:
            return f"Cpk = {self.cpk:.2f}: Poor capability (process not capable)"


@dataclass
class SPCAnalysis:
    """Complete SPC analysis results."""
    parameter_name: str
    chart_type: ControlChartType
    limits: ControlLimits
    points: List[ControlChartPoint]
    capability: Optional[ProcessCapability] = None
    
    # Summary statistics
    n_points: int = 0
    n_violations: int = 0
    violation_rate: float = 0.0
    
    # Trend analysis
    has_trend: bool = False
    trend_direction: Optional[str] = None  # 'increasing', 'decreasing'
    trend_slope: Optional[float] = None
    
    def __post_init__(self):
        self.n_points = len(self.points)
        self.n_violations = sum(1 for p in self.points if not p.in_control)
        self.violation_rate = self.n_violations / self.n_points if self.n_points > 0 else 0
    
    def get_out_of_control_points(self) -> List[ControlChartPoint]:
        """Get all points with violations."""
        return [p for p in self.points if not p.in_control]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameter_name': self.parameter_name,
            'chart_type': self.chart_type.value,
            'limits': self.limits.to_dict(),
            'n_points': self.n_points,
            'n_violations': self.n_violations,
            'violation_rate': self.violation_rate,
            'has_trend': self.has_trend,
            'trend_direction': self.trend_direction,
            'capability': self.capability.to_dict() if self.capability else None,
        }


# =============================================================================
# CONTROL LIMIT CALCULATIONS
# =============================================================================

# Control chart constants (for small sample sizes)
CHART_CONSTANTS = {
    2: {'d2': 1.128, 'd3': 0.853, 'D3': 0, 'D4': 3.267, 'A2': 1.880, 'A3': 2.659, 'c4': 0.7979, 'B3': 0, 'B4': 3.267},
    3: {'d2': 1.693, 'd3': 0.888, 'D3': 0, 'D4': 2.574, 'A2': 1.023, 'A3': 1.954, 'c4': 0.8862, 'B3': 0, 'B4': 2.568},
    4: {'d2': 2.059, 'd3': 0.880, 'D3': 0, 'D4': 2.282, 'A2': 0.729, 'A3': 1.628, 'c4': 0.9213, 'B3': 0, 'B4': 2.266},
    5: {'d2': 2.326, 'd3': 0.864, 'D3': 0, 'D4': 2.114, 'A2': 0.577, 'A3': 1.427, 'c4': 0.9400, 'B3': 0, 'B4': 2.089},
    6: {'d2': 2.534, 'd3': 0.848, 'D3': 0, 'D4': 2.004, 'A2': 0.483, 'A3': 1.287, 'c4': 0.9515, 'B3': 0.030, 'B4': 1.970},
    7: {'d2': 2.704, 'd3': 0.833, 'D3': 0.076, 'D4': 1.924, 'A2': 0.419, 'A3': 1.182, 'c4': 0.9594, 'B3': 0.118, 'B4': 1.882},
    8: {'d2': 2.847, 'd3': 0.820, 'D3': 0.136, 'D4': 1.864, 'A2': 0.373, 'A3': 1.099, 'c4': 0.9650, 'B3': 0.185, 'B4': 1.815},
    9: {'d2': 2.970, 'd3': 0.808, 'D3': 0.184, 'D4': 1.816, 'A2': 0.337, 'A3': 1.032, 'c4': 0.9693, 'B3': 0.239, 'B4': 1.761},
    10: {'d2': 3.078, 'd3': 0.797, 'D3': 0.223, 'D4': 1.777, 'A2': 0.308, 'A3': 0.975, 'c4': 0.9727, 'B3': 0.284, 'B4': 1.716},
}


def calculate_imr_limits(values: np.ndarray) -> Tuple[ControlLimits, ControlLimits]:
    """
    Calculate Individual and Moving Range control limits.
    
    Args:
        values: Array of individual measurements
        
    Returns:
        Tuple of (individual_limits, mr_limits)
    """
    n = len(values)
    
    if n < 2:
        raise ValueError("Need at least 2 points for I-MR chart")
    
    # Moving ranges
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr)
    
    # d2 for n=2 (moving range of 2 consecutive points)
    d2 = CHART_CONSTANTS[2]['d2']
    D3 = CHART_CONSTANTS[2]['D3']
    D4 = CHART_CONSTANTS[2]['D4']
    
    # Estimate sigma from moving range
    sigma_hat = mr_bar / d2
    
    # Individual chart limits
    x_bar = np.mean(values)
    i_limits = ControlLimits(
        center_line=x_bar,
        ucl=x_bar + 3 * sigma_hat,
        lcl=x_bar - 3 * sigma_hat,
    )
    i_limits.calculate_zones()
    
    # Moving range limits
    mr_limits = ControlLimits(
        center_line=mr_bar,
        ucl=D4 * mr_bar,
        lcl=D3 * mr_bar,
    )
    
    return i_limits, mr_limits


def calculate_xbar_r_limits(
    subgroups: List[np.ndarray]
) -> Tuple[ControlLimits, ControlLimits]:
    """
    Calculate X-bar and R control limits.
    
    Args:
        subgroups: List of arrays, each containing subgroup measurements
        
    Returns:
        Tuple of (xbar_limits, r_limits)
    """
    n_subgroups = len(subgroups)
    subgroup_size = len(subgroups[0])
    
    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Subgroup size must be between 2 and 10")
    
    if n_subgroups < 2:
        raise ValueError("Need at least 2 subgroups")
    
    # Calculate subgroup means and ranges
    x_bars = np.array([np.mean(sg) for sg in subgroups])
    ranges = np.array([np.max(sg) - np.min(sg) for sg in subgroups])
    
    x_bar_bar = np.mean(x_bars)
    r_bar = np.mean(ranges)
    
    # Get constants for subgroup size
    constants = CHART_CONSTANTS[subgroup_size]
    A2 = constants['A2']
    D3 = constants['D3']
    D4 = constants['D4']
    
    # X-bar limits
    xbar_limits = ControlLimits(
        center_line=x_bar_bar,
        ucl=x_bar_bar + A2 * r_bar,
        lcl=x_bar_bar - A2 * r_bar,
    )
    xbar_limits.calculate_zones()
    
    # R limits
    r_limits = ControlLimits(
        center_line=r_bar,
        ucl=D4 * r_bar,
        lcl=D3 * r_bar,
    )
    
    return xbar_limits, r_limits


# =============================================================================
# WESTERN ELECTRIC RULES
# =============================================================================

def check_western_electric_rules(
    values: np.ndarray,
    limits: ControlLimits
) -> List[List[ViolationType]]:
    """
    Apply Western Electric rules to detect out-of-control conditions.
    
    Args:
        values: Array of measurements
        limits: Control limits with zone boundaries
        
    Returns:
        List of violation lists for each point
    """
    n = len(values)
    violations = [[] for _ in range(n)]
    
    cl = limits.center_line
    
    # Ensure zones are calculated
    if limits.zone_a_upper is None:
        limits.calculate_zones()
    
    sigma = (limits.ucl - cl) / 3
    
    for i, val in enumerate(values):
        point_violations = []
        
        # Rule 1: Beyond 3σ
        if val > limits.ucl or val < limits.lcl:
            point_violations.append(ViolationType.BEYOND_3SIGMA)
        
        # Rule 2: 2 of 3 in Zone A (same side)
        if i >= 2:
            recent = values[i-2:i+1]
            above_2sigma = sum(1 for v in recent if v > cl + 2*sigma)
            below_2sigma = sum(1 for v in recent if v < cl - 2*sigma)
            if above_2sigma >= 2 or below_2sigma >= 2:
                point_violations.append(ViolationType.ZONE_A_2OF3)
        
        # Rule 3: 4 of 5 beyond 1σ (same side)
        if i >= 4:
            recent = values[i-4:i+1]
            above_1sigma = sum(1 for v in recent if v > cl + sigma)
            below_1sigma = sum(1 for v in recent if v < cl - sigma)
            if above_1sigma >= 4 or below_1sigma >= 4:
                point_violations.append(ViolationType.ZONE_B_4OF5)
        
        # Rule 4: 8 consecutive on same side
        if i >= 7:
            recent = values[i-7:i+1]
            all_above = all(v > cl for v in recent)
            all_below = all(v < cl for v in recent)
            if all_above or all_below:
                point_violations.append(ViolationType.RUN_OF_8)
        
        # Rule 5: 6 consecutive increasing or decreasing
        if i >= 5:
            recent = values[i-5:i+1]
            diffs = np.diff(recent)
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                point_violations.append(ViolationType.TREND_6)
        
        # Rule 6: 14 alternating
        if i >= 13:
            recent = values[i-13:i+1]
            diffs = np.diff(recent)
            signs = np.sign(diffs)
            alternating = all(signs[j] != signs[j+1] for j in range(len(signs)-1))
            if alternating:
                point_violations.append(ViolationType.ALTERNATING_14)
        
        violations[i] = point_violations
    
    return violations


# =============================================================================
# PROCESS CAPABILITY
# =============================================================================

def calculate_capability(
    values: np.ndarray,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    target: Optional[float] = None,
) -> ProcessCapability:
    """
    Calculate process capability indices.
    
    Args:
        values: Array of measurements
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value (defaults to midpoint of specs)
    
    Returns:
        ProcessCapability with all indices
    """
    n = len(values)
    
    if n < 2:
        return ProcessCapability()
    
    mean = np.mean(values)
    
    # Overall standard deviation (for Pp/Ppk)
    s = np.std(values, ddof=1)
    
    # Within-subgroup sigma estimate from moving range (for Cp/Cpk)
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr)
    d2 = CHART_CONSTANTS[2]['d2']
    sigma_within = mr_bar / d2 if mr_bar > 0 else s
    
    capability = ProcessCapability(usl=usl, lsl=lsl, target=target)
    
    # Set target to midpoint if not specified
    if target is None and usl is not None and lsl is not None:
        target = (usl + lsl) / 2
        capability.target = target
    
    # Calculate Cp/Cpk (potential capability - using within sigma)
    if usl is not None and lsl is not None and sigma_within > 0:
        capability.cp = (usl - lsl) / (6 * sigma_within)
    
    if usl is not None and sigma_within > 0:
        capability.cpu = (usl - mean) / (3 * sigma_within)
    
    if lsl is not None and sigma_within > 0:
        capability.cpl = (mean - lsl) / (3 * sigma_within)
    
    if capability.cpu is not None and capability.cpl is not None:
        capability.cpk = min(capability.cpu, capability.cpl)
    elif capability.cpu is not None:
        capability.cpk = capability.cpu
    elif capability.cpl is not None:
        capability.cpk = capability.cpl
    
    # Calculate Pp/Ppk (performance - using overall sigma)
    if usl is not None and lsl is not None and s > 0:
        capability.pp = (usl - lsl) / (6 * s)
    
    if usl is not None and s > 0:
        capability.ppu = (usl - mean) / (3 * s)
    
    if lsl is not None and s > 0:
        capability.ppl = (mean - lsl) / (3 * s)
    
    if capability.ppu is not None and capability.ppl is not None:
        capability.ppk = min(capability.ppu, capability.ppl)
    elif capability.ppu is not None:
        capability.ppk = capability.ppu
    elif capability.ppl is not None:
        capability.ppk = capability.ppl
    
    # Cpm (Taguchi) - incorporates deviation from target
    if target is not None and usl is not None and lsl is not None:
        tau = np.sqrt(s**2 + (mean - target)**2)
        if tau > 0:
            capability.cpm = (usl - lsl) / (6 * tau)
    
    # Sigma level
    if capability.cpk is not None:
        capability.sigma_level = 3 * capability.cpk
    
    return capability


# =============================================================================
# TREND ANALYSIS
# =============================================================================

def detect_trend(
    values: np.ndarray,
    min_points: int = 6,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Detect linear trend in data using linear regression.
    
    Args:
        values: Array of measurements
        min_points: Minimum points required for trend detection
        
    Returns:
        Tuple of (has_trend, direction, slope)
    """
    n = len(values)
    
    if n < min_points:
        return False, None, None
    
    # Simple linear regression
    x = np.arange(n)
    
    x_mean = np.mean(x)
    y_mean = np.mean(values)
    
    numerator = np.sum((x - x_mean) * (values - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return False, None, None
    
    slope = numerator / denominator
    
    # Calculate R-squared
    y_pred = slope * x + (y_mean - slope * x_mean)
    ss_res = np.sum((values - y_pred) ** 2)
    ss_tot = np.sum((values - y_mean) ** 2)
    
    if ss_tot == 0:
        return False, None, None
    
    r_squared = 1 - (ss_res / ss_tot)
    
    # Trend is significant if R² > 0.5 and slope is meaningful
    data_range = np.max(values) - np.min(values)
    relative_slope = abs(slope * n) / data_range if data_range > 0 else 0
    
    has_trend = r_squared > 0.5 and relative_slope > 0.1
    
    direction = None
    if has_trend:
        direction = 'increasing' if slope > 0 else 'decreasing'
    
    return has_trend, direction, slope


# =============================================================================
# CUSUM AND EWMA CONTROL CHARTS
# =============================================================================

@dataclass
class CUSUMResult:
    """CUSUM control chart results."""
    parameter_name: str
    target: float
    k: float  # Allowable slack (typically 0.5 * shift_to_detect)
    h: float  # Decision interval
    c_plus: np.ndarray  # Upper CUSUM
    c_minus: np.ndarray  # Lower CUSUM
    signals_upper: List[int]  # Indices where C+ > h
    signals_lower: List[int]  # Indices where C- > h
    n_signals: int = 0

    def __post_init__(self):
        self.n_signals = len(self.signals_upper) + len(self.signals_lower)


@dataclass
class EWMAResult:
    """EWMA control chart results."""
    parameter_name: str
    lambda_param: float  # Smoothing parameter (0 < lambda <= 1)
    ewma_values: np.ndarray
    center_line: float
    ucl: np.ndarray  # Time-varying UCL
    lcl: np.ndarray  # Time-varying LCL
    signals: List[int]  # Indices where EWMA exceeds limits
    n_signals: int = 0

    def __post_init__(self):
        self.n_signals = len(self.signals)


def create_cusum_chart(
    values: np.ndarray,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    k: Optional[float] = None,
    h: Optional[float] = None,
    parameter_name: str = '',
) -> CUSUMResult:
    """
    Create a tabular CUSUM (Page's procedure) control chart.

    The CUSUM chart accumulates deviations from a target value and is
    particularly effective at detecting small, sustained shifts in the
    process mean that Shewhart charts may miss.

    Two one-sided statistics are maintained:
        C_plus[i]  = max(0, C_plus[i-1]  + (x[i] - target - k))
        C_minus[i] = max(0, C_minus[i-1] + (target - k - x[i]))

    A signal is generated when C_plus > h (upward shift) or
    C_minus > h (downward shift).

    Args:
        values: Array of individual measurements (chronological order).
        target: Process target value. Defaults to the mean of values.
        sigma: Process standard deviation estimate. Defaults to
            mean(moving_range) / d2 (d2 = 1.128 for n=2).
        k: Allowable slack (reference value). Defaults to 0.5 * sigma,
            which is optimal for detecting a 1-sigma shift.
        h: Decision interval. Defaults to 5 * sigma, giving an
            in-control ARL of approximately 465.
        parameter_name: Name of the parameter being charted.

    Returns:
        CUSUMResult with upper/lower cumulative sums and signal indices.

    Raises:
        ValueError: If fewer than 2 data points are provided.

    Example:
        >>> vals = np.array([10.1, 10.0, 10.3, 10.2, 11.0, 11.1, 11.2])
        >>> result = create_cusum_chart(vals, target=10.0, parameter_name='pressure')
        >>> print(f"Signals detected: {result.n_signals}")
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 data points for CUSUM chart")

    # Default target: process mean
    if target is None:
        target = float(np.mean(values))

    # Default sigma: estimated from moving range
    if sigma is None:
        mr = np.abs(np.diff(values))
        mr_bar = np.mean(mr)
        d2 = CHART_CONSTANTS[2]['d2']
        sigma = mr_bar / d2

    # Protect against zero sigma (constant data)
    if sigma <= 0:
        sigma = 1.0

    # Default CUSUM parameters
    if k is None:
        k = 0.5 * sigma
    if h is None:
        h = 5.0 * sigma

    # Compute tabular CUSUM
    c_plus = np.zeros(n)
    c_minus = np.zeros(n)

    for i in range(n):
        if i == 0:
            c_plus[i] = max(0.0, values[i] - target - k)
            c_minus[i] = max(0.0, target - k - values[i])
        else:
            c_plus[i] = max(0.0, c_plus[i - 1] + (values[i] - target - k))
            c_minus[i] = max(0.0, c_minus[i - 1] + (target - k - values[i]))

    # Identify signal points
    signals_upper = [i for i in range(n) if c_plus[i] > h]
    signals_lower = [i for i in range(n) if c_minus[i] > h]

    return CUSUMResult(
        parameter_name=parameter_name,
        target=target,
        k=k,
        h=h,
        c_plus=c_plus,
        c_minus=c_minus,
        signals_upper=signals_upper,
        signals_lower=signals_lower,
    )


def create_ewma_chart(
    values: np.ndarray,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    lambda_param: float = 0.2,
    L: float = 3.0,
    parameter_name: str = '',
) -> EWMAResult:
    """
    Create an Exponentially Weighted Moving Average (EWMA) control chart.

    The EWMA chart applies geometrically decaying weights to past
    observations, making it sensitive to small and moderate shifts
    in the process mean while remaining robust to non-normality.

    The EWMA statistic is:
        z[i] = lambda * x[i] + (1 - lambda) * z[i-1],  z[0] = target

    Time-varying control limits are:
        UCL[i] = target + L * sigma * sqrt(lambda/(2-lambda) * (1-(1-lambda)^(2*i)))
        LCL[i] = target - L * sigma * sqrt(lambda/(2-lambda) * (1-(1-lambda)^(2*i)))

    As i -> infinity, the limits converge to steady-state values.

    Args:
        values: Array of individual measurements (chronological order).
        target: Process target value. Defaults to the mean of values.
        sigma: Process standard deviation estimate. Defaults to
            mean(moving_range) / d2 (d2 = 1.128 for n=2).
        lambda_param: Smoothing parameter (0 < lambda <= 1). Smaller values
            give more weight to historical data and detect smaller shifts.
            Default 0.2 is a common choice.
        L: Width of control limits in sigma units. Default 3.0.
        parameter_name: Name of the parameter being charted.

    Returns:
        EWMAResult with EWMA values, time-varying control limits, and
        signal indices.

    Raises:
        ValueError: If fewer than 2 data points are provided, or if
            lambda_param is not in (0, 1].

    Example:
        >>> vals = np.array([10.1, 10.0, 10.3, 10.2, 11.0, 11.1, 11.2])
        >>> result = create_ewma_chart(vals, target=10.0, parameter_name='pressure')
        >>> print(f"Signals detected: {result.n_signals}")
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 data points for EWMA chart")

    if not (0.0 < lambda_param <= 1.0):
        raise ValueError(
            f"lambda_param must be in (0, 1], got {lambda_param}"
        )

    # Default target: process mean
    if target is None:
        target = float(np.mean(values))

    # Default sigma: estimated from moving range
    if sigma is None:
        mr = np.abs(np.diff(values))
        mr_bar = np.mean(mr)
        d2 = CHART_CONSTANTS[2]['d2']
        sigma = mr_bar / d2

    # Protect against zero sigma (constant data)
    if sigma <= 0:
        sigma = 1.0

    # Compute EWMA statistic
    ewma_values = np.zeros(n)
    ewma_values[0] = lambda_param * values[0] + (1.0 - lambda_param) * target

    for i in range(1, n):
        ewma_values[i] = (
            lambda_param * values[i]
            + (1.0 - lambda_param) * ewma_values[i - 1]
        )

    # Compute time-varying control limits
    # Factor that grows with i: lambda/(2-lambda) * (1 - (1-lambda)^(2*i))
    i_indices = np.arange(1, n + 1)  # 1-based for the formula
    variance_factor = (
        (lambda_param / (2.0 - lambda_param))
        * (1.0 - (1.0 - lambda_param) ** (2 * i_indices))
    )
    limit_width = L * sigma * np.sqrt(variance_factor)

    ucl = target + limit_width
    lcl = target - limit_width

    # Identify signal points (EWMA outside control limits)
    signals = [
        i for i in range(n)
        if ewma_values[i] > ucl[i] or ewma_values[i] < lcl[i]
    ]

    return EWMAResult(
        parameter_name=parameter_name,
        lambda_param=lambda_param,
        ewma_values=ewma_values,
        center_line=target,
        ucl=ucl,
        lcl=lcl,
        signals=signals,
    )


# =============================================================================
# MAIN SPC FUNCTIONS
# =============================================================================

def create_imr_chart(
    df: pd.DataFrame,
    parameter: str,
    test_id_col: str = 'test_id',
    timestamp_col: Optional[str] = 'test_timestamp',
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    target: Optional[float] = None,
) -> SPCAnalysis:
    """
    Create Individual-Moving Range control chart from campaign data.
    
    Args:
        df: DataFrame with campaign test results
        parameter: Column name of parameter to analyze
        test_id_col: Column containing test IDs
        timestamp_col: Column containing timestamps
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value
        
    Returns:
        SPCAnalysis with complete results
    """
    if parameter not in df.columns:
        raise ValueError(f"Parameter '{parameter}' not found in data")
    
    # Get valid data
    valid_mask = df[parameter].notna()
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) < 2:
        raise ValueError(f"Need at least 2 valid points, got {len(df_valid)}")
    
    values = df_valid[parameter].values
    test_ids = df_valid[test_id_col].values
    timestamps = df_valid[timestamp_col].values if timestamp_col in df_valid.columns else [None] * len(values)
    
    # Calculate control limits
    i_limits, mr_limits = calculate_imr_limits(values)
    
    # Check Western Electric rules
    violations = check_western_electric_rules(values, i_limits)
    
    # Create points
    points = []
    for i, (val, test_id, ts, viols) in enumerate(zip(values, test_ids, timestamps, violations)):
        point = ControlChartPoint(
            index=i,
            value=float(val),
            test_id=str(test_id),
            timestamp=str(ts) if ts is not None else None,
            in_control=len(viols) == 0,
            violations=viols,
        )
        points.append(point)
    
    # Calculate capability if specs provided
    capability = None
    if usl is not None or lsl is not None:
        capability = calculate_capability(values, usl, lsl, target)
    
    # Detect trend
    has_trend, trend_direction, trend_slope = detect_trend(values)
    
    analysis = SPCAnalysis(
        parameter_name=parameter,
        chart_type=ControlChartType.I_MR,
        limits=i_limits,
        points=points,
        capability=capability,
        has_trend=has_trend,
        trend_direction=trend_direction,
        trend_slope=trend_slope,
    )
    
    return analysis


def analyze_campaign_spc(
    df: pd.DataFrame,
    parameters: List[str],
    specs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, SPCAnalysis]:
    """
    Run SPC analysis on multiple parameters from a campaign.
    
    Args:
        df: DataFrame with campaign test results
        parameters: List of parameter column names
        specs: Dict of parameter -> {'usl': ..., 'lsl': ..., 'target': ...}
        
    Returns:
        Dictionary mapping parameter names to SPCAnalysis
    """
    results = {}
    specs = specs or {}
    
    for param in parameters:
        try:
            param_specs = specs.get(param, {})
            analysis = create_imr_chart(
                df, param,
                usl=param_specs.get('usl'),
                lsl=param_specs.get('lsl'),
                target=param_specs.get('target'),
            )
            results[param] = analysis
        except Exception as e:
            print(f"Warning: SPC analysis failed for {param}: {e}")
    
    return results


def format_spc_summary(analysis: SPCAnalysis) -> str:
    """Format SPC analysis as markdown summary."""
    lines = [
        f"## SPC Analysis: {analysis.parameter_name}",
        "",
        f"**Chart Type:** {analysis.chart_type.value}",
        f"**Points Analyzed:** {analysis.n_points}",
        "",
        "### Control Limits",
        f"- Center Line: {analysis.limits.center_line:.4f}",
        f"- UCL (3σ): {analysis.limits.ucl:.4f}",
        f"- LCL (3σ): {analysis.limits.lcl:.4f}",
        "",
        "### Statistical Control",
        f"- Out of Control: {analysis.n_violations} points ({analysis.violation_rate*100:.1f}%)",
    ]
    
    if analysis.n_violations > 0:
        lines.append("- **Violations detected:**")
        for point in analysis.get_out_of_control_points():
            viols = ', '.join(v.value for v in point.violations)
            lines.append(f"  - {point.test_id}: {viols}")
    
    if analysis.has_trend:
        lines.extend([
            "",
            f"### [WARN] Trend Detected",
            f"- Direction: {analysis.trend_direction}",
            f"- Slope: {analysis.trend_slope:.6f} per test",
        ])
    
    if analysis.capability:
        lines.extend([
            "",
            "### Process Capability",
            analysis.capability.summary(),
        ])
        if analysis.capability.cpk is not None:
            lines.append(f"- Cpk: {analysis.capability.cpk:.2f}")
        if analysis.capability.ppk is not None:
            lines.append(f"- Ppk: {analysis.capability.ppk:.2f}")
    
    return "\n".join(lines)
