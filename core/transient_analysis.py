"""
Transient / Multi-Phase Test Analysis Engine
=============================================
Segments rocket engine test data into discrete phases and computes
phase-specific metrics for startup, steady-state, and shutdown transients.

Typical rocket engine test timeline:
    PRETEST -> STARTUP -> [TRANSIENT] -> STEADY_STATE -> SHUTDOWN -> COOLDOWN

Key capabilities:
- Automatic phase segmentation using derivative analysis, CV stability,
  and threshold crossings
- Startup transient characterization (rise time, overshoot, settling time)
- Shutdown transient characterization (decay time, tail-off impulse)
- Per-phase statistical metrics (mean, std, min, max, ramp rate)

Design Principles:
- No Streamlit imports (core module - business logic only)
- All data structures use dataclasses with type hints
- Time column in seconds; phase windows reported in milliseconds
- Pressure in bar, mass flow in g/s (standard HDA units)

Usage:
    from core.transient_analysis import (
        segment_test_phases,
        analyze_startup_transient,
        analyze_shutdown_transient,
        compute_phase_metrics,
        TestPhase,
    )

    # Segment a test into phases
    result = segment_test_phases(df, signal_col='PC-01', time_col='time_s')
    for phase in result.phases:
        print(f"{phase.phase.value}: {phase.start_ms:.0f} - {phase.end_ms:.0f} ms")

    # Analyze startup transient
    startup_df = df[(df['time_s'] >= t_start) & (df['time_s'] <= t_end)]
    metrics = analyze_startup_transient(startup_df, 'PC-01', steady_value=10.0)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import warnings


# =============================================================================
# Data Structures
# =============================================================================

class TestPhase(Enum):
    """
    Enumeration of discrete phases in a rocket engine test.

    Phase ordering follows the typical test timeline:
        PRETEST -> STARTUP -> TRANSIENT -> STEADY_STATE -> SHUTDOWN -> COOLDOWN

    Not all phases are present in every test. Short-duration tests may lack
    a distinct TRANSIENT phase, and some tests may not reach STEADY_STATE.
    """
    PRETEST = "pretest"
    STARTUP = "startup"
    TRANSIENT = "transient"
    STEADY_STATE = "steady_state"
    SHUTDOWN = "shutdown"
    COOLDOWN = "cooldown"


@dataclass
class PhaseResult:
    """
    Analysis result for a single test phase.

    Attributes:
        phase: The phase type (from TestPhase enum)
        start_ms: Phase start time in milliseconds
        end_ms: Phase end time in milliseconds
        duration_s: Phase duration in seconds
        metrics: Dictionary of computed metrics for this phase
            (keys depend on phase type, e.g., 'mean', 'std', 'ramp_rate')
        quality: Quality assessment string ('good', 'marginal', 'poor')
            based on data density and signal characteristics
    """
    phase: TestPhase
    start_ms: float
    end_ms: float
    duration_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality: str = "good"

    def __post_init__(self):
        """Validate phase window consistency."""
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"Phase {self.phase.value}: end_ms ({self.end_ms}) "
                f"must be >= start_ms ({self.start_ms})"
            )

    @property
    def window(self) -> Tuple[float, float]:
        """Return (start_ms, end_ms) tuple for convenience."""
        return (self.start_ms, self.end_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage or export."""
        return {
            'phase': self.phase.value,
            'start_ms': self.start_ms,
            'end_ms': self.end_ms,
            'duration_s': self.duration_s,
            'metrics': self.metrics,
            'quality': self.quality,
        }


@dataclass
class MultiPhaseResult:
    """
    Complete multi-phase segmentation result for an entire test.

    Attributes:
        phases: Ordered list of PhaseResult objects (in chronological order)
        total_duration_s: Total test duration in seconds
        transition_times_ms: List of phase transition times in milliseconds,
            representing the boundary between consecutive phases
        signal_col: The signal column used for segmentation
        segmentation_params: Parameters used for the segmentation algorithm
    """
    phases: List[PhaseResult]
    total_duration_s: float
    transition_times_ms: List[float] = field(default_factory=list)
    signal_col: str = ""
    segmentation_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_phases(self) -> int:
        """Number of detected phases."""
        return len(self.phases)

    def get_phase(self, phase_type: TestPhase) -> Optional[PhaseResult]:
        """
        Get the first phase matching the given type.

        Args:
            phase_type: The TestPhase to search for

        Returns:
            PhaseResult if found, None otherwise
        """
        for phase in self.phases:
            if phase.phase == phase_type:
                return phase
        return None

    def get_phases(self, phase_type: TestPhase) -> List[PhaseResult]:
        """
        Get all phases matching the given type.

        Some tests may have multiple transient phases (e.g., step tests).

        Args:
            phase_type: The TestPhase to search for

        Returns:
            List of matching PhaseResult objects
        """
        return [p for p in self.phases if p.phase == phase_type]

    def has_phase(self, phase_type: TestPhase) -> bool:
        """Check if a specific phase type was detected."""
        return any(p.phase == phase_type for p in self.phases)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage or export."""
        return {
            'phases': [p.to_dict() for p in self.phases],
            'total_duration_s': self.total_duration_s,
            'transition_times_ms': self.transition_times_ms,
            'signal_col': self.signal_col,
            'n_phases': self.n_phases,
        }


# =============================================================================
# Phase Segmentation
# =============================================================================

def segment_test_phases(
    df: pd.DataFrame,
    signal_col: str,
    time_col: str = 'time_s',
    threshold_pct: float = 10.0,
    cv_threshold: float = 0.02,
    min_phase_duration_s: float = 0.1,
) -> MultiPhaseResult:
    """
    Segment a test into discrete phases using signal analysis.

    Uses a combination of three detection methods:
    1. **Derivative analysis**: Detects rapid signal changes (startup/shutdown)
       by computing the smoothed absolute derivative and comparing against
       a dynamic threshold.
    2. **CV-based stability**: Identifies steady-state regions where the
       rolling coefficient of variation falls below cv_threshold.
    3. **Threshold crossings**: Determines when the signal crosses above/below
       a percentage of its operating range to mark ignition and shutdown events.

    Algorithm overview:
        1. Normalize signal to [0, 100] percent of its range
        2. Find "active" region where signal exceeds threshold_pct
        3. Within the active region, detect steady-state using rolling CV
        4. Classify pre-active as PRETEST, post-active as COOLDOWN
        5. Classify rapid rise before steady-state as STARTUP
        6. Classify rapid decline after steady-state as SHUTDOWN
        7. Any active region that is neither startup, steady, nor shutdown
           is labeled TRANSIENT

    Phases shorter than min_phase_duration_s are merged into adjacent phases
    to avoid spurious micro-phases from sensor noise.

    Args:
        df: DataFrame with time-series sensor data. Must contain at least
            the signal_col and time_col columns.
        signal_col: Column name of the primary signal to segment on
            (e.g., chamber pressure 'PC-01' or upstream pressure 'PT-01')
        time_col: Column name for time in seconds (default: 'time_s')
        threshold_pct: Percentage of signal range that defines the boundary
            between inactive (pretest/cooldown) and active (startup through
            shutdown) regions. Range: 1.0 to 50.0. Default: 10.0
        cv_threshold: Maximum coefficient of variation (std/mean) for a
            region to be considered steady state. Lower values require
            more stability. Default: 0.02 (2%)
        min_phase_duration_s: Minimum duration in seconds for a phase to
            be retained. Phases shorter than this are merged into their
            neighbors. Default: 0.1

    Returns:
        MultiPhaseResult containing the ordered list of detected phases,
        overall test duration, and transition times.

    Raises:
        ValueError: If signal_col or time_col is not in the DataFrame,
            or if the DataFrame has fewer than 10 rows.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 1000)
        >>> signal = np.concatenate([
        ...     np.zeros(100),           # pretest
        ...     np.linspace(0, 10, 100), # startup
        ...     10 + 0.1*np.random.randn(500),  # steady state
        ...     np.linspace(10, 0, 100), # shutdown
        ...     np.zeros(200),           # cooldown
        ... ])
        >>> df = pd.DataFrame({'time_s': t, 'PC-01': signal})
        >>> result = segment_test_phases(df, 'PC-01')
        >>> for phase in result.phases:
        ...     print(f"{phase.phase.value}: {phase.duration_s:.2f}s")
    """
    # --- Input validation ---
    if signal_col not in df.columns:
        raise ValueError(
            f"Signal column '{signal_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if time_col not in df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if len(df) < 10:
        raise ValueError(
            f"DataFrame has only {len(df)} rows. "
            f"Need at least 10 rows for phase segmentation."
        )

    times = df[time_col].values.astype(float)
    signal = df[signal_col].values.astype(float)
    n = len(times)

    total_duration_s = float(times[-1] - times[0])

    # Handle all-NaN signals
    if np.all(np.isnan(signal)):
        single_phase = PhaseResult(
            phase=TestPhase.PRETEST,
            start_ms=float(times[0]) * 1000.0,
            end_ms=float(times[-1]) * 1000.0,
            duration_s=total_duration_s,
            metrics={},
            quality='poor',
        )
        return MultiPhaseResult(
            phases=[single_phase],
            total_duration_s=total_duration_s,
            transition_times_ms=[],
            signal_col=signal_col,
            segmentation_params={
                'threshold_pct': threshold_pct,
                'cv_threshold': cv_threshold,
                'min_phase_duration_s': min_phase_duration_s,
            },
        )

    # Interpolate NaNs for analysis (don't modify original df)
    signal_clean = pd.Series(signal).interpolate(method='linear').bfill().ffill().values

    # --- Step 1: Normalize signal to percentage of range ---
    sig_min = float(np.nanmin(signal_clean))
    sig_max = float(np.nanmax(signal_clean))
    sig_range = sig_max - sig_min

    if sig_range < 1e-12:
        # Flat signal -- entire test is one phase
        single_phase = PhaseResult(
            phase=TestPhase.STEADY_STATE,
            start_ms=float(times[0]) * 1000.0,
            end_ms=float(times[-1]) * 1000.0,
            duration_s=total_duration_s,
            metrics=_compute_basic_stats(signal_clean, times),
            quality='good',
        )
        return MultiPhaseResult(
            phases=[single_phase],
            total_duration_s=total_duration_s,
            transition_times_ms=[],
            signal_col=signal_col,
            segmentation_params={
                'threshold_pct': threshold_pct,
                'cv_threshold': cv_threshold,
                'min_phase_duration_s': min_phase_duration_s,
            },
        )

    signal_pct = (signal_clean - sig_min) / sig_range * 100.0

    # --- Step 2: Find "active" region (signal above threshold) ---
    active_mask = signal_pct >= threshold_pct

    if not np.any(active_mask):
        # Signal never exceeds threshold -- everything is PRETEST
        single_phase = PhaseResult(
            phase=TestPhase.PRETEST,
            start_ms=float(times[0]) * 1000.0,
            end_ms=float(times[-1]) * 1000.0,
            duration_s=total_duration_s,
            metrics=_compute_basic_stats(signal_clean, times),
            quality='good',
        )
        return MultiPhaseResult(
            phases=[single_phase],
            total_duration_s=total_duration_s,
            transition_times_ms=[],
            signal_col=signal_col,
            segmentation_params={
                'threshold_pct': threshold_pct,
                'cv_threshold': cv_threshold,
                'min_phase_duration_s': min_phase_duration_s,
            },
        )

    active_indices = np.where(active_mask)[0]
    active_start_idx = int(active_indices[0])
    active_end_idx = int(active_indices[-1])

    # --- Step 3: Detect steady-state region using rolling CV ---
    window_size = max(10, n // 50)  # Adaptive window: ~2% of data
    rolling_mean = pd.Series(signal_clean).rolling(
        window=window_size, center=True, min_periods=max(3, window_size // 3)
    ).mean()
    rolling_std = pd.Series(signal_clean).rolling(
        window=window_size, center=True, min_periods=max(3, window_size // 3)
    ).std()

    with np.errstate(divide='ignore', invalid='ignore'):
        rolling_cv = (rolling_std / rolling_mean.abs()).fillna(1.0).values

    # Steady-state: CV below threshold AND signal is active
    steady_mask = (rolling_cv < cv_threshold) & active_mask

    # --- Step 4: Compute smoothed derivative for transient detection ---
    dt = np.diff(times)
    dt[dt == 0] = 1e-10  # Prevent division by zero
    raw_deriv = np.diff(signal_clean) / dt

    # Smooth the derivative (adaptive window)
    smooth_window = max(5, n // 100)
    smoothed_deriv = pd.Series(raw_deriv).rolling(
        window=smooth_window, center=True, min_periods=1
    ).mean().values

    # Pad derivative to match signal length
    smoothed_deriv = np.concatenate([smoothed_deriv, [smoothed_deriv[-1]]])
    abs_deriv = np.abs(smoothed_deriv)

    # Dynamic derivative threshold: median of active-region derivative * factor
    active_deriv = abs_deriv[active_start_idx:active_end_idx + 1]
    if len(active_deriv) > 0:
        deriv_threshold = float(np.median(active_deriv)) + 2.0 * float(np.std(active_deriv))
        deriv_threshold = max(deriv_threshold, sig_range * 0.01)  # At least 1%/sample of range
    else:
        deriv_threshold = sig_range * 0.05

    high_deriv_mask = abs_deriv > deriv_threshold

    # --- Step 5: Classify each sample into a phase ---
    phase_labels = np.full(n, TestPhase.TRANSIENT, dtype=object)

    # Pre-active region -> PRETEST
    if active_start_idx > 0:
        phase_labels[:active_start_idx] = TestPhase.PRETEST

    # Post-active region -> COOLDOWN
    if active_end_idx < n - 1:
        phase_labels[active_end_idx + 1:] = TestPhase.COOLDOWN

    # Steady-state samples
    phase_labels[steady_mask] = TestPhase.STEADY_STATE

    # Find the boundaries of the steady-state block
    steady_indices = np.where(steady_mask)[0]
    if len(steady_indices) > 0:
        # Find the longest contiguous steady-state block
        ss_start_idx, ss_end_idx = _find_longest_contiguous(steady_indices)

        # STARTUP: active region before steady-state with rising signal
        startup_region = np.arange(active_start_idx, ss_start_idx)
        if len(startup_region) > 0:
            phase_labels[startup_region] = TestPhase.STARTUP

        # SHUTDOWN: active region after steady-state with falling signal
        shutdown_region = np.arange(ss_end_idx + 1, active_end_idx + 1)
        if len(shutdown_region) > 0:
            phase_labels[shutdown_region] = TestPhase.SHUTDOWN

        # Mark the steady-state block cleanly
        phase_labels[ss_start_idx:ss_end_idx + 1] = TestPhase.STEADY_STATE

        # Re-check: if a supposed STARTUP region has significant steady
        # sub-regions or vice versa, mark the intermediate part as TRANSIENT
        for idx in startup_region:
            if not active_mask[idx]:
                phase_labels[idx] = TestPhase.PRETEST

        for idx in shutdown_region:
            if not active_mask[idx]:
                phase_labels[idx] = TestPhase.COOLDOWN
    else:
        # No steady-state detected: use derivative to split startup/shutdown
        # Find the peak of the signal in the active region
        active_signal = signal_clean[active_start_idx:active_end_idx + 1]
        peak_local_idx = int(np.argmax(active_signal))
        peak_idx = active_start_idx + peak_local_idx

        # Before peak -> STARTUP, after peak -> SHUTDOWN
        phase_labels[active_start_idx:peak_idx + 1] = TestPhase.STARTUP
        if peak_idx + 1 <= active_end_idx:
            phase_labels[peak_idx + 1:active_end_idx + 1] = TestPhase.SHUTDOWN

    # --- Step 6: Build phase list from contiguous label runs ---
    raw_phases = _labels_to_phases(phase_labels, times, signal_clean)

    # --- Step 7: Merge short phases ---
    merged_phases = _merge_short_phases(raw_phases, min_phase_duration_s)

    # --- Step 8: Compute metrics for each phase ---
    for phase_result in merged_phases:
        t_start_s = phase_result.start_ms / 1000.0
        t_end_s = phase_result.end_ms / 1000.0
        mask = (times >= t_start_s) & (times <= t_end_s)
        phase_signal = signal_clean[mask]
        phase_times = times[mask]

        if len(phase_signal) > 0:
            phase_result.metrics = compute_phase_metrics(
                phase_signal, phase_times, phase_result.phase
            )

        # Assess quality based on sample count
        n_samples = int(np.sum(mask))
        if n_samples < 5:
            phase_result.quality = 'poor'
        elif n_samples < 20:
            phase_result.quality = 'marginal'
        else:
            phase_result.quality = 'good'

    # --- Step 9: Compute transition times ---
    transition_times_ms: List[float] = []
    for i in range(len(merged_phases) - 1):
        transition_times_ms.append(merged_phases[i].end_ms)

    return MultiPhaseResult(
        phases=merged_phases,
        total_duration_s=total_duration_s,
        transition_times_ms=transition_times_ms,
        signal_col=signal_col,
        segmentation_params={
            'threshold_pct': threshold_pct,
            'cv_threshold': cv_threshold,
            'min_phase_duration_s': min_phase_duration_s,
        },
    )


# =============================================================================
# Startup Transient Analysis
# =============================================================================

def analyze_startup_transient(
    df: pd.DataFrame,
    signal_col: str,
    time_col: str = 'time_s',
    steady_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute startup transient metrics from a signal trace.

    Characterizes the startup phase by computing standard transient
    response metrics commonly used in propulsion test analysis.

    Metrics computed:
    - **rise_time_s**: Time for signal to go from 10% to 90% of steady value
    - **rise_time_10_50_s**: Time from 10% to 50% (early rise characterization)
    - **rise_time_50_90_s**: Time from 50% to 90% (late rise characterization)
    - **overshoot_pct**: Maximum overshoot as percentage of steady value.
      0.0 if the signal never exceeds steady_value.
    - **peak_value**: Maximum signal value during startup
    - **time_to_peak_s**: Time from start of data to peak value
    - **settling_time_s**: Time for signal to enter and remain within a
      +/-5% band around steady_value. None if the signal never settles.
    - **steady_value**: The steady-state reference value used for calculations

    If steady_value is not provided, it is estimated as the mean of the
    last 20% of the provided data (assuming the data ends near steady state).

    Args:
        df: DataFrame containing the startup region data. Should start
            near the beginning of the startup and extend through early
            steady state if possible.
        signal_col: Column name of the signal to analyze
            (e.g., chamber pressure 'PC-01')
        time_col: Column name for time in seconds (default: 'time_s')
        steady_value: Known steady-state value for reference. If None,
            estimated from the tail of the data.

    Returns:
        Dictionary of startup transient metrics. All time values are
        in seconds. Returns empty dict if insufficient data.

    Example:
        >>> startup_df = df[(df['time_s'] >= 1.0) & (df['time_s'] <= 3.0)]
        >>> metrics = analyze_startup_transient(startup_df, 'PC-01', steady_value=10.0)
        >>> print(f"Rise time: {metrics['rise_time_s']:.3f} s")
        >>> print(f"Overshoot: {metrics['overshoot_pct']:.1f}%")
    """
    if signal_col not in df.columns or time_col not in df.columns:
        return {}

    times = df[time_col].values.astype(float)
    signal = df[signal_col].values.astype(float)

    # Remove NaNs
    valid = ~(np.isnan(times) | np.isnan(signal))
    times = times[valid]
    signal = signal[valid]

    if len(signal) < 3:
        return {}

    t0 = float(times[0])

    # Estimate steady value if not provided
    if steady_value is None:
        tail_count = max(1, len(signal) // 5)
        steady_value = float(np.mean(signal[-tail_count:]))

    # Baseline value (start of data)
    head_count = max(1, len(signal) // 10)
    baseline = float(np.mean(signal[:head_count]))

    # Effective range for threshold calculations
    value_range = steady_value - baseline
    if abs(value_range) < 1e-12:
        # No appreciable signal change
        return {
            'rise_time_s': 0.0,
            'rise_time_10_50_s': 0.0,
            'rise_time_50_90_s': 0.0,
            'overshoot_pct': 0.0,
            'peak_value': float(np.max(signal)),
            'time_to_peak_s': 0.0,
            'settling_time_s': None,
            'steady_value': steady_value,
        }

    # Threshold levels (relative to baseline -> steady_value)
    level_10 = baseline + 0.10 * value_range
    level_50 = baseline + 0.50 * value_range
    level_90 = baseline + 0.90 * value_range

    # Find crossing times using linear interpolation
    t_10 = _find_crossing_time(times, signal, level_10, direction='rising')
    t_50 = _find_crossing_time(times, signal, level_50, direction='rising')
    t_90 = _find_crossing_time(times, signal, level_90, direction='rising')

    # Rise times
    rise_time_s = None
    if t_10 is not None and t_90 is not None:
        rise_time_s = t_90 - t_10

    rise_time_10_50_s = None
    if t_10 is not None and t_50 is not None:
        rise_time_10_50_s = t_50 - t_10

    rise_time_50_90_s = None
    if t_50 is not None and t_90 is not None:
        rise_time_50_90_s = t_90 - t_50

    # Peak and overshoot
    peak_idx = int(np.argmax(signal))
    peak_value = float(signal[peak_idx])
    time_to_peak_s = float(times[peak_idx] - t0)

    overshoot_pct = 0.0
    if abs(steady_value) > 1e-12 and peak_value > steady_value:
        overshoot_pct = (peak_value - steady_value) / abs(steady_value) * 100.0

    # Settling time: time for signal to enter and stay within +/-5% band
    settling_band = 0.05 * abs(steady_value) if abs(steady_value) > 1e-12 else 0.05 * abs(value_range)
    settling_time_s = _compute_settling_time(
        times, signal, steady_value, settling_band, t0
    )

    return {
        'rise_time_s': rise_time_s,
        'rise_time_10_50_s': rise_time_10_50_s,
        'rise_time_50_90_s': rise_time_50_90_s,
        'overshoot_pct': overshoot_pct,
        'peak_value': peak_value,
        'time_to_peak_s': time_to_peak_s,
        'settling_time_s': settling_time_s,
        'steady_value': steady_value,
    }


# =============================================================================
# Shutdown Transient Analysis
# =============================================================================

def analyze_shutdown_transient(
    df: pd.DataFrame,
    signal_col: str,
    time_col: str = 'time_s',
    steady_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute shutdown transient metrics from a signal trace.

    Characterizes the shutdown phase by computing standard transient
    response metrics for the signal decay.

    Metrics computed:
    - **decay_time_s**: Time for signal to fall from 90% to 10% of
      the steady-state value (measured from baseline)
    - **decay_time_90_50_s**: Time from 90% to 50% (early decay)
    - **decay_time_50_10_s**: Time from 50% to 10% (late decay)
    - **tail_off_impulse**: Integral of the signal from shutdown start
      to the end of the data. For thrust signals, this gives the tail-off
      impulse in N*s; for pressure signals, it gives the pressure-time
      integral in bar*s.
    - **min_value**: Minimum signal value during shutdown
    - **time_to_min_s**: Time from start of data to minimum value
    - **residual_pct**: Final signal level as a percentage of steady value.
      Non-zero residual may indicate incomplete shutdown or sensor offset.
    - **steady_value**: The steady-state reference value used

    If steady_value is not provided, it is estimated as the mean of the
    first 20% of the provided data (assuming the data starts at steady state).

    Args:
        df: DataFrame containing the shutdown region data. Should start
            near the end of steady state and extend through cooldown.
        signal_col: Column name of the signal to analyze
        time_col: Column name for time in seconds (default: 'time_s')
        steady_value: Known steady-state value for reference. If None,
            estimated from the head of the data.

    Returns:
        Dictionary of shutdown transient metrics. All time values are
        in seconds. Returns empty dict if insufficient data.

    Example:
        >>> shutdown_df = df[(df['time_s'] >= 8.0) & (df['time_s'] <= 10.0)]
        >>> metrics = analyze_shutdown_transient(shutdown_df, 'PC-01', steady_value=10.0)
        >>> print(f"Decay time: {metrics['decay_time_s']:.3f} s")
        >>> print(f"Tail-off impulse: {metrics['tail_off_impulse']:.2f} bar*s")
    """
    if signal_col not in df.columns or time_col not in df.columns:
        return {}

    times = df[time_col].values.astype(float)
    signal = df[signal_col].values.astype(float)

    # Remove NaNs
    valid = ~(np.isnan(times) | np.isnan(signal))
    times = times[valid]
    signal = signal[valid]

    if len(signal) < 3:
        return {}

    t0 = float(times[0])

    # Estimate steady value if not provided (from head of shutdown data)
    if steady_value is None:
        head_count = max(1, len(signal) // 5)
        steady_value = float(np.mean(signal[:head_count]))

    # Baseline value (end of data, should be near zero/ambient)
    tail_count = max(1, len(signal) // 10)
    baseline = float(np.mean(signal[-tail_count:]))

    # Effective range for threshold calculations
    value_range = steady_value - baseline
    if abs(value_range) < 1e-12:
        return {
            'decay_time_s': 0.0,
            'decay_time_90_50_s': 0.0,
            'decay_time_50_10_s': 0.0,
            'tail_off_impulse': 0.0,
            'min_value': float(np.min(signal)),
            'time_to_min_s': 0.0,
            'residual_pct': 0.0,
            'steady_value': steady_value,
        }

    # Threshold levels (relative to baseline -> steady_value)
    level_90 = baseline + 0.90 * value_range
    level_50 = baseline + 0.50 * value_range
    level_10 = baseline + 0.10 * value_range

    # Find crossing times for falling signal
    t_90 = _find_crossing_time(times, signal, level_90, direction='falling')
    t_50 = _find_crossing_time(times, signal, level_50, direction='falling')
    t_10 = _find_crossing_time(times, signal, level_10, direction='falling')

    # Decay times
    decay_time_s = None
    if t_90 is not None and t_10 is not None:
        decay_time_s = t_10 - t_90

    decay_time_90_50_s = None
    if t_90 is not None and t_50 is not None:
        decay_time_90_50_s = t_50 - t_90

    decay_time_50_10_s = None
    if t_50 is not None and t_10 is not None:
        decay_time_50_10_s = t_10 - t_50

    # Tail-off impulse: integral of signal over the shutdown region
    # Use trapezoidal integration (numpy >=2.0 renamed trapz -> trapezoid)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    if _trapz is not None:
        tail_off_impulse = float(_trapz(signal, times))
    else:
        # Manual trapezoidal fallback
        tail_off_impulse = float(np.sum(
            0.5 * (signal[:-1] + signal[1:]) * np.diff(times)
        ))

    # Minimum value
    min_idx = int(np.argmin(signal))
    min_value = float(signal[min_idx])
    time_to_min_s = float(times[min_idx] - t0)

    # Residual: final value as percentage of steady value
    residual_pct = 0.0
    if abs(steady_value) > 1e-12:
        final_value = float(signal[-1])
        residual_pct = final_value / abs(steady_value) * 100.0

    return {
        'decay_time_s': decay_time_s,
        'decay_time_90_50_s': decay_time_90_50_s,
        'decay_time_50_10_s': decay_time_50_10_s,
        'tail_off_impulse': tail_off_impulse,
        'min_value': min_value,
        'time_to_min_s': time_to_min_s,
        'residual_pct': residual_pct,
        'steady_value': steady_value,
    }


# =============================================================================
# Per-Phase Metrics
# =============================================================================

def compute_phase_metrics(
    signal: np.ndarray,
    times: np.ndarray,
    phase_type: TestPhase,
) -> Dict[str, Any]:
    """
    Compute relevant statistics for a single phase.

    The set of metrics varies by phase type:
    - All phases: mean, std, min, max, n_samples
    - STARTUP / SHUTDOWN / TRANSIENT: ramp_rate (average rate of change),
      peak_ramp_rate, total_change
    - STEADY_STATE: cv (coefficient of variation), stability_pct
    - PRETEST / COOLDOWN: drift_rate (slow signal drift)

    Args:
        signal: 1-D numpy array of signal values for this phase
        times: 1-D numpy array of corresponding time values (seconds)
        phase_type: The TestPhase enum value for context-aware metric selection

    Returns:
        Dictionary of computed metrics. Keys and values depend on phase_type.
        Returns empty dict if signal is empty.

    Example:
        >>> signal = np.array([0.1, 5.0, 9.8, 10.2, 10.0])
        >>> times = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        >>> metrics = compute_phase_metrics(signal, times, TestPhase.STARTUP)
        >>> print(f"Ramp rate: {metrics['ramp_rate']:.2f} units/s")
    """
    if len(signal) == 0:
        return {}

    metrics: Dict[str, Any] = {}

    # --- Basic statistics (all phases) ---
    metrics['mean'] = float(np.nanmean(signal))
    metrics['std'] = float(np.nanstd(signal, ddof=1)) if len(signal) > 1 else 0.0
    metrics['min'] = float(np.nanmin(signal))
    metrics['max'] = float(np.nanmax(signal))
    metrics['n_samples'] = int(len(signal))

    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    metrics['duration_s'] = duration

    # --- Phase-specific metrics ---
    if phase_type in (TestPhase.STARTUP, TestPhase.SHUTDOWN, TestPhase.TRANSIENT):
        # Rate of change metrics
        total_change = float(signal[-1] - signal[0])
        metrics['total_change'] = total_change

        if duration > 1e-12:
            metrics['ramp_rate'] = total_change / duration
        else:
            metrics['ramp_rate'] = 0.0

        # Peak instantaneous ramp rate
        if len(signal) > 1:
            dt = np.diff(times)
            dt[dt == 0] = 1e-10
            inst_rates = np.diff(signal) / dt
            metrics['peak_ramp_rate'] = float(np.max(np.abs(inst_rates)))
        else:
            metrics['peak_ramp_rate'] = 0.0

    elif phase_type == TestPhase.STEADY_STATE:
        # Stability metrics
        mean_val = metrics['mean']
        if abs(mean_val) > 1e-12:
            metrics['cv'] = metrics['std'] / abs(mean_val)
        else:
            metrics['cv'] = 0.0

        # Stability percentage: fraction of samples within +/-2% of mean
        if abs(mean_val) > 1e-12:
            band = 0.02 * abs(mean_val)
            within_band = np.sum(np.abs(signal - mean_val) <= band)
            metrics['stability_pct'] = float(within_band) / len(signal) * 100.0
        else:
            metrics['stability_pct'] = 100.0

    elif phase_type in (TestPhase.PRETEST, TestPhase.COOLDOWN):
        # Drift rate (slow change over time)
        if duration > 1e-12 and len(signal) > 1:
            # Linear fit to detect drift
            coeffs = np.polyfit(times - times[0], signal, 1)
            metrics['drift_rate'] = float(coeffs[0])  # units/s
        else:
            metrics['drift_rate'] = 0.0

    return metrics


# =============================================================================
# Private Helper Functions
# =============================================================================

def _find_crossing_time(
    times: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    direction: str = 'rising',
) -> Optional[float]:
    """
    Find the time at which the signal crosses a threshold level.

    Uses linear interpolation between samples for sub-sample accuracy.

    Args:
        times: Time array in seconds
        signal: Signal array
        threshold: Threshold level to detect crossing of
        direction: 'rising' for low-to-high crossing,
                   'falling' for high-to-low crossing

    Returns:
        Interpolated crossing time in seconds, or None if no crossing found
    """
    if len(signal) < 2:
        return None

    if direction == 'rising':
        # Find first index where signal goes from below to above threshold
        for i in range(len(signal) - 1):
            if signal[i] <= threshold < signal[i + 1]:
                # Linear interpolation
                frac = (threshold - signal[i]) / (signal[i + 1] - signal[i])
                return float(times[i] + frac * (times[i + 1] - times[i]))
    elif direction == 'falling':
        # Find first index where signal goes from above to below threshold
        for i in range(len(signal) - 1):
            if signal[i] >= threshold > signal[i + 1]:
                frac = (signal[i] - threshold) / (signal[i] - signal[i + 1])
                return float(times[i] + frac * (times[i + 1] - times[i]))

    return None


def _compute_settling_time(
    times: np.ndarray,
    signal: np.ndarray,
    target: float,
    band: float,
    t_reference: float,
) -> Optional[float]:
    """
    Compute settling time: time for signal to enter and remain within a band.

    Scans from the end of the data backward to find the last excursion
    outside the band, then reports the time from t_reference to that point.

    Args:
        times: Time array in seconds
        signal: Signal array
        target: Target value (center of the settling band)
        band: Half-width of the settling band (absolute)
        t_reference: Reference time (start of startup) for computing offset

    Returns:
        Settling time in seconds from t_reference, or None if signal
        never stays within the band
    """
    if len(signal) < 2 or band <= 0:
        return None

    within_band = np.abs(signal - target) <= band

    if not np.any(within_band):
        return None

    # Find the last point outside the band
    outside_indices = np.where(~within_band)[0]

    if len(outside_indices) == 0:
        # Signal is always within band
        return 0.0

    last_outside_idx = int(outside_indices[-1])

    # Settling time is the time just after the last excursion
    if last_outside_idx + 1 < len(times):
        return float(times[last_outside_idx + 1] - t_reference)
    else:
        # Never fully settled within the data
        return None


def _find_longest_contiguous(indices: np.ndarray) -> Tuple[int, int]:
    """
    Find the start and end values of the longest contiguous run in an index array.

    Args:
        indices: Sorted array of integer indices

    Returns:
        Tuple of (start_index, end_index) of the longest contiguous block
    """
    if len(indices) == 0:
        return (0, 0)

    if len(indices) == 1:
        return (int(indices[0]), int(indices[0]))

    # Find breaks in continuity
    diffs = np.diff(indices)
    breaks = np.where(diffs > 1)[0]

    if len(breaks) == 0:
        # All contiguous
        return (int(indices[0]), int(indices[-1]))

    # Build segments
    seg_starts = [0] + [b + 1 for b in breaks]
    seg_ends = [b for b in breaks] + [len(indices) - 1]

    # Find longest
    best_len = 0
    best_start = 0
    best_end = 0
    for s, e in zip(seg_starts, seg_ends):
        seg_len = e - s + 1
        if seg_len > best_len:
            best_len = seg_len
            best_start = s
            best_end = e

    return (int(indices[best_start]), int(indices[best_end]))


def _labels_to_phases(
    labels: np.ndarray,
    times: np.ndarray,
    signal: np.ndarray,
) -> List[PhaseResult]:
    """
    Convert a per-sample label array into a list of PhaseResult objects.

    Groups contiguous samples with the same label into phases.

    Args:
        labels: Array of TestPhase values, one per sample
        times: Time array in seconds
        signal: Signal array (for metrics computation)

    Returns:
        Ordered list of PhaseResult objects
    """
    if len(labels) == 0:
        return []

    phases: List[PhaseResult] = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # End of current phase
            t_start_s = float(times[start_idx])
            t_end_s = float(times[i - 1])
            duration_s = t_end_s - t_start_s

            phases.append(PhaseResult(
                phase=current_label,
                start_ms=t_start_s * 1000.0,
                end_ms=t_end_s * 1000.0,
                duration_s=duration_s,
            ))

            current_label = labels[i]
            start_idx = i

    # Final phase
    t_start_s = float(times[start_idx])
    t_end_s = float(times[-1])
    duration_s = t_end_s - t_start_s

    phases.append(PhaseResult(
        phase=current_label,
        start_ms=t_start_s * 1000.0,
        end_ms=t_end_s * 1000.0,
        duration_s=duration_s,
    ))

    return phases


def _merge_short_phases(
    phases: List[PhaseResult],
    min_duration_s: float,
) -> List[PhaseResult]:
    """
    Merge phases shorter than min_duration_s into adjacent phases.

    Short phases are merged into their predecessor if one exists, or
    their successor otherwise. This prevents spurious micro-phases
    caused by sensor noise near phase boundaries.

    Args:
        phases: Ordered list of PhaseResult objects
        min_duration_s: Minimum allowed phase duration in seconds

    Returns:
        New list of PhaseResult objects with short phases removed
    """
    if len(phases) <= 1:
        return phases

    merged: List[PhaseResult] = [phases[0]]

    for i in range(1, len(phases)):
        current = phases[i]

        if current.duration_s < min_duration_s and len(merged) > 0:
            # Merge into the previous phase by extending its end time
            prev = merged[-1]
            merged[-1] = PhaseResult(
                phase=prev.phase,
                start_ms=prev.start_ms,
                end_ms=current.end_ms,
                duration_s=(current.end_ms - prev.start_ms) / 1000.0,
                metrics=prev.metrics,
                quality=prev.quality,
            )
        else:
            merged.append(current)

    return merged


def _compute_basic_stats(
    signal: np.ndarray,
    times: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute basic statistics for a signal segment.

    Args:
        signal: Signal values
        times: Corresponding time values

    Returns:
        Dictionary with mean, std, min, max, n_samples, duration_s
    """
    if len(signal) == 0:
        return {}

    return {
        'mean': float(np.nanmean(signal)),
        'std': float(np.nanstd(signal, ddof=1)) if len(signal) > 1 else 0.0,
        'min': float(np.nanmin(signal)),
        'max': float(np.nanmax(signal)),
        'n_samples': int(len(signal)),
        'duration_s': float(times[-1] - times[0]) if len(times) > 1 else 0.0,
    }
