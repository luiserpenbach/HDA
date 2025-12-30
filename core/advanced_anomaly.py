"""
Advanced Anomaly Detection (P2)
===============================
Enhanced anomaly detection capabilities:
- Multi-variate anomaly detection
- Sensor health monitoring
- Transient event detection
- Anomaly classification and severity scoring
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    SPIKE = "spike"
    DROPOUT = "dropout"
    DRIFT = "drift"
    OSCILLATION = "oscillation"
    SATURATION = "saturation"
    FLATLINE = "flatline"
    CORRELATION_BREAK = "correlation_break"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    channel: str
    start_index: int
    end_index: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    score: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_samples(self) -> int:
        return self.end_index - self.start_index + 1
    
    @property
    def duration_time(self) -> Optional[float]:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class AnomalyReport:
    """Complete anomaly detection report."""
    channel_reports: Dict[str, List[Anomaly]]
    multivariate_anomalies: List[Anomaly]
    sensor_health: Dict[str, float]  # Channel -> health score (0-1)
    total_anomalies: int
    critical_count: int
    warning_count: int
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0
    
    def get_all_anomalies(self) -> List[Anomaly]:
        """Get all anomalies flattened."""
        all_anomalies = []
        for anomalies in self.channel_reports.values():
            all_anomalies.extend(anomalies)
        all_anomalies.extend(self.multivariate_anomalies)
        return all_anomalies
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "ANOMALY DETECTION REPORT",
            "=" * 60,
            f"Total Anomalies: {self.total_anomalies}",
            f"  Critical: {self.critical_count}",
            f"  Warning: {self.warning_count}",
            f"  Info: {self.total_anomalies - self.critical_count - self.warning_count}",
            "",
            "Sensor Health Scores:",
        ]
        
        for channel, health in sorted(self.sensor_health.items()):
            status = "" if health > 0.9 else ("[WARN]" if health > 0.7 else "")
            lines.append(f"  {status} {channel}: {health:.1%}")
        
        if self.has_critical:
            lines.extend(["", "CRITICAL ANOMALIES:"])
            for anomaly in self.get_all_anomalies():
                if anomaly.severity == AnomalySeverity.CRITICAL:
                    lines.append(f"  - {anomaly.channel}: {anomaly.anomaly_type.value} "
                               f"at indices {anomaly.start_index}-{anomaly.end_index}")
        
        return "\n".join(lines)


# =============================================================================
# SPIKE DETECTION
# =============================================================================

def detect_spikes(
    data: np.ndarray,
    threshold_sigma: float = 4.0,
    min_spike_duration: int = 1,
    max_spike_duration: int = 10,
) -> List[Tuple[int, int, float]]:
    """
    Detect spikes using z-score method.
    
    Returns:
        List of (start_index, end_index, max_deviation) tuples
    """
    if len(data) < 10:
        return []
    
    # Use robust statistics
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    robust_std = 1.4826 * mad  # Scale MAD to approximate std
    
    if robust_std < 1e-10:
        return []
    
    # Calculate z-scores
    z_scores = np.abs((data - median) / robust_std)
    
    # Find spike regions
    spike_mask = z_scores > threshold_sigma
    
    # Group contiguous spikes
    spikes = []
    in_spike = False
    start_idx = 0
    
    for i, is_spike in enumerate(spike_mask):
        if is_spike and not in_spike:
            in_spike = True
            start_idx = i
        elif not is_spike and in_spike:
            in_spike = False
            duration = i - start_idx
            if min_spike_duration <= duration <= max_spike_duration:
                max_dev = np.max(z_scores[start_idx:i])
                spikes.append((start_idx, i - 1, float(max_dev)))
    
    # Handle spike at end
    if in_spike:
        duration = len(data) - start_idx
        if min_spike_duration <= duration <= max_spike_duration:
            max_dev = np.max(z_scores[start_idx:])
            spikes.append((start_idx, len(data) - 1, float(max_dev)))
    
    return spikes


# =============================================================================
# DROPOUT DETECTION
# =============================================================================

def detect_dropouts(
    data: np.ndarray,
    dropout_threshold: float = 0.1,
    min_duration: int = 3,
) -> List[Tuple[int, int]]:
    """
    Detect signal dropouts (sudden drops to near-zero or NaN).
    
    Returns:
        List of (start_index, end_index) tuples
    """
    if len(data) < 5:
        return []
    
    # Calculate baseline (robust)
    baseline = np.nanpercentile(data, 75)
    
    if baseline < 1e-10:
        return []
    
    # Find dropout regions
    dropout_mask = (np.abs(data) < baseline * dropout_threshold) | np.isnan(data)
    
    # Group contiguous dropouts
    dropouts = []
    in_dropout = False
    start_idx = 0
    
    for i, is_dropout in enumerate(dropout_mask):
        if is_dropout and not in_dropout:
            in_dropout = True
            start_idx = i
        elif not is_dropout and in_dropout:
            in_dropout = False
            if i - start_idx >= min_duration:
                dropouts.append((start_idx, i - 1))
    
    if in_dropout and len(data) - start_idx >= min_duration:
        dropouts.append((start_idx, len(data) - 1))
    
    return dropouts


# =============================================================================
# DRIFT DETECTION
# =============================================================================

def detect_drift(
    data: np.ndarray,
    window_size: int = 100,
    drift_threshold: float = 0.05,
) -> Optional[Tuple[str, float, float]]:
    """
    Detect systematic drift in signal.
    
    Returns:
        Tuple of (direction, slope, r_squared) or None if no drift
    """
    if len(data) < window_size * 2:
        return None
    
    # Remove NaNs
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_data) < window_size:
        return None
    
    # Linear regression
    x = valid_indices.astype(float)
    y = valid_data
    
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-10:
        return None
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    if ss_tot < 1e-10:
        return None
    
    r_squared = 1 - ss_res / ss_tot
    
    # Normalize slope by data range
    data_range = np.ptp(valid_data)
    if data_range > 0:
        normalized_slope = slope * len(data) / data_range
    else:
        normalized_slope = 0
    
    # Check if drift is significant
    if abs(normalized_slope) > drift_threshold and r_squared > 0.5:
        direction = "increasing" if slope > 0 else "decreasing"
        return (direction, float(slope), float(r_squared))
    
    return None


# =============================================================================
# OSCILLATION DETECTION
# =============================================================================

def detect_oscillation(
    data: np.ndarray,
    sample_rate_hz: float = 100.0,
    min_freq_hz: float = 0.5,
    max_freq_hz: float = 20.0,
    power_threshold: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """
    Detect periodic oscillations using FFT.
    
    Returns:
        Tuple of (dominant_frequency_hz, relative_power) or None
    """
    if len(data) < 64:
        return None
    
    # Remove NaNs and detrend
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < 64:
        return None
    
    detrended = valid_data - np.mean(valid_data)
    
    # FFT
    n = len(detrended)
    fft_result = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    power = np.abs(fft_result) ** 2
    
    # Filter to frequency range of interest
    freq_mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    
    if not np.any(freq_mask):
        return None
    
    filtered_power = power[freq_mask]
    filtered_freqs = freqs[freq_mask]
    
    # Find dominant frequency
    if len(filtered_power) == 0:
        return None
    
    max_idx = np.argmax(filtered_power)
    dominant_freq = filtered_freqs[max_idx]
    dominant_power = filtered_power[max_idx]
    
    # Calculate relative power
    total_power = np.sum(power[1:])  # Exclude DC
    if total_power > 0:
        relative_power = dominant_power / total_power
    else:
        return None
    
    if relative_power > power_threshold:
        return (float(dominant_freq), float(relative_power))
    
    return None


# =============================================================================
# SATURATION DETECTION
# =============================================================================

def detect_saturation(
    data: np.ndarray,
    saturation_percentile: float = 99.5,
    min_duration: int = 5,
) -> List[Tuple[int, int, str]]:
    """
    Detect sensor saturation (stuck at min or max).
    
    Returns:
        List of (start_index, end_index, 'high'|'low') tuples
    """
    if len(data) < min_duration * 2:
        return []
    
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < min_duration:
        return []
    
    # Calculate saturation thresholds
    high_thresh = np.percentile(valid_data, saturation_percentile)
    low_thresh = np.percentile(valid_data, 100 - saturation_percentile)
    
    saturations = []
    
    # Check for high saturation
    high_sat = data >= high_thresh
    in_sat = False
    start_idx = 0
    
    for i, is_sat in enumerate(high_sat):
        if is_sat and not in_sat:
            in_sat = True
            start_idx = i
        elif not is_sat and in_sat:
            in_sat = False
            if i - start_idx >= min_duration:
                saturations.append((start_idx, i - 1, 'high'))
    
    if in_sat and len(data) - start_idx >= min_duration:
        saturations.append((start_idx, len(data) - 1, 'high'))
    
    # Check for low saturation
    low_sat = data <= low_thresh
    in_sat = False
    
    for i, is_sat in enumerate(low_sat):
        if is_sat and not in_sat:
            in_sat = True
            start_idx = i
        elif not is_sat and in_sat:
            in_sat = False
            if i - start_idx >= min_duration:
                saturations.append((start_idx, i - 1, 'low'))
    
    if in_sat and len(data) - start_idx >= min_duration:
        saturations.append((start_idx, len(data) - 1, 'low'))
    
    return saturations


# =============================================================================
# FLATLINE DETECTION
# =============================================================================

def detect_flatline(
    data: np.ndarray,
    min_duration: int = 10,
    variance_threshold: float = 1e-6,
) -> List[Tuple[int, int]]:
    """
    Detect flatline regions (no variation).
    
    Returns:
        List of (start_index, end_index) tuples
    """
    if len(data) < min_duration:
        return []
    
    flatlines = []
    window = min_duration
    
    i = 0
    while i < len(data) - window + 1:
        segment = data[i:i + window]
        
        # Skip if has NaNs
        if np.any(np.isnan(segment)):
            i += 1
            continue
        
        variance = np.var(segment)
        
        if variance < variance_threshold:
            # Extend flatline region
            end_idx = i + window - 1
            while end_idx < len(data) - 1:
                extended = data[i:end_idx + 2]
                if np.any(np.isnan(extended)):
                    break
                if np.var(extended) < variance_threshold:
                    end_idx += 1
                else:
                    break
            
            flatlines.append((i, end_idx))
            i = end_idx + 1
        else:
            i += 1
    
    return flatlines


# =============================================================================
# CORRELATION BREAK DETECTION
# =============================================================================

def detect_correlation_breaks(
    data1: np.ndarray,
    data2: np.ndarray,
    window_size: int = 50,
    correlation_threshold: float = 0.5,
    expected_correlation: float = 0.9,
) -> List[Tuple[int, int, float]]:
    """
    Detect breaks in expected correlation between two signals.
    
    Returns:
        List of (start_index, end_index, correlation) tuples
    """
    if len(data1) != len(data2) or len(data1) < window_size * 2:
        return []
    
    breaks = []
    step = window_size // 2
    
    for i in range(0, len(data1) - window_size, step):
        seg1 = data1[i:i + window_size]
        seg2 = data2[i:i + window_size]
        
        # Skip if has NaNs
        valid_mask = ~(np.isnan(seg1) | np.isnan(seg2))
        if np.sum(valid_mask) < window_size // 2:
            continue
        
        seg1_valid = seg1[valid_mask]
        seg2_valid = seg2[valid_mask]
        
        # Calculate correlation
        if np.std(seg1_valid) < 1e-10 or np.std(seg2_valid) < 1e-10:
            continue
        
        correlation = np.corrcoef(seg1_valid, seg2_valid)[0, 1]
        
        # Check if correlation is unexpectedly low
        if abs(correlation) < correlation_threshold < expected_correlation:
            breaks.append((i, i + window_size - 1, float(correlation)))
    
    # Merge adjacent breaks
    if len(breaks) < 2:
        return breaks
    
    merged = [breaks[0]]
    for start, end, corr in breaks[1:]:
        prev_start, prev_end, prev_corr = merged[-1]
        if start <= prev_end + step:
            # Merge
            merged[-1] = (prev_start, end, min(prev_corr, corr))
        else:
            merged.append((start, end, corr))
    
    return merged


# =============================================================================
# TRANSIENT DETECTION
# =============================================================================

def detect_transients(
    data: np.ndarray,
    derivative_threshold: float = 3.0,
    min_duration: int = 2,
    max_duration: int = 20,
) -> List[Tuple[int, int, float]]:
    """
    Detect transient events (rapid changes).
    
    Returns:
        List of (start_index, end_index, max_derivative) tuples
    """
    if len(data) < 10:
        return []
    
    # Calculate derivative
    derivative = np.diff(data)
    derivative = np.concatenate([[0], derivative])  # Pad to match length
    
    # Handle NaNs
    derivative[np.isnan(derivative)] = 0
    
    # Calculate threshold based on robust statistics
    median_deriv = np.median(np.abs(derivative))
    mad_deriv = np.median(np.abs(np.abs(derivative) - median_deriv))
    threshold = median_deriv + derivative_threshold * 1.4826 * mad_deriv
    
    if threshold < 1e-10:
        return []
    
    # Find transient regions
    transient_mask = np.abs(derivative) > threshold
    
    transients = []
    in_transient = False
    start_idx = 0
    
    for i, is_trans in enumerate(transient_mask):
        if is_trans and not in_transient:
            in_transient = True
            start_idx = i
        elif not is_trans and in_transient:
            in_transient = False
            duration = i - start_idx
            if min_duration <= duration <= max_duration:
                max_deriv = np.max(np.abs(derivative[start_idx:i]))
                transients.append((start_idx, i - 1, float(max_deriv)))
    
    if in_transient:
        duration = len(data) - start_idx
        if min_duration <= duration <= max_duration:
            max_deriv = np.max(np.abs(derivative[start_idx:]))
            transients.append((start_idx, len(data) - 1, float(max_deriv)))
    
    return transients


# =============================================================================
# SENSOR HEALTH SCORING
# =============================================================================

def calculate_sensor_health(
    data: np.ndarray,
    sample_rate_hz: float = 100.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate overall health score for a sensor channel.
    
    Returns:
        Tuple of (health_score 0-1, details_dict)
    """
    if len(data) < 10:
        return 0.0, {'error': 'insufficient data'}
    
    details = {}
    penalties = []
    
    # NaN ratio
    nan_ratio = np.sum(np.isnan(data)) / len(data)
    details['nan_ratio'] = nan_ratio
    if nan_ratio > 0:
        penalties.append(min(nan_ratio * 2, 0.5))  # Up to 50% penalty
    
    # Spike count
    spikes = detect_spikes(data)
    spike_ratio = len(spikes) / (len(data) / 100)  # Per 100 samples
    details['spike_count'] = len(spikes)
    if spike_ratio > 0.1:
        penalties.append(min(spike_ratio * 0.1, 0.2))
    
    # Dropout check
    dropouts = detect_dropouts(data)
    dropout_samples = sum(e - s + 1 for s, e in dropouts)
    dropout_ratio = dropout_samples / len(data)
    details['dropout_ratio'] = dropout_ratio
    if dropout_ratio > 0:
        penalties.append(min(dropout_ratio * 3, 0.4))
    
    # Flatline check
    flatlines = detect_flatline(data)
    flatline_samples = sum(e - s + 1 for s, e in flatlines)
    flatline_ratio = flatline_samples / len(data)
    details['flatline_ratio'] = flatline_ratio
    if flatline_ratio > 0.1:
        penalties.append(min(flatline_ratio * 0.5, 0.3))
    
    # Saturation check
    saturations = detect_saturation(data)
    sat_samples = sum(e - s + 1 for s, e, _ in saturations)
    sat_ratio = sat_samples / len(data)
    details['saturation_ratio'] = sat_ratio
    if sat_ratio > 0:
        penalties.append(min(sat_ratio * 2, 0.3))
    
    # Drift check
    drift = detect_drift(data)
    if drift:
        details['drift'] = drift
        penalties.append(0.1)
    
    # Oscillation check
    oscillation = detect_oscillation(data, sample_rate_hz)
    if oscillation:
        freq, power = oscillation
        details['oscillation'] = {'frequency_hz': freq, 'power': power}
        if power > 0.5:
            penalties.append(0.1)
    
    # Calculate final health score
    total_penalty = min(sum(penalties), 1.0)
    health_score = 1.0 - total_penalty
    
    details['penalties'] = penalties
    
    return health_score, details


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_anomaly_detection(
    df: pd.DataFrame,
    channels: Optional[List[str]] = None,
    timestamp_col: str = 'timestamp',
    sample_rate_hz: float = 100.0,
    correlation_pairs: Optional[List[Tuple[str, str]]] = None,
) -> AnomalyReport:
    """
    Run comprehensive anomaly detection on a DataFrame.
    
    Args:
        df: DataFrame with sensor data
        channels: List of channels to analyze (None = all numeric)
        timestamp_col: Name of timestamp column
        sample_rate_hz: Sample rate for frequency analysis
        correlation_pairs: List of (channel1, channel2) to check correlation
        
    Returns:
        AnomalyReport with all findings
    """
    # Determine channels
    if channels is None:
        channels = [c for c in df.columns 
                   if c != timestamp_col and df[c].dtype in ['float64', 'int64']]
    
    # Get timestamps if available
    timestamps = df[timestamp_col].values if timestamp_col in df.columns else None
    
    channel_reports = {}
    sensor_health = {}
    total_anomalies = 0
    critical_count = 0
    warning_count = 0
    
    # Analyze each channel
    for channel in channels:
        if channel not in df.columns:
            continue
        
        data = df[channel].values.astype(float)
        anomalies = []
        
        # Spike detection
        spikes = detect_spikes(data)
        for start, end, score in spikes:
            severity = AnomalySeverity.CRITICAL if score > 6 else AnomalySeverity.WARNING
            anomaly = Anomaly(
                anomaly_type=AnomalyType.SPIKE,
                severity=severity,
                channel=channel,
                start_index=start,
                end_index=end,
                start_time=timestamps[start] if timestamps is not None else None,
                end_time=timestamps[end] if timestamps is not None else None,
                score=score,
                description=f"Spike detected with z-score {score:.1f}",
            )
            anomalies.append(anomaly)
        
        # Dropout detection
        dropouts = detect_dropouts(data)
        for start, end in dropouts:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.DROPOUT,
                severity=AnomalySeverity.CRITICAL,
                channel=channel,
                start_index=start,
                end_index=end,
                start_time=timestamps[start] if timestamps is not None else None,
                end_time=timestamps[end] if timestamps is not None else None,
                description=f"Signal dropout for {end - start + 1} samples",
            )
            anomalies.append(anomaly)
        
        # Saturation detection
        saturations = detect_saturation(data)
        for start, end, direction in saturations:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.SATURATION,
                severity=AnomalySeverity.WARNING,
                channel=channel,
                start_index=start,
                end_index=end,
                start_time=timestamps[start] if timestamps is not None else None,
                end_time=timestamps[end] if timestamps is not None else None,
                description=f"Sensor saturated {direction} for {end - start + 1} samples",
                metadata={'direction': direction},
            )
            anomalies.append(anomaly)
        
        # Flatline detection
        flatlines = detect_flatline(data)
        for start, end in flatlines:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.FLATLINE,
                severity=AnomalySeverity.WARNING,
                channel=channel,
                start_index=start,
                end_index=end,
                start_time=timestamps[start] if timestamps is not None else None,
                end_time=timestamps[end] if timestamps is not None else None,
                description=f"Signal flatline for {end - start + 1} samples",
            )
            anomalies.append(anomaly)
        
        # Drift detection
        drift = detect_drift(data)
        if drift:
            direction, slope, r2 = drift
            anomaly = Anomaly(
                anomaly_type=AnomalyType.DRIFT,
                severity=AnomalySeverity.INFO,
                channel=channel,
                start_index=0,
                end_index=len(data) - 1,
                start_time=timestamps[0] if timestamps is not None else None,
                end_time=timestamps[-1] if timestamps is not None else None,
                score=r2,
                description=f"Signal {direction} with R²={r2:.2f}",
                metadata={'direction': direction, 'slope': slope, 'r_squared': r2},
            )
            anomalies.append(anomaly)
        
        # Oscillation detection
        oscillation = detect_oscillation(data, sample_rate_hz)
        if oscillation:
            freq, power = oscillation
            anomaly = Anomaly(
                anomaly_type=AnomalyType.OSCILLATION,
                severity=AnomalySeverity.WARNING if power > 0.5 else AnomalySeverity.INFO,
                channel=channel,
                start_index=0,
                end_index=len(data) - 1,
                start_time=timestamps[0] if timestamps is not None else None,
                end_time=timestamps[-1] if timestamps is not None else None,
                score=power,
                description=f"Oscillation at {freq:.1f} Hz (power={power:.1%})",
                metadata={'frequency_hz': freq, 'relative_power': power},
            )
            anomalies.append(anomaly)
        
        # Transient detection
        transients = detect_transients(data)
        for start, end, max_deriv in transients:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.TRANSIENT,
                severity=AnomalySeverity.INFO,
                channel=channel,
                start_index=start,
                end_index=end,
                start_time=timestamps[start] if timestamps is not None else None,
                end_time=timestamps[end] if timestamps is not None else None,
                score=max_deriv,
                description=f"Transient event with max derivative {max_deriv:.2f}",
            )
            anomalies.append(anomaly)
        
        # Calculate sensor health
        health, _ = calculate_sensor_health(data, sample_rate_hz)
        sensor_health[channel] = health
        
        # Store results
        channel_reports[channel] = anomalies
        
        # Count anomalies
        for a in anomalies:
            total_anomalies += 1
            if a.severity == AnomalySeverity.CRITICAL:
                critical_count += 1
            elif a.severity == AnomalySeverity.WARNING:
                warning_count += 1
    
    # Multivariate analysis (correlation breaks)
    multivariate_anomalies = []
    
    if correlation_pairs:
        for ch1, ch2 in correlation_pairs:
            if ch1 in df.columns and ch2 in df.columns:
                data1 = df[ch1].values.astype(float)
                data2 = df[ch2].values.astype(float)
                
                breaks = detect_correlation_breaks(data1, data2)
                for start, end, corr in breaks:
                    anomaly = Anomaly(
                        anomaly_type=AnomalyType.CORRELATION_BREAK,
                        severity=AnomalySeverity.WARNING,
                        channel=f"{ch1}:{ch2}",
                        start_index=start,
                        end_index=end,
                        start_time=timestamps[start] if timestamps is not None else None,
                        end_time=timestamps[end] if timestamps is not None else None,
                        score=1 - abs(corr),
                        description=f"Correlation break between {ch1} and {ch2}: r={corr:.2f}",
                        metadata={'correlation': corr},
                    )
                    multivariate_anomalies.append(anomaly)
                    total_anomalies += 1
                    warning_count += 1
    
    return AnomalyReport(
        channel_reports=channel_reports,
        multivariate_anomalies=multivariate_anomalies,
        sensor_health=sensor_health,
        total_anomalies=total_anomalies,
        critical_count=critical_count,
        warning_count=warning_count,
    )


def format_anomaly_table(report: AnomalyReport) -> pd.DataFrame:
    """Convert anomaly report to DataFrame for display."""
    rows = []
    
    for anomaly in report.get_all_anomalies():
        rows.append({
            'Channel': anomaly.channel,
            'Type': anomaly.anomaly_type.value,
            'Severity': anomaly.severity.name,
            'Start': anomaly.start_index,
            'End': anomaly.end_index,
            'Duration': anomaly.duration_samples,
            'Score': f"{anomaly.score:.2f}" if anomaly.score else "-",
            'Description': anomaly.description,
        })
    
    if not rows:
        return pd.DataFrame(columns=['Channel', 'Type', 'Severity', 'Start', 'End', 
                                     'Duration', 'Score', 'Description'])
    
    return pd.DataFrame(rows)
