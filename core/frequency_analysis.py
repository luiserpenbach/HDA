"""
Frequency / Spectral Analysis Toolkit
======================================
Comprehensive spectral analysis for rocket propulsion test data.

Provides power spectral density estimation, spectrogram computation,
harmonic detection, cross-spectral analysis, resonance detection, and
frequency-band power integration.

Typical use-cases:
- Combustion instability detection (acoustic modes in chamber pressure)
- Feed system coupling identification (coherence between injector & chamber)
- Vibration characterisation from accelerometer data
- Noise floor assessment during cold flow testing

All functions operate on raw numpy arrays so they stay UI-agnostic
(no Streamlit imports) and can be unit-tested independently.

Dependencies:
    numpy, scipy (signal module)
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.signal import find_peaks

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


# ---------------------------------------------------------------------------
# Default frequency bands for rocket propulsion test data
# ---------------------------------------------------------------------------
DEFAULT_FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    'low': (0.1, 5.0),
    'mid': (5.0, 50.0),
    'high': (50.0, 500.0),
    'acoustic': (500.0, 5000.0),
}

# Minimum number of samples to attempt any spectral analysis
_MIN_SAMPLES = 8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpectralResult:
    """
    Result of a power spectral density computation.

    Attributes:
        frequencies: Array of frequency bin centres (Hz).
        power_spectral_density: Array of PSD values (unit^2/Hz).
        dominant_frequency: Frequency with the highest PSD (Hz).
        dominant_power: PSD value at the dominant frequency.
        bandwidth: 3-dB bandwidth around the dominant peak (Hz).
            Set to 0.0 when the bandwidth cannot be determined.
        total_power: Integrated power across all frequencies.
        sample_rate_hz: Sample rate used for the computation.
        method: Estimation method (e.g. 'welch').
    """
    frequencies: np.ndarray
    power_spectral_density: np.ndarray
    dominant_frequency: float
    dominant_power: float
    bandwidth: float
    total_power: float
    sample_rate_hz: float
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary (arrays become lists)."""
        return {
            'frequencies': self.frequencies.tolist(),
            'power_spectral_density': self.power_spectral_density.tolist(),
            'dominant_frequency': float(self.dominant_frequency),
            'dominant_power': float(self.dominant_power),
            'bandwidth': float(self.bandwidth),
            'total_power': float(self.total_power),
            'sample_rate_hz': float(self.sample_rate_hz),
            'method': self.method,
        }


@dataclass
class HarmonicInfo:
    """
    Information about a single harmonic component.

    Attributes:
        frequency: Centre frequency of the harmonic (Hz).
        power: PSD value at the harmonic.
        harmonic_number: Harmonic index (1 = fundamental).
        relative_power: Power relative to the fundamental (linear scale,
            fundamental = 1.0).
    """
    frequency: float
    power: float
    harmonic_number: int
    relative_power: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            'frequency': float(self.frequency),
            'power': float(self.power),
            'harmonic_number': self.harmonic_number,
            'relative_power': float(self.relative_power),
        }


@dataclass
class CrossSpectralResult:
    """
    Result of cross-spectral analysis between two channels.

    Attributes:
        frequencies: Array of frequency bin centres (Hz).
        coherence: Magnitude-squared coherence (0 to 1).
        phase: Phase angle between channels (radians).
        cross_power: Cross power spectral density (complex).
    """
    frequencies: np.ndarray
    coherence: np.ndarray
    phase: np.ndarray
    cross_power: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            'frequencies': self.frequencies.tolist(),
            'coherence': self.coherence.tolist(),
            'phase': self.phase.tolist(),
            'cross_power_real': np.real(self.cross_power).tolist(),
            'cross_power_imag': np.imag(self.cross_power).tolist(),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_signal(data: np.ndarray, name: str = 'data') -> np.ndarray:
    """
    Validate and clean a 1-D signal array.

    Args:
        data: Input signal.
        name: Name used in error messages.

    Returns:
        Cleaned 1-D float64 array.

    Raises:
        ValueError: If the signal is too short, entirely NaN, or zero-length.
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    if data.size == 0:
        raise ValueError(f"{name} is empty.")

    # Replace NaN with linear interpolation where possible
    nan_mask = np.isnan(data)
    if nan_mask.all():
        raise ValueError(f"{name} is entirely NaN.")

    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        data = np.interp(
            np.arange(len(data)),
            valid_idx,
            data[valid_idx],
        )
        warnings.warn(
            f"{name} contained {int(nan_mask.sum())} NaN values; "
            "replaced via linear interpolation.",
            stacklevel=3,
        )

    if data.size < _MIN_SAMPLES:
        raise ValueError(
            f"{name} has {data.size} samples, need at least {_MIN_SAMPLES} "
            "for spectral analysis."
        )

    return data


def _compute_bandwidth_3db(
    frequencies: np.ndarray,
    psd: np.ndarray,
    peak_idx: int,
) -> float:
    """
    Estimate the 3-dB bandwidth around a spectral peak.

    The bandwidth is defined as the frequency span where the PSD stays
    within 3 dB (factor of 2) of the peak value.

    Args:
        frequencies: Frequency array.
        psd: PSD array.
        peak_idx: Index of the peak in the PSD array.

    Returns:
        Bandwidth in Hz, or 0.0 if it cannot be determined.
    """
    if peak_idx < 0 or peak_idx >= len(psd):
        return 0.0

    peak_power = psd[peak_idx]
    if peak_power <= 0:
        return 0.0

    half_power = peak_power / 2.0  # -3 dB point

    # Search left for the -3 dB crossing
    left_freq = frequencies[0]
    for i in range(peak_idx - 1, -1, -1):
        if psd[i] < half_power:
            # Linear interpolation between i and i+1
            if psd[i + 1] - psd[i] != 0:
                frac = (half_power - psd[i]) / (psd[i + 1] - psd[i])
                left_freq = frequencies[i] + frac * (frequencies[i + 1] - frequencies[i])
            else:
                left_freq = frequencies[i]
            break

    # Search right for the -3 dB crossing
    right_freq = frequencies[-1]
    for i in range(peak_idx + 1, len(psd)):
        if psd[i] < half_power:
            if psd[i - 1] - psd[i] != 0:
                frac = (half_power - psd[i]) / (psd[i - 1] - psd[i])
                right_freq = frequencies[i] - frac * (frequencies[i] - frequencies[i - 1])
            else:
                right_freq = frequencies[i]
            break

    bw = float(right_freq - left_freq)
    return max(bw, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_power_spectral_density(
    data: np.ndarray,
    sample_rate_hz: float = 100.0,
    method: str = 'welch',
    nperseg: Optional[int] = None,
    window: str = 'hann',
    detrend: str = 'linear',
) -> SpectralResult:
    """
    Estimate the power spectral density of a signal.

    Uses Welch's method by default for robust, low-variance PSD estimates
    suitable for noisy propulsion test data.

    Args:
        data: 1-D time-domain signal.
        sample_rate_hz: Sampling frequency in Hz.
        method: PSD estimation method. Currently supports 'welch' and
            'periodogram'.
        nperseg: Length of each Welch segment.  Defaults to
            ``min(256, len(data))`` for Welch or the full signal length
            for the periodogram.
        window: Window function name (any ``scipy.signal.get_window`` name).
        detrend: Detrend mode passed to ``scipy.signal.welch`` ('linear',
            'constant', or False).

    Returns:
        SpectralResult with the estimated PSD and summary statistics.

    Raises:
        ValueError: If the signal is too short, all NaN, or the method
            is unsupported.

    Example:
        >>> import numpy as np
        >>> t = np.arange(0, 1, 1/1000)
        >>> sig = np.sin(2 * np.pi * 50 * t)
        >>> result = compute_power_spectral_density(sig, sample_rate_hz=1000)
        >>> abs(result.dominant_frequency - 50.0) < 2.0
        True
    """
    data = _validate_signal(data, 'data')

    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

    # Handle zero-variance signal
    if np.std(data) == 0:
        warnings.warn(
            "Signal has zero variance; PSD will be trivially zero.",
            stacklevel=2,
        )
        n_freqs = len(data) // 2 + 1
        freqs = np.linspace(0, sample_rate_hz / 2.0, n_freqs)
        psd = np.zeros(n_freqs)
        return SpectralResult(
            frequencies=freqs,
            power_spectral_density=psd,
            dominant_frequency=0.0,
            dominant_power=0.0,
            bandwidth=0.0,
            total_power=0.0,
            sample_rate_hz=sample_rate_hz,
            method=method,
        )

    if method == 'welch':
        if nperseg is None:
            nperseg = min(256, len(data))
        # Ensure nperseg does not exceed signal length
        nperseg = min(nperseg, len(data))

        freqs, psd = signal.welch(
            data,
            fs=sample_rate_hz,
            window=window,
            nperseg=nperseg,
            detrend=detrend,
            scaling='density',
        )

    elif method == 'periodogram':
        freqs, psd = signal.periodogram(
            data,
            fs=sample_rate_hz,
            window=window,
            detrend=detrend,
            scaling='density',
        )

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Choose 'welch' or 'periodogram'."
        )

    # Summary statistics
    peak_idx = int(np.argmax(psd))
    dominant_frequency = float(freqs[peak_idx])
    dominant_power = float(psd[peak_idx])
    bandwidth = _compute_bandwidth_3db(freqs, psd, peak_idx)
    total_power = float(_trapezoid(psd, freqs))

    return SpectralResult(
        frequencies=freqs,
        power_spectral_density=psd,
        dominant_frequency=dominant_frequency,
        dominant_power=dominant_power,
        bandwidth=bandwidth,
        total_power=total_power,
        sample_rate_hz=sample_rate_hz,
        method=method,
    )


def compute_spectrogram(
    data: np.ndarray,
    sample_rate_hz: float = 100.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = 'hann',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a time-frequency spectrogram via the Short-Time Fourier Transform.

    Useful for visualising how frequency content evolves during a test --
    e.g. detecting combustion instability onset or valve chatter that
    appears only during transients.

    Args:
        data: 1-D time-domain signal.
        sample_rate_hz: Sampling frequency in Hz.
        nperseg: Number of samples per STFT segment.
        noverlap: Number of overlapping samples between segments.
            Defaults to ``nperseg // 2``.
        window: Window function name.

    Returns:
        Tuple of (frequencies, times, Sxx) where:
            - frequencies: 1-D array of frequency bins (Hz).
            - times: 1-D array of time segment centres (s).
            - Sxx: 2-D magnitude-squared spectrogram (frequencies x times).

    Raises:
        ValueError: If the signal is too short or invalid.

    Example:
        >>> import numpy as np
        >>> sig = np.random.randn(2000)
        >>> f, t, Sxx = compute_spectrogram(sig, sample_rate_hz=1000, nperseg=128)
        >>> f.shape[0] == 128 // 2 + 1
        True
    """
    data = _validate_signal(data, 'data')

    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

    # Clamp nperseg to signal length
    nperseg = min(nperseg, len(data))

    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = min(noverlap, nperseg - 1)

    freqs, times, Zxx = signal.stft(
        data,
        fs=sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Return magnitude-squared (power spectrogram)
    Sxx = np.abs(Zxx) ** 2

    return freqs, times, Sxx


def detect_harmonics(
    spectral_result: SpectralResult,
    n_harmonics: int = 5,
    tolerance_hz: float = 1.0,
) -> List[HarmonicInfo]:
    """
    Identify the fundamental frequency and its harmonics in a PSD.

    The fundamental is taken as the dominant frequency reported in
    ``spectral_result``.  Harmonics are then sought at integer multiples
    of the fundamental (within ``tolerance_hz``).

    Args:
        spectral_result: PSD result from ``compute_power_spectral_density``.
        n_harmonics: Maximum harmonic number to search for (including the
            fundamental, which counts as harmonic 1).
        tolerance_hz: Frequency tolerance when matching harmonics.

    Returns:
        List of ``HarmonicInfo`` for each detected harmonic.  The
        fundamental (harmonic_number=1) is always included first when it
        has non-zero power.

    Raises:
        ValueError: If n_harmonics < 1 or tolerance_hz < 0.

    Example:
        >>> result = compute_power_spectral_density(sig, sample_rate_hz=1000)
        >>> harmonics = detect_harmonics(result, n_harmonics=3)
        >>> harmonics[0].harmonic_number
        1
    """
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
    if tolerance_hz < 0:
        raise ValueError(f"tolerance_hz must be >= 0, got {tolerance_hz}")

    freqs = spectral_result.frequencies
    psd = spectral_result.power_spectral_density
    fundamental = spectral_result.dominant_frequency

    # Guard: no meaningful fundamental
    if fundamental <= 0 or spectral_result.dominant_power <= 0:
        return []

    fundamental_power = spectral_result.dominant_power
    harmonics: List[HarmonicInfo] = []

    for n in range(1, n_harmonics + 1):
        target_freq = fundamental * n

        # Skip if target exceeds Nyquist
        if target_freq > freqs[-1]:
            break

        # Find the closest frequency bin within tolerance
        freq_diff = np.abs(freqs - target_freq)
        min_diff_idx = int(np.argmin(freq_diff))

        if freq_diff[min_diff_idx] <= tolerance_hz:
            # Refine by picking the local maximum within the tolerance window
            lo = np.searchsorted(freqs, target_freq - tolerance_hz, side='left')
            hi = np.searchsorted(freqs, target_freq + tolerance_hz, side='right')
            lo = max(lo, 0)
            hi = min(hi, len(psd))

            if hi > lo:
                local_peak_idx = lo + int(np.argmax(psd[lo:hi]))
                peak_freq = float(freqs[local_peak_idx])
                peak_power = float(psd[local_peak_idx])
            else:
                peak_freq = float(freqs[min_diff_idx])
                peak_power = float(psd[min_diff_idx])

            relative = peak_power / fundamental_power if fundamental_power > 0 else 0.0

            harmonics.append(HarmonicInfo(
                frequency=peak_freq,
                power=peak_power,
                harmonic_number=n,
                relative_power=float(relative),
            ))

    return harmonics


def compute_cross_spectrum(
    data_a: np.ndarray,
    data_b: np.ndarray,
    sample_rate_hz: float = 100.0,
    nperseg: Optional[int] = None,
) -> CrossSpectralResult:
    """
    Compute the cross-spectral quantities between two sensor channels.

    Returns magnitude-squared coherence, phase angle, and cross power
    spectral density.  Coherence close to 1.0 at a given frequency
    indicates strong linear coupling between the two channels at that
    frequency -- useful for detecting feed-system-coupled combustion
    instabilities.

    Args:
        data_a: First channel time-domain signal.
        data_b: Second channel time-domain signal (same length as data_a).
        sample_rate_hz: Sampling frequency in Hz.
        nperseg: Segment length for Welch-based estimation.  Defaults to
            ``min(256, len(data_a))``.

    Returns:
        CrossSpectralResult with coherence, phase, and cross power arrays.

    Raises:
        ValueError: If the signals have different lengths, are too short,
            or are invalid.

    Example:
        >>> a = np.sin(2 * np.pi * 100 * np.arange(0, 1, 1/1000))
        >>> b = np.sin(2 * np.pi * 100 * np.arange(0, 1, 1/1000) + 0.5)
        >>> result = compute_cross_spectrum(a, b, sample_rate_hz=1000)
        >>> result.coherence.max() > 0.9
        True
    """
    data_a = _validate_signal(data_a, 'data_a')
    data_b = _validate_signal(data_b, 'data_b')

    if len(data_a) != len(data_b):
        raise ValueError(
            f"data_a ({len(data_a)} samples) and data_b ({len(data_b)} samples) "
            "must have the same length."
        )

    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

    if nperseg is None:
        nperseg = min(256, len(data_a))
    nperseg = min(nperseg, len(data_a))

    # Magnitude-squared coherence
    freq_coh, coherence = signal.coherence(
        data_a, data_b,
        fs=sample_rate_hz,
        nperseg=nperseg,
    )

    # Cross power spectral density (complex)
    freq_csd, csd = signal.csd(
        data_a, data_b,
        fs=sample_rate_hz,
        nperseg=nperseg,
    )

    # Phase angle from the CSD
    phase = np.angle(csd)

    return CrossSpectralResult(
        frequencies=freq_coh,
        coherence=coherence,
        phase=phase,
        cross_power=csd,
    )


def detect_resonance(
    spectral_result: SpectralResult,
    prominence: float = 3.0,
    min_freq_hz: float = 0.5,
    max_freq_hz: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Find resonance peaks in a PSD and estimate their Q-factors.

    A resonance is identified as a peak in the PSD that exceeds a minimum
    prominence (in dB above the local baseline).  The quality factor Q is
    estimated as ``f_centre / bandwidth_3dB``.

    Args:
        spectral_result: PSD result from ``compute_power_spectral_density``.
        prominence: Minimum peak prominence in dB for a peak to qualify as
            a resonance.  Higher values yield fewer, more prominent peaks.
        min_freq_hz: Ignore peaks below this frequency (filters out DC
            leakage and very-low-frequency trends).
        max_freq_hz: Ignore peaks above this frequency.  Defaults to
            the Nyquist frequency.

    Returns:
        List of dicts, each with keys: ``frequency``, ``power``,
        ``q_factor``, ``bandwidth``.  Sorted by descending power.

    Raises:
        ValueError: If prominence < 0 or min_freq_hz < 0.

    Example:
        >>> result = compute_power_spectral_density(sig, sample_rate_hz=1000)
        >>> peaks = detect_resonance(result, prominence=5.0)
        >>> all('q_factor' in p for p in peaks)
        True
    """
    if prominence < 0:
        raise ValueError(f"prominence must be >= 0, got {prominence}")
    if min_freq_hz < 0:
        raise ValueError(f"min_freq_hz must be >= 0, got {min_freq_hz}")

    freqs = spectral_result.frequencies
    psd = spectral_result.power_spectral_density

    if max_freq_hz is None:
        max_freq_hz = float(freqs[-1])

    # Work in dB for prominence-based peak finding
    # Clamp to a floor to avoid log(0)
    psd_safe = np.maximum(psd, np.finfo(float).tiny)
    psd_db = 10.0 * np.log10(psd_safe)

    # Restrict search to the requested frequency range
    freq_mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    valid_indices = np.where(freq_mask)[0]

    if len(valid_indices) == 0:
        return []

    # Find peaks in the dB-scale PSD within the valid range
    # We operate on the full array but filter afterwards
    peak_indices, peak_properties = find_peaks(
        psd_db,
        prominence=prominence,
    )

    resonances: List[Dict[str, float]] = []

    for idx in peak_indices:
        if idx not in valid_indices:
            continue

        freq = float(freqs[idx])
        power = float(psd[idx])
        bw = _compute_bandwidth_3db(freqs, psd, idx)

        if bw > 0:
            q_factor = freq / bw
        else:
            q_factor = 0.0

        resonances.append({
            'frequency': freq,
            'power': power,
            'q_factor': float(q_factor),
            'bandwidth': float(bw),
        })

    # Sort by descending power
    resonances.sort(key=lambda r: r['power'], reverse=True)
    return resonances


def compute_frequency_bands(
    spectral_result: SpectralResult,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Integrate PSD power within defined frequency bands.

    This is the spectral equivalent of an "energy budget" -- for
    propulsion tests it reveals whether energy is concentrated in
    low-frequency bulk flow oscillations, mid-frequency feed-system
    modes, high-frequency turbulence, or acoustic combustion modes.

    Args:
        spectral_result: PSD result from ``compute_power_spectral_density``.
        bands: Dictionary mapping band names to (low_hz, high_hz) tuples.
            Defaults to ``DEFAULT_FREQUENCY_BANDS`` which covers
            low (0.1-5 Hz), mid (5-50 Hz), high (50-500 Hz), and
            acoustic (500-5000 Hz).

    Returns:
        Dictionary mapping band names to integrated power values.  An
        additional ``'total'`` key gives the power across all bands
        combined.  Bands that fall outside the available frequency range
        will have a power of 0.0.

    Example:
        >>> result = compute_power_spectral_density(sig, sample_rate_hz=1000)
        >>> band_power = compute_frequency_bands(result)
        >>> 'low' in band_power and 'total' in band_power
        True
    """
    if bands is None:
        bands = DEFAULT_FREQUENCY_BANDS

    freqs = spectral_result.frequencies
    psd = spectral_result.power_spectral_density

    band_power: Dict[str, float] = {}
    cumulative = 0.0

    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs <= f_high)
        if mask.any():
            power = float(_trapezoid(psd[mask], freqs[mask]))
        else:
            power = 0.0

        band_power[band_name] = power
        cumulative += power

    band_power['total'] = cumulative
    return band_power
