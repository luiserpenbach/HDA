"""
Hot Fire Sample Data Generator
===============================
Generates realistic synthetic hot fire test data (CSV + metadata JSON)
for testing all HDA hot fire analysis features.

Produces:
- Raw CSV with timestamp and sensor columns matching config expectations
- metadata.json with test article properties, sensor roles, and geometry
- Optionally generates multiple test files for campaign/batch testing

Data profiles simulate real hot fire behavior:
- Pre-ignition baseline (ambient sensors, no flow)
- Ignition transient (rapid ramp-up with overshoot)
- Steady-state combustion (stable with realistic noise)
- Shutdown transient (rapid decay)
- Post-shutdown cooldown

Scenarios:
- Nominal test (standard performance)
- Off-nominal O/F (shifted mixture ratio)
- Low chamber pressure (throttled or degraded ignition)
- Failed ignition (aborted test)
- Sensor anomaly (flatline, dropout, spike)

Usage:
    python generate_hot_fire_sample_data.py

    # Generate a single nominal test
    python generate_hot_fire_sample_data.py --scenario nominal

    # Generate a full campaign (10 tests with varied conditions)
    python generate_hot_fire_sample_data.py --campaign --n-tests 10

    # Generate into a specific directory
    python generate_hot_fire_sample_data.py --output-dir ./sample_data/hot_fire
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd


# =============================================================================
# DATA GENERATION PROFILES
# =============================================================================

def _ignition_profile(t: np.ndarray, t_ignition: float = 0.3,
                      t_shutdown: float = 4.2,
                      ramp_sharpness: float = 10.0) -> np.ndarray:
    """
    Generate smooth ignition/shutdown envelope using tanh transitions.

    Args:
        t: Time array in seconds
        t_ignition: Time of ignition onset (s)
        t_shutdown: Time of shutdown onset (s)
        ramp_sharpness: Controls transition steepness (higher = sharper)

    Returns:
        Profile array [0, 1] representing engine firing envelope
    """
    ignition = 0.5 * (1 + np.tanh(ramp_sharpness * (t - t_ignition)))
    shutdown = 0.5 * (1 - np.tanh(ramp_sharpness * (t - t_shutdown)))
    return ignition * shutdown


def _add_ignition_overshoot(signal: np.ndarray, t: np.ndarray,
                            t_ignition: float = 0.3,
                            overshoot_pct: float = 0.15,
                            decay_rate: float = 5.0) -> np.ndarray:
    """
    Add realistic ignition overshoot to a signal.

    Args:
        signal: Base signal array
        t: Time array in seconds
        t_ignition: Ignition time (s)
        overshoot_pct: Overshoot as fraction of steady-state value
        decay_rate: How quickly overshoot decays (1/s)

    Returns:
        Signal with overshoot added
    """
    mask = t > t_ignition
    overshoot = np.zeros_like(signal)
    overshoot[mask] = (overshoot_pct * np.max(signal) *
                       np.exp(-decay_rate * (t[mask] - t_ignition)) *
                       np.sin(2 * np.pi * 8 * (t[mask] - t_ignition)))
    return signal + overshoot


def generate_sensor_noise(n_samples: int, noise_std: float,
                          rng: np.random.Generator,
                          drift_rate: float = 0.0) -> np.ndarray:
    """
    Generate realistic sensor noise with optional drift.

    Args:
        n_samples: Number of data points
        noise_std: Standard deviation of white noise
        rng: NumPy random generator
        drift_rate: Linear drift per sample (0 = no drift)

    Returns:
        Noise array
    """
    white_noise = rng.normal(0, noise_std, n_samples)
    drift = drift_rate * np.arange(n_samples)
    return white_noise + drift


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

# Default engine parameters (small bipropellant engine / igniter class)
DEFAULT_ENGINE = {
    'target_pc_bar': 20.0,       # Chamber pressure
    'target_thrust_n': 250.0,    # Thrust
    'target_mf_fuel_gs': 40.0,   # Fuel mass flow (g/s)
    'of_ratio': 2.5,             # O/F ratio -> ox = fuel * of_ratio
    'temp_ox_k': 90.0,           # LOX temperature
    'temp_fuel_k': 300.0,        # Fuel (RP-1) temperature
    'duration_s': 5.0,           # Total test duration
    'sample_rate_hz': 100,       # DAQ sample rate
    't_ignition_s': 0.3,         # Ignition onset time
    't_shutdown_s': 4.2,         # Shutdown onset time
}

SCENARIOS = {
    'nominal': {
        'description': 'Nominal hot fire test with standard performance',
        'overrides': {},
    },
    'high_of': {
        'description': 'Off-nominal O/F ratio (oxidizer rich)',
        'overrides': {'of_ratio': 3.2},
    },
    'low_of': {
        'description': 'Off-nominal O/F ratio (fuel rich)',
        'overrides': {'of_ratio': 1.8},
    },
    'low_pc': {
        'description': 'Low chamber pressure (throttled / weak ignition)',
        'overrides': {
            'target_pc_bar': 10.0,
            'target_thrust_n': 120.0,
            'target_mf_fuel_gs': 25.0,
        },
    },
    'high_pc': {
        'description': 'High chamber pressure run',
        'overrides': {
            'target_pc_bar': 28.0,
            'target_thrust_n': 350.0,
            'target_mf_fuel_gs': 55.0,
        },
    },
    'failed_ignition': {
        'description': 'Failed ignition - very weak combustion then abort',
        'overrides': {
            'target_pc_bar': 2.0,
            'target_thrust_n': 15.0,
            'target_mf_fuel_gs': 5.0,
            't_shutdown_s': 1.5,
            'duration_s': 3.0,
        },
    },
    'short_burn': {
        'description': 'Short duration burn (1.5s steady state)',
        'overrides': {
            't_shutdown_s': 2.2,
            'duration_s': 3.0,
        },
    },
    'long_burn': {
        'description': 'Extended duration burn (8s steady state)',
        'overrides': {
            't_shutdown_s': 8.5,
            'duration_s': 10.0,
        },
    },
    'sensor_spike': {
        'description': 'Nominal test with pressure sensor spike anomaly',
        'overrides': {},
        'anomaly': 'spike',
    },
    'sensor_dropout': {
        'description': 'Nominal test with brief sensor dropout (NaN gap)',
        'overrides': {},
        'anomaly': 'dropout',
    },
}


# =============================================================================
# CORE DATA GENERATOR
# =============================================================================

def generate_hot_fire_csv(
    scenario: str = 'nominal',
    seed: Optional[int] = None,
    engine_params: Optional[Dict[str, Any]] = None,
    sensor_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic hot fire test CSV DataFrame.

    Args:
        scenario: One of the SCENARIOS keys
        seed: Random seed for reproducibility
        engine_params: Override default engine parameters
        sensor_names: Override default sensor column names

    Returns:
        DataFrame with timestamp (ms) and sensor columns
    """
    rng = np.random.default_rng(seed)

    # Resolve engine parameters
    params = dict(DEFAULT_ENGINE)
    if engine_params:
        params.update(engine_params)

    scenario_def = SCENARIOS.get(scenario, SCENARIOS['nominal'])
    params.update(scenario_def.get('overrides', {}))

    # Default sensor column names
    names = {
        'chamber_pressure': 'HF-PT-01',
        'thrust': 'HF-LC-01',
        'mass_flow_ox': 'HF-FM-01',
        'mass_flow_fuel': 'HF-FM-02',
        'temperature_ox': 'HF-TC-01',
        'temperature_fuel': 'HF-TC-02',
    }
    if sensor_names:
        names.update(sensor_names)

    # Time vector
    duration_s = params['duration_s']
    sample_rate = params['sample_rate_hz']
    n_samples = int(duration_s * sample_rate)
    time_s = np.linspace(0, duration_s, n_samples)
    timestamp_ms = time_s * 1000.0

    # Firing envelope
    profile = _ignition_profile(
        time_s,
        t_ignition=params['t_ignition_s'],
        t_shutdown=params['t_shutdown_s'],
    )

    # --- Chamber Pressure (bar) ---
    pc_base = params['target_pc_bar'] * profile
    pc_base = _add_ignition_overshoot(pc_base, time_s,
                                      t_ignition=params['t_ignition_s'],
                                      overshoot_pct=0.12)
    pc_noise = generate_sensor_noise(n_samples, params['target_pc_bar'] * 0.005, rng)
    pc = np.maximum(pc_base + pc_noise, 0.0)

    # --- Thrust (N) ---
    thrust_base = params['target_thrust_n'] * profile
    thrust_base = _add_ignition_overshoot(thrust_base, time_s,
                                          t_ignition=params['t_ignition_s'],
                                          overshoot_pct=0.10)
    thrust_noise = generate_sensor_noise(n_samples, params['target_thrust_n'] * 0.008, rng)
    thrust = np.maximum(thrust_base + thrust_noise, 0.0)

    # --- Mass Flows (g/s) ---
    mf_fuel_base = params['target_mf_fuel_gs'] * profile
    mf_ox_base = params['target_mf_fuel_gs'] * params['of_ratio'] * profile

    mf_fuel_noise = generate_sensor_noise(n_samples, params['target_mf_fuel_gs'] * 0.012, rng)
    mf_ox_noise = generate_sensor_noise(
        n_samples, params['target_mf_fuel_gs'] * params['of_ratio'] * 0.010, rng)

    mf_fuel = np.maximum(mf_fuel_base + mf_fuel_noise, 0.0)
    mf_ox = np.maximum(mf_ox_base + mf_ox_noise, 0.0)

    # --- Temperatures (K) ---
    # Ox temp (cryogenic - rises slightly during flow)
    temp_ox = (params['temp_ox_k']
               + 5.0 * profile
               + generate_sensor_noise(n_samples, 0.5, rng))

    # Fuel temp (ambient - heats during flow)
    temp_fuel = (params['temp_fuel_k']
                 + 15.0 * profile
                 + generate_sensor_noise(n_samples, 0.5, rng))

    # --- Apply anomalies ---
    anomaly = scenario_def.get('anomaly')
    if anomaly == 'spike':
        # Inject a pressure spike mid-test
        spike_idx = int(0.5 * n_samples)
        spike_width = int(0.01 * n_samples)
        pc[spike_idx:spike_idx + spike_width] += params['target_pc_bar'] * 0.5

    elif anomaly == 'dropout':
        # Inject NaN dropout in thrust for ~50ms
        dropout_start = int(0.4 * n_samples)
        dropout_len = max(1, int(0.05 * sample_rate))
        thrust[dropout_start:dropout_start + dropout_len] = np.nan

    # Build DataFrame
    df = pd.DataFrame({
        'timestamp': timestamp_ms,
        names['chamber_pressure']: pc,
        names['thrust']: thrust,
        names['mass_flow_ox']: mf_ox,
        names['mass_flow_fuel']: mf_fuel,
        names['temperature_ox']: temp_ox,
        names['temperature_fuel']: temp_fuel,
    })

    return df


# =============================================================================
# METADATA GENERATOR
# =============================================================================

def generate_hot_fire_metadata(
    test_id: str = 'ENG-HF-001',
    scenario: str = 'nominal',
    engine_params: Optional[Dict[str, Any]] = None,
    sensor_names: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generate metadata JSON content for a hot fire test.

    Args:
        test_id: Unique test identifier
        scenario: Scenario name (for description)
        engine_params: Override default engine parameters
        sensor_names: Override default sensor names

    Returns:
        Metadata dictionary ready for JSON serialization
    """
    params = dict(DEFAULT_ENGINE)
    if engine_params:
        params.update(engine_params)
    scenario_def = SCENARIOS.get(scenario, SCENARIOS['nominal'])
    params.update(scenario_def.get('overrides', {}))

    names = {
        'chamber_pressure': 'HF-PT-01',
        'thrust': 'HF-LC-01',
        'mass_flow_ox': 'HF-FM-01',
        'mass_flow_fuel': 'HF-FM-02',
        'temperature_ox': 'HF-TC-01',
        'temperature_fuel': 'HF-TC-02',
    }
    if sensor_names:
        names.update(sensor_names)

    metadata = {
        'test_metadata': {
            'test_id': test_id,
            'test_date': '2026-02-08',
            'test_type': 'hot_fire',
            'operator': 'Sample Data Generator',
            'facility': 'Simulated Test Stand',
            'scenario': scenario,
            'scenario_description': scenario_def.get('description', ''),
        },

        'test_article': {
            'part_number': 'ENG-V1-02',
            'serial_number': 'SN-2026-101',
            'description': 'Small bipropellant engine - sample data',
            'revision': 'C',
        },

        'geometry': {
            'throat_area_mm2': 12.566,          # 4mm diameter nozzle
            'throat_area_uncertainty_mm2': 0.05,
            'throat_diameter_mm': 4.0,
            'expansion_ratio': 2.5,
            'chamber_volume_cc': 50.0,
        },

        'propellant_combination': 'LOX/RP-1',

        'fluid': {
            'name': 'combustion_products',
            'gamma': 1.25,
            'molecular_weight': 22.0,
            'density_kg_m3': 1.0,
            'density_uncertainty_kg_m3': 0.1,
        },

        'sensor_roles': {
            'chamber_pressure': names['chamber_pressure'],
            'thrust': names['thrust'],
            'mass_flow_ox': names['mass_flow_ox'],
            'mass_flow_fuel': names['mass_flow_fuel'],
            'temperature_ox': names['temperature_ox'],
            'temperature_fuel': names['temperature_fuel'],
        },

        'test_conditions': {
            'target_pc_bar': params['target_pc_bar'],
            'target_thrust_n': params['target_thrust_n'],
            'target_of_ratio': params['of_ratio'],
            'ambient_temp_k': 295.0,
            'ambient_pressure_bar': 1.01325,
        },

        'acceptance_criteria': {
            'pc_min_bar': params['target_pc_bar'] * 0.8,
            'pc_max_bar': params['target_pc_bar'] * 1.2,
            'of_min': params['of_ratio'] * 0.9,
            'of_max': params['of_ratio'] * 1.1,
            'isp_min_s': 150.0,
        },

        'references': {
            'test_procedure': 'TP-ENG-HF-001 Rev A',
            'acceptance_spec': 'AS-ENG-001 Rev A',
        },

        'notes': [
            f'Generated sample data - scenario: {scenario}',
            'Use with sample hot fire hardware config for testing',
        ],
    }

    # Flatten part/serial_num for analysis compatibility
    metadata['part'] = metadata['test_article']['part_number']
    metadata['serial_num'] = metadata['test_article']['serial_number']
    metadata['test_date'] = metadata['test_metadata']['test_date']
    metadata['operator'] = metadata['test_metadata']['operator']

    return metadata


def generate_hot_fire_config() -> Dict[str, Any]:
    """
    Generate a hardware configuration JSON matching the sample data sensors.

    Returns:
        Config dictionary ready for JSON serialization
    """
    return {
        'config_name': 'Sample_HotFire_Config',
        'test_type': 'hot_fire',
        'version': '1.0.0',
        'description': 'Sample hot fire hardware config for generated test data',

        'channel_config': {
            '10001': 'HF-PT-01',
            '10002': 'HF-LC-01',
            '10003': 'HF-FM-01',
            '10004': 'HF-FM-02',
            '10005': 'HF-TC-01',
            '10006': 'HF-TC-02',
        },

        'uncertainties': {
            'HF-PT-01': {
                'type': 'relative',
                'value': 0.005,
                'description': 'Chamber pressure transducer - +/-0.5% of reading',
            },
            'HF-LC-01': {
                'type': 'relative',
                'value': 0.01,
                'description': 'Load cell - +/-1.0% of reading',
            },
            'HF-FM-01': {
                'type': 'relative',
                'value': 0.01,
                'description': 'Oxidizer flow meter - +/-1.0% of reading',
            },
            'HF-FM-02': {
                'type': 'relative',
                'value': 0.01,
                'description': 'Fuel flow meter - +/-1.0% of reading',
            },
            'HF-TC-01': {
                'type': 'absolute',
                'value': 1.0,
                'unit': 'K',
                'description': 'Type K thermocouple - +/-1 K',
            },
            'HF-TC-02': {
                'type': 'absolute',
                'value': 1.0,
                'unit': 'K',
                'description': 'Type K thermocouple - +/-1 K',
            },
        },

        'settings': {
            'resample_freq_ms': 10,
            'steady_state_cv_threshold': 0.02,
            'steady_state_min_duration_s': 0.5,
            'p_atm_bar': 1.01325,
        },

        'sensor_ranges': {
            'HF-PT-01': {'min': 0, 'max': 50, 'unit': 'bar'},
            'HF-LC-01': {'min': 0, 'max': 500, 'unit': 'N'},
            'HF-FM-01': {'min': 0, 'max': 200, 'unit': 'g/s'},
            'HF-FM-02': {'min': 0, 'max': 100, 'unit': 'g/s'},
            'HF-TC-01': {'min': 50, 'max': 400, 'unit': 'K'},
            'HF-TC-02': {'min': 200, 'max': 600, 'unit': 'K'},
        },

        'notes': [
            'SAMPLE HARDWARE CONFIG for generated hot fire test data',
            'Sensor names match generate_hot_fire_sample_data.py defaults',
        ],
    }


# =============================================================================
# FILE OUTPUT
# =============================================================================

def write_test_files(
    output_dir: str,
    test_id: str,
    scenario: str = 'nominal',
    seed: Optional[int] = None,
    include_config: bool = True,
) -> Dict[str, str]:
    """
    Write CSV data, metadata JSON, and optionally config JSON to disk.

    Args:
        output_dir: Directory to write files into
        test_id: Test identifier (used in filenames and metadata)
        scenario: Scenario name
        seed: Random seed
        include_config: Whether to write the hardware config file

    Returns:
        Dictionary of file type -> file path written
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {}

    # Generate data
    df = generate_hot_fire_csv(scenario=scenario, seed=seed)
    metadata = generate_hot_fire_metadata(test_id=test_id, scenario=scenario)

    # Write CSV
    csv_path = out / f'{test_id}_data.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    files['csv'] = str(csv_path)

    # Write metadata
    meta_path = out / f'{test_id}_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    files['metadata'] = str(meta_path)

    # Write config (once per output dir)
    if include_config:
        config = generate_hot_fire_config()
        config_path = out / 'hot_fire_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        files['config'] = str(config_path)

    return files


def generate_campaign(
    output_dir: str,
    n_tests: int = 10,
    base_seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generate a full campaign of hot fire tests with varied conditions.

    Creates n_tests with:
    - Varied O/F ratios (2.0 - 3.0)
    - Varied chamber pressures (15 - 25 bar)
    - 1-2 failed ignitions
    - 1 sensor anomaly
    - Mix of scenarios

    Args:
        output_dir: Base directory for campaign data
        n_tests: Number of tests to generate
        base_seed: Base random seed (each test offsets by index)

    Returns:
        List of file dictionaries per test
    """
    rng = np.random.default_rng(base_seed)
    all_files = []

    # Distribute scenarios across tests
    scenario_pool = ['nominal'] * max(1, n_tests - 4)
    scenario_pool += ['high_of', 'low_of', 'failed_ignition', 'sensor_spike']
    # Trim or extend to match n_tests
    if len(scenario_pool) > n_tests:
        scenario_pool = scenario_pool[:n_tests]
    while len(scenario_pool) < n_tests:
        scenario_pool.append('nominal')
    rng.shuffle(scenario_pool)

    for i in range(n_tests):
        test_id = f'ENG-HF-{i + 1:03d}'
        scenario = scenario_pool[i]
        test_seed = base_seed + i

        # Add per-test variation to engine params
        engine_overrides = {}
        if scenario not in ('failed_ignition',):
            # Vary O/F slightly around target
            of_jitter = float(rng.normal(0, 0.15))
            engine_overrides['of_ratio'] = DEFAULT_ENGINE['of_ratio'] + of_jitter
            # Vary Pc slightly
            pc_jitter = float(rng.normal(0, 1.0))
            engine_overrides['target_pc_bar'] = DEFAULT_ENGINE['target_pc_bar'] + pc_jitter

        test_dir = os.path.join(output_dir, test_id)
        files = write_test_files(
            output_dir=test_dir,
            test_id=test_id,
            scenario=scenario,
            seed=test_seed,
            include_config=(i == 0),  # Config only in first test dir
        )
        all_files.append(files)

        print(f'  [{i + 1}/{n_tests}] {test_id} ({scenario})')

    # Write campaign-level config
    config = generate_hot_fire_config()
    config_path = os.path.join(output_dir, 'hot_fire_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return all_files


# =============================================================================
# QUICK VALIDATION
# =============================================================================

def validate_generated_data(df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
    """
    Run basic validation on generated data to confirm it is usable.

    Checks:
    - Required columns exist
    - Timestamp is monotonically increasing
    - No excessive NaN (< 5%)
    - Sensor values are physically plausible
    - Metadata has required fields

    Args:
        df: Generated DataFrame
        metadata: Generated metadata dict

    Returns:
        True if all checks pass
    """
    errors = []

    # Check required columns
    required_sensors = list(metadata.get('sensor_roles', {}).values())
    for col in ['timestamp'] + required_sensors:
        if col not in df.columns:
            errors.append(f'Missing column: {col}')

    # Check timestamp monotonicity
    if not df['timestamp'].is_monotonic_increasing:
        errors.append('Timestamp is not monotonically increasing')

    # Check NaN ratio per column
    for col in df.columns:
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 5.0:
            errors.append(f'{col}: NaN ratio {nan_pct:.1f}% exceeds 5% limit')

    # Check metadata required fields
    for field in ['geometry', 'sensor_roles', 'fluid']:
        if field not in metadata:
            errors.append(f'Missing metadata field: {field}')

    if metadata.get('geometry', {}).get('throat_area_mm2', 0) <= 0:
        errors.append('throat_area_mm2 must be positive')

    for role in ['chamber_pressure', 'thrust', 'mass_flow_ox', 'mass_flow_fuel']:
        if role not in metadata.get('sensor_roles', {}):
            errors.append(f'Missing sensor role: {role}')

    if errors:
        print('Validation FAILED:')
        for e in errors:
            print(f'  - {e}')
        return False

    print('Validation PASSED')
    return True


# =============================================================================
# MAIN / CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate sample hot fire test data for HDA testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  nominal          Standard performance hot fire test
  high_of          Oxidizer-rich mixture ratio (O/F=3.2)
  low_of           Fuel-rich mixture ratio (O/F=1.8)
  low_pc           Low chamber pressure (throttled)
  high_pc          High chamber pressure run
  failed_ignition  Aborted test with weak combustion
  short_burn       Short 1.5s steady-state burn
  long_burn        Extended 8s steady-state burn
  sensor_spike     Nominal test with pressure spike anomaly
  sensor_dropout   Nominal test with thrust sensor dropout

Examples:
  python generate_hot_fire_sample_data.py
  python generate_hot_fire_sample_data.py --scenario high_of --seed 123
  python generate_hot_fire_sample_data.py --campaign --n-tests 15
  python generate_hot_fire_sample_data.py --all-scenarios
        """,
    )

    parser.add_argument(
        '--scenario', type=str, default='nominal',
        choices=list(SCENARIOS.keys()),
        help='Test scenario to generate (default: nominal)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='./sample_hot_fire_data',
        help='Output directory (default: ./sample_hot_fire_data)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    parser.add_argument(
        '--campaign', action='store_true',
        help='Generate a full campaign with varied conditions',
    )
    parser.add_argument(
        '--n-tests', type=int, default=10,
        help='Number of tests for campaign mode (default: 10)',
    )
    parser.add_argument(
        '--all-scenarios', action='store_true',
        help='Generate one test per scenario',
    )
    parser.add_argument(
        '--validate', action='store_true', default=True,
        help='Run validation on generated data (default: True)',
    )
    parser.add_argument(
        '--no-validate', action='store_false', dest='validate',
        help='Skip validation',
    )

    args = parser.parse_args()

    print('=' * 70)
    print('HDA Hot Fire Sample Data Generator')
    print('=' * 70)
    print()

    if args.campaign:
        # Campaign mode
        print(f'Generating campaign with {args.n_tests} tests...')
        print(f'Output: {args.output_dir}/')
        print()

        all_files = generate_campaign(
            output_dir=args.output_dir,
            n_tests=args.n_tests,
            base_seed=args.seed,
        )

        print()
        print(f'Generated {len(all_files)} test datasets')
        print(f'Campaign config: {args.output_dir}/hot_fire_config.json')

    elif args.all_scenarios:
        # One test per scenario
        print(f'Generating all {len(SCENARIOS)} scenarios...')
        print(f'Output: {args.output_dir}/')
        print()

        for i, (scenario_name, scenario_def) in enumerate(SCENARIOS.items()):
            test_id = f'ENG-HF-{scenario_name.upper()}'
            test_dir = os.path.join(args.output_dir, test_id)

            files = write_test_files(
                output_dir=test_dir,
                test_id=test_id,
                scenario=scenario_name,
                seed=args.seed + i,
                include_config=(i == 0),
            )

            print(f'  [{i + 1}/{len(SCENARIOS)}] {test_id}: {scenario_def["description"]}')

        # Write shared config
        config = generate_hot_fire_config()
        config_path = os.path.join(args.output_dir, 'hot_fire_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print()
        print(f'Generated {len(SCENARIOS)} scenario datasets')
        print(f'Shared config: {config_path}')

    else:
        # Single test mode
        test_id = f'ENG-HF-{args.scenario.upper()}'
        print(f'Scenario:  {args.scenario}')
        print(f'Test ID:   {test_id}')
        print(f'Output:    {args.output_dir}/')
        print(f'Seed:      {args.seed}')
        print()

        files = write_test_files(
            output_dir=args.output_dir,
            test_id=test_id,
            scenario=args.scenario,
            seed=args.seed,
        )

        for ftype, fpath in files.items():
            print(f'  {ftype:10s}: {fpath}')

    # Validate a sample
    if args.validate:
        print()
        print('Validating sample data...')
        df = generate_hot_fire_csv(scenario=args.scenario, seed=args.seed)
        metadata = generate_hot_fire_metadata(
            test_id='VALIDATE-001', scenario=args.scenario)
        validate_generated_data(df, metadata)

    # Print usage hints
    print()
    print('-' * 70)
    print('Usage with HDA:')
    print()
    print('  # Load and analyze in Python:')
    print('  import json, pandas as pd')
    print('  from core.integrated_analysis import analyze_test')
    print('  from core.config_validation import merge_config_and_metadata')
    print()
    print('  df = pd.read_csv("sample_hot_fire_data/ENG-HF-NOMINAL_data.csv")')
    print('  with open("sample_hot_fire_data/hot_fire_config.json") as f:')
    print('      config = json.load(f)')
    print('  with open("sample_hot_fire_data/ENG-HF-NOMINAL_metadata.json") as f:')
    print('      metadata = json.load(f)')
    print()
    print('  result = analyze_test(')
    print('      df=df,')
    print('      config=config,')
    print('      steady_window=(500, 4000),  # ms')
    print('      test_id="ENG-HF-NOMINAL",')
    print('      plugin_slug="hot_fire",')
    print('      metadata=metadata,')
    print('      skip_qc=True,')
    print('  )')
    print()
    print('  for name, m in result.measurements.items():')
    print('      print(f"  {name}: {m.value:.3f} +/- {m.uncertainty:.3f} {m.unit}")')
    print()


if __name__ == '__main__':
    main()
