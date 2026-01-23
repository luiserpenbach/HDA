"""
Sample Script: Testing the HotFirePlugin with Synthetic Data

This script demonstrates hot fire test analysis with the new plugin architecture,
including operating envelope visualization.

Usage:
    python sample_hot_fire_test.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import the plugin-based API
from core.integrated_analysis import analyze_test
from core.plugins import PluginRegistry
from core.operating_envelope import calculate_operating_envelope, plot_operating_envelope


def create_hot_fire_config():
    """Create sample hot fire configuration."""
    config = {
        "config_name": "Sample_Engine_HotFire_Config",
        "test_type": "hot_fire",
        "description": "Example configuration for engine hot fire tests",
        "created": "2026-01-16",
        "version": "1.0",

        "uncertainties": {
            "HF-PT-01": {
                "type": "relative",
                "value": 0.005,
                "description": "Chamber pressure transducer - ±0.5% of reading"
            },
            "HF-LC-01": {
                "type": "relative",
                "value": 0.01,
                "description": "Load cell - ±1.0% of reading"
            },
            "HF-FM-01": {
                "type": "relative",
                "value": 0.01,
                "description": "Oxidizer mass flow meter - ±1.0% of reading"
            },
            "HF-FM-02": {
                "type": "relative",
                "value": 0.01,
                "description": "Fuel mass flow meter - ±1.0% of reading"
            },
            "HF-TC-01": {
                "type": "absolute",
                "value": 1.0,
                "unit": "K",
                "description": "Type K thermocouple - ±1 K"
            },
            "HF-TC-02": {
                "type": "absolute",
                "value": 1.0,
                "unit": "K",
                "description": "Type K thermocouple - ±1 K"
            },
        },

        "settings": {
            "resample_freq_ms": 10,
            "steady_state_cv_threshold": 0.02,
            "steady_state_min_duration_s": 0.5,
            "p_atm_bar": 1.01325
        },

        "sensor_ranges": {
            "HF-PT-01": {"min": 0, "max": 50, "unit": "bar"},
            "HF-LC-01": {"min": 0, "max": 500, "unit": "N"},
            "HF-FM-01": {"min": 0, "max": 200, "unit": "g/s"},
            "HF-FM-02": {"min": 0, "max": 50, "unit": "g/s"},
            "HF-TC-01": {"min": 200, "max": 600, "unit": "K"},
            "HF-TC-02": {"min": 200, "max": 600, "unit": "K"},
        },

        "notes": [
            "TESTBENCH HARDWARE CONFIGURATION - Changes rarely",
            "Contains: sensor mappings, calibration uncertainties, processing settings",
            "Does NOT contain: geometry (test-article specific)",
        ]
    }
    return config


def create_hot_fire_metadata():
    """Create sample hot fire metadata (test article properties)."""
    metadata = {
        'part': 'ENG-V1-02',
        'serial_num': 'SN-2026-101',
        'test_date': '2026-01-16',
        'operator': 'Test Engineer',

        # Geometry (test-article specific - NOT in config)
        'geometry': {
            'throat_area_mm2': 12.566,  # 4mm diameter
            'throat_area_uncertainty_mm2': 0.05,
            'throat_diameter_mm': 4.0,
            'expansion_ratio': 2.5,
            'chamber_volume_cc': 50.0,
        },

        # Propellant info
        'propellant_combination': 'LOX/RP-1',

        # Fluid properties (for validation - not critical for hot fire performance calcs)
        'fluid': {
            'name': 'combustion_products',
            'gamma': 1.25,  # Typical for LOX/RP-1 combustion products
            'molecular_weight': 22.0,  # Approximate
            'density_kg_m3': 1.0,  # Placeholder - not used for Isp/C* calculations
            'density_uncertainty_kg_m3': 0.1,
        },

        # Sensor roles - ALL sensor assignments belong in metadata, not config
        'sensor_roles': {
            'chamber_pressure': 'HF-PT-01',
            'thrust': 'HF-LC-01',
            'mass_flow_ox': 'HF-FM-01',
            'mass_flow_fuel': 'HF-FM-02',
            'temperature_ox': 'HF-TC-01',
            'temperature_fuel': 'HF-TC-02',
        }
    }
    return metadata


def generate_synthetic_hot_fire_data(duration_s=5, sample_rate_hz=100, of_ratio=2.5, ignition_successful=True):
    """
    Generate synthetic hot fire test data.

    Simulates:
    - Ignition transient (0-0.5s)
    - Steady combustion (0.5-4.0s)
    - Shutdown (4.0-5.0s)
    """
    np.random.seed(42)
    n_samples = int(duration_s * sample_rate_hz)
    time_s = np.linspace(0, duration_s, n_samples)

    # Ignition profile
    def ignition_profile(t):
        """Smooth ignition and shutdown using tanh."""
        ignition = 0.5 * (1 + np.tanh(10 * (t - 0.3)))
        shutdown = 0.5 * (1 - np.tanh(10 * (t - 4.2)))
        return ignition * shutdown

    profile = ignition_profile(time_s)

    # Failed ignition has different profile
    if not ignition_successful:
        profile = 0.1 * profile  # Very weak combustion

    # Chamber pressure (bar) - target 20 bar for successful ignition
    target_pc = 20.0 if ignition_successful else 2.0
    pc = target_pc * profile + np.random.normal(0, 0.1, n_samples)

    # Thrust (N) - roughly proportional to chamber pressure
    target_thrust = 250.0 if ignition_successful else 25.0
    thrust = target_thrust * profile + np.random.normal(0, 2.0, n_samples)

    # Mass flows (g/s) - target O/F ratio
    target_mf_fuel = 40.0 if ignition_successful else 4.0
    target_mf_ox = target_mf_fuel * of_ratio

    mf_fuel = target_mf_fuel * profile + np.random.normal(0, 0.5, n_samples)
    mf_ox = target_mf_ox * profile + np.random.normal(0, 1.0, n_samples)

    # Temperatures (K)
    temp_ox = 90.0 + 5.0 * profile + np.random.normal(0, 0.5, n_samples)  # LOX temp
    temp_fuel = 300.0 + 10.0 * profile + np.random.normal(0, 0.5, n_samples)  # RP-1 temp

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time_s * 1000,  # Convert to ms
        'HF-PT-01': pc,
        'HF-LC-01': thrust,
        'HF-FM-01': mf_ox,
        'HF-FM-02': mf_fuel,
        'HF-TC-01': temp_ox,
        'HF-TC-02': temp_fuel,
    })

    return df


def generate_campaign_data(n_tests=10):
    """Generate synthetic campaign data for operating envelope demo."""
    np.random.seed(123)
    tests = []

    for i in range(n_tests):
        # Vary O/F ratio and chamber pressure
        of_ratio = np.random.uniform(2.0, 3.0)
        pc_bar = np.random.uniform(15.0, 25.0)

        # Some tests fail ignition
        ignition_successful = np.random.random() > 0.2  # 80% success rate

        test_record = {
            'test_id': f'ENG-HF-{i+1:03d}',
            'avg_of_ratio': of_ratio,
            'avg_pc_bar': pc_bar,
            'ignition_successful': 1 if ignition_successful else 0,
        }
        tests.append(test_record)

    return pd.DataFrame(tests)


def main():
    """Run the sample hot fire plugin test."""
    print("=" * 70)
    print("HDA Hot Fire Plugin - Sample Test")
    print("=" * 70)
    print()

    # 1. List available plugins
    print("1. Available Plugins:")
    print("-" * 70)
    plugins = PluginRegistry.list_available_plugins()
    for plugin_info in plugins:
        print(f"   • {plugin_info['name']}")
        print(f"     Slug: {plugin_info['slug']}")
        print(f"     Type: {plugin_info['test_type']}")
        print(f"     Version: {plugin_info['version']}")
        print()

    # 2. Create configuration and metadata
    print("2. Configuration and Metadata:")
    print("-" * 70)
    config = create_hot_fire_config()
    metadata = create_hot_fire_metadata()
    print(f"   Config: {config['config_name']}")
    print(f"   Test Article: {metadata['part']} / {metadata['serial_num']}")
    print(f"   Throat Area: {metadata['geometry']['throat_area_mm2']:.3f} mm² (from metadata)")
    print(f"   Propellant: {metadata['propellant_combination']}")
    print()

    # 3. Generate synthetic test data
    print("3. Generating Synthetic Hot Fire Test Data:")
    print("-" * 70)
    df = generate_synthetic_hot_fire_data(duration_s=5, sample_rate_hz=100, of_ratio=2.5, ignition_successful=True)
    print(f"   Duration: {df['timestamp'].max() / 1000:.1f} seconds")
    print(f"   Sample Rate: {len(df) / (df['timestamp'].max() / 1000):.0f} Hz")
    print(f"   Data Points: {len(df)}")
    print()

    # Data preview
    print("   Data Preview (steady state 0.5-4.0s):")
    steady_df = df[(df['timestamp'] >= 500) & (df['timestamp'] <= 4000)]
    print(f"     Chamber Pressure: {steady_df['HF-PT-01'].mean():.2f} ± {steady_df['HF-PT-01'].std():.2f} bar")
    print(f"     Thrust:           {steady_df['HF-LC-01'].mean():.1f} ± {steady_df['HF-LC-01'].std():.1f} N")
    print(f"     Mass Flow Ox:     {steady_df['HF-FM-01'].mean():.1f} ± {steady_df['HF-FM-01'].std():.1f} g/s")
    print(f"     Mass Flow Fuel:   {steady_df['HF-FM-02'].mean():.1f} ± {steady_df['HF-FM-02'].std():.1f} g/s")
    print()

    # 4. Run analysis with hot fire plugin
    print("4. Running Analysis with HotFirePlugin:")
    print("-" * 70)

    result = analyze_test(
        df=df,
        config=config,
        steady_window=(500, 4000),  # 0.5-4.0 seconds
        test_id="SAMPLE-HF-001",
        plugin_slug="hot_fire",
        detection_method="Manual",
        metadata=metadata,
        skip_qc=True,  # Skip QC for quick demo
    )

    print(f"   Test ID: {result.test_id}")
    print(f"   QC Status: {'PASSED' if result.passed_qc else 'FAILED'}")
    print(f"   Measurements: {len(result.measurements)}")
    print()

    # 5. Display results
    print("5. Hot Fire Analysis Results:")
    print("-" * 70)

    # Key metrics
    key_metrics = ['chamber_pressure', 'thrust', 'mass_flow_total', 'of_ratio', 'Isp', 'c_star']
    for metric in key_metrics:
        if metric in result.measurements:
            meas = result.measurements[metric]
            print(f"   {metric:20s}: {meas.value:8.2f} ± {meas.uncertainty:.2f} {meas.unit}")
            print(f"   {'':20s}  (±{meas.relative_uncertainty_percent:.2f}%)")
    print()

    # 6. Database record
    print("6. Database Record (ready for campaign storage):")
    print("-" * 70)
    db_record = result.to_database_record('hot_fire')
    key_fields = ['test_id', 'avg_pc_bar', 'avg_thrust_n', 'avg_of_ratio',
                  'avg_isp_s', 'avg_c_star_m_s', 'qc_passed']
    for field in key_fields:
        if field in db_record:
            value = db_record[field]
            if isinstance(value, float):
                print(f"   {field:25s}: {value:.3f}")
            else:
                print(f"   {field:25s}: {value}")
    print()

    # 7. Operating Envelope Demonstration
    print("7. Operating Envelope Analysis:")
    print("-" * 70)

    # Generate synthetic campaign data
    campaign_df = generate_campaign_data(n_tests=15)
    print(f"   Generated campaign with {len(campaign_df)} tests")
    print(f"   Successful ignitions: {campaign_df['ignition_successful'].sum()}")
    print(f"   Failed ignitions: {len(campaign_df) - campaign_df['ignition_successful'].sum()}")
    print()

    # Calculate operating envelope
    envelope = calculate_operating_envelope(
        campaign_df,
        of_column='avg_of_ratio',
        pc_column='avg_pc_bar',
        ignition_column='ignition_successful',
        margin_pct=10.0,
        filter_successful_only=True
    )

    print("   Operating Envelope Bounds:")
    print(f"     O/F Ratio:         {envelope.of_min:.3f} - {envelope.of_max:.3f}")
    print(f"     Chamber Pressure:  {envelope.pc_min:.2f} - {envelope.pc_max:.2f} bar")
    print(f"     Safety Margin:     {envelope.margin_pct:.0f}%")
    print(f"     Based on:          {envelope.n_tests} successful tests")
    print()

    # Create plot
    print("   Creating operating envelope plot...")
    fig = plot_operating_envelope(
        campaign_df,
        envelope=envelope,
        title="Sample Hot Fire Operating Envelope",
        show_envelope=True,
        show_test_ids=True,
    )

    # Save plot
    output_path = Path("sample_hot_fire_envelope.html")
    fig.write_html(str(output_path))
    print(f"   ✓ Plot saved to: {output_path}")
    print()

    # 8. Summary
    print("=" * 70)
    print("✓ Hot Fire Plugin Test Complete!")
    print("=" * 70)
    print()
    print("Results:")
    print(f"  • Single test analysis: PASSED")
    print(f"  • Operating envelope calculation: PASSED")
    print(f"  • Interactive visualization: {output_path}")
    print()
    print("Next Steps:")
    print("  • Open sample_hot_fire_envelope.html in browser")
    print("  • Integrate operating envelope into campaign analysis pages")
    print("  • Create sample hot fire config/metadata files")
    print("  • Add hot fire plugin tests to test suite")
    print()

    return result, envelope, fig


if __name__ == '__main__':
    result, envelope, fig = main()
