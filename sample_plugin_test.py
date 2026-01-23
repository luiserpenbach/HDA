"""
Sample Script: Testing the ColdFlowPlugin with Sample Data

This script demonstrates how to use the new plugin architecture
with the sample configuration and metadata files.

Usage:
    python sample_plugin_test.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import the new plugin-based API
from core.integrated_analysis import analyze_test
from core.plugins import PluginRegistry


def load_sample_config():
    """Load the sample configuration file."""
    with open('sample_cold_flow_config.json', 'r') as f:
        config = json.load(f)
    return config


def load_sample_metadata():
    """Load the sample metadata file."""
    with open('sample_test_metadata.json', 'r') as f:
        metadata_file = json.load(f)

    # Flatten for analysis - include geometry, fluid, and sensor_roles (test-specific)
    flattened = {
        'part': metadata_file['test_article']['part_number'],
        'serial_num': metadata_file['test_article']['serial_number'],
        'test_date': metadata_file['test_metadata']['test_date'],
        'operator': metadata_file['test_metadata']['operator'],

        # Geometry (test-article specific - NOT in config)
        'geometry': metadata_file['geometry'],

        # Fluid properties (test-specific - NOT in config)
        'fluid': metadata_file['fluid_properties']['oxidizer'],

        # Sensor roles (ALL sensor assignments - NOT in config)
        'sensor_roles': metadata_file['sensor_roles'],
    }
    return flattened


def generate_synthetic_test_data(duration_s=10, sample_rate_hz=100):
    """
    Generate synthetic cold flow test data for demonstration.

    Simulates a typical cold flow test with:
    - Ramp up (0-2s)
    - Steady state (2-8s)
    - Ramp down (8-10s)
    """
    np.random.seed(42)
    n_samples = int(duration_s * sample_rate_hz)
    time_s = np.linspace(0, duration_s, n_samples)

    # Create realistic test profile
    def ramp_profile(t):
        """Smooth ramp profile using tanh."""
        ramp_up = 0.5 * (1 + np.tanh(5 * (t - 1.5)))
        ramp_down = 0.5 * (1 - np.tanh(5 * (t - 8.5)))
        return ramp_up * ramp_down

    profile = ramp_profile(time_s)

    # Upstream pressure (bar) - target 10 bar
    p_upstream = 10.0 * profile + np.random.normal(0, 0.05, n_samples)

    # Downstream pressure (bar) - target 1 bar
    p_downstream = 1.0 * profile + np.random.normal(0, 0.01, n_samples)

    # Mass flow (g/s) - target 125 g/s, correlates with sqrt(delta_P)
    delta_p = p_upstream - p_downstream
    mass_flow = 125.0 * np.sqrt(np.maximum(delta_p, 0) / 9.0) + np.random.normal(0, 1.0, n_samples)

    # Temperature (K)
    temperature = 300.0 + np.random.normal(0, 0.5, n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time_s * 1000,  # Convert to ms for HDA convention
        'IG-PT-01': p_upstream,
        'IG-PT-02': p_downstream,
        'FM-01': mass_flow,
        'TC-01': temperature,
    })

    return df


def main():
    """Run the sample plugin test."""
    print("=" * 70)
    print("HDA Plugin Architecture - Sample Test")
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
        print(f"     Description: {plugin_info['description']}")
        print()

    # 2. Load configuration and metadata
    print("2. Loading Configuration and Metadata:")
    print("-" * 70)
    config = load_sample_config()
    metadata = load_sample_metadata()
    print(f"   Config: {config['config_name']} (testbench hardware)")
    print(f"   Test Article: {metadata['part']} / {metadata['serial_num']}")
    print(f"   Orifice Area: {metadata['geometry']['orifice_area_mm2']:.5f} mm² (from metadata)")
    print()

    # 3. Generate synthetic test data
    print("3. Generating Synthetic Test Data:")
    print("-" * 70)
    df = generate_synthetic_test_data(duration_s=10, sample_rate_hz=100)
    print(f"   Duration: {df['timestamp'].max() / 1000:.1f} seconds")
    print(f"   Sample Rate: {len(df) / (df['timestamp'].max() / 1000):.0f} Hz")
    print(f"   Data Points: {len(df)}")
    print(f"   Channels: {', '.join([c for c in df.columns if c != 'timestamp'])}")
    print()

    # Show data preview
    print("   Data Preview (steady state):")
    steady_df = df[(df['timestamp'] >= 2000) & (df['timestamp'] <= 8000)]
    print(f"     P_upstream:   {steady_df['IG-PT-01'].mean():.3f} ± {steady_df['IG-PT-01'].std():.3f} bar")
    print(f"     P_downstream: {steady_df['IG-PT-02'].mean():.3f} ± {steady_df['IG-PT-02'].std():.3f} bar")
    print(f"     Mass Flow:    {steady_df['FM-01'].mean():.2f} ± {steady_df['FM-01'].std():.2f} g/s")
    print()

    # 4. Run analysis with plugin
    print("4. Running Analysis with ColdFlowPlugin:")
    print("-" * 70)

    result = analyze_test(
        df=df,
        config=config,
        steady_window=(2000, 8000),  # 2-8 seconds
        test_id="SAMPLE-CF-001",
        plugin_slug="cold_flow",
        detection_method="Manual",
        metadata=metadata,
        skip_qc=True,  # Skip QC for quick demo
    )

    print(f"   Test ID: {result.test_id}")
    print(f"   QC Status: {'PASSED' if result.passed_qc else 'FAILED'}")
    print(f"   Measurements: {len(result.measurements)}")
    print()

    # 5. Display results
    print("5. Analysis Results:")
    print("-" * 70)

    # Key measurements
    key_metrics = ['pressure_upstream', 'delta_p', 'mass_flow', 'Cd']
    for metric in key_metrics:
        if metric in result.measurements:
            meas = result.measurements[metric]
            print(f"   {metric:20s}: {meas.value:8.4f} ± {meas.uncertainty:.4f} {meas.unit}")
            print(f"   {'':20s}  (±{meas.relative_uncertainty_percent:.2f}%)")
    print()

    # 6. Traceability info
    print("6. Traceability Information:")
    print("-" * 70)
    print(f"   Config Hash: {result.traceability.get('config_hash', 'N/A')[:20]}...")
    print(f"   Processing Version: {result.traceability.get('processing_version', 'N/A')}")
    print(f"   Analysis Timestamp: {result.traceability.get('analysis_timestamp_utc', 'N/A')}")
    print()

    # 7. Database record preview
    print("7. Database Record (ready for campaign storage):")
    print("-" * 70)
    db_record = result.to_database_record('cold_flow')
    key_fields = ['test_id', 'avg_cd_CALC', 'u_cd_CALC', 'cd_rel_uncertainty_pct',
                  'avg_mf_g_s', 'u_mf_g_s', 'qc_passed']
    for field in key_fields:
        if field in db_record:
            print(f"   {field:25s}: {db_record[field]}")
    print()

    # 8. Summary
    print("=" * 70)
    print("✓ Plugin Architecture Test Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  • Review PLUGIN_ARCHITECTURE.md for detailed documentation")
    print("  • Modify sample_cold_flow_config.json for your hardware")
    print("  • Modify sample_test_metadata.json for your test article")
    print("  • Use analyze_test() in your analysis workflows")
    print("  • Create custom plugins for new test types")
    print()

    return result


if __name__ == '__main__':
    result = main()
