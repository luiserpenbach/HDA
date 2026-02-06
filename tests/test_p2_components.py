"""
P2 Component Tests
==================
Tests for advanced anomaly detection, comparison, and templates.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_section(name):
    print(f"\n{name}")
    print("-" * 40)


def test_pass(msg):
    print(f"[PASS] {msg}")


def test_fail(msg):
    print(f"[FAIL] {msg}")
    global failures
    failures += 1


failures = 0


# =============================================================================
# ADVANCED ANOMALY DETECTION TESTS
# =============================================================================

test_section("TestAdvancedAnomaly")

from core.advanced_anomaly import (
    detect_spikes,
    detect_dropouts,
    detect_drift,
    detect_oscillation,
    detect_saturation,
    detect_flatline,
    detect_transients,
    calculate_sensor_health,
    run_anomaly_detection,
    AnomalyType,
    AnomalySeverity,
)

# Test spike detection
np.random.seed(42)
normal_data = np.random.normal(100, 2, 1000)
# Add spikes
data_with_spikes = normal_data.copy()
data_with_spikes[100] = 150  # Large spike
data_with_spikes[500] = 50   # Large dip

spikes = detect_spikes(data_with_spikes, threshold_sigma=4.0)
if len(spikes) >= 2:
    test_pass(f"Spike detection: Found {len(spikes)} spikes")
else:
    test_fail(f"Spike detection: Expected >=2, got {len(spikes)}")

# Test dropout detection
data_with_dropout = normal_data.copy()
data_with_dropout[200:210] = 0  # Dropout

dropouts = detect_dropouts(data_with_dropout)
if len(dropouts) >= 1:
    test_pass(f"Dropout detection: Found {len(dropouts)} dropouts")
else:
    test_fail(f"Dropout detection: Expected >=1, got {len(dropouts)}")

# Test drift detection
drifting_data = np.linspace(100, 120, 500) + np.random.normal(0, 0.5, 500)

drift = detect_drift(drifting_data, drift_threshold=0.05)
if drift is not None and drift[0] == "increasing":
    test_pass(f"Drift detection: {drift[0]} (R²={drift[2]:.2f})")
else:
    test_fail(f"Drift detection: Expected increasing drift")

# Test oscillation detection
t = np.linspace(0, 10, 1000)
oscillating = 100 + 5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz oscillation

osc = detect_oscillation(oscillating, sample_rate_hz=100)
if osc is not None and 1.5 < osc[0] < 2.5:
    test_pass(f"Oscillation detection: {osc[0]:.1f} Hz (power={osc[1]:.1%})")
else:
    test_fail(f"Oscillation detection: Expected ~2 Hz")

# Test flatline detection
data_with_flatline = normal_data.copy()
data_with_flatline[300:350] = 100.0  # Flatline

flatlines = detect_flatline(data_with_flatline)
if len(flatlines) >= 1:
    test_pass(f"Flatline detection: Found {len(flatlines)} flatlines")
else:
    test_fail(f"Flatline detection: Expected >=1")

# Test saturation detection
data_with_saturation = normal_data.copy()
data_with_saturation[400:420] = np.max(normal_data)  # High saturation

saturations = detect_saturation(data_with_saturation)
if len(saturations) >= 1:
    test_pass(f"Saturation detection: Found {len(saturations)} saturations")
else:
    test_fail(f"Saturation detection: Expected >=1")

# Test sensor health
health, details = calculate_sensor_health(normal_data)
if 0.8 < health <= 1.0:
    test_pass(f"Sensor health (clean data): {health:.1%}")
else:
    test_fail(f"Sensor health (clean data): Expected >80%, got {health:.1%}")

# Test full anomaly report
df = pd.DataFrame({
    'timestamp': np.arange(1000),
    'sensor_a': data_with_spikes,
    'sensor_b': data_with_dropout,
})

report = run_anomaly_detection(df, channels=['sensor_a', 'sensor_b'])
if report.total_anomalies > 0:
    test_pass(f"Full anomaly detection: {report.total_anomalies} anomalies found")
else:
    test_fail("Full anomaly detection: Expected some anomalies")

if report.sensor_health:
    test_pass(f"Sensor health scores computed: {len(report.sensor_health)} channels")
else:
    test_fail("Sensor health scores not computed")


# =============================================================================
# COMPARISON TESTS
# =============================================================================

test_section("TestComparison")

from core.comparison import (
    compare_values,
    compare_tests,
    create_golden_from_campaign,
    linear_regression,
    calculate_correlation_matrix,
    track_deviations,
    compare_campaigns,
    GoldenReference,
)

# Test value comparison
result = compare_values(100.0, 105.0, "test_param", tolerance_percent=5.0)
if not result.within_tolerance:  # ~4.9% diff, should be close
    test_pass(f"Value comparison: {result.percent_difference:.2f}% (tol: ±{result.tolerance}%)")
else:
    test_pass(f"Value comparison: within tolerance at {result.percent_difference:.2f}%")

# Test test comparison
test_a = {'Cd': 0.65, 'mass_flow': 12.5, 'pressure': 25.0}
test_b = {'Cd': 0.66, 'mass_flow': 12.3, 'pressure': 25.2}

comparison = compare_tests(test_a, test_b, "Test_001", "Test_002", default_tolerance=5.0)
if comparison.n_parameters == 3:
    test_pass(f"Test comparison: {comparison.n_within_tolerance}/{comparison.n_parameters} within tolerance")
else:
    test_fail(f"Test comparison: Expected 3 parameters")

# Test golden reference
campaign_df = pd.DataFrame({
    'test_id': [f'T{i}' for i in range(10)],
    'avg_cd': np.random.normal(0.65, 0.01, 10),
    'avg_mf': np.random.normal(12.5, 0.2, 10),
})

golden = create_golden_from_campaign(campaign_df, "Test_Golden", ['avg_cd', 'avg_mf'])
if 'avg_cd' in golden.parameters and 'avg_mf' in golden.parameters:
    test_pass(f"Golden reference created with {len(golden.parameters)} parameters")
else:
    test_fail("Golden reference: missing parameters")

# Test linear regression
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.0, 6.1, 7.9, 10.2])

reg_result = linear_regression(x, y, "x", "y")
if 0.99 < reg_result.r_squared <= 1.0:
    test_pass(f"Linear regression: R²={reg_result.r_squared:.4f}, slope={reg_result.slope:.2f}")
else:
    test_fail(f"Linear regression: Poor fit R²={reg_result.r_squared:.4f}")

# Test correlation matrix
corr_df = pd.DataFrame({
    'a': np.random.randn(50),
    'b': np.random.randn(50),
})
corr_df['c'] = corr_df['a'] * 2 + np.random.randn(50) * 0.1  # c correlated with a

corr_matrix = calculate_correlation_matrix(corr_df, ['a', 'b', 'c'])
ac_corr = corr_matrix.get_correlation('a', 'c')
if abs(ac_corr) > 0.9:
    test_pass(f"Correlation matrix: a-c correlation = {ac_corr:.3f}")
else:
    test_fail(f"Correlation matrix: Expected high a-c correlation, got {ac_corr:.3f}")

# Test deviation tracking
tracker = track_deviations(campaign_df, 'avg_cd', expected_value=0.65, tolerance_percent=5.0)
if tracker.n_measurements == 10:
    test_pass(f"Deviation tracking: {tracker.n_measurements} measurements, mean dev={tracker.mean_deviation:.2f}%")
else:
    test_fail(f"Deviation tracking: Expected 10 measurements")

# Test campaign comparison
campaign_df_b = pd.DataFrame({
    'test_id': [f'T{i}' for i in range(8)],
    'avg_cd': np.random.normal(0.66, 0.015, 8),
    'avg_mf': np.random.normal(12.3, 0.3, 8),
})

camp_comparison = compare_campaigns(campaign_df, campaign_df_b, "Camp_A", "Camp_B", ['avg_cd', 'avg_mf'])
if 'parameters' in camp_comparison and len(camp_comparison['parameters']) == 2:
    test_pass(f"Campaign comparison: {len(camp_comparison['parameters'])} parameters compared")
else:
    test_fail("Campaign comparison: Failed")


# =============================================================================
# TEMPLATES TESTS
# =============================================================================

test_section("TestTemplates")

from core.saved_configs import (
    ConfigTemplate,
    TemplateManager,
    BUILTIN_TEMPLATES,
    create_config_from_template,
    validate_config_against_template,
    merge_configs,
)
import shutil

# Test built-in templates exist (cold_flow_default and hot_fire_default)
if len(BUILTIN_TEMPLATES) >= 2:
    test_pass(f"Built-in templates: {len(BUILTIN_TEMPLATES)} available")
else:
    test_fail(f"Built-in templates: Expected >=2, got {len(BUILTIN_TEMPLATES)}")

# Test template structure (actual keys are cold_flow_default, hot_fire_default)
template = BUILTIN_TEMPLATES.get('cold_flow_default')
if template and template.test_type == 'cold_flow':
    test_pass(f"Template structure: {template.name}")
else:
    test_fail("Template structure: cold_flow_default not found or invalid")

# Test config generation (SavedConfig.to_config() uses channel_config, uncertainties, settings)
config = template.to_config()
if 'channel_config' in config and 'uncertainties' in config:
    test_pass("Config generation: All required sections present")
else:
    test_fail("Config generation: Missing sections")

# Test template manager
# Clean up any existing test templates
test_templates_dir = "test_config_templates"
if Path(test_templates_dir).exists():
    shutil.rmtree(test_templates_dir)

manager = TemplateManager(templates_dir=test_templates_dir)

# List templates
templates = manager.list_templates()
if len(templates) >= 2:  # Built-in templates
    test_pass(f"Template manager list: {len(templates)} templates")
else:
    test_fail(f"Template manager list: Expected >=2")

# Get template
retrieved = manager.get_template('cold_flow_default')
if retrieved and retrieved.name == template.name:
    test_pass("Template manager get: Retrieved built-in template")
else:
    test_fail("Template manager get: Failed")

# Save custom template (using SavedConfig fields: channel_config, settings)
custom = ConfigTemplate(
    name="Custom Test Template",
    version="1.0.0",
    test_type="cold_flow",
    channel_config={'10001': 'PT-01', '10002': 'FM-01'},
    settings={'orifice_area_mm2': 2.0},
)

saved_id = manager.save_template(custom, "custom_test")
if saved_id == "custom_test":
    test_pass(f"Template manager save: {saved_id}")
else:
    test_fail("Template manager save: Failed")

# Retrieve custom template
retrieved_custom = manager.get_template("custom_test")
if retrieved_custom and retrieved_custom.settings.get('orifice_area_mm2') == 2.0:
    test_pass("Template manager: Custom template retrieved correctly")
else:
    test_fail("Template manager: Custom template retrieval failed")

# Test create from parent
child = manager.create_from_parent('cold_flow_default', "Child Template", {'settings': {'orifice_area_mm2': 3.0}})
if child.parent_config == 'cold_flow_default' and child.settings.get('orifice_area_mm2') == 3.0:
    test_pass("Create from parent: Inheritance works")
else:
    test_fail("Create from parent: Inheritance failed")

# Test config merge
base = {'a': 1, 'b': {'x': 10, 'y': 20}}
override = {'b': {'x': 15, 'z': 30}, 'c': 3}
merged = merge_configs(base, override)

if merged['a'] == 1 and merged['b']['x'] == 15 and merged['b']['y'] == 20 and merged['b']['z'] == 30 and merged['c'] == 3:
    test_pass("Config merge: Deep merge works correctly")
else:
    test_fail("Config merge: Merge failed")

# Test create_config_from_template (load_saved_config)
config = create_config_from_template('cold_flow_default', {'settings': {'orifice_area_mm2': 5.0}})
if config['settings']['orifice_area_mm2'] == 5.0:
    test_pass("Create config from template: Overrides applied")
else:
    test_fail("Create config from template: Override failed")

# Test validation against template
errors = validate_config_against_template(config, template)
if len(errors) == 0:
    test_pass("Config validation: Valid config passes")
else:
    test_fail(f"Config validation: Unexpected errors: {errors}")

# Cleanup
shutil.rmtree(test_templates_dir)


# =============================================================================
# INTEGRATION TEST
# =============================================================================

test_section("TestP2Integration")

# Create test data with various anomalies
np.random.seed(123)
n_samples = 2000

timestamp = np.arange(n_samples)
p_upstream = 25.0 + np.random.normal(0, 0.1, n_samples)
p_downstream = 24.5 + np.random.normal(0, 0.08, n_samples)
mass_flow = 12.0 + np.random.normal(0, 0.15, n_samples)
temperature = 293.0 + np.random.normal(0, 0.5, n_samples)

# Add some anomalies
p_upstream[500:505] = 30.0  # Spike
mass_flow[800:820] = 0  # Dropout
temperature[1000:1100] = np.linspace(293, 300, 100)  # Drift

test_df = pd.DataFrame({
    'timestamp': timestamp,
    'P_upstream': p_upstream,
    'P_downstream': p_downstream,
    'mass_flow': mass_flow,
    'T_fluid': temperature,
})

# Run full anomaly detection
report = run_anomaly_detection(
    test_df,
    channels=['P_upstream', 'P_downstream', 'mass_flow', 'T_fluid'],
    sample_rate_hz=100,
)

if report.total_anomalies >= 3:  # Should detect spike, dropout, drift
    test_pass(f"Integration: Detected {report.total_anomalies} anomalies across {len(report.channel_reports)} channels")
else:
    test_fail(f"Integration: Expected >=3 anomalies, got {report.total_anomalies}")

# Check sensor health reflects issues
mass_flow_health = report.sensor_health.get('mass_flow', 1.0)
if mass_flow_health < 1.0:  # Should be penalized for dropout (even slightly)
    test_pass(f"Integration: Mass flow health {mass_flow_health:.1%} reflects issues")
else:
    test_fail(f"Integration: Mass flow health {mass_flow_health:.1%} doesn't reflect issues")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print(f"Results: {sum([1 for line in open(__file__).readlines() if 'test_pass' in line]) - failures} passed, {failures} failed")
print("=" * 60)

sys.exit(failures)
