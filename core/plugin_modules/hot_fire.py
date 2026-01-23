"""
Hot Fire Analysis Plugin

Implements hot fire test analysis for HDA's plugin architecture.

This plugin handles:
- Engine hot fire tests
- Igniter characterization
- Performance metrics (Isp, C*, thrust, O/F ratio)
- Operating envelope visualization

Measurements calculated:
- Chamber pressure (bar)
- Thrust (N)
- Total mass flow rate (g/s)
- O/F ratio (mixture ratio)
- Specific impulse (Isp, s)
- Characteristic velocity (C*, m/s)

Version: 1.0.0
Created: 2026-01-16
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

from core.plugins import (
    AnalysisPlugin,
    PluginMetadata,
    PluginRegistry,
    UncertaintySpec,
    UncertaintySourceType,
    ColumnSpec,
    InputSpec,
)

from core.qc_checks import QCReport, run_qc_checks, run_quick_qc
from core.uncertainty import (
    calculate_hot_fire_uncertainties,
    MeasurementWithUncertainty,
)
from core.config_validation import validate_config_simple


class HotFirePlugin:
    """
    Hot fire test analysis plugin.

    Handles analysis of engine hot fire tests, calculating performance
    metrics like Isp, C*, thrust, and O/F ratio.
    """

    def __init__(self):
        """Initialize the hot fire plugin with metadata."""
        self.metadata = PluginMetadata(
            name="Hot Fire Engine Analysis",
            slug="hot_fire",
            version="1.0.0",
            test_type="hot_fire",
            description="Analyze hot fire tests for engine performance (Isp, C*, thrust, O/F ratio, operating envelope)",
            author="HDA Core Team",
            required_hda_version=">=2.4.0",

            # Required sensors (column names from config)
            required_sensors=[
                "chamber_pressure",
                "thrust",
                "mass_flow_ox",
                "mass_flow_fuel",
            ],
            optional_sensors=[
                "temperature_ox",
                "temperature_fuel",
                "pressure_ox",
                "pressure_fuel",
            ],

            # Database columns (prefixed with hf_)
            database_columns=[
                ColumnSpec("avg_pc_bar", "REAL", nullable=False, description="Chamber pressure (bar)"),
                ColumnSpec("u_pc_bar", "REAL", nullable=False, description="Chamber pressure uncertainty (bar)"),
                ColumnSpec("avg_thrust_n", "REAL", nullable=False, description="Thrust (N)"),
                ColumnSpec("u_thrust_n", "REAL", nullable=False, description="Thrust uncertainty (N)"),
                ColumnSpec("avg_mf_total_g_s", "REAL", nullable=False, description="Total mass flow rate (g/s)"),
                ColumnSpec("u_mf_total_g_s", "REAL", nullable=False, description="Total mass flow uncertainty (g/s)"),
                ColumnSpec("avg_of_ratio", "REAL", nullable=True, description="O/F mixture ratio"),
                ColumnSpec("u_of_ratio", "REAL", nullable=True, description="O/F ratio uncertainty"),
                ColumnSpec("of_rel_uncertainty_pct", "REAL", nullable=True, description="O/F relative uncertainty (%)"),
                ColumnSpec("avg_isp_s", "REAL", nullable=True, description="Specific impulse (s)"),
                ColumnSpec("u_isp_s", "REAL", nullable=True, description="Specific impulse uncertainty (s)"),
                ColumnSpec("isp_rel_uncertainty_pct", "REAL", nullable=True, description="Isp relative uncertainty (%)"),
                ColumnSpec("avg_c_star_m_s", "REAL", nullable=True, description="Characteristic velocity (m/s)"),
                ColumnSpec("u_c_star_m_s", "REAL", nullable=True, description="C* uncertainty (m/s)"),
                ColumnSpec("c_star_rel_uncertainty_pct", "REAL", nullable=True, description="C* relative uncertainty (%)"),
                ColumnSpec("ignition_successful", "INTEGER", nullable=True, description="1 if ignition successful, 0 if failed"),
            ],

            # UI inputs (for dynamic form generation in future phases)
            ui_inputs=[
                InputSpec(
                    name="throat_area_mm2",
                    label="Throat Area (mm²)",
                    input_type="number",
                    min_value=0.01,
                    help_text="Nozzle throat cross-sectional area"
                ),
                InputSpec(
                    name="expansion_ratio",
                    label="Expansion Ratio",
                    input_type="number",
                    min_value=1.0,
                    help_text="Nozzle exit area / throat area"
                ),
                InputSpec(
                    name="propellant_combination",
                    label="Propellant",
                    input_type="select",
                    options=["LOX/RP-1", "LOX/Methane", "LOX/Ethanol", "N2O/HTPB", "H2O2/Kerosene"],
                    default="LOX/RP-1",
                    help_text="Propellant combination"
                ),
            ],

            # Uncertainty specifications
            uncertainty_specs=[
                UncertaintySpec(
                    name="Isp",
                    formula="Isp = F / (ṁ_total × g₀)",
                    sources=[
                        ("thrust", UncertaintySourceType.SENSOR),
                        ("mass_flow_total", UncertaintySourceType.CALCULATION),
                    ],
                    propagation_method="analytical",
                ),
                UncertaintySpec(
                    name="c_star",
                    formula="C* = (Pc × At) / ṁ_total",
                    sources=[
                        ("chamber_pressure", UncertaintySourceType.SENSOR),
                        ("throat_area", UncertaintySourceType.GEOMETRY),
                        ("mass_flow_total", UncertaintySourceType.CALCULATION),
                    ],
                    propagation_method="analytical",
                ),
                UncertaintySpec(
                    name="of_ratio",
                    formula="O/F = ṁ_ox / ṁ_fuel",
                    sources=[
                        ("mass_flow_ox", UncertaintySourceType.SENSOR),
                        ("mass_flow_fuel", UncertaintySourceType.SENSOR),
                    ],
                    propagation_method="analytical",
                ),
            ],
        )

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate hot fire specific configuration.

        Checks:
        - Test type is 'hot_fire'
        - Required sensors are mapped
        - Geometry parameters present (throat area for C*)
        - Uncertainty specifications present

        Args:
            config: Configuration dictionary (merged hardware + metadata)

        Raises:
            ValueError: If configuration is invalid
        """
        # Use existing validation (already comprehensive)
        validated = validate_config_simple(config, 'hot_fire')

        # Check sensor_roles in metadata (ALL sensor assignments should be in metadata, NOT config)
        sensor_roles = config.get('sensor_roles', {})

        if not sensor_roles:
            raise ValueError(
                "Hot fire analysis requires 'sensor_roles' in metadata. "
                "Add to metadata.json: \"sensor_roles\": {\"chamber_pressure\": \"YOUR-PT\", \"thrust\": \"YOUR-LC\", ...}"
            )

        # Check for required chamber pressure sensor
        if not sensor_roles.get('chamber_pressure'):
            raise ValueError(
                "Hot fire analysis requires 'chamber_pressure' in metadata sensor_roles. "
                "Add to metadata.json: \"sensor_roles\": {\"chamber_pressure\": \"YOUR-PT-SENSOR\"}"
            )

        # Check for required thrust sensor
        if not sensor_roles.get('thrust'):
            raise ValueError(
                "Hot fire analysis requires 'thrust' in metadata sensor_roles. "
                "Add to metadata.json: \"sensor_roles\": {\"thrust\": \"YOUR-LC-SENSOR\"}"
            )

        # Check for required mass flow sensors
        has_ox_flow = sensor_roles.get('mass_flow_ox') is not None
        has_fuel_flow = sensor_roles.get('mass_flow_fuel') is not None

        if not (has_ox_flow and has_fuel_flow):
            raise ValueError(
                "Hot fire analysis requires both 'mass_flow_ox' and 'mass_flow_fuel' in metadata sensor_roles. "
                "Add: \"mass_flow_ox\": \"YOUR-OX-FM\", \"mass_flow_fuel\": \"YOUR-FUEL-FM\""
            )

        # Check geometry (needed for C* calculation) - should be in metadata
        geom = config.get('geometry', {})
        if not geom.get('throat_area_mm2'):
            # Warning, not error - can still analyze without C*
            pass

    def run_qc_checks(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        quick: bool = False
    ) -> QCReport:
        """
        Run hot fire specific QC checks.

        Leverages core QC system with hot-fire specific considerations.

        Args:
            df: Preprocessed test data
            config: Active configuration
            quick: If True, run lightweight checks

        Returns:
            QCReport with check results
        """
        # Use existing comprehensive QC system
        if quick:
            return run_quick_qc(df, time_col='timestamp')
        else:
            return run_qc_checks(df, config, time_col='timestamp')

    def extract_steady_state(
        self,
        df: pd.DataFrame,
        steady_window: Tuple[float, float],
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract steady-state data from hot fire test.

        For hot fire, steady state is the time window where
        chamber pressure and thrust have stabilized.

        Args:
            df: Full preprocessed test data
            steady_window: (start_ms, end_ms) for steady state
            config: Active configuration

        Returns:
            DataFrame subset containing steady-state data
        """
        # Simple time-based extraction
        steady_df = df[
            (df['timestamp'] >= steady_window[0]) &
            (df['timestamp'] <= steady_window[1])
        ].copy()

        return steady_df

    def compute_raw_metrics(
        self,
        steady_df: pd.DataFrame,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Compute raw hot fire metrics (averages over steady state).

        Args:
            steady_df: Steady-state data subset
            config: Active configuration
            metadata: Test article properties

        Returns:
            Dictionary of averaged sensor values
        """
        # Calculate average values over steady window
        avg_values = steady_df.mean(numeric_only=True).to_dict()

        # Convert numpy types to Python floats
        avg_values = {k: float(v) for k, v in avg_values.items() if not pd.isna(v)}

        return avg_values

    def get_uncertainty_specs(self) -> Dict[str, UncertaintySpec]:
        """
        Get uncertainty specifications for hot fire metrics.

        Returns:
            Dictionary mapping metric names to uncertainty specs
        """
        return {spec.name: spec for spec in self.metadata.uncertainty_specs}

    def calculate_measurements_with_uncertainties(
        self,
        avg_values: Dict[str, float],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, MeasurementWithUncertainty]:
        """
        Calculate hot fire measurements WITH uncertainties.

        This method bridges to the existing uncertainty calculation system.
        In Phase 1, we leverage the existing calculate_hot_fire_uncertainties()
        to maintain backward compatibility.

        Args:
            avg_values: Averaged raw sensor values
            config: Active configuration (merged with metadata)
            metadata: Test metadata

        Returns:
            Dictionary of measurements with uncertainties
        """
        # Delegate to existing uncertainty calculation
        measurements = calculate_hot_fire_uncertainties(
            avg_values=avg_values,
            config=config
        )

        return measurements

    def generate_report_sections(
        self,
        result: Any  # AnalysisResult, avoiding circular import
    ) -> Dict[str, str]:
        """
        Generate hot fire specific report sections.

        Args:
            result: Complete AnalysisResult

        Returns:
            Dictionary of section_name -> HTML content
        """
        sections = {}

        # Hot fire specific metrics section
        if 'Isp' in result.measurements and 'c_star' in result.measurements:
            isp = result.measurements['Isp']
            c_star = result.measurements['c_star']
            sections['hot_fire_metrics'] = f"""
            <div class="metric-highlight">
                <h3>Engine Performance</h3>
                <p class="value">Isp = {isp.value:.2f} ± {isp.uncertainty:.2f} s</p>
                <p class="uncertainty">Relative uncertainty: {isp.relative_uncertainty_percent:.2f}%</p>
                <p class="value">C* = {c_star.value:.1f} ± {c_star.uncertainty:.1f} m/s</p>
                <p class="uncertainty">Relative uncertainty: {c_star.relative_uncertainty_percent:.2f}%</p>
            </div>
            """

        return sections

    def get_display_order(self) -> List[str]:
        """
        Get preferred display order for hot fire measurements.

        Returns:
            List of metric names in display order
        """
        return [
            'chamber_pressure',
            'thrust',
            'mass_flow_total',
            'of_ratio',
            'Isp',
            'c_star',
        ]


# =============================================================================
# Auto-register plugin
# =============================================================================

# Create instance and register with PluginRegistry
_plugin_instance = HotFirePlugin()
PluginRegistry.register(_plugin_instance)
