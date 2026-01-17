"""
Cold Flow Analysis Plugin

Implements cold flow test analysis for HDA's plugin architecture.

This plugin handles:
- Injector characterization tests
- Discharge coefficient (Cd) measurement
- Flow capacity verification
- Pressure drop characterization

Measurements calculated:
- Upstream pressure (bar)
- Downstream pressure (bar)
- Delta P (bar)
- Mass flow rate (g/s)
- Discharge coefficient (Cd)

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
    calculate_cold_flow_uncertainties,
    MeasurementWithUncertainty,
)
from core.config_validation import validate_config_simple


class ColdFlowPlugin:
    """
    Cold flow test analysis plugin.

    Handles analysis of cold flow injector characterization tests,
    calculating discharge coefficient and other flow metrics.
    """

    def __init__(self):
        """Initialize the cold flow plugin with metadata."""
        self.metadata = PluginMetadata(
            name="Cold Flow Injector Analysis",
            slug="cold_flow",
            version="1.0.0",
            test_type="cold_flow",
            description="Analyze cold flow tests to characterize injector performance (Cd, flow capacity, pressure drop)",
            author="HDA Core Team",
            required_hda_version=">=2.3.0",

            # Required sensors (column names from config)
            required_sensors=[
                "upstream_pressure",  # or inlet_pressure
                "mass_flow",          # or mf
            ],
            optional_sensors=[
                "downstream_pressure",
                "temperature",
            ],

            # Database columns (prefixed with cf_)
            database_columns=[
                ColumnSpec("avg_p_up_bar", "REAL", nullable=False, description="Upstream pressure (bar)"),
                ColumnSpec("u_p_up_bar", "REAL", nullable=False, description="Upstream pressure uncertainty (bar)"),
                ColumnSpec("avg_p_down_bar", "REAL", nullable=True, description="Downstream pressure (bar)"),
                ColumnSpec("u_p_down_bar", "REAL", nullable=True, description="Downstream pressure uncertainty (bar)"),
                ColumnSpec("avg_mf_g_s", "REAL", nullable=False, description="Mass flow rate (g/s)"),
                ColumnSpec("u_mf_g_s", "REAL", nullable=False, description="Mass flow rate uncertainty (g/s)"),
                ColumnSpec("avg_delta_p_bar", "REAL", nullable=True, description="Pressure drop (bar)"),
                ColumnSpec("u_delta_p_bar", "REAL", nullable=True, description="Pressure drop uncertainty (bar)"),
                ColumnSpec("avg_cd_CALC", "REAL", nullable=True, description="Discharge coefficient"),
                ColumnSpec("u_cd_CALC", "REAL", nullable=True, description="Discharge coefficient uncertainty"),
                ColumnSpec("cd_rel_uncertainty_pct", "REAL", nullable=True, description="Cd relative uncertainty (%)"),
            ],

            # UI inputs (for dynamic form generation in future phases)
            ui_inputs=[
                InputSpec(
                    name="orifice_area_mm2",
                    label="Orifice Area (mm²)",
                    input_type="number",
                    min_value=0.01,
                    help_text="Cross-sectional area of the orifice"
                ),
                InputSpec(
                    name="fluid_name",
                    label="Fluid",
                    input_type="select",
                    options=["nitrogen", "helium", "oxygen", "argon", "nitrous_oxide"],
                    default="nitrogen",
                    help_text="Working fluid for the test"
                ),
            ],

            # Uncertainty specifications
            uncertainty_specs=[
                UncertaintySpec(
                    name="Cd",
                    formula="Cd = ṁ / (A × √(2ρΔP))",
                    sources=[
                        ("mass_flow", UncertaintySourceType.SENSOR),
                        ("orifice_area", UncertaintySourceType.GEOMETRY),
                        ("delta_p", UncertaintySourceType.CALCULATION),
                        ("density", UncertaintySourceType.FLUID_PROPERTY),
                    ],
                    propagation_method="analytical",
                ),
            ],
        )

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate cold flow specific configuration.

        Checks:
        - Test type is 'cold_flow'
        - Required sensors are mapped
        - Geometry parameters present
        - Uncertainty specifications present

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Use existing validation (already comprehensive)
        validated = validate_config_simple(config, 'cold_flow')

        # Additional plugin-specific checks
        # Support both 'columns' (semantic mapping) and 'channel_config' (legacy DAQ mapping)
        # Prefer 'columns' as it provides semantic role → sensor name mapping
        cols = config.get('columns', {})

        # If no 'columns', check 'channel_config' as fallback
        # but warn that it should be updated to include 'columns'
        if not cols:
            cols = config.get('channel_config', {})
            if cols:
                import warnings
                warnings.warn(
                    "Config uses legacy 'channel_config' without 'columns'. "
                    "Please add 'columns' with semantic mappings (e.g., 'mass_flow': 'OX-FM-01'). "
                    "Attempting to infer from sensor names...",
                    DeprecationWarning
                )

        # Check for pressure sensor (semantic keys)
        has_pressure = (
            cols.get('upstream_pressure') is not None or
            cols.get('inlet_pressure') is not None
        )
        if not has_pressure:
            raise ValueError(
                "Cold flow config must specify 'upstream_pressure' or 'inlet_pressure' in columns. "
                "Add: \"columns\": {\"upstream_pressure\": \"YOUR-PT-SENSOR\", \"mass_flow\": \"YOUR-FM-SENSOR\"}"
            )

        # Check for flow sensor (semantic keys)
        has_flow = (
            cols.get('mass_flow') is not None or
            cols.get('mf') is not None
        )
        if not has_flow:
            raise ValueError(
                "Cold flow config must specify 'mass_flow' or 'mf' in columns. "
                "Add: \"columns\": {\"upstream_pressure\": \"YOUR-PT-SENSOR\", \"mass_flow\": \"YOUR-FM-SENSOR\"}"
            )

        # Check geometry (needed for Cd calculation)
        geom = config.get('geometry', {})
        if not geom.get('orifice_area_mm2'):
            # Warning, not error - can still analyze without Cd
            pass

    def run_qc_checks(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        quick: bool = False
    ) -> QCReport:
        """
        Run cold flow specific QC checks.

        Leverages core QC system with cold-flow specific considerations.

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
        Extract steady-state data from cold flow test.

        For cold flow, steady state is simply the time window where
        flow has stabilized (constant pressure and mass flow).

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
        Compute raw cold flow metrics (averages over steady state).

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
        Get uncertainty specifications for cold flow metrics.

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
        Calculate cold flow measurements WITH uncertainties.

        This method bridges to the existing uncertainty calculation system.
        In Phase 1, we leverage the existing calculate_cold_flow_uncertainties()
        to maintain backward compatibility.

        Args:
            avg_values: Averaged raw sensor values
            config: Active configuration
            metadata: Test metadata

        Returns:
            Dictionary of measurements with uncertainties
        """
        # Delegate to existing uncertainty calculation
        # (Future phases may refactor this to be more modular)
        measurements = calculate_cold_flow_uncertainties(
            avg_values=avg_values,
            config=config,
            metadata=metadata
        )

        return measurements

    def generate_report_sections(
        self,
        result: Any  # AnalysisResult, avoiding circular import
    ) -> Dict[str, str]:
        """
        Generate cold flow specific report sections.

        Args:
            result: Complete AnalysisResult

        Returns:
            Dictionary of section_name -> HTML content
        """
        sections = {}

        # Cold flow specific metrics section
        if 'Cd' in result.measurements:
            cd = result.measurements['Cd']
            sections['cold_flow_metrics'] = f"""
            <div class="metric-highlight">
                <h3>Discharge Coefficient</h3>
                <p class="value">Cd = {cd.value:.4f} ± {cd.uncertainty:.4f}</p>
                <p class="uncertainty">Relative uncertainty: {cd.relative_uncertainty_percent:.2f}%</p>
            </div>
            """

        return sections

    def get_display_order(self) -> List[str]:
        """
        Get preferred display order for cold flow measurements.

        Returns:
            List of metric names in display order
        """
        return [
            'pressure_upstream',
            'pressure_downstream',
            'delta_p',
            'mass_flow',
            'Cd',
        ]


# =============================================================================
# Auto-register plugin
# =============================================================================

# Create instance and register with PluginRegistry
_plugin_instance = ColdFlowPlugin()
PluginRegistry.register(_plugin_instance)
