"""
Plugin System for HDA - Modular Analysis Architecture

This module provides the foundation for the plugin-based architecture that allows
HDA to support multiple test types (cold flow, hot fire, valve timing, etc.) without
hardcoding test-specific logic in core modules.

Key Features:
- Protocol-based plugin interface (no forced inheritance)
- Automatic plugin discovery (local files + entry-points for pip-installable plugins)
- Core P0 guarantees enforced by wrapper (traceability, uncertainty, QC)
- Plugin metadata for UI generation and schema coordination

Version: 1.0.0
Created: 2026-01-16
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional, Tuple, runtime_checkable
from enum import Enum
import pandas as pd
import importlib
import importlib.util
from pathlib import Path
import logging

# Import core integrity components
from core.qc_checks import QCReport, run_qc_checks, run_quick_qc
from core.uncertainty import MeasurementWithUncertainty
from core.traceability import DataTraceability

logger = logging.getLogger(__name__)


# ============================================================================
# Plugin Metadata Structures
# ============================================================================

class UncertaintySourceType(Enum):
    """Types of uncertainty sources for plugin declarations."""
    SENSOR = "sensor"                    # Sensor calibration uncertainty
    GEOMETRY = "geometry"                # Geometric tolerance
    FLUID_PROPERTY = "fluid_property"    # Fluid property uncertainty
    CALCULATION = "calculation"          # Derived calculation uncertainty


@dataclass
class UncertaintySpec:
    """
    Specification for uncertainty calculation.

    Plugins declare their uncertainty sources using this spec,
    and the core wrapper enforces proper propagation.
    """
    name: str                                    # Measurement name (e.g., 'Cd')
    formula: str                                 # Human-readable formula (for docs)
    sources: List[Tuple[str, UncertaintySourceType]]  # [(source_id, type), ...]
    propagation_method: str = "analytical"       # 'analytical' or 'monte_carlo'
    monte_carlo_samples: int = 10000            # If using MC


@dataclass
class ColumnSpec:
    """Database column specification for plugin storage."""
    name: str                    # Column name (will be prefixed with plugin slug)
    type: str                    # SQL type: REAL, TEXT, INTEGER, BLOB
    nullable: bool = True        # Can be NULL?
    description: str = ""        # Human-readable description


@dataclass
class InputSpec:
    """UI input specification for dynamic form generation."""
    name: str                    # Parameter name
    label: str                   # Display label
    input_type: str              # 'number', 'text', 'select', 'multiselect', 'slider'
    default: Any = None          # Default value
    options: Optional[List[Any]] = None          # For select/multiselect
    min_value: Optional[float] = None            # For number/slider
    max_value: Optional[float] = None            # For number/slider
    help_text: str = ""          # Tooltip/help text


@dataclass
class PluginMetadata:
    """
    Complete plugin metadata for discovery, validation, and UI generation.

    This metadata allows the core system to:
    - Validate plugin compatibility
    - Auto-generate configuration forms
    - Auto-migrate database schemas
    - Display plugin information to users
    """
    name: str                                    # Human-readable name
    slug: str                                    # Machine-readable identifier (lowercase, underscores)
    version: str                                 # Plugin version (semantic versioning)
    test_type: str                               # Test type identifier (e.g., 'cold_flow', 'hot_fire')
    description: str = ""                        # Brief description
    author: str = ""                             # Plugin author

    # Requirements
    required_hda_version: str = ">=2.3.0"       # Minimum HDA version

    # Configuration & validation
    config_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema for config validation
    required_sensors: List[str] = field(default_factory=list)     # Sensor IDs that must be present
    optional_sensors: List[str] = field(default_factory=list)     # Optional sensor IDs

    # Database schema
    database_columns: List[ColumnSpec] = field(default_factory=list)  # Custom columns for this plugin

    # UI generation
    ui_inputs: List[InputSpec] = field(default_factory=list)      # Custom UI inputs

    # Uncertainty declarations
    uncertainty_specs: List[UncertaintySpec] = field(default_factory=list)


# ============================================================================
# Plugin Protocol
# ============================================================================

@runtime_checkable
class AnalysisPlugin(Protocol):
    """
    Protocol defining the interface for analysis plugins.

    Plugins implement domain-specific logic (cold flow, hot fire, etc.)
    while the core wrapper enforces P0 guarantees (traceability, uncertainty, QC).

    Design Philosophy:
    - Plugins own domain logic, core owns integrity
    - Plugins return RAW values, core adds uncertainties (ensures no bypass)
    - Plugins declare their requirements, core validates
    - Plugins are UI-agnostic (can be used in scripts, notebooks, or Streamlit)

    Implementation Pattern:
    1. Plugin validates config (test-specific checks)
    2. Plugin runs QC (test-specific checks, core adds common checks)
    3. Plugin extracts steady state (test-specific windowing)
    4. Plugin computes raw metrics (domain calculations)
    5. Core wraps with uncertainties (enforced P0)
    6. Core creates traceability (enforced P0)
    7. Core packages result
    """

    # Required attribute: metadata
    metadata: PluginMetadata

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate plugin-specific configuration.

        Should raise ValueError with clear message if config is invalid.
        Core will call this before analysis.

        Args:
            config: Configuration dictionary (merged hardware + metadata)

        Raises:
            ValueError: If configuration is invalid
        """
        ...

    def run_qc_checks(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        quick: bool = False
    ) -> QCReport:
        """
        Run plugin-specific quality control checks.

        Core will also run common QC checks (timestamp monotonic, sampling rate, etc.)
        and merge results. Plugins should add test-specific checks here.

        Args:
            df: Preprocessed test data (resampled, NaN-handled)
            config: Active configuration
            quick: If True, run lightweight checks only

        Returns:
            QCReport with plugin-specific check results
        """
        ...

    def extract_steady_state(
        self,
        df: pd.DataFrame,
        steady_window: Tuple[float, float],
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract steady-state subset from preprocessed data.

        Args:
            df: Full preprocessed test data
            steady_window: (start_ms, end_ms) window for steady state
            config: Active configuration

        Returns:
            Filtered DataFrame containing only steady-state data
        """
        ...

    def compute_raw_metrics(
        self,
        steady_df: pd.DataFrame,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Compute raw metric values WITHOUT uncertainties.

        Core will add uncertainties based on plugin's uncertainty_specs.
        This enforces that plugins cannot skip uncertainty calculation.

        Args:
            steady_df: Steady-state data subset
            config: Active configuration (hardware + metadata merged)
            metadata: Test article properties (optional, may be in config)

        Returns:
            Dictionary of {metric_name: raw_value}
            Example: {'p_up_bar': 10.5, 'mf_g_s': 125.3, 'Cd': 0.654}

        Note:
            - All values must be Python floats (not numpy types)
            - Values should use standard HDA units (bar, g/s, K, mm²)
            - Names should match database column conventions
        """
        ...

    def get_uncertainty_specs(self) -> Dict[str, UncertaintySpec]:
        """
        Declare uncertainty sources for each metric.

        Core uses these specs to enforce proper uncertainty propagation.

        Returns:
            Dictionary mapping metric_name -> UncertaintySpec
        """
        ...

    # Optional methods

    def generate_report_sections(
        self,
        result: 'AnalysisResult'
    ) -> Dict[str, str]:
        """
        Generate plugin-specific HTML report sections.

        Optional. If not implemented, core uses default formatting.

        Args:
            result: Complete AnalysisResult object

        Returns:
            Dictionary of {section_name: html_content}
        """
        return {}

    def get_display_order(self) -> List[str]:
        """
        Specify display order for measurements in UI/reports.

        Optional. If not implemented, alphabetical order used.

        Returns:
            List of metric names in display order
        """
        return []


# ============================================================================
# Plugin Registry
# ============================================================================

class PluginRegistry:
    """
    Central registry for plugin discovery and management.

    Supports two discovery mechanisms:
    1. Local file-based discovery (core/plugins/*.py) - for development
    2. Entry-point based discovery (via setuptools) - for pip-installable plugins

    Plugins are loaded lazily on first access and cached.
    """

    _plugins: List[AnalysisPlugin] = []
    _plugins_loaded: bool = False
    _plugin_map: Dict[str, AnalysisPlugin] = {}  # slug -> plugin

    @classmethod
    def register(cls, plugin: AnalysisPlugin) -> None:
        """
        Manually register a plugin instance.

        Args:
            plugin: Plugin instance implementing AnalysisPlugin protocol

        Raises:
            ValueError: If plugin with same slug already registered
        """
        # Validate plugin implements protocol
        if not isinstance(plugin, AnalysisPlugin):
            raise TypeError(f"Plugin must implement AnalysisPlugin protocol, got {type(plugin)}")

        # Check for duplicate slug
        if plugin.metadata.slug in cls._plugin_map:
            existing = cls._plugin_map[plugin.metadata.slug]
            raise ValueError(
                f"Plugin slug '{plugin.metadata.slug}' already registered "
                f"(existing: {existing.metadata.name} v{existing.metadata.version})"
            )

        cls._plugins.append(plugin)
        cls._plugin_map[plugin.metadata.slug] = plugin
        logger.info(
            f"Registered plugin: {plugin.metadata.name} "
            f"v{plugin.metadata.version} (slug: {plugin.metadata.slug})"
        )

    @classmethod
    def get_plugins(cls) -> List[AnalysisPlugin]:
        """
        Get all registered plugins.

        Triggers plugin discovery on first call.

        Returns:
            List of all registered plugin instances
        """
        if not cls._plugins_loaded:
            cls._load_plugins()
        return cls._plugins.copy()

    @classmethod
    def get_plugin(cls, slug: str) -> AnalysisPlugin:
        """
        Get plugin by slug.

        Args:
            slug: Plugin slug (e.g., 'cold_flow', 'hot_fire')

        Returns:
            Plugin instance

        Raises:
            KeyError: If plugin not found
        """
        if not cls._plugins_loaded:
            cls._load_plugins()

        if slug not in cls._plugin_map:
            available = [p.metadata.slug for p in cls._plugins]
            raise KeyError(
                f"Plugin '{slug}' not found. "
                f"Available plugins: {', '.join(available)}"
            )

        return cls._plugin_map[slug]

    @classmethod
    def get_plugins_by_test_type(cls, test_type: str) -> List[AnalysisPlugin]:
        """
        Get all plugins for a specific test type.

        Args:
            test_type: Test type identifier (e.g., 'cold_flow')

        Returns:
            List of plugins supporting that test type
        """
        if not cls._plugins_loaded:
            cls._load_plugins()

        return [p for p in cls._plugins if p.metadata.test_type == test_type]

    @classmethod
    def list_available_plugins(cls) -> List[Dict[str, str]]:
        """
        List all available plugins with metadata.

        Returns:
            List of dicts with plugin info (name, slug, version, description)
        """
        if not cls._plugins_loaded:
            cls._load_plugins()

        return [
            {
                'name': p.metadata.name,
                'slug': p.metadata.slug,
                'version': p.metadata.version,
                'test_type': p.metadata.test_type,
                'description': p.metadata.description,
                'author': p.metadata.author,
            }
            for p in cls._plugins
        ]

    @classmethod
    def disable(cls, slug: str) -> None:
        """
        Disable a plugin (remove from registry).

        Args:
            slug: Plugin slug to disable
        """
        if slug in cls._plugin_map:
            plugin = cls._plugin_map[slug]
            cls._plugins.remove(plugin)
            del cls._plugin_map[slug]
            logger.info(f"Disabled plugin: {slug}")

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins (mainly for testing)."""
        cls._plugins.clear()
        cls._plugin_map.clear()
        cls._plugins_loaded = False
        logger.info("Cleared plugin registry")

    @classmethod
    def _load_plugins(cls) -> None:
        """
        Load plugins from all discovery mechanisms.

        Discovery order:
        1. Local files in core/plugins/*.py (development)
        2. Entry-points 'hda.analysis_plugins' (pip-installable)
        """
        if cls._plugins_loaded:
            return

        logger.info("Starting plugin discovery...")

        # 1. Local file-based discovery
        cls._load_local_plugins()

        # 2. Entry-point based discovery (for pip-installable plugins)
        cls._load_entrypoint_plugins()

        cls._plugins_loaded = True
        logger.info(f"Plugin discovery complete. Loaded {len(cls._plugins)} plugins.")

    @classmethod
    def _load_local_plugins(cls) -> None:
        """
        Load plugins from core/plugin_modules/*.py directory.

        Searches for all .py files (except __init__.py and _*.py) and imports them.
        Plugins self-register by calling PluginRegistry.register() during import.
        """
        plugins_dir = Path(__file__).parent / "plugin_modules"

        if not plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return

        logger.info(f"Scanning local plugins in: {plugins_dir}")

        for py_file in sorted(plugins_dir.glob("*.py")):
            # Skip private modules and __init__
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue

            try:
                # Import module
                module_name = f"core.plugin_modules.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    logger.info(f"Loaded local plugin module: {module_name}")
            except Exception as e:
                logger.error(f"Failed to load plugin from {py_file}: {e}", exc_info=True)

    @classmethod
    def _load_entrypoint_plugins(cls) -> None:
        """
        Load plugins from setuptools entry points.

        Looks for entry point group 'hda.analysis_plugins'.
        Plugins installed via pip should declare:

        [project.entry-points.'hda.analysis_plugins']
        my_plugin = "my_package.plugins:MyPlugin"
        """
        try:
            # Try importlib.metadata (Python 3.8+)
            try:
                from importlib.metadata import entry_points
            except ImportError:
                # Fallback to pkg_resources (older Python)
                import pkg_resources
                entry_points = lambda group: pkg_resources.iter_entry_points(group)

            # Load entry points
            eps = entry_points(group='hda.analysis_plugins')

            # Handle both dict and list return types (API changed in Python 3.10)
            if hasattr(eps, 'select'):  # Python 3.10+
                eps = eps.select(group='hda.analysis_plugins')

            for ep in eps:
                try:
                    plugin_class = ep.load()
                    # Instantiate and register
                    plugin_instance = plugin_class()
                    cls.register(plugin_instance)
                    logger.info(f"Loaded entry-point plugin: {ep.name}")
                except Exception as e:
                    logger.error(f"Failed to load entry-point plugin {ep.name}: {e}", exc_info=True)

        except Exception as e:
            logger.debug(f"Entry-point discovery not available or failed: {e}")


# ============================================================================
# Plugin Validation Utilities
# ============================================================================

def validate_plugin(plugin: AnalysisPlugin) -> List[str]:
    """
    Validate that a plugin correctly implements the AnalysisPlugin protocol.

    Args:
        plugin: Plugin instance to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required attributes
    if not hasattr(plugin, 'metadata'):
        errors.append("Plugin missing 'metadata' attribute")
    else:
        metadata = plugin.metadata
        if not metadata.name:
            errors.append("Plugin metadata missing 'name'")
        if not metadata.slug:
            errors.append("Plugin metadata missing 'slug'")
        if not metadata.version:
            errors.append("Plugin metadata missing 'version'")
        if not metadata.test_type:
            errors.append("Plugin metadata missing 'test_type'")

    # Check required methods
    required_methods = [
        'validate_config',
        'run_qc_checks',
        'extract_steady_state',
        'compute_raw_metrics',
        'get_uncertainty_specs'
    ]

    for method_name in required_methods:
        if not hasattr(plugin, method_name):
            errors.append(f"Plugin missing required method: {method_name}")
        elif not callable(getattr(plugin, method_name)):
            errors.append(f"Plugin attribute '{method_name}' is not callable")

    return errors


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core protocol and registry
    'AnalysisPlugin',
    'PluginRegistry',

    # Metadata structures
    'PluginMetadata',
    'UncertaintySpec',
    'UncertaintySourceType',
    'ColumnSpec',
    'InputSpec',

    # Utilities
    'validate_plugin',
]
