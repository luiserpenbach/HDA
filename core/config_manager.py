"""
Unified Configuration Manager
==============================
Centralized configuration management for Hopper Data Studio.

This module provides a single source of truth for:
- Default configurations (cold flow, hot fire)
- Configuration validation
- Recent configuration tracking
- Template integration

Eliminates duplication of default configs across multiple pages.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ConfigInfo:
    """Metadata about a configuration."""
    config_name: str
    test_type: str
    source: str  # 'default', 'template', 'uploaded', 'custom'
    timestamp: str
    version: Optional[str] = None


class ConfigManager:
    """
    Centralized configuration management.

    Usage:
        # Get default config
        config = ConfigManager.get_default_config('cold_flow')

        # Save to recent configs
        ConfigManager.save_to_recent(config, 'template', 'my_nitrogen_config')

        # Get recent configs
        recent = ConfigManager.get_recent_configs(limit=5)
    """

    # Session state keys
    RECENT_CONFIGS_KEY = 'recent_configs'
    ACTIVE_CONFIG_KEY = 'active_config'

    @staticmethod
    def get_default_config(test_type: str) -> Dict[str, Any]:
        """
        Get default configuration for test type.

        Single source of truth for default configurations.

        Args:
            test_type: 'cold_flow' or 'hot_fire'

        Returns:
            Default configuration dictionary

        Raises:
            ValueError: If test_type is not recognized
        """
        if test_type == 'cold_flow':
            return ConfigManager._default_cold_flow_config()
        elif test_type == 'hot_fire':
            return ConfigManager._default_hot_fire_config()
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    @staticmethod
    def _default_cold_flow_config() -> Dict[str, Any]:
        """Create default cold flow configuration."""
        return {
            'config_name': 'Default Cold Flow',
            'version': '1.0.0',
            'test_type': 'cold_flow',
            'description': 'Default configuration for cold flow testing',
            'channel_config': {
                # Example mapping - user should customize
                # "10001": "PT-UP-01",
                # "10002": "PT-DN-01",
                # "10003": "TC-01",
                # "10004": "FM-01",
            },
            'columns': {
                'timestamp': 'timestamp',
                'upstream_pressure': 'P_upstream',
                'downstream_pressure': 'P_downstream',
                'temperature': 'T_fluid',
                'mass_flow': 'mass_flow',
            },
            'fluid': {
                'name': 'nitrogen',
                'gamma': 1.4,
                'R': 296.8,  # J/(kg·K) for nitrogen
            },
            'geometry': {
                'orifice_area_mm2': 1.0,
                'orifice_diameter_mm': 1.128,
            },
            'uncertainties': {
                'pressure': {'type': 'relative', 'value': 0.005},
                'temperature': {'type': 'absolute', 'value': 1.0},
                'mass_flow': {'type': 'relative', 'value': 0.01},
            },
            'geometry_uncertainties': {
                'area': {'type': 'relative', 'value': 0.02},
            },
            'settings': {
                'sample_rate_hz': 100,
                'resample_freq_ms': 10,
                'steady_window_ms': 1000,
                'cv_threshold': 1.5,
            },
            'qc': {
                'max_nan_ratio': 0.05,
                'min_correlation': 0.3,
            }
        }

    @staticmethod
    def _default_hot_fire_config() -> Dict[str, Any]:
        """Create default hot fire configuration."""
        return {
            'config_name': 'Default Hot Fire',
            'version': '1.0.0',
            'test_type': 'hot_fire',
            'description': 'Default configuration for hot fire testing',
            'channel_config': {
                # Example mapping - user should customize
                # "10001": "IG-PT-01",
                # "10002": "FU-PT-01",
                # "10003": "FU-FM-01",
                # "10004": "OX-FM-01",
                # "10009": "LC-01",
            },
            'columns': {
                'timestamp': 'timestamp',
                'chamber_pressure': 'P_chamber',
                'ox_mass_flow': 'mf_ox',
                'fuel_mass_flow': 'mf_fuel',
                'thrust': 'thrust',
            },
            'propellants': {
                'oxidizer': 'LOX',
                'fuel': 'RP-1',
            },
            'geometry': {
                'throat_area_mm2': 100.0,
                'nozzle_expansion_ratio': 10.0,
            },
            'uncertainties': {
                'chamber_pressure': {'type': 'relative', 'value': 0.005},
                'mass_flow': {'type': 'relative', 'value': 0.01},
                'thrust': {'type': 'relative', 'value': 0.02},
            },
            'geometry_uncertainties': {
                'throat_area': {'type': 'relative', 'value': 0.02},
            },
            'settings': {
                'sample_rate_hz': 1000,
                'target_c_star': 1800,
                'target_cf': 1.5,
                'target_isp': 300,
            },
            'qc': {
                'max_nan_ratio': 0.05,
                'min_correlation': 0.3,
            }
        }

    @staticmethod
    def save_to_recent(
        config: Dict[str, Any],
        source: str,
        name: Optional[str] = None
    ) -> None:
        """
        Save configuration to recent configs list.

        Args:
            config: Configuration dictionary
            source: Source of config ('default', 'template', 'uploaded', 'custom')
            name: Optional name override (defaults to config_name from dict)
        """
        import streamlit as st

        if ConfigManager.RECENT_CONFIGS_KEY not in st.session_state:
            st.session_state[ConfigManager.RECENT_CONFIGS_KEY] = []

        config_info = ConfigInfo(
            config_name=name or config.get('config_name', 'Unnamed'),
            test_type=config.get('test_type', 'unknown'),
            source=source,
            timestamp=datetime.now().isoformat(),
            version=config.get('version')
        )

        recent = st.session_state[ConfigManager.RECENT_CONFIGS_KEY]

        # Add to front of list (most recent first)
        recent.insert(0, {
            'info': asdict(config_info),
            'config': config
        })

        # Keep only last 10
        st.session_state[ConfigManager.RECENT_CONFIGS_KEY] = recent[:10]

    @staticmethod
    def get_recent_configs(limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get list of recently used configurations.

        Args:
            limit: Maximum number of recent configs to return

        Returns:
            List of recent config dictionaries with metadata
        """
        import streamlit as st

        if ConfigManager.RECENT_CONFIGS_KEY not in st.session_state:
            return []

        recent = st.session_state[ConfigManager.RECENT_CONFIGS_KEY]
        return recent[:limit]

    @staticmethod
    def set_active_config(config: Dict[str, Any]) -> None:
        """
        Set the currently active configuration.

        Args:
            config: Configuration dictionary to set as active
        """
        import streamlit as st
        st.session_state[ConfigManager.ACTIVE_CONFIG_KEY] = config

    @staticmethod
    def get_active_config() -> Optional[Dict[str, Any]]:
        """
        Get the currently active configuration.

        Returns:
            Active configuration dictionary, or None if not set
        """
        import streamlit as st
        return st.session_state.get(ConfigManager.ACTIVE_CONFIG_KEY)

    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(path, 'r') as f:
            config = json.load(f)

        return config

    @staticmethod
    def save_to_file(config: Dict[str, Any], file_path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
            file_path: Path to save JSON file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Required top-level keys
        required_keys = ['test_type', 'columns']
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Validate test_type
        if 'test_type' in config:
            valid_types = ['cold_flow', 'hot_fire']
            if config['test_type'] not in valid_types:
                errors.append(f"Invalid test_type: {config['test_type']}. Must be one of {valid_types}")

        # Validate columns (must have timestamp)
        if 'columns' in config:
            if 'timestamp' not in config['columns']:
                errors.append("columns dict must include 'timestamp' mapping")

        # Validate uncertainties structure (if present)
        if 'uncertainties' in config:
            for sensor, unc in config['uncertainties'].items():
                if not isinstance(unc, dict):
                    errors.append(f"Uncertainty for '{sensor}' must be a dict")
                elif 'type' not in unc or 'value' not in unc:
                    errors.append(f"Uncertainty for '{sensor}' must have 'type' and 'value'")

        return len(errors) == 0, errors

    @staticmethod
    def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations (deep merge).

        Args:
            base: Base configuration
            overrides: Configuration overrides to apply

        Returns:
            Merged configuration dictionary
        """
        import copy

        result = copy.deepcopy(base)

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary information about a configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Summary dictionary with key stats
        """
        return {
            'config_name': config.get('config_name', 'Unnamed'),
            'test_type': config.get('test_type', 'unknown'),
            'version': config.get('version', 'N/A'),
            'fluid': config.get('fluid', {}).get('name', 'N/A'),
            'sensor_count': len(config.get('columns', {})),
            'uncertainty_count': len(config.get('uncertainties', {})),
            'has_qc': 'qc' in config and bool(config['qc']),
        }
