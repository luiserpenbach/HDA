"""
Saved Configurations Module (v2.3.0)
=====================================
Manage reusable test configurations (formerly "templates"):
- Pre-built configurations for common test types
- Configuration validation and versioning
- Configuration inheritance
- Import/export functionality

Terminology (v2.3.0):
- "Saved Config" = Reusable configuration stored in saved_configs/ folder
- "Active Configuration" = Currently selected config for analysis
- Previously called "templates" in v2.0-v2.2
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import copy


# =============================================================================
# SAVED CONFIGURATION DEFINITIONS (v2.3.0)
# =============================================================================

@dataclass
class UncertaintySpec:
    """Specification for a measurement uncertainty."""
    type: str  # 'absolute' or 'relative'
    value: float
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {'type': self.type, 'value': self.value}
        if self.unit:
            d['unit'] = self.unit
        return d


@dataclass
class SavedConfig:
    """
    Saved Configuration for reuse across tests (v2.4.0).

    A saved config contains ONLY testbench hardware configuration:
    - Channel mappings (DAQ channel ID -> sensor name)
    - Measurement uncertainties
    - Sampling/processing settings
    - Quality control thresholds

    Fluid properties, geometry, and test-specific info belong in metadata files,
    NOT in the hardware config.
    """
    name: str
    version: str
    test_type: str  # 'cold_flow' or 'hot_fire'
    description: str = ""

    # Hardware channel mappings (DAQ channel ID -> sensor name)
    channel_config: Dict[str, str] = field(default_factory=dict)

    # Measurement uncertainties (sensor type -> uncertainty spec)
    uncertainties: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Sampling and processing settings
    settings: Dict[str, Any] = field(default_factory=dict)

    # Quality control thresholds
    qc: Dict[str, Any] = field(default_factory=dict)

    # Config metadata (for management, not analysis)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    parent_config: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config_name': self.name,
            'version': self.version,
            'test_type': self.test_type,
            'description': self.description,
            'channel_config': self.channel_config,
            'uncertainties': self.uncertainties,
            'settings': self.settings,
            'qc': self.qc,
            'created_date': self.created_date,
            'author': self.author,
            'parent_config': self.parent_config,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SavedConfig':
        """Create from dictionary."""
        # Support both 'channel_config' (new) and 'columns' (legacy)
        channel_config = data.get('channel_config', data.get('columns', {}))

        return cls(
            name=data.get('config_name', data.get('name', 'Unknown')),
            version=data.get('version', '1.0.0'),
            test_type=data.get('test_type', 'cold_flow'),
            description=data.get('description', ''),
            channel_config=channel_config,
            uncertainties=data.get('uncertainties', {}),
            settings=data.get('settings', {}),
            qc=data.get('qc', {}),
            created_date=data.get('created_date', datetime.now().isoformat()),
            author=data.get('author', ''),
            parent_config=data.get('parent_config') or data.get('parent_template'),
            tags=data.get('tags', []),
        )

    def to_config(self) -> Dict[str, Any]:
        """Convert to analysis-ready config dict."""
        return {
            'config_name': self.name,
            'version': self.version,
            'test_type': self.test_type,
            'description': self.description,
            'channel_config': self.channel_config,
            'uncertainties': self.uncertainties,
            'settings': self.settings,
            'qc': self.qc,
        }


# =============================================================================
# BUILT-IN TEMPLATES
# =============================================================================

# Default Cold Flow Template - Hardware config only (no fluid/geometry)
COLD_FLOW_DEFAULT_TEMPLATE = SavedConfig(
    name="Default Cold Flow",
    version="1.0.0",
    test_type="cold_flow",
    description="Default configuration for cold flow testing",
    channel_config={
        # Example channel mappings - user should customize
        "10001": "IG-PT-01",
        "10002": "FU-PT-01",
        "10003": "OX-PT-15",
        "10004": "OX-PT-12",
        "10009": "OX-FM-01",
        "10011": "OX-TS-13",
    },
    uncertainties={
        'pressure': {'type': 'relative', 'value': 0.005},  # 0.5%
        'temperature': {'type': 'absolute', 'value': 1.0},  # 1 K
        'mass_flow': {'type': 'relative', 'value': 0.01},  # 1%
    },
    settings={
        'sample_rate_hz': 100,
        'resample_freq_ms': 10,
        'steady_window_ms': 1000,
        'cv_threshold': 1.5,
    },
    qc={
        'max_nan_ratio': 0.05,
        'min_correlation': 0.3,
    },
    tags=['cold_flow', 'default'],
)

# Default Hot Fire Template - Hardware config only (no propellants/geometry)
HOT_FIRE_DEFAULT_TEMPLATE = SavedConfig(
    name="Default Hot Fire",
    version="1.0.0",
    test_type="hot_fire",
    description="Default configuration for hot fire testing",
    channel_config={
        # Example channel mappings - user should customize
        "10001": "PC-PT-01",
        "10002": "OX-PT-01",
        "10003": "FU-PT-01",
        "10004": "OX-FM-01",
        "10005": "FU-FM-01",
        "10006": "LC-01",
    },
    uncertainties={
        'chamber_pressure': {'type': 'relative', 'value': 0.005},  # 0.5%
        'mass_flow': {'type': 'relative', 'value': 0.01},  # 1%
        'thrust': {'type': 'relative', 'value': 0.02},  # 2%
    },
    settings={
        'sample_rate_hz': 1000,
        'resample_freq_ms': 1,
        'steady_window_ms': 500,
        'cv_threshold': 2.0,
        'g0': 9.80665,
    },
    qc={
        'max_nan_ratio': 0.02,
        'min_pc_bar': 5.0,
    },
    tags=['hot_fire', 'default'],
)

# Registry of built-in templates
BUILTIN_SAVED_CONFIGS: Dict[str, SavedConfig] = {
    'cold_flow_default': COLD_FLOW_DEFAULT_TEMPLATE,
    'hot_fire_default': HOT_FIRE_DEFAULT_TEMPLATE,
}


# =============================================================================
# TEMPLATE MANAGER
# =============================================================================

class SavedConfigManager:
    """Manage configuration templates."""

    def __init__(self, saved_configs_dir: str = "saved_configs", templates_dir: str = None):
        # Backward compatibility: accept both saved_configs_dir and templates_dir
        if templates_dir is not None:
            self.saved_configs_dir = templates_dir
        else:
            self.saved_configs_dir = saved_configs_dir
        self._ensure_dir()
    
    def _ensure_dir(self):
        """Ensure templates directory exists."""
        if not os.path.exists(self.saved_configs_dir):
            os.makedirs(self.saved_configs_dir)
    
    def list_templates(self, include_builtin: bool = True) -> List[Dict[str, Any]]:
        """List all available templates."""
        templates = []
        
        # Add built-in templates
        if include_builtin:
            for key, template in BUILTIN_SAVED_CONFIGS.items():
                templates.append({
                    'id': key,
                    'name': template.name,
                    'version': template.version,
                    'test_type': template.test_type,
                    'description': template.description,
                    'builtin': True,
                    'tags': template.tags,
                })
        
        # Add custom templates
        for filename in os.listdir(self.saved_configs_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.saved_configs_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    templates.append({
                        'id': filename[:-5],  # Remove .json
                        'name': data.get('config_name', data.get('name', filename)),
                        'version': data.get('version', '1.0.0'),
                        'test_type': data.get('test_type', 'unknown'),
                        'description': data.get('description', ''),
                        'builtin': False,
                        'tags': data.get('tags', []),
                    })
                except Exception:
                    continue
        
        return templates
    
    def get_template(self, saved_config_id: str) -> Optional[SavedConfig]:
        """Get a template by ID."""
        # Check built-in
        if saved_config_id in BUILTIN_SAVED_CONFIGS:
            return copy.deepcopy(BUILTIN_SAVED_CONFIGS[saved_config_id])
        
        # Check custom
        filepath = os.path.join(self.saved_configs_dir, f"{saved_config_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return SavedConfig.from_dict(data)
        
        return None
    
    def save_template(self, template: SavedConfig, saved_config_id: Optional[str] = None) -> str:
        """Save a template to disk."""
        if saved_config_id is None:
            # Generate ID from name
            saved_config_id = template.name.lower().replace(' ', '_').replace('-', '_')
            saved_config_id = ''.join(c for c in saved_config_id if c.isalnum() or c == '_')
        
        filepath = os.path.join(self.saved_configs_dir, f"{saved_config_id}.json")
        
        with open(filepath, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)
        
        return saved_config_id
    
    def delete_template(self, saved_config_id: str) -> bool:
        """Delete a custom template."""
        if saved_config_id in BUILTIN_SAVED_CONFIGS:
            raise ValueError("Cannot delete built-in templates")
        
        filepath = os.path.join(self.saved_configs_dir, f"{saved_config_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    def create_from_parent(
        self,
        parent_id: str,
        new_name: str,
        overrides: Dict[str, Any],
    ) -> SavedConfig:
        """Create a new template inheriting from a parent."""
        parent = self.get_template(parent_id)
        if parent is None:
            raise ValueError(f"Parent template '{parent_id}' not found")
        
        # Create copy
        new_template = copy.deepcopy(parent)
        new_template.name = new_name
        new_template.parent_template = parent_id
        new_template.created_date = datetime.now().isoformat()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(new_template, key):
                if isinstance(value, dict) and isinstance(getattr(new_template, key), dict):
                    # Merge dicts
                    getattr(new_template, key).update(value)
                else:
                    setattr(new_template, key, value)
        
        return new_template
    
    def export_template(self, saved_config_id: str, output_path: str):
        """Export template to a file."""
        template = self.get_template(saved_config_id)
        if template is None:
            raise ValueError(f"Template '{saved_config_id}' not found")
        
        with open(output_path, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)
    
    def import_template(self, input_path: str, saved_config_id: Optional[str] = None) -> str:
        """Import template from a file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        template = SavedConfig.from_dict(data)
        return self.save_template(template, saved_config_id)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_template_for_test_type(test_type: str) -> List[SavedConfig]:
    """Get all templates matching a test type."""
    return [t for t in BUILTIN_SAVED_CONFIGS.values() if t.test_type == test_type]


def validate_config_against_template(
    config: Dict[str, Any],
    template: SavedConfig,
) -> List[str]:
    """
    Validate a config dict against a template.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Check test type matches
    if config.get('test_type') != template.test_type:
        errors.append(f"Test type mismatch: expected '{template.test_type}', got '{config.get('test_type')}'")

    # Check uncertainties are defined
    config_uncertainties = config.get('uncertainties', {})
    for key in template.uncertainties:
        if key not in config_uncertainties:
            errors.append(f"Missing uncertainty specification: {key}")

    # Check required settings
    config_settings = config.get('settings', {})
    for key in template.settings:
        if key not in config_settings:
            errors.append(f"Missing setting: {key}")

    return errors


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Deep merge two config dicts."""
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_saved_config(
    saved_config_id: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an analysis-ready config from a template.
    
    Args:
        saved_config_id: ID of template to use
        overrides: Dict of values to override
        
    Returns:
        Config dict ready for analysis
    """
    manager = TemplateManager()
    template = manager.get_template(saved_config_id)
    
    if template is None:
        raise ValueError(f"Template '{saved_config_id}' not found")
    
    config = template.to_config()
    
    if overrides:
        config = merge_configs(config, overrides)
    
    return config


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES (v2.3.0)
# =============================================================================
# These allow old code using "template" terminology to continue working

# Class aliases
ConfigTemplate = SavedConfig  # Old name → New name
TemplateManager = SavedConfigManager  # Old name → New name

# Dictionary aliases
BUILTIN_TEMPLATES = BUILTIN_SAVED_CONFIGS  # Old name → New name

# Function aliases
create_config_from_template = load_saved_config  # Old name → New name

