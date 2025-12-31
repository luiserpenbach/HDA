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
    Saved Configuration for reuse across tests.

    Note (v2.3.0): In the new architecture, saved configs should only contain
    testbench hardware info. Geometry and fluid will be migrated to metadata.
    This class maintains backward compatibility with v2.0-v2.2 format.
    """
    name: str
    version: str
    test_type: str  # 'cold_flow' or 'hot_fire'
    description: str = ""

    # Column mappings
    columns: Dict[str, str] = field(default_factory=dict)

    # Fluid/propellant properties (will be migrated to metadata in v2.3.0+)
    fluid: Dict[str, Any] = field(default_factory=dict)
    propellants: Dict[str, Any] = field(default_factory=dict)

    # Geometry (will be migrated to metadata in v2.3.0+)
    geometry: Dict[str, float] = field(default_factory=dict)

    # Uncertainties
    uncertainties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    geometry_uncertainties: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)

    # QC settings
    qc: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    parent_config: Optional[str] = None  # Renamed from parent_template
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config_name': self.name,
            'version': self.version,
            'test_type': self.test_type,
            'description': self.description,
            'columns': self.columns,
            'fluid': self.fluid,
            'propellants': self.propellants,
            'geometry': self.geometry,
            'uncertainties': self.uncertainties,
            'geometry_uncertainties': self.geometry_uncertainties,
            'settings': self.settings,
            'qc': self.qc,
            'created_date': self.created_date,
            'author': self.author,
            'parent_config': self.parent_config,  # Renamed from parent_template
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SavedConfig':
        """Create from dictionary."""
        return cls(
            name=data.get('config_name', data.get('name', 'Unknown')),
            version=data.get('version', '1.0.0'),
            test_type=data.get('test_type', 'cold_flow'),
            description=data.get('description', ''),
            columns=data.get('columns', {}),
            fluid=data.get('fluid', {}),
            propellants=data.get('propellants', {}),
            geometry=data.get('geometry', {}),
            uncertainties=data.get('uncertainties', {}),
            geometry_uncertainties=data.get('geometry_uncertainties', {}),
            settings=data.get('settings', {}),
            qc=data.get('qc', {}),
            created_date=data.get('created_date', datetime.now().isoformat()),
            author=data.get('author', ''),
            parent_template=data.get('parent_template'),
            tags=data.get('tags', []),
        )
    
    def to_config(self) -> Dict[str, Any]:
        """Convert to analysis-ready config dict."""
        config = {
            'config_name': self.name,
            'test_type': self.test_type,
            'columns': self.columns,
            'geometry': self.geometry,
            'uncertainties': self.uncertainties,
            'geometry_uncertainties': self.geometry_uncertainties,
            'settings': self.settings,
        }
        
        if self.fluid:
            config['fluid'] = self.fluid
        if self.propellants:
            config['propellants'] = self.propellants
        if self.qc:
            config['qc'] = self.qc
        
        return config


# =============================================================================
# BUILT-IN TEMPLATES
# =============================================================================

COLD_FLOW_N2_TEMPLATE = ConfigTemplate(
    name="Cold Flow - Nitrogen (Standard)",
    version="2.0.0",
    test_type="cold_flow",
    description="Standard cold flow test with nitrogen gas",
    columns={
        'timestamp': 'timestamp',
        'upstream_pressure': 'P_upstream',
        'downstream_pressure': 'P_downstream',
        'temperature': 'T_fluid',
        'mass_flow': 'mass_flow',
    },
    fluid={
        'name': 'nitrogen',
        'gamma': 1.4,
        'R': 296.8,  # J/(kg·K)
        'molecular_weight': 28.0134,  # g/mol
    },
    geometry={
        'orifice_area_mm2': 1.0,
        'orifice_diameter_mm': 1.128,
    },
    uncertainties={
        'pressure': {'type': 'relative', 'value': 0.005},  # 0.5%
        'temperature': {'type': 'absolute', 'value': 1.0},  # 1 K
        'mass_flow': {'type': 'relative', 'value': 0.01},  # 1%
    },
    geometry_uncertainties={
        'area': {'type': 'relative', 'value': 0.02},  # 2%
    },
    settings={
        'sample_rate_hz': 100,
        'steady_state_cv_threshold': 0.02,
        'min_steady_duration_ms': 500,
    },
    qc={
        'max_nan_ratio': 0.05,
        'min_correlation': 0.3,
        'max_cv_percent': 5.0,
    },
    tags=['cold_flow', 'nitrogen', 'standard'],
)

COLD_FLOW_WATER_TEMPLATE = ConfigTemplate(
    name="Cold Flow - Water (Incompressible)",
    version="2.0.0",
    test_type="cold_flow",
    description="Water flow test for hydraulic characterization",
    columns={
        'timestamp': 'timestamp',
        'upstream_pressure': 'P_inlet',
        'downstream_pressure': 'P_outlet',
        'temperature': 'T_water',
        'mass_flow': 'mass_flow',
    },
    fluid={
        'name': 'water',
        'gamma': 1.0,  # Incompressible
        'density_kg_m3': 998.0,  # At 20°C
        'viscosity_pa_s': 0.001,
    },
    geometry={
        'orifice_area_mm2': 1.0,
        'orifice_diameter_mm': 1.128,
    },
    uncertainties={
        'pressure': {'type': 'relative', 'value': 0.005},
        'temperature': {'type': 'absolute', 'value': 0.5},
        'mass_flow': {'type': 'relative', 'value': 0.005},
    },
    geometry_uncertainties={
        'area': {'type': 'relative', 'value': 0.02},
    },
    settings={
        'sample_rate_hz': 100,
        'use_incompressible_model': True,
    },
    qc={
        'max_nan_ratio': 0.05,
        'min_correlation': 0.5,
    },
    tags=['cold_flow', 'water', 'incompressible'],
)

HOT_FIRE_LOX_RP1_TEMPLATE = ConfigTemplate(
    name="Hot Fire - LOX/RP-1 (Standard)",
    version="2.0.0",
    test_type="hot_fire",
    description="Standard hot fire test with LOX/RP-1 propellants",
    columns={
        'timestamp': 'timestamp',
        'chamber_pressure': 'P_chamber',
        'ox_mass_flow': 'mf_ox',
        'fuel_mass_flow': 'mf_fuel',
        'thrust': 'thrust',
        'ox_inlet_pressure': 'P_ox_inlet',
        'fuel_inlet_pressure': 'P_fuel_inlet',
    },
    propellants={
        'oxidizer': 'LOX',
        'fuel': 'RP-1',
        'optimal_of_ratio': 2.56,
        'theoretical_cstar_m_s': 1780,
        'theoretical_isp_s': 311,
    },
    geometry={
        'throat_area_mm2': 100.0,
        'throat_diameter_mm': 11.28,
        'nozzle_expansion_ratio': 10.0,
        'chamber_volume_cc': 50.0,
    },
    uncertainties={
        'chamber_pressure': {'type': 'relative', 'value': 0.005},
        'mass_flow': {'type': 'relative', 'value': 0.01},
        'thrust': {'type': 'relative', 'value': 0.02},
    },
    geometry_uncertainties={
        'throat_area': {'type': 'relative', 'value': 0.01},
    },
    settings={
        'sample_rate_hz': 1000,
        'g0': 9.80665,
    },
    qc={
        'max_nan_ratio': 0.02,
        'min_pc_bar': 5.0,
        'max_of_deviation': 0.5,
    },
    tags=['hot_fire', 'LOX', 'RP-1', 'bipropellant'],
)

HOT_FIRE_N2O_HTPB_TEMPLATE = ConfigTemplate(
    name="Hot Fire - N2O/HTPB Hybrid",
    version="2.0.0",
    test_type="hot_fire",
    description="Hybrid motor test with N2O oxidizer and HTPB fuel grain",
    columns={
        'timestamp': 'timestamp',
        'chamber_pressure': 'P_c',
        'ox_mass_flow': 'mf_n2o',
        'thrust': 'thrust',
        'ox_tank_pressure': 'P_tank',
    },
    propellants={
        'oxidizer': 'N2O',
        'fuel': 'HTPB',
        'theoretical_of_ratio': 7.0,
        'theoretical_cstar_m_s': 1550,
    },
    geometry={
        'throat_area_mm2': 50.0,
        'nozzle_expansion_ratio': 8.0,
        'port_diameter_mm': 30.0,
        'grain_length_mm': 200.0,
    },
    uncertainties={
        'chamber_pressure': {'type': 'relative', 'value': 0.01},
        'mass_flow': {'type': 'relative', 'value': 0.02},
        'thrust': {'type': 'relative', 'value': 0.03},
    },
    settings={
        'sample_rate_hz': 500,
        'g0': 9.80665,
    },
    tags=['hot_fire', 'N2O', 'HTPB', 'hybrid'],
)

# Registry of built-in templates
BUILTIN_SAVED_CONFIGS: Dict[str, ConfigTemplate] = {
    'cold_flow_n2': COLD_FLOW_N2_TEMPLATE,
    'cold_flow_water': COLD_FLOW_WATER_TEMPLATE,
    'hot_fire_lox_rp1': HOT_FIRE_LOX_RP1_TEMPLATE,
    'hot_fire_n2o_htpb': HOT_FIRE_N2O_HTPB_TEMPLATE,
}


# =============================================================================
# TEMPLATE MANAGER
# =============================================================================

class SavedConfigManager:
    """Manage configuration templates."""
    
    def __init__(self, saved_configs_dir: str = "saved_configs"):
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
    
    def get_template(self, saved_config_id: str) -> Optional[ConfigTemplate]:
        """Get a template by ID."""
        # Check built-in
        if saved_config_id in BUILTIN_SAVED_CONFIGS:
            return copy.deepcopy(BUILTIN_SAVED_CONFIGS[saved_config_id])
        
        # Check custom
        filepath = os.path.join(self.saved_configs_dir, f"{saved_config_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return ConfigTemplate.from_dict(data)
        
        return None
    
    def save_template(self, template: ConfigTemplate, saved_config_id: Optional[str] = None) -> str:
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
    ) -> ConfigTemplate:
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
        
        template = ConfigTemplate.from_dict(data)
        return self.save_template(template, saved_config_id)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_template_for_test_type(test_type: str) -> List[ConfigTemplate]:
    """Get all templates matching a test type."""
    return [t for t in BUILTIN_SAVED_CONFIGS.values() if t.test_type == test_type]


def validate_config_against_template(
    config: Dict[str, Any],
    template: ConfigTemplate,
) -> List[str]:
    """
    Validate a config dict against a template.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    # Check required columns
    template_cols = template.columns
    config_cols = config.get('columns', {})
    
    for key in template_cols:
        if key not in config_cols:
            errors.append(f"Missing column mapping: {key}")
    
    # Check geometry
    for key in template.geometry:
        if key not in config.get('geometry', {}):
            errors.append(f"Missing geometry parameter: {key}")
    
    # Check uncertainties
    for key in template.uncertainties:
        if key not in config.get('uncertainties', {}):
            errors.append(f"Missing uncertainty specification: {key}")
    
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

# Function aliases
create_config_from_template = load_saved_config  # Old name → New name

