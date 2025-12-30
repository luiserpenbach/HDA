"""
Configuration Templates Page (P2)
=================================
Manage configuration templates for test analysis.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime

from core.templates import (
    TemplateManager,
    ConfigTemplate,
    BUILTIN_TEMPLATES,
    create_config_from_template,
    validate_config_against_template,
)

st.set_page_config(page_title="Config Templates", page_icon="📝", layout="wide")

st.title("📝 Configuration Templates")
st.markdown("Manage configuration templates for consistent test analysis.")

# Initialize template manager
manager = TemplateManager()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Actions")
    
    action = st.radio(
        "Select action",
        ["Browse Templates", "Create Template", "Import/Export"]
    )

# =============================================================================
# BROWSE TEMPLATES
# =============================================================================

if action == "Browse Templates":
    st.header("📚 Available Templates")
    
    templates = manager.list_templates()
    
    if not templates:
        st.info("No templates found")
    else:
        # Filter
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox("Filter by type", ["All", "cold_flow", "hot_fire"])
        with col2:
            filter_builtin = st.selectbox("Source", ["All", "Built-in", "Custom"])
        
        # Apply filters
        filtered = templates
        if filter_type != "All":
            filtered = [t for t in filtered if t['test_type'] == filter_type]
        if filter_builtin == "Built-in":
            filtered = [t for t in filtered if t['builtin']]
        elif filter_builtin == "Custom":
            filtered = [t for t in filtered if not t['builtin']]
        
        # Display templates
        for template_info in filtered:
            with st.expander(f"{'📦' if template_info['builtin'] else '📄'} {template_info['name']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**ID:** `{template_info['id']}`")
                    st.markdown(f"**Type:** {template_info['test_type']}")
                
                with col2:
                    st.markdown(f"**Version:** {template_info['version']}")
                    source = "Built-in" if template_info['builtin'] else "Custom"
                    st.markdown(f"**Source:** {source}")
                
                with col3:
                    if template_info.get('tags'):
                        st.markdown(f"**Tags:** {', '.join(template_info['tags'])}")
                
                if template_info.get('description'):
                    st.markdown(f"_{template_info['description']}_")
                
                # View full template
                if st.button("View Details", key=f"view_{template_info['id']}"):
                    template = manager.get_template(template_info['id'])
                    if template:
                        st.json(template.to_dict())
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Use Template", key=f"use_{template_info['id']}"):
                        config = create_config_from_template(template_info['id'])
                        st.session_state['active_config'] = config
                        st.success("Template loaded to active config!")
                
                with col2:
                    if st.button("📥 Export", key=f"export_{template_info['id']}"):
                        template = manager.get_template(template_info['id'])
                        if template:
                            st.download_button(
                                "Download JSON",
                                json.dumps(template.to_dict(), indent=2),
                                file_name=f"{template_info['id']}.json",
                                mime="application/json",
                                key=f"download_{template_info['id']}"
                            )
                
                with col3:
                    if not template_info['builtin']:
                        if st.button("🗑️ Delete", key=f"delete_{template_info['id']}"):
                            if manager.delete_template(template_info['id']):
                                st.success("Deleted!")
                                st.rerun()

# =============================================================================
# CREATE TEMPLATE
# =============================================================================

elif action == "Create Template":
    st.header("➕ Create New Template")
    
    tab1, tab2 = st.tabs(["From Scratch", "From Existing"])
    
    with tab1:
        st.subheader("Create from Scratch")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Template Name", value="My Template")
            version = st.text_input("Version", value="1.0.0")
            test_type = st.selectbox("Test Type", ["cold_flow", "hot_fire"])
        
        with col2:
            description = st.text_area("Description", height=100)
            tags = st.text_input("Tags (comma-separated)")
        
        st.divider()
        st.subheader("Column Mappings")
        
        if test_type == "cold_flow":
            default_cols = {
                'timestamp': 'timestamp',
                'upstream_pressure': 'P_upstream',
                'downstream_pressure': 'P_downstream',
                'temperature': 'T_fluid',
                'mass_flow': 'mass_flow',
            }
        else:
            default_cols = {
                'timestamp': 'timestamp',
                'chamber_pressure': 'P_chamber',
                'ox_mass_flow': 'mf_ox',
                'fuel_mass_flow': 'mf_fuel',
                'thrust': 'thrust',
            }
        
        columns = {}
        for key, default in default_cols.items():
            columns[key] = st.text_input(f"{key}", value=default, key=f"col_{key}")
        
        st.divider()
        st.subheader("Geometry")
        
        geometry = {}
        if test_type == "cold_flow":
            geometry['orifice_area_mm2'] = st.number_input("Orifice Area (mm²)", value=1.0, format="%.4f")
            geometry['orifice_diameter_mm'] = st.number_input("Orifice Diameter (mm)", value=1.128, format="%.4f")
        else:
            geometry['throat_area_mm2'] = st.number_input("Throat Area (mm²)", value=100.0, format="%.2f")
            geometry['nozzle_expansion_ratio'] = st.number_input("Expansion Ratio", value=10.0, format="%.1f")
        
        st.divider()
        st.subheader("Uncertainties")
        
        uncertainties = {}
        st.markdown("Specify as percentage for relative, or absolute value")
        
        col1, col2 = st.columns(2)
        with col1:
            u_pressure_type = st.selectbox("Pressure uncertainty type", ["relative", "absolute"])
            u_pressure_val = st.number_input("Pressure uncertainty value", value=0.005 if u_pressure_type == "relative" else 0.1)
            uncertainties['pressure'] = {'type': u_pressure_type, 'value': u_pressure_val}
        
        with col2:
            u_mf_type = st.selectbox("Mass flow uncertainty type", ["relative", "absolute"])
            u_mf_val = st.number_input("Mass flow uncertainty value", value=0.01 if u_mf_type == "relative" else 0.1)
            uncertainties['mass_flow'] = {'type': u_mf_type, 'value': u_mf_val}
        
        st.divider()
        st.subheader("Settings")
        
        settings = {}
        settings['sample_rate_hz'] = st.number_input("Sample Rate (Hz)", value=100, min_value=1)
        
        # Create template
        if st.button("💾 Save Template", type="primary"):
            try:
                template = ConfigTemplate(
                    name=name,
                    version=version,
                    test_type=test_type,
                    description=description,
                    columns=columns,
                    geometry=geometry,
                    uncertainties=uncertainties,
                    settings=settings,
                    tags=[t.strip() for t in tags.split(',')] if tags else [],
                )
                
                template_id = manager.save_template(template)
                st.success(f"Saved template: {template_id}")
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Create from Existing Template")
        
        templates = manager.list_templates()
        template_options = {t['name']: t['id'] for t in templates}
        
        parent_name = st.selectbox("Base template", list(template_options.keys()))
        parent_id = template_options.get(parent_name)
        
        new_name = st.text_input("New template name", value=f"{parent_name} (Modified)")
        
        st.markdown("**Specify overrides (JSON format):**")
        overrides_json = st.text_area(
            "Overrides",
            value='{\n  "geometry": {\n    "orifice_area_mm2": 2.0\n  }\n}',
            height=200
        )
        
        if st.button("📋 Create from Parent"):
            try:
                overrides = json.loads(overrides_json)
                
                new_template = manager.create_from_parent(parent_id, new_name, overrides)
                template_id = manager.save_template(new_template)
                
                st.success(f"Created template: {template_id}")
                
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

# =============================================================================
# IMPORT/EXPORT
# =============================================================================

elif action == "Import/Export":
    st.header("📤 Import / Export Templates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Import Template")
        
        uploaded = st.file_uploader("Upload template JSON", type=['json'])
        
        if uploaded:
            try:
                data = json.load(uploaded)
                
                st.json(data)
                
                import_id = st.text_input("Template ID (optional)", value="")
                
                if st.button("📥 Import"):
                    # Save to temp file and import
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(data, f)
                        temp_path = f.name
                    
                    template_id = manager.import_template(temp_path, import_id or None)
                    st.success(f"Imported as: {template_id}")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Export All Templates")
        
        if st.button("📦 Export All Custom Templates"):
            templates = manager.list_templates(include_builtin=False)
            
            if not templates:
                st.info("No custom templates to export")
            else:
                export_data = {}
                for t in templates:
                    template = manager.get_template(t['id'])
                    if template:
                        export_data[t['id']] = template.to_dict()
                
                st.download_button(
                    "Download All Templates",
                    json.dumps(export_data, indent=2),
                    file_name="all_templates.json",
                    mime="application/json"
                )
        
        st.divider()
        
        st.subheader("Built-in Templates Reference")
        
        st.markdown("""
        Built-in templates available:
        - `cold_flow_n2` - Nitrogen cold flow (standard)
        - `cold_flow_water` - Water flow (incompressible)
        - `hot_fire_lox_rp1` - LOX/RP-1 bipropellant
        - `hot_fire_n2o_htpb` - N2O/HTPB hybrid
        """)
        
        if st.button("📥 Export Built-in Templates"):
            export_data = {}
            for tid, template in BUILTIN_TEMPLATES.items():
                export_data[tid] = template.to_dict()
            
            st.download_button(
                "Download Built-in Templates",
                json.dumps(export_data, indent=2),
                file_name="builtin_templates.json",
                mime="application/json"
            )
