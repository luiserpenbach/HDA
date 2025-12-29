# pages/07_Config_Manager.py

import streamlit as st
import json
import os
from pathlib import Path
import yaml

st.set_page_config(page_title="Config Manager", page_icon="⚙", layout="wide")

st.title("Configuration Manager")
st.markdown("Create and manage test configurations for cold flow and hot fire analysis")

# --- CONFIG DIRECTORY SETUP ---
CONFIG_DIR = os.path.join(os.getcwd(), "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)


def get_config_files():
    """Get list of existing config files"""
    configs = []
    for file in os.listdir(CONFIG_DIR):
        if file.endswith(('.json', '.yaml', '.yml')):
            configs.append(file)
    return sorted(configs)


def load_config_file(filename):
    """Load config from file"""
    filepath = os.path.join(CONFIG_DIR, filename)

    if filename.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filename.endswith(('.yaml', '.yml')):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)


def save_config_file(filename, config_data):
    """Save config to file"""
    filepath = os.path.join(CONFIG_DIR, filename)

    if filename.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=4)
    elif filename.endswith(('.yaml', '.yml')):
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)


def delete_config_file(filename):
    """Delete config file"""
    filepath = os.path.join(CONFIG_DIR, filename)
    os.remove(filepath)


# --- SIDEBAR: CONFIG SELECTION ---
with st.sidebar:
    st.header("Configuration Library")

    configs = get_config_files()

    if configs:
        st.info(f"Found {len(configs)} configurations")

        selected_config = st.selectbox("Select Configuration", configs)

        col_btn1, col_btn2 = st.columns(2)

        load_config = col_btn1.button("Load", type="primary")
        delete_config = col_btn2.button("Delete", type="secondary")

        if delete_config:
            if st.sidebar.checkbox(f"Confirm delete {selected_config}?"):
                delete_config_file(selected_config)
                st.success(f"Deleted {selected_config}")
                st.rerun()
    else:
        st.warning("No configurations found")
        selected_config = None
        load_config = False

    st.markdown("---")
    st.markdown("**Quick Actions**")

    if st.button("Create New Config"):
        st.session_state['mode'] = 'create'
        st.session_state['config_data'] = None

    if st.button("View All Configs"):
        st.session_state['mode'] = 'view_all'

# --- LOAD EXISTING CONFIG ---
if selected_config and load_config:
    st.session_state['mode'] = 'edit'
    st.session_state['config_data'] = load_config_file(selected_config)
    st.session_state['config_filename'] = selected_config

# --- MAIN INTERFACE ---
mode = st.session_state.get('mode', 'home')

# --- HOME VIEW ---
if mode == 'home':
    st.subheader("Welcome to Configuration Manager")

    st.markdown("""
    Use this tool to create and manage test configurations for your analysis workflows.

    **Configuration Types:**
    - **Cold Flow**: Injector characterization, valve testing
    - **Hot Fire**: Engine performance testing

    **What's in a Configuration:**
    - Channel mappings (sensor names to standard labels)
    - Fluid properties (density, temperature)
    - Geometry (orifice areas, throat diameter)
    - Theoretical values (CEA predictions)

    **Get Started:**
    - Click "Create New Config" to start from scratch
    - Or select an existing config from the sidebar to edit
    """)

    if configs:
        st.markdown("---")
        st.subheader("Recent Configurations")

        config_list = []
        for cfg in configs[:5]:
            try:
                data = load_config_file(cfg)
                config_list.append({
                    'Name': cfg,
                    'Type': data.get('test_type', 'Unknown'),
                    'Description': data.get('description', 'No description')[:50]
                })
            except:
                pass

        if config_list:
            import pandas as pd

            df_configs = pd.DataFrame(config_list)
            st.dataframe(df_configs, use_container_width=True, hide_index=True)

# --- VIEW ALL CONFIGS ---
elif mode == 'view_all':
    st.subheader("All Configurations")

    if configs:
        config_details = []
        for cfg in configs:
            try:
                data = load_config_file(cfg)
                config_details.append({
                    'Filename': cfg,
                    'Name': data.get('name', 'Unnamed'),
                    'Type': data.get('test_type', 'Unknown'),
                    'Description': data.get('description', 'No description'),
                    'Version': data.get('version', '1.0')
                })
            except Exception as e:
                config_details.append({
                    'Filename': cfg,
                    'Name': 'Error',
                    'Type': 'Error',
                    'Description': str(e),
                    'Version': 'N/A'
                })

        import pandas as pd

        df_all = pd.DataFrame(config_details)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

        # Export all configs
        if st.button("Export Configuration Summary"):
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "config_summary.csv",
                "text/csv"
            )
    else:
        st.info("No configurations found")

    if st.button("Back to Home"):
        st.session_state['mode'] = 'home'
        st.rerun()

# --- CREATE/EDIT CONFIG ---
elif mode in ['create', 'edit']:
    is_edit = (mode == 'edit')

    st.subheader("Edit Configuration" if is_edit else "Create New Configuration")

    # Initialize config data
    if st.session_state.get('config_data') is None:
        config_data = {
            'name': '',
            'version': '1.0',
            'test_type': 'cold_flow',
            'description': '',
            'channel_config': {},
            'columns': {},
            'fluid': {},
            'geometry': {},
            'propellants': {},
            'chamber': {},
            'theoretical': {}
        }
    else:
        config_data = st.session_state['config_data'].copy()

    # --- BASIC INFORMATION ---
    st.markdown("### Basic Information")

    col_basic1, col_basic2 = st.columns(2)

    with col_basic1:
        config_data['name'] = st.text_input(
            "Configuration Name",
            value=config_data.get('name', ''),
            placeholder="INJ_001_N2_Config"
        )

        config_data['test_type'] = st.selectbox(
            "Test Type",
            ['cold_flow', 'hot_fire'],
            index=['cold_flow', 'hot_fire'].index(config_data.get('test_type', 'cold_flow'))
        )

    with col_basic2:
        config_data['version'] = st.text_input(
            "Version",
            value=config_data.get('version', '1.0')
        )

        config_data['description'] = st.text_area(
            "Description",
            value=config_data.get('description', ''),
            placeholder="Brief description of this configuration"
        )

    # --- CHANNEL MAPPING ---
    st.markdown("---")
    st.markdown("### Channel Mapping")
    st.caption("Map raw data column names to standardized names used by analysis functions")

    col_chan1, col_chan2 = st.columns(2)

    if 'channel_config' not in config_data:
        config_data['channel_config'] = {}

    with col_chan1:
        st.markdown("**Input Channels (Raw Data)**")

        # Common mappings
        mappings = []
        if config_data['test_type'] == 'cold_flow':
            mappings = [
                ('upstream_pressure', 'Upstream Pressure'),
                ('downstream_pressure', 'Downstream Pressure'),
                ('mass_flow', 'Mass Flow'),
                ('temperature', 'Temperature'),
                ('timestamp', 'Timestamp')
            ]
        else:  # hot_fire
            mappings = [
                ('chamber_pressure', 'Chamber Pressure'),
                ('thrust', 'Thrust/Force'),
                ('mf_ox', 'Oxidizer Mass Flow'),
                ('mf_fuel', 'Fuel Mass Flow'),
                ('p_ox', 'Ox Inlet Pressure'),
                ('p_fuel', 'Fuel Inlet Pressure'),
                ('timestamp', 'Timestamp')
            ]

        for key, label in mappings:
            config_data['channel_config'][key] = st.text_input(
                label,
                value=config_data['channel_config'].get(key, ''),
                key=f"chan_{key}"
            )

    with col_chan2:
        st.markdown("**Additional Channels**")

        if 'custom_channels' not in st.session_state:
            st.session_state['custom_channels'] = list(config_data['channel_config'].keys())

        new_channel_name = st.text_input("Add Custom Channel Name", placeholder="my_sensor")
        new_channel_mapping = st.text_input("Raw Column Name", placeholder="CH_05")

        if st.button("Add Custom Channel"):
            if new_channel_name and new_channel_mapping:
                config_data['channel_config'][new_channel_name] = new_channel_mapping
                st.success(f"Added: {new_channel_name} -> {new_channel_mapping}")
                st.rerun()

        # Show current custom channels
        if config_data['channel_config']:
            st.markdown("**Current Mappings:**")
            for k, v in config_data['channel_config'].items():
                if v:  # Only show non-empty
                    st.caption(f"`{k}` → `{v}`")

    # --- COLD FLOW SPECIFIC ---
    if config_data['test_type'] == 'cold_flow':
        st.markdown("---")
        st.markdown("### Cold Flow Configuration")

        col_cf1, col_cf2 = st.columns(2)

        with col_cf1:
            st.markdown("**Fluid Properties**")

            if 'fluid' not in config_data:
                config_data['fluid'] = {}

            config_data['fluid']['name'] = st.text_input(
                "Fluid Name",
                value=config_data['fluid'].get('name', ''),
                placeholder="Nitrogen"
            )

            config_data['fluid']['density_kg_m3'] = st.number_input(
                "Density (kg/m³)",
                value=float(config_data['fluid'].get('density_kg_m3', 1.0)),
                min_value=0.0,
                format="%.3f"
            )

            config_data['fluid']['temperature_K'] = st.number_input(
                "Temperature (K)",
                value=float(config_data['fluid'].get('temperature_K', 293.15)),
                min_value=0.0,
                format="%.2f"
            )

        with col_cf2:
            st.markdown("**Geometry**")

            if 'geometry' not in config_data:
                config_data['geometry'] = {}

            config_data['geometry']['orifice_area_mm2'] = st.number_input(
                "Orifice Area (mm²)",
                value=float(config_data['geometry'].get('orifice_area_mm2', 1.0)),
                min_value=0.0,
                format="%.4f"
            )

            config_data['geometry']['orifice_diameter_mm'] = st.number_input(
                "Orifice Diameter (mm)",
                value=float(config_data['geometry'].get('orifice_diameter_mm', 1.0)),
                min_value=0.0,
                format="%.3f"
            )

            # Calculate area from diameter option
            if st.checkbox("Calculate area from diameter"):
                import math

                diameter = config_data['geometry']['orifice_diameter_mm']
                area = math.pi * (diameter / 2) ** 2
                config_data['geometry']['orifice_area_mm2'] = area
                st.info(f"Calculated area: {area:.4f} mm²")

    # --- HOT FIRE SPECIFIC ---
    elif config_data['test_type'] == 'hot_fire':
        st.markdown("---")
        st.markdown("### Hot Fire Configuration")

        col_hf1, col_hf2 = st.columns(2)

        with col_hf1:
            st.markdown("**Propellants**")

            if 'propellants' not in config_data:
                config_data['propellants'] = {}

            config_data['propellants']['oxidizer'] = st.text_input(
                "Oxidizer",
                value=config_data['propellants'].get('oxidizer', ''),
                placeholder="LOX"
            )

            config_data['propellants']['fuel'] = st.text_input(
                "Fuel",
                value=config_data['propellants'].get('fuel', ''),
                placeholder="Ethanol"
            )

            config_data['propellants']['of_ratio_design'] = st.number_input(
                "Design O/F Ratio",
                value=float(config_data['propellants'].get('of_ratio_design', 2.0)),
                min_value=0.0,
                format="%.2f"
            )

        with col_hf2:
            st.markdown("**Chamber Geometry**")

            if 'chamber' not in config_data:
                config_data['chamber'] = {}

            config_data['chamber']['throat_diameter_mm'] = st.number_input(
                "Throat Diameter (mm)",
                value=float(config_data['chamber'].get('throat_diameter_mm', 10.0)),
                min_value=0.0,
                format="%.3f"
            )

            config_data['chamber']['throat_area_mm2'] = st.number_input(
                "Throat Area (mm²)",
                value=float(config_data['chamber'].get('throat_area_mm2', 78.5)),
                min_value=0.0,
                format="%.4f"
            )

            config_data['chamber']['expansion_ratio'] = st.number_input(
                "Expansion Ratio (ε)",
                value=float(config_data['chamber'].get('expansion_ratio', 10.0)),
                min_value=1.0,
                format="%.2f"
            )

            # Calculate throat area from diameter
            if st.checkbox("Calculate throat area from diameter"):
                import math

                diameter = config_data['chamber']['throat_diameter_mm']
                area = math.pi * (diameter / 2) ** 2
                config_data['chamber']['throat_area_mm2'] = area
                st.info(f"Calculated area: {area:.4f} mm²")

        # Theoretical values (CEA)
        st.markdown("---")
        st.markdown("**Theoretical Performance (from CEA)**")
        st.caption("Optional: Enter CEA-predicted values for efficiency calculations")

        col_theo1, col_theo2, col_theo3 = st.columns(3)

        if 'theoretical' not in config_data:
            config_data['theoretical'] = {}

        with col_theo1:
            config_data['theoretical']['c_star'] = st.number_input(
                "C* (m/s)",
                value=float(config_data['theoretical'].get('c_star', 0.0)),
                min_value=0.0,
                format="%.1f"
            )

        with col_theo2:
            config_data['theoretical']['isp_vac'] = st.number_input(
                "Isp Vacuum (s)",
                value=float(config_data['theoretical'].get('isp_vac', 0.0)),
                min_value=0.0,
                format="%.1f"
            )

        with col_theo3:
            config_data['theoretical']['cf_vac'] = st.number_input(
                "Cf Vacuum",
                value=float(config_data['theoretical'].get('cf_vac', 0.0)),
                min_value=0.0,
                format="%.3f"
            )

    # --- SAVE CONFIGURATION ---
    st.markdown("---")
    st.markdown("### Save Configuration")

    col_save1, col_save2, col_save3 = st.columns([2, 1, 1])

    with col_save1:
        if is_edit:
            default_filename = st.session_state.get('config_filename', 'config.json')
        else:
            default_filename = f"{config_data.get('name', 'config').replace(' ', '_')}.json"

        save_filename = st.text_input(
            "Filename",
            value=default_filename,
            help="Use .json or .yaml extension"
        )

    with col_save2:
        file_format = st.radio("Format", ["JSON", "YAML"], horizontal=True)

        # Ensure correct extension
        if file_format == "JSON" and not save_filename.endswith('.json'):
            save_filename = os.path.splitext(save_filename)[0] + '.json'
        elif file_format == "YAML" and not save_filename.endswith(('.yaml', '.yml')):
            save_filename = os.path.splitext(save_filename)[0] + '.yaml'

    with col_save3:
        st.write("")  # Spacing
        st.write("")

        if st.button("Save Configuration", type="primary"):
            if not config_data['name']:
                st.error("Configuration name is required")
            else:
                try:
                    save_config_file(save_filename, config_data)
                    st.success(f"Saved: {save_filename}")
                    st.balloons()

                    # Clear session state
                    st.session_state['mode'] = 'home'
                    st.session_state['config_data'] = None

                except Exception as e:
                    st.error(f"Error saving: {e}")

    # Preview
    with st.expander("Preview Configuration"):
        if file_format == "JSON":
            st.json(config_data)
        else:
            st.code(yaml.dump(config_data, default_flow_style=False, sort_keys=False))

    # Cancel button
    if st.button("Cancel"):
        st.session_state['mode'] = 'home'
        st.session_state['config_data'] = None
        st.rerun()

# --- CONFIG TEMPLATES ---
with st.sidebar:
    st.markdown("---")
    st.subheader("Templates")

    if st.button("Load Cold Flow Template"):
        st.session_state['mode'] = 'create'
        st.session_state['config_data'] = {
            'name': 'CF_Template',
            'version': '1.0',
            'test_type': 'cold_flow',
            'description': 'Template for cold flow testing',
            'channel_config': {
                'upstream_pressure': 'P_up',
                'downstream_pressure': 'P_down',
                'mass_flow': 'MF',
                'temperature': 'T_up',
                'timestamp': 'Time'
            },
            'columns': {
                'upstream_pressure': 'P_up',
                'mass_flow': 'MF',
                'temperature': 'T_up'
            },
            'fluid': {
                'name': 'Nitrogen',
                'density_kg_m3': 1.165,
                'temperature_K': 293.15
            },
            'geometry': {
                'orifice_area_mm2': 1.0,
                'orifice_diameter_mm': 1.128
            }
        }
        st.rerun()

    if st.button("Load Hot Fire Template"):
        st.session_state['mode'] = 'create'
        st.session_state['config_data'] = {
            'name': 'HF_Template',
            'version': '1.0',
            'test_type': 'hot_fire',
            'description': 'Template for hot fire testing',
            'channel_config': {
                'chamber_pressure': 'Pc',
                'thrust': 'Thrust',
                'mf_ox': 'MF_Ox',
                'mf_fuel': 'MF_Fuel',
                'timestamp': 'Time'
            },
            'columns': {
                'chamber_pressure': 'Pc',
                'thrust': 'Thrust',
                'mf_ox': 'MF_Ox',
                'mf_fuel': 'MF_Fuel'
            },
            'propellants': {
                'oxidizer': 'LOX',
                'fuel': 'Ethanol',
                'of_ratio_design': 2.0
            },
            'chamber': {
                'throat_diameter_mm': 10.0,
                'throat_area_mm2': 78.54,
                'expansion_ratio': 10.0
            },
            'theoretical': {
                'c_star': 1800.0,
                'isp_vac': 320.0,
                'cf_vac': 1.8
            }
        }
        st.rerun()