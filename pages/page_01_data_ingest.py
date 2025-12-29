# pages/01_📥_Data_Ingest.py

import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import re

st.set_page_config(page_title="Data Ingest", page_icon="📥", layout="wide")

# --- CONFIGURATION ---
if 'data_root' not in st.session_state:
    st.session_state['data_root'] = os.path.join(os.getcwd(), "test_data")

st.title("📥 Test Data Ingest")
st.markdown("Standardized test data ingestion with automatic folder structure and metadata generation.")


# --- UTILITY FUNCTIONS ---

def get_subfolders(path):
    """Returns a list of immediate subfolders in a directory."""
    if not os.path.exists(path):
        return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def parse_run_id(folder_name):
    match = re.search(r'Run(\d+)', folder_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def get_next_run_id(campaign_path):
    if not os.path.exists(campaign_path):
        return "RUN01"
    subfolders = get_subfolders(campaign_path)
    max_id = 0
    for folder in subfolders:
        parts = folder.split('-')
        for part in parts:
            rid = parse_run_id(part)
            if rid > max_id:
                max_id = rid
    return f"RUN{max_id + 1:02d}"


def save_metadata(path, data):
    with open(os.path.join(path, "metadata.json"), 'w') as f:
        json.dump(data, f, indent=4)


def process_file(uploaded_file, target_folder, new_filename_base, convert_parquet):
    raw_path = os.path.join(target_folder, "raw_data")
    os.makedirs(raw_path, exist_ok=True)
    original_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if convert_parquet and original_ext == '.csv':
        save_name = f"{new_filename_base}.parquet"
        full_path = os.path.join(raw_path, save_name)
        try:
            df = pd.read_csv(uploaded_file)
            df.to_parquet(full_path, engine='pyarrow', compression='snappy')
            return True, f"Converted to Parquet: {save_name}", full_path
        except Exception as e:
            return False, f"Error converting parquet: {e}", None
    else:
        save_name = f"{new_filename_base}{original_ext}"
        full_path = os.path.join(raw_path, save_name)
        with open(full_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, f"Saved raw file: {save_name}", full_path


# --- ROOT DIRECTORY ---
with st.sidebar:
    st.header("📂 Storage Root")
    root_path = st.text_input("Data Root Path", value=st.session_state['data_root'])
    if root_path != st.session_state['data_root']:
        st.session_state['data_root'] = root_path

    if not os.path.exists(root_path):
        st.warning(f"Path does not exist. Will be created on first ingest.")

# --- HIERARCHICAL SELECTION ---
st.subheader("1️⃣ Test Hierarchy")
st.caption("Define the test location in your data structure")

col1, col2, col3 = st.columns(3)

# SYSTEM
with col1:
    st.markdown("**System**")
    systems = get_subfolders(root_path) if os.path.exists(root_path) else []
    systems.insert(0, "➕ Create New...")
    selected_system = st.selectbox("Select System", systems, key='sys_select')

    if selected_system == "➕ Create New...":
        system_name = st.text_input("New System Name", key='sys_new').upper().strip()
    else:
        system_name = selected_system

# TYPE
with col2:
    st.markdown("**Test Type**")
    if system_name:
        sys_path = os.path.join(root_path, system_name)
        existing_types = []
        if os.path.exists(sys_path):
            folders = get_subfolders(sys_path)
            for f in folders:
                if f.startswith(f"{system_name}-"):
                    existing_types.append(f.replace(f"{system_name}-", ""))
                else:
                    existing_types.append(f)

        existing_types.insert(0, "➕ Create New...")
        selected_type = st.selectbox("Select Test Type", existing_types, key='type_select')

        if selected_type == "➕ Create New...":
            type_name = st.text_input("New Type Name", key='type_new').upper().strip()
        else:
            type_name = selected_type
    else:
        st.info("Select System first")
        type_name = None

# CAMPAIGN
with col3:
    st.markdown("**Campaign**")
    if system_name and type_name:
        type_folder_name = f"{system_name}-{type_name}"
        type_path = os.path.join(root_path, system_name, type_folder_name)

        campaigns = set()
        if os.path.exists(type_path):
            run_folders = get_subfolders(type_path)
            for rf in run_folders:
                parts = rf.split('-')
                if len(parts) >= 3:
                    campaigns.add(parts[-2])

        sorted_campaigns = sorted(list(campaigns))
        sorted_campaigns.insert(0, "➕ Create New...")
        selected_campaign = st.selectbox("Select Campaign", sorted_campaigns, key='camp_select')

        if selected_campaign == "➕ Create New...":
            campaign_id = st.text_input("New Campaign ID", key='camp_new').upper().strip()
        else:
            campaign_id = selected_campaign
    else:
        st.info("Select Type first")
        campaign_id = None

st.markdown("---")

# --- RUN CONFIGURATION ---
st.subheader("2️⃣ Test Configuration")

col_meta, col_file = st.columns([1, 1])

with col_meta:
    suggested_run_id = "RUN01"
    if system_name and type_name and campaign_id:
        type_folder_name = f"{system_name}-{type_name}"
        target_base_path = os.path.join(root_path, system_name, type_folder_name)
        suggested_run_id = get_next_run_id(target_base_path)

    run_id = st.text_input("Run ID", value=suggested_run_id, key='run_id').upper().strip()

    st.markdown("**Metadata Passport**")
    meta_operator = st.text_input("Operator", placeholder="J. Doe", key='meta_op')
    meta_sn = st.text_input("Serial Number", placeholder="SN-001", key='meta_sn')
    meta_config = st.text_input("Config Used", placeholder="Config name", key='meta_cfg')
    meta_propellants = st.text_input("Propellants/Fluid", placeholder="LOX/Ethanol or N2", key='meta_prop')
    meta_notes = st.text_area("Notes", placeholder="Nominal start, throttle 50%...", key='meta_notes')

with col_file:
    uploaded_file = st.file_uploader("Upload Raw Test File", type=['csv', 'parquet', 'txt', 'tdms'], key='file_up')
    convert_parquet = st.checkbox("Convert CSV to Parquet", value=True, key='convert_pq')

    if system_name and type_name and campaign_id and run_id:
        type_folder = f"{system_name}-{type_name}"
        test_folder = f"{type_folder}-{campaign_id}-{run_id}"

        test_id = f"{system_name}-{type_name}-{campaign_id}-{run_id}"

        st.success(f"**Test ID:** `{test_id}`")
        st.info(f"📍 **Target Path:**\n`.../{test_folder}`")

# --- SUBMIT ---
st.markdown("---")

col_submit, col_link = st.columns([1, 2])

with col_submit:
    submit_button = st.button("🚀 INGEST DATA", type="primary", use_container_width=True)

with col_link:
    link_to_campaign = st.checkbox("Link to Campaign Database", value=True, key='link_db')
    if link_to_campaign:
        from data_lib.campaign_manager import get_available_campaigns

        campaigns = get_available_campaigns()
        if campaigns:
            target_campaign = st.selectbox("Target Campaign DB", campaigns, key='target_camp')
        else:
            st.warning("No campaigns found. Create one in CF/HF Campaign pages first.")
            target_campaign = None

if submit_button:
    if not uploaded_file:
        st.error("❌ Please upload a file first.")
    elif not (system_name and type_name and campaign_id and run_id):
        st.error("❌ Please fill in all hierarchy fields.")
    else:
        with st.spinner("Processing..."):
            # Create folder structure
            final_sys_folder = os.path.join(root_path, system_name)
            final_type_folder = os.path.join(final_sys_folder, f"{system_name}-{type_name}")
            final_test_folder = os.path.join(final_type_folder, f"{system_name}-{type_name}-{campaign_id}-{run_id}")

            subfolders = ["raw_data", "config", "logs", "processed_data", "plots", "media", "reports"]
            for sf in subfolders:
                os.makedirs(os.path.join(final_test_folder, sf), exist_ok=True)

            # Build metadata
            metadata = {
                "test_id": test_id,
                "system": system_name,
                "type": type_name,
                "campaign": campaign_id,
                "run": run_id,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "operator": meta_operator,
                "serial_num": meta_sn,
                "config_used": meta_config,
                "propellants_fluid": meta_propellants,
                "notes": meta_notes,
                "original_filename": uploaded_file.name,
                "test_folder_path": final_test_folder
            }
            save_metadata(final_test_folder, metadata)

            # Process file
            new_base_name = f"{test_id}_raw"
            success, msg, file_path = process_file(uploaded_file, final_test_folder, new_base_name, convert_parquet)

            if success:
                st.success(f"✅ Ingest Complete! {msg}")
                st.balloons()

                # Show folder structure
                with st.expander("📁 Created Structure"):
                    st.code(f"""
{final_test_folder}/
├── metadata.json
├── raw_data/
│   └── {os.path.basename(file_path)}
├── config/
├── logs/
├── processed_data/
├── plots/
├── media/
└── reports/
                    """)

                # Store in session for quick access
                st.session_state['last_ingested_test'] = {
                    'test_id': test_id,
                    'metadata_path': os.path.join(final_test_folder, 'metadata.json'),
                    'data_file': file_path,
                    'test_folder': final_test_folder
                }

                st.info(f"💾 Test data saved and ready for analysis!")

            else:
                st.error(f"❌ Error: {msg}")

# --- ARCHIVE VIEWER ---
st.markdown("---")
st.subheader("📚 Recent Tests Archive")

if os.path.exists(root_path):
    scan_data = []
    for root, dirs, files in os.walk(root_path):
        if "metadata.json" in files:
            try:
                with open(os.path.join(root, "metadata.json"), 'r') as f:
                    m = json.load(f)
                    scan_data.append({
                        "Test ID": m.get("test_id", "N/A"),
                        "Date": m.get("timestamp_utc", "")[:10],
                        "System": m.get("system", "-"),
                        "Type": m.get("type", "-"),
                        "Campaign": m.get("campaign", "-"),
                        "Run": m.get("run", "-"),
                        "Operator": m.get("operator", "-"),
                        "Notes": m.get("notes", "")[:50] + "..." if len(m.get("notes", "")) > 50 else m.get("notes", "")
                    })
            except:
                pass

    if scan_data:
        # Show only most recent 50
        df_archive = pd.DataFrame(scan_data).sort_values("Date", ascending=False).head(50)
        st.dataframe(df_archive, use_container_width=True, hide_index=True)

        # Download full archive
        csv = df_archive.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Archive",
            csv,
            "test_archive.csv",
            "text/csv"
        )
    else:
        st.info("No tests found in archive.")
else:
    st.warning("Data root directory not found.")