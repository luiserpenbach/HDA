import streamlit as st
import os
import json
import shutil
import pandas as pd
from datetime import datetime
import re
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURATION ---
# Default fallback path
DEFAULT_ROOT_PATH = r"C:\Rocket_Data\Storage"

st.set_page_config(page_title="Hopper Data Ingest", page_icon="", layout="wide")


# --- UTILITY FUNCTIONS ---

def get_folder_dialog():
    """Opens a native system dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    root.wm_attributes('-topmost', 1)  # Make the dialog appear on top
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path


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
        if len(parts) > 0:
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
            return True, f"Converted to Parquet: {save_name}"
        except Exception as e:
            return False, f"Error converting parquet: {e}"
    else:
        save_name = f"{new_filename_base}{original_ext}"
        full_path = os.path.join(raw_path, save_name)
        with open(full_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, f"Saved raw file: {save_name}"


# --- APP LAYOUT ---

st.title("Hopper Test Data Ingest")

# --- ROOT DIRECTORY SELECTION ---
st.sidebar.header("📂 Storage Configuration")

# Initialize session state for the root path if it doesn't exist
if 'root_path' not in st.session_state:
    st.session_state['root_path'] = DEFAULT_ROOT_PATH

col_dir_btn, col_dir_txt = st.sidebar.columns([1, 3])

with col_dir_btn:
    if st.button("📂 Browse"):
        selected_folder = get_folder_dialog()
        if selected_folder:
            st.session_state['root_path'] = selected_folder
            st.rerun()  # Refresh app with new path

with col_dir_txt:
    # Display the path (read-only or editable)
    root_path = st.text_input("Root Path", value=st.session_state['root_path'], label_visibility="collapsed")
    # Update state if user types manually
    if root_path != st.session_state['root_path']:
        st.session_state['root_path'] = root_path

if not os.path.exists(root_path):
    st.error(f"Root path does not exist: {root_path}")
    st.stop()

# --- CASCADING DROPDOWNS ---
col1, col2, col3 = st.columns(3)

# 1. SYSTEM
with col1:
    st.subheader("1. System")
    systems = get_subfolders(root_path)
    systems.insert(0, "➕ Create New...")
    selected_system = st.selectbox("Select System", systems)
    system_name = st.text_input(
        "New System Name").upper().strip() if selected_system == "➕ Create New..." else selected_system

# 2. TYPE
with col2:
    st.subheader("2. Test Type")
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
        selected_type = st.selectbox("Select Test Type", existing_types)
        type_name = st.text_input(
            "New Type Name").upper().strip() if selected_type == "➕ Create New..." else selected_type
    else:
        st.info("Select System first")
        type_name = None

# 3. CAMPAIGN
with col3:
    st.subheader("3. Campaign")
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
        selected_campaign = st.selectbox("Select Campaign", sorted_campaigns)
        campaign_id = st.text_input(
            "New Campaign ID").upper().strip() if selected_campaign == "➕ Create New..." else selected_campaign
    else:
        st.info("Select Type first")
        campaign_id = None

# --- RUN CONFIGURATION ---
st.markdown("---")
st.subheader("4. Run Configuration & Upload")

col_meta, col_file = st.columns([1, 1])

with col_meta:
    suggested_run_id = "RUN01"
    if system_name and type_name and campaign_id:
        type_folder_name = f"{system_name}-{type_name}"
        target_base_path = os.path.join(root_path, system_name, type_folder_name)
        suggested_run_id = get_next_run_id(target_base_path)

    run_id = st.text_input("Run ID", value=suggested_run_id).upper().strip()
    st.markdown("**Metadata Passport**")
    meta_operator = st.text_input("Operator", placeholder="J. Doe")
    meta_sn = st.text_input("Engine Serial Number", placeholder="E-001")
    meta_fuel = st.text_input("Propellants", placeholder="LOX / Ethanol")
    meta_notes = st.text_area("Notes", placeholder="Nominal start...")

with col_file:
    uploaded_file = st.file_uploader("Upload Raw Test File", type=['csv', 'txt', 'tdms', 'log'])
    convert_parquet = st.checkbox("Convert CSV to Parquet", value=True)

    if system_name and type_name and campaign_id and run_id:
        type_folder = f"{system_name}-{type_name}"
        test_folder = f"{type_folder}-{campaign_id}-{run_id}"
        st.info(f"📍 Target Path: ...\\{test_folder}")

# --- SUBMIT ---
if st.button("🚀 INGEST DATA", type="primary"):
    if not uploaded_file:
        st.error("Please upload a file first.")
    elif not (system_name and type_name and campaign_id and run_id):
        st.error("Please fill in all System/Type/Campaign/Run fields.")
    else:
        final_sys_folder = os.path.join(root_path, system_name)
        final_type_folder = os.path.join(final_sys_folder, f"{system_name}-{type_name}")
        final_test_folder = os.path.join(final_type_folder, f"{system_name}-{type_name}-{campaign_id}-{run_id}")

        subfolders = ["raw_data", "config", "logs", "processed_data", "plots", "media", "reports"]
        for sf in subfolders:
            os.makedirs(os.path.join(final_test_folder, sf), exist_ok=True)

        metadata = {
            "test_id": f"{system_name}-{type_name}-{campaign_id}-{run_id}",
            "system": system_name,
            "type": type_name,
            "campaign": campaign_id,
            "run": run_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "operator": meta_operator,
            "engine_sn": meta_sn,
            "propellants": meta_fuel,
            "notes": meta_notes,
            "original_filename": uploaded_file.name
        }
        save_metadata(final_test_folder, metadata)

        new_base_name = f"{metadata['test_id']}_raw"
        success, msg = process_file(uploaded_file, final_test_folder, new_base_name, convert_parquet)

        if success:
            st.success(f"✅ Ingest Complete! {msg}")
            st.balloons()
        else:
            st.error(f"❌ Error processing file: {msg}")

# --- ARCHIVE VIEWER ---
st.markdown("---")
st.subheader("Existing Test Archive")
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
                        "Campaign": m.get("campaign", "-"),
                        "Engine": m.get("engine_sn", "-"),
                        "Notes": m.get("notes", "")
                    })
            except:
                pass
    if scan_data:
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True)
    else:
        st.caption("No tests found.")