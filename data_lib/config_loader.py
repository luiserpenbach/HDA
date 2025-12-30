import json
import os
import glob

# Path to your config folder
CONFIG_DIR = "configs"


def get_available_configs():
    """
    Scans the config directory and returns a list of available config names.
    """
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        return []

    # Find all .json files
    files = glob.glob(os.path.join(CONFIG_DIR, "*.json"))
    # Return filenames without extension (e.g., 'Engine_V1_HotFire')
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def load_config(config_name):
    """
    Loads a specific configuration file by name.
    """
    path = os.path.join(CONFIG_DIR, f"{config_name}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def validate_columns(df, config):
    """
    Checks if the dataframe contains the required columns.
    Supports both direct mapping and channel_config mapping.
    """
    col_map = config.get('columns', {})
    channel_map = config.get('channel_config', {})
    # If we have a channel map, the DF columns should match the VALUES of channel_map
    # (assuming we renamed them already).
    # If we haven't renamed them yet, we should check against keys.
    # BUT: The standard workflow is to rename immediately.


    missing = []

    for key, sensor_id in col_map.items():
        # 1. Check if the Sensor ID is in the DataFrame
        if sensor_id not in df.columns:
            # Debug hint: Check if it's in the channel map but missing from CSV
            raw_id = None
            if channel_map:
                # Reverse lookup for error message
                raw_id = next((k for k, v in channel_map.items() if v == sensor_id), "Unknown")

            error_msg = f"{key} -> {sensor_id}"
            if raw_id:
                error_msg += f" (Raw Channel: {raw_id})"
            missing.append(error_msg)

    return missing