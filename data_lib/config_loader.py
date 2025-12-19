import json
import os
import glob

# Path to your config folder
CONFIG_DIR = "../test_configs"


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
    Checks if the loaded CSV actually contains the columns defined in the config.
    Returns a list of missing columns.
    """
    col_map = config.get('columns', {})
    missing = []

    for key, csv_col_name in col_map.items():
        if csv_col_name not in df.columns:
            missing.append(f"{key} -> {csv_col_name}")

    return missing