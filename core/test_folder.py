"""
Test Folder Management Module
=============================
Handles the standardized test folder structure for organizing test data,
metadata, configurations, and results.

Folder Structure:
    SYSTEM/
        SYSTEM-TEST_TYPE/
            SYSTEM-TEST_TYPE-CAMPAIGN_ID/
                TEST_ID/
                    config/         - Test configuration files
                    logs/           - DAQ logs, event logs
                    media/          - Photos, videos
                    plots/          - Generated plots
                    processed_data/ - Resampled, filtered data
                    raw_data/       - Original sensor data
                    reports/        - Analysis reports
                    metadata.json   - Test metadata

Example:
    RCS/
        RCS-CF/
            RCS-CF-C01/
                RCS-CF-C01-RUN01/
                    config/
                    raw_data/
                    metadata.json
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import shutil


@dataclass
class TestMetadata:
    """
    Metadata for a single test run.

    This is stored as metadata.json in the test folder.
    """
    # Identifiers - all have defaults for flexibility when loading partial metadata
    test_id: str = ""                     # e.g., "RCS-CF-C01-RUN01"
    system: str = ""                      # e.g., "RCS"
    test_type: str = ""                   # e.g., "CF" (cold flow), "HF" (hot fire)
    campaign_id: str = ""                 # e.g., "C01"
    run_id: str = ""                      # e.g., "RUN01"

    # Test article info
    part_name: str = ""                   # e.g., "Injector Assembly"
    part_number: str = ""                 # e.g., "INJ-001-A"
    serial_number: str = ""               # e.g., "SN-0042"

    # Test info
    test_date: str = ""                   # ISO format date
    test_time: str = ""                   # ISO format time
    operator: str = ""
    facility: str = ""
    test_stand: str = ""

    # Configuration reference
    config_file: str = ""                 # Relative path to config file
    config_name: str = ""

    # Data files
    raw_data_files: List[str] = field(default_factory=list)

    # Test conditions (nominal)
    nominal_conditions: Dict[str, Any] = field(default_factory=dict)

    # Notes and observations
    notes: str = ""
    anomalies: str = ""

    # Status
    status: str = "pending"               # pending, analyzed, approved, rejected

    # Analysis info (filled after analysis)
    analysis_date: str = ""
    analysis_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMetadata':
        """Create from dictionary, handling missing fields gracefully."""
        if data is None:
            return cls()

        # Get valid field names
        field_names = {f.name for f in cls.__dataclass_fields__.values()}

        # Filter to only valid fields
        filtered = {k: v for k, v in data.items() if k in field_names}

        return cls(**filtered)

    @classmethod
    def from_test_id(cls, test_id: str) -> 'TestMetadata':
        """
        Create metadata from a test ID string.

        Expected format: SYSTEM-TEST_TYPE-CAMPAIGN_ID-RUN_ID
        Example: RCS-CF-C01-RUN01
        """
        parts = test_id.split('-')
        if len(parts) >= 4:
            return cls(
                test_id=test_id,
                system=parts[0],
                test_type=parts[1],
                campaign_id=parts[2],
                run_id='-'.join(parts[3:]),  # Handle run IDs with dashes
            )
        else:
            # Fallback for non-standard IDs
            return cls(
                test_id=test_id,
                system=parts[0] if parts else "UNKNOWN",
                test_type=parts[1] if len(parts) > 1 else "XX",
                campaign_id=parts[2] if len(parts) > 2 else "C00",
                run_id=parts[3] if len(parts) > 3 else "RUN00",
            )

    def save(self, folder_path: Union[str, Path]) -> Path:
        """Save metadata to JSON file in folder."""
        folder_path = Path(folder_path)
        metadata_file = folder_path / "metadata.json"

        with open(metadata_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        return metadata_file

    @classmethod
    def load(cls, folder_path: Union[str, Path]) -> 'TestMetadata':
        """Load metadata from folder's metadata.json."""
        folder_path = Path(folder_path)
        metadata_file = folder_path / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata.json found in {folder_path}")

        with open(metadata_file, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)


# Standard subfolders for a test
TEST_SUBFOLDERS = [
    'config',
    'logs',
    'media',
    'plots',
    'processed_data',
    'raw_data',
    'reports',
]


def create_test_folder(
    base_path: Union[str, Path],
    test_id: str,
    metadata: Optional[TestMetadata] = None,
    create_subfolders: bool = True,
) -> Path:
    """
    Create a test folder with the standard structure.

    Args:
        base_path: Base directory for all tests (e.g., /data/tests)
        test_id: Test identifier (e.g., RCS-CF-C01-RUN01)
        metadata: Optional metadata to save
        create_subfolders: Whether to create standard subfolders

    Returns:
        Path to the created test folder
    """
    base_path = Path(base_path)

    # Parse test ID to get folder hierarchy
    if metadata is None:
        metadata = TestMetadata.from_test_id(test_id)

    # Build path: SYSTEM/SYSTEM-TEST_TYPE/SYSTEM-TEST_TYPE-CAMPAIGN_ID/TEST_ID
    system = metadata.system
    test_type_folder = f"{system}-{metadata.test_type}"
    campaign_folder = f"{system}-{metadata.test_type}-{metadata.campaign_id}"

    test_folder = base_path / system / test_type_folder / campaign_folder / test_id

    # Create folder structure
    test_folder.mkdir(parents=True, exist_ok=True)

    if create_subfolders:
        for subfolder in TEST_SUBFOLDERS:
            (test_folder / subfolder).mkdir(exist_ok=True)

    # Save metadata
    if metadata:
        metadata.save(test_folder)

    return test_folder


def get_test_folder_path(base_path: Union[str, Path], test_id: str) -> Path:
    """
    Get the expected path for a test folder.

    Args:
        base_path: Base directory for all tests
        test_id: Test identifier

    Returns:
        Expected path (may not exist)
    """
    metadata = TestMetadata.from_test_id(test_id)

    system = metadata.system
    test_type_folder = f"{system}-{metadata.test_type}"
    campaign_folder = f"{system}-{metadata.test_type}-{metadata.campaign_id}"

    return Path(base_path) / system / test_type_folder / campaign_folder / test_id


def find_test_folder(
    base_path: Union[str, Path],
    test_id: str,
) -> Optional[Path]:
    """
    Find a test folder by ID, searching the directory structure.

    Args:
        base_path: Base directory to search
        test_id: Test identifier to find

    Returns:
        Path to test folder if found, None otherwise
    """
    base_path = Path(base_path)

    # First try the expected path
    expected = get_test_folder_path(base_path, test_id)
    if expected.exists():
        return expected

    # Fall back to recursive search
    for folder in base_path.rglob(test_id):
        if folder.is_dir():
            return folder

    return None


def load_test_metadata(folder_path: Union[str, Path]) -> Optional[TestMetadata]:
    """
    Load metadata from a test folder.

    Args:
        folder_path: Path to test folder

    Returns:
        TestMetadata if found, None otherwise
    """
    try:
        return TestMetadata.load(folder_path)
    except FileNotFoundError:
        return None


def find_raw_data_file(folder_path: Union[str, Path]) -> Optional[Path]:
    """
    Find the raw data CSV file in a test folder.

    Looks in raw_data/ subfolder first, then root.

    Args:
        folder_path: Path to test folder

    Returns:
        Path to raw data file if found
    """
    folder_path = Path(folder_path)

    # Check raw_data subfolder first
    raw_data_dir = folder_path / "raw_data"
    if raw_data_dir.exists():
        csv_files = list(raw_data_dir.glob("*.csv"))
        if csv_files:
            # Prefer files with 'raw' in name
            raw_files = [f for f in csv_files if 'raw' in f.name.lower()]
            return raw_files[0] if raw_files else csv_files[0]

    # Check root folder
    csv_files = list(folder_path.glob("*.csv"))
    if csv_files:
        raw_files = [f for f in csv_files if 'raw' in f.name.lower()]
        return raw_files[0] if raw_files else csv_files[0]

    return None


def find_config_file(folder_path: Union[str, Path]) -> Optional[Path]:
    """
    Find the configuration file in a test folder.

    Looks in config/ subfolder first, then root.

    Args:
        folder_path: Path to test folder

    Returns:
        Path to config file if found
    """
    folder_path = Path(folder_path)

    # Check config subfolder first
    config_dir = folder_path / "config"
    if config_dir.exists():
        json_files = list(config_dir.glob("*.json"))
        if json_files:
            # Prefer files with 'config' in name
            config_files = [f for f in json_files if 'config' in f.name.lower()]
            return config_files[0] if config_files else json_files[0]

    # Check root folder
    json_files = list(folder_path.glob("*config*.json"))
    if json_files:
        return json_files[0]

    return None


def load_test_from_folder(folder_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load all test data from a folder.

    Args:
        folder_path: Path to test folder

    Returns:
        Dictionary with metadata, config, and data file paths
    """
    folder_path = Path(folder_path)

    result = {
        'folder_path': str(folder_path),
        'metadata': None,
        'config': None,
        'raw_data_file': None,
        'config_file': None,
    }

    # Load metadata
    metadata = load_test_metadata(folder_path)
    if metadata:
        result['metadata'] = metadata.to_dict()

    # Find config file
    config_file = find_config_file(folder_path)
    if config_file:
        result['config_file'] = str(config_file)
        with open(config_file, 'r') as f:
            result['config'] = json.load(f)

    # Find raw data file
    raw_data_file = find_raw_data_file(folder_path)
    if raw_data_file:
        result['raw_data_file'] = str(raw_data_file)

    return result


def save_analysis_results(
    folder_path: Union[str, Path],
    result: Any,
    processed_df: Optional[Any] = None,
    plots: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Save analysis results to the test folder structure.

    Args:
        folder_path: Path to test folder
        result: Analysis result object
        processed_df: Processed DataFrame to save
        plots: Dictionary of plot figures to save

    Returns:
        Dictionary of saved file paths
    """
    import pandas as pd

    folder_path = Path(folder_path)
    saved = {}

    # Save processed data
    if processed_df is not None:
        processed_dir = folder_path / "processed_data"
        processed_dir.mkdir(exist_ok=True)

        processed_file = processed_dir / "processed_data.csv"
        processed_df.to_csv(processed_file, index=False)
        saved['processed_data'] = processed_file

    # Save result as JSON
    reports_dir = folder_path / "reports"
    reports_dir.mkdir(exist_ok=True)

    if hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    elif hasattr(result, '__dict__'):
        result_dict = {k: v for k, v in result.__dict__.items()
                      if not k.startswith('_')}
    else:
        result_dict = {'result': str(result)}

    result_file = reports_dir / "analysis_result.json"
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    saved['analysis_result'] = result_file

    # Save plots
    if plots:
        plots_dir = folder_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        for name, fig in plots.items():
            if hasattr(fig, 'write_html'):  # Plotly figure
                plot_file = plots_dir / f"{name}.html"
                fig.write_html(str(plot_file))
                saved[f'plot_{name}'] = plot_file
            elif hasattr(fig, 'savefig'):  # Matplotlib figure
                plot_file = plots_dir / f"{name}.png"
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                saved[f'plot_{name}'] = plot_file

    # Update metadata status
    metadata = load_test_metadata(folder_path)
    if metadata:
        metadata.status = "analyzed"
        metadata.analysis_date = datetime.now().isoformat()
        metadata.save(folder_path)

    return saved


def list_campaigns(base_path: Union[str, Path], system: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all campaigns in the base directory.

    Args:
        base_path: Base directory for tests
        system: Optional system filter

    Returns:
        List of campaign info dictionaries
    """
    base_path = Path(base_path)
    campaigns = []

    if not base_path.exists():
        return campaigns

    # Iterate through systems
    for system_dir in base_path.iterdir():
        if not system_dir.is_dir():
            continue
        if system and system_dir.name != system:
            continue

        # Iterate through test types
        for type_dir in system_dir.iterdir():
            if not type_dir.is_dir():
                continue

            # Iterate through campaigns
            for campaign_dir in type_dir.iterdir():
                if not campaign_dir.is_dir():
                    continue

                # Count tests in campaign
                test_count = sum(1 for t in campaign_dir.iterdir()
                               if t.is_dir() and (t / "metadata.json").exists())

                campaigns.append({
                    'path': str(campaign_dir),
                    'name': campaign_dir.name,
                    'system': system_dir.name,
                    'test_type': type_dir.name.split('-')[-1] if '-' in type_dir.name else '',
                    'test_count': test_count,
                })

    return campaigns


def list_tests_in_campaign(campaign_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List all tests in a campaign folder.

    Args:
        campaign_path: Path to campaign folder

    Returns:
        List of test info dictionaries
    """
    campaign_path = Path(campaign_path)
    tests = []

    if not campaign_path.exists():
        return tests

    for test_dir in campaign_path.iterdir():
        if not test_dir.is_dir():
            continue

        metadata = load_test_metadata(test_dir)
        if metadata:
            tests.append({
                'path': str(test_dir),
                'test_id': metadata.test_id,
                'status': metadata.status,
                'test_date': metadata.test_date,
                'part_number': metadata.part_number,
                'serial_number': metadata.serial_number,
                'metadata': metadata.to_dict(),
            })
        else:
            # Folder exists but no metadata
            tests.append({
                'path': str(test_dir),
                'test_id': test_dir.name,
                'status': 'no_metadata',
                'test_date': '',
                'part_number': '',
                'serial_number': '',
                'metadata': None,
            })

    return sorted(tests, key=lambda x: x['test_id'])