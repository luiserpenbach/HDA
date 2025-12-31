"""
Metadata Manager Module
========================
Handles loading and validation of test metadata from folders or UI input.

Version: 2.3.0

Key Principle: Test article properties (geometry, fluid) belong in metadata,
not in configuration. Configuration is for testbench hardware only.

Usage:
    from core.metadata_manager import MetadataManager, load_metadata_from_folder

    # Load from folder (auto-detects metadata.json)
    metadata = load_metadata_from_folder("/path/to/test/data")

    # Or use manager for more control
    manager = MetadataManager()
    metadata = manager.load(folder_path="/path/to/test/data")

    # Check if complete for campaign save
    if metadata.is_complete_for_campaign():
        save_to_campaign(...)
    else:
        missing = metadata.get_missing_for_campaign()
        st.warning(f"Missing for campaign: {missing}")
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from dataclasses import dataclass

from .config_validation import TestMetadata, validate_test_metadata


@dataclass
class MetadataSource:
    """Records where metadata came from for traceability."""
    source_type: str  # 'file', 'ui', 'merged', 'empty'
    file_path: Optional[str] = None
    has_geometry: bool = False
    has_fluid: bool = False
    is_complete: bool = False


class MetadataManager:
    """
    Manages loading and validation of test metadata.

    Supports loading from:
    - metadata.json file in test folder (primary)
    - UI-provided metadata (fallback)
    - Merged file + UI metadata (UI overrides file)
    """

    @staticmethod
    def load_from_folder(
        folder_path: Union[str, Path],
        filename: str = "metadata.json"
    ) -> tuple[Optional[TestMetadata], MetadataSource]:
        """
        Load metadata from test data folder.

        Args:
            folder_path: Path to test data folder
            filename: Metadata filename (default: metadata.json)

        Returns:
            Tuple of (TestMetadata object or None, MetadataSource)

        Examples:
            >>> metadata, source = MetadataManager.load_from_folder("/data/test_001")
            >>> if metadata:
            ...     print(f"Loaded from {source.file_path}")
        """
        folder = Path(folder_path)
        metadata_file = folder / filename

        if not metadata_file.exists():
            return None, MetadataSource(
                source_type='empty',
                file_path=None,
                has_geometry=False,
                has_fluid=False,
                is_complete=False
            )

        try:
            metadata = TestMetadata.from_file(metadata_file)

            source = MetadataSource(
                source_type='file',
                file_path=str(metadata_file),
                has_geometry=metadata.geometry is not None,
                has_fluid=metadata.fluid is not None,
                is_complete=metadata.is_complete_for_campaign()
            )

            return metadata, source

        except Exception as e:
            # File exists but couldn't be loaded - return error info
            raise ValueError(
                f"Failed to load metadata from {metadata_file}: {str(e)}"
            )

    @staticmethod
    def load_from_dict(
        metadata_dict: Dict[str, Any],
        validate: bool = True,
        require_complete: bool = False
    ) -> TestMetadata:
        """
        Load metadata from dictionary (typically from UI input).

        Args:
            metadata_dict: Metadata dictionary
            validate: If True, validate the metadata
            require_complete: If True, require all fields for campaign save

        Returns:
            TestMetadata object

        Raises:
            ValueError: If validation fails
        """
        if validate:
            return validate_test_metadata(
                metadata_dict,
                require_complete=require_complete
            )
        else:
            return TestMetadata.from_dict(metadata_dict)

    @staticmethod
    def merge_file_and_ui(
        file_metadata: Optional[TestMetadata],
        ui_metadata: Dict[str, Any],
        ui_overrides: bool = True
    ) -> TestMetadata:
        """
        Merge metadata from file and UI input.

        Args:
            file_metadata: Metadata loaded from file (or None)
            ui_metadata: Metadata provided via UI
            ui_overrides: If True, UI values override file values

        Returns:
            Merged TestMetadata object

        Example:
            >>> file_meta, _ = MetadataManager.load_from_folder("/data/test_001")
            >>> ui_meta = {"part_number": "INJ-B1-05"}  # User changed part number
            >>> merged = MetadataManager.merge_file_and_ui(file_meta, ui_meta)
        """
        if file_metadata is None:
            # No file metadata, use UI only
            return TestMetadata.from_dict(ui_metadata)

        # Start with file metadata
        merged_dict = file_metadata.to_dict()

        # Merge UI metadata
        if ui_overrides:
            # UI overrides file
            merged_dict.update({k: v for k, v in ui_metadata.items() if v is not None})
        else:
            # File takes precedence, UI only fills gaps
            for key, value in ui_metadata.items():
                if key not in merged_dict or merged_dict[key] is None:
                    merged_dict[key] = value

        return TestMetadata.from_dict(merged_dict)

    @staticmethod
    def create_empty() -> TestMetadata:
        """Create empty metadata object for UI entry."""
        return TestMetadata()

    @staticmethod
    def extract_from_old_config(config: Dict[str, Any]) -> TestMetadata:
        """
        Extract metadata from old-format config (v2.0-v2.2).

        Args:
            config: Old config with geometry/fluid mixed in

        Returns:
            TestMetadata with extracted geometry/fluid
        """
        from .config_validation import split_old_config

        _, metadata_dict = split_old_config(config)
        return TestMetadata.from_dict(metadata_dict)

    @staticmethod
    def check_campaign_requirements(
        metadata: TestMetadata,
        verbose: bool = False
    ) -> tuple[bool, list[str]]:
        """
        Check if metadata meets requirements for campaign save.

        Args:
            metadata: TestMetadata to check
            verbose: If True, return detailed messages

        Returns:
            Tuple of (is_complete, list of missing fields)

        Example:
            >>> is_complete, missing = MetadataManager.check_campaign_requirements(metadata)
            >>> if not is_complete:
            ...     st.error(f"Cannot save to campaign. Missing: {', '.join(missing)}")
        """
        is_complete = metadata.is_complete_for_campaign()
        missing = metadata.get_missing_for_campaign()

        if verbose and not is_complete:
            detailed_missing = []
            for field in missing:
                if field == "part_number or test_id":
                    detailed_missing.append("Test identifier (part_number or test_id)")
                elif field == "geometry":
                    detailed_missing.append("Geometry (orifice_area_mm2, throat_area_mm2, etc.)")
                elif field == "fluid":
                    detailed_missing.append("Fluid properties (name, gamma, density, etc.)")
                else:
                    detailed_missing.append(field)
            return is_complete, detailed_missing

        return is_complete, missing


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_metadata_from_folder(
    folder_path: Union[str, Path],
    filename: str = "metadata.json",
    require_complete: bool = False
) -> Optional[TestMetadata]:
    """
    Convenience function to load metadata from folder.

    Args:
        folder_path: Path to test data folder
        filename: Metadata filename (default: metadata.json)
        require_complete: If True, raise error if metadata incomplete for campaign

    Returns:
        TestMetadata object if file exists, None otherwise

    Raises:
        ValueError: If file exists but invalid, or if require_complete=True and incomplete

    Example:
        >>> metadata = load_metadata_from_folder("/data/test_001")
        >>> if metadata:
        ...     print(f"Part: {metadata.part_number}, Fluid: {metadata.fluid['name']}")
    """
    metadata, source = MetadataManager.load_from_folder(folder_path, filename)

    if metadata is None:
        return None

    if require_complete and not metadata.is_complete_for_campaign():
        missing = metadata.get_missing_for_campaign()
        raise ValueError(
            f"Metadata incomplete for campaign save. Missing: {', '.join(missing)}"
        )

    return metadata


def create_metadata_template(
    test_type: str = "cold_flow",
    include_examples: bool = True
) -> Dict[str, Any]:
    """
    Create a metadata template for users to fill out.

    Args:
        test_type: "cold_flow" or "hot_fire"
        include_examples: If True, include example values as comments

    Returns:
        Metadata template dictionary

    Example:
        >>> template = create_metadata_template("cold_flow")
        >>> with open("metadata.json", "w") as f:
        ...     json.dump(template, f, indent=2)
    """
    if test_type == "cold_flow":
        template = {
            "part_number": "INJ-XX-XX" if include_examples else "",
            "serial_number": "SN-YYYY-NNN" if include_examples else "",
            "test_datetime": "2025-12-31T14:30:00" if include_examples else "",
            "analyst": "Your Name" if include_examples else "",
            "test_type": "cold_flow",
            "geometry": {
                "orifice_area_mm2": 3.14159 if include_examples else None,
                "num_orifices": 12 if include_examples else None,
                "orifice_diameter_mm": 2.0 if include_examples else None,
            },
            "fluid": {
                "name": "nitrogen" if include_examples else "",
                "gamma": 1.4 if include_examples else None,
                "molecular_weight": 28.014 if include_examples else None,
                "density_kg_m3": 1.165 if include_examples else None,
            },
            "test_conditions": {
                "ambient_pressure_psi": 14.7 if include_examples else None,
                "target_pressure_psi": 100.0 if include_examples else None,
                "notes": "" if include_examples else "",
            }
        }
    elif test_type == "hot_fire":
        template = {
            "part_number": "THR-XX-XX" if include_examples else "",
            "serial_number": "SN-YYYY-NNN" if include_examples else "",
            "test_datetime": "2025-12-31T14:30:00" if include_examples else "",
            "analyst": "Your Name" if include_examples else "",
            "test_type": "hot_fire",
            "geometry": {
                "throat_area_mm2": 12.566 if include_examples else None,
                "throat_diameter_mm": 4.0 if include_examples else None,
                "expansion_ratio": 4.0 if include_examples else None,
            },
            "fluid": {
                "name": "LOX/RP-1" if include_examples else "",
                "oxidizer": "LOX" if include_examples else "",
                "fuel": "RP-1" if include_examples else "",
            },
            "test_conditions": {
                "ambient_pressure_psi": 14.7 if include_examples else None,
                "target_chamber_pressure_psi": 300.0 if include_examples else None,
                "target_of_ratio": 2.5 if include_examples else None,
                "notes": "" if include_examples else "",
            }
        }
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    # Remove None values if not including examples
    if not include_examples:
        template = {k: v for k, v in template.items() if v is not None}

    return template


def save_metadata_template(
    folder_path: Union[str, Path],
    test_type: str = "cold_flow",
    filename: str = "metadata_template.json",
    overwrite: bool = False
) -> Path:
    """
    Save a metadata template to a folder for users to fill out.

    Args:
        folder_path: Folder to save template in
        test_type: "cold_flow" or "hot_fire"
        filename: Filename to save (default: metadata_template.json)
        overwrite: If True, overwrite existing file

    Returns:
        Path to saved template file

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    template_file = folder / filename

    if template_file.exists() and not overwrite:
        raise FileExistsError(
            f"Template file already exists: {template_file}. "
            "Set overwrite=True to replace it."
        )

    template = create_metadata_template(test_type, include_examples=True)

    with open(template_file, 'w') as f:
        json.dump(template, f, indent=2)

    return template_file
