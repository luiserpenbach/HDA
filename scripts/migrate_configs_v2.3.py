#!/usr/bin/env python3
"""
Configuration Migration Script (v2.3.0)
=======================================
Migrates old-format configs (v2.0-v2.2) to new v2.3.0 format.

Old format: Single config with geometry + fluid mixed in
New format: Active Configuration (hardware) + Test Metadata (test article)

Usage:
    python scripts/migrate_configs_v2.3.py
    python scripts/migrate_configs_v2.3.py --dry-run
    python scripts/migrate_configs_v2.3.py --config-dir configs/ --output-dir migrated/
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_validation import (
    detect_config_format,
    split_old_config,
    validate_active_configuration,
    validate_test_metadata,
)


@dataclass
class MigrationResult:
    """Result of migrating a single config file."""
    original_file: str
    format_detected: str
    migrated: bool
    active_config_path: str = ""
    metadata_path: str = ""
    error: str = ""


class ConfigMigrator:
    """Handles migration of configs from v2.0-v2.2 to v2.3.0 format."""

    def __init__(
        self,
        config_dir: str = "configs",
        saved_configs_dir: str = "saved_configs",
        metadata_output_dir: str = "example_metadata",
        dry_run: bool = False
    ):
        self.config_dir = Path(config_dir)
        self.saved_configs_dir = Path(saved_configs_dir)
        self.metadata_output_dir = Path(metadata_output_dir)
        self.dry_run = dry_run

        # Create output directories
        if not dry_run:
            self.saved_configs_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_output_dir.mkdir(parents=True, exist_ok=True)

    def discover_configs(self) -> List[Path]:
        """Find all JSON config files."""
        if not self.config_dir.exists():
            print(f"⚠️  Config directory not found: {self.config_dir}")
            return []

        configs = list(self.config_dir.glob("*.json"))
        print(f"📁 Found {len(configs)} config files in {self.config_dir}/")
        return configs

    def migrate_config(self, config_path: Path) -> MigrationResult:
        """
        Migrate a single config file.

        Returns:
            MigrationResult with details of migration
        """
        try:
            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Detect format
            format_type = detect_config_format(config)

            result = MigrationResult(
                original_file=str(config_path),
                format_detected=format_type,
                migrated=False
            )

            # If already new format, skip
            if format_type == 'active_config':
                result.error = "Already in v2.3.0 format (no migration needed)"
                return result

            # If old format, split it
            if format_type == 'old':
                active_config_dict, metadata_dict = split_old_config(config)

                # Validate
                active_config = validate_active_configuration(active_config_dict, auto_migrate=False)
                metadata = validate_test_metadata(metadata_dict, require_complete=False)

                # Generate output filenames
                base_name = config_path.stem
                active_config_filename = f"{base_name}_active_config.json"
                metadata_filename = f"{base_name}_metadata_example.json"

                active_config_path = self.saved_configs_dir / active_config_filename
                metadata_path = self.metadata_output_dir / metadata_filename

                # Save (if not dry run)
                if not self.dry_run:
                    # Save active config
                    with open(active_config_path, 'w') as f:
                        json.dump(active_config.to_dict(), f, indent=2)

                    # Save metadata example
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata.to_dict(), f, indent=2)

                result.migrated = True
                result.active_config_path = str(active_config_path)
                result.metadata_path = str(metadata_path)

                return result

        except Exception as e:
            return MigrationResult(
                original_file=str(config_path),
                format_detected="error",
                migrated=False,
                error=str(e)
            )

    def migrate_all(self) -> Tuple[List[MigrationResult], Dict[str, int]]:
        """
        Migrate all configs in the directory.

        Returns:
            Tuple of (results list, summary dict)
        """
        configs = self.discover_configs()

        if not configs:
            return [], {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

        results = []
        for config_path in configs:
            result = self.migrate_config(config_path)
            results.append(result)

        # Generate summary
        summary = {
            "total": len(results),
            "migrated": sum(1 for r in results if r.migrated),
            "skipped": sum(1 for r in results if not r.migrated and not r.error),
            "errors": sum(1 for r in results if r.error and r.format_detected == "error"),
            "already_new": sum(1 for r in results if "Already in v2.3.0" in r.error),
        }

        return results, summary

    def print_report(self, results: List[MigrationResult], summary: Dict[str, int]):
        """Print migration report."""
        print("\n" + "=" * 80)
        print("MIGRATION REPORT - v2.3.0 Config/Metadata Separation")
        print("=" * 80)

        if self.dry_run:
            print("\n🔍 DRY RUN MODE - No files were actually modified\n")

        # Summary
        print(f"\n📊 Summary:")
        print(f"   Total configs found:     {summary['total']}")
        print(f"   ✅ Migrated:             {summary['migrated']}")
        print(f"   ⏭️  Already v2.3.0 format: {summary['already_new']}")
        print(f"   ⚠️  Errors:               {summary['errors']}")

        # Details
        if results:
            print(f"\n📝 Details:")
            for result in results:
                status_icon = "✅" if result.migrated else ("⚠️" if result.error else "⏭️")
                print(f"\n{status_icon} {Path(result.original_file).name}")
                print(f"   Format detected: {result.format_detected}")

                if result.migrated:
                    print(f"   → Active Config: {result.active_config_path}")
                    print(f"   → Metadata Example: {result.metadata_path}")
                elif result.error:
                    print(f"   Error: {result.error}")

        # Instructions
        print("\n" + "=" * 80)
        print("📖 What to do next:")
        print("=" * 80)

        if summary['migrated'] > 0:
            print(f"""
✅ Migrated {summary['migrated']} config(s) successfully!

Active Configurations (testbench hardware):
   → Saved to: {self.saved_configs_dir}/
   → Use these in the UI via 'Saved Configs' section
   → Contains: sensor mappings, uncertainties, processing settings

Metadata Examples (test article properties):
   → Saved to: {self.metadata_output_dir}/
   → These are EXAMPLES showing geometry/fluid from original config
   → For each test, create metadata.json in your test data folder
   → Or enter metadata via UI (optional for analysis, required for campaign)

Next Steps:
   1. Review migrated active configs in {self.saved_configs_dir}/
   2. Copy metadata examples to your test folders (rename to metadata.json)
   3. Customize metadata for each specific test (part numbers, serial numbers, etc.)
   4. In UI, select Active Configuration → provide test metadata → analyze
""")

        if summary['already_new'] > 0:
            print(f"\n✅ {summary['already_new']} config(s) already in v2.3.0 format - no action needed")

        if summary['errors'] > 0:
            print(f"\n⚠️  {summary['errors']} config(s) had errors - review details above")

        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate configs from v2.0-v2.2 to v2.3.0 format"
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing configs to migrate (default: configs/)"
    )
    parser.add_argument(
        "--saved-configs-dir",
        default="saved_configs",
        help="Output directory for active configs (default: saved_configs/)"
    )
    parser.add_argument(
        "--metadata-dir",
        default="example_metadata",
        help="Output directory for metadata examples (default: example_metadata/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually migrating"
    )

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                   Config Migration to v2.3.0                               ║
║                   Active Configuration + Test Metadata Separation          ║
╚═══════════════════════════════════════════════════════════════════════════╝

Input:  {args.config_dir}/
Output: {args.saved_configs_dir}/ (Active Configurations)
        {args.metadata_dir}/ (Metadata Examples)
Mode:   {'🔍 DRY RUN (no files modified)' if args.dry_run else '✍️  LIVE (files will be created)'}
""")

    migrator = ConfigMigrator(
        config_dir=args.config_dir,
        saved_configs_dir=args.saved_configs_dir,
        metadata_output_dir=args.metadata_dir,
        dry_run=args.dry_run
    )

    results, summary = migrator.migrate_all()

    migrator.print_report(results, summary)

    # Exit code
    if summary['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
