#!/usr/bin/env python3
"""
Schema version detection for automated conversion workflow.

Identifies YAML files that need schema conversion by checking their
schema_version field against the latest available templates.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SchemaVersionDetector:
    """Detects schema versions and identifies files needing conversion."""

    # Latest schema versions for each metadata type
    LATEST_VERSIONS = {
        "parameter": "v3",
        "test_statistic": "v2",
        "quick_estimate": "v1"  # Quick estimates don't have versioning yet
    }

    # Template paths for each schema version
    # Note: Only the latest template is kept in the repo
    # Older versions must be retrieved from git history if needed for conversion
    SCHEMA_TEMPLATES = {
        "parameter": {
            "v3": "templates/parameter_metadata_template.yaml"
        },
        "test_statistic": {
            "v2": "templates/test_statistic_template.yaml"
        },
        "quick_estimate": {
            "v1": "templates/quick_estimate_template.yaml"
        }
    }

    def __init__(self, base_dir: Path, storage_dir: Path):
        """
        Initialize schema version detector.

        Args:
            base_dir: Base directory of qsp-llm-workflows
            storage_dir: Path to qsp-metadata-storage repository
        """
        self.base_dir = Path(base_dir)
        self.storage_dir = Path(storage_dir)

    def detect_schema_version(self, yaml_file: Path) -> Optional[str]:
        """
        Detect schema version from YAML file.

        Args:
            yaml_file: Path to YAML file

        Returns:
            Schema version string (e.g., "v3") or None if not found
        """
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Check for schema_version field
            version = data.get('schema_version')
            if version:
                return str(version)

            # If no schema_version field, assume v1 (old files)
            return "v1"

        except Exception as e:
            print(f"Warning: Could not read {yaml_file.name}: {e}")
            return None

    def get_metadata_type_from_directory(self, yaml_file: Path) -> Optional[str]:
        """
        Determine metadata type (parameter/test_statistic/quick_estimate) from file location.

        Args:
            yaml_file: Path to YAML file

        Returns:
            Metadata type string or None
        """
        # Check if file is in storage directory
        try:
            relative_path = yaml_file.relative_to(self.storage_dir)
        except ValueError:
            return None

        # Determine type from directory
        if "parameter_estimates" in relative_path.parts:
            return "parameter"
        elif "test_statistics" in relative_path.parts:
            return "test_statistic"
        elif "quick_estimates" in relative_path.parts:
            return "quick_estimate"

        return None

    def needs_conversion(self, yaml_file: Path, metadata_type: str) -> bool:
        """
        Check if file needs schema conversion.

        Args:
            yaml_file: Path to YAML file
            metadata_type: Type of metadata (parameter/test_statistic/quick_estimate)

        Returns:
            True if file needs conversion, False otherwise
        """
        current_version = self.detect_schema_version(yaml_file)
        if current_version is None:
            return False

        latest_version = self.LATEST_VERSIONS.get(metadata_type)
        if latest_version is None:
            return False

        return current_version != latest_version

    def scan_directory(self, directory: Path, metadata_type: str) -> List[Tuple[Path, str, str]]:
        """
        Scan directory for files needing schema conversion.

        Args:
            directory: Directory to scan
            metadata_type: Type of metadata to scan for

        Returns:
            List of tuples: (file_path, current_version, target_version)
        """
        files_to_convert = []
        yaml_files = list(directory.glob("*.yaml"))

        for yaml_file in yaml_files:
            current_version = self.detect_schema_version(yaml_file)
            if current_version is None:
                continue

            latest_version = self.LATEST_VERSIONS.get(metadata_type)
            if latest_version is None:
                continue

            if current_version != latest_version:
                files_to_convert.append((yaml_file, current_version, latest_version))

        return files_to_convert

    def scan_all_directories(self, metadata_types: Optional[List[str]] = None) -> Dict[str, List[Tuple[Path, str, str]]]:
        """
        Scan all metadata directories for files needing conversion.

        Args:
            metadata_types: List of types to scan (default: all types)

        Returns:
            Dictionary mapping metadata type to list of files needing conversion
        """
        if metadata_types is None:
            metadata_types = ["parameter", "test_statistic", "quick_estimate"]

        results = {}

        for metadata_type in metadata_types:
            # Determine directory
            if metadata_type == "parameter":
                directory = self.storage_dir / "parameter_estimates"
            elif metadata_type == "test_statistic":
                directory = self.storage_dir / "test_statistics"
            elif metadata_type == "quick_estimate":
                directory = self.storage_dir / "quick_estimates"
            else:
                continue

            if not directory.exists():
                print(f"Warning: Directory not found: {directory}")
                continue

            # Scan directory
            files_to_convert = self.scan_directory(directory, metadata_type)
            if files_to_convert:
                results[metadata_type] = files_to_convert

        return results

    def get_template_paths(self, metadata_type: str, from_version: str, to_version: str) -> Tuple[Path, Path]:
        """
        Get paths to schema templates for conversion.

        Args:
            metadata_type: Type of metadata
            from_version: Current schema version
            to_version: Target schema version

        Returns:
            Tuple of (old_template_path, new_template_path)
        """
        import tempfile
        import subprocess

        templates = self.SCHEMA_TEMPLATES.get(metadata_type, {})

        # New template should exist in current repo
        new_template = templates.get(to_version)
        if new_template is None:
            raise ValueError(f"No template found for {metadata_type} {to_version}")

        new_path = self.base_dir / new_template
        if not new_path.exists():
            raise FileNotFoundError(f"New template not found: {new_path}")

        # For old template, check if it exists in current repo first
        old_template = templates.get(from_version)
        if old_template:
            old_path = self.base_dir / old_template
            if old_path.exists():
                return (old_path, new_path)

        # Old template doesn't exist in current repo - need to get from git history
        # Determine the old template filename based on version
        if metadata_type == "parameter":
            if from_version == "v1":
                old_filename = "templates/parameter_metadata_template.yaml"
                git_ref = "b18a37e"  # Commit before we removed old templates
            elif from_version == "v2":
                old_filename = "templates/parameter_metadata_template_v2.yaml"
                git_ref = "b18a37e"
            else:
                raise ValueError(f"Cannot convert from unknown version: {from_version}")
        elif metadata_type == "test_statistic":
            if from_version == "v1":
                old_filename = "templates/test_statistic_template.yaml"
                git_ref = "b18a37e"
            else:
                raise ValueError(f"Cannot convert from unknown version: {from_version}")
        else:
            raise ValueError(f"Cannot convert {metadata_type} from {from_version}")

        # Extract old template from git history
        try:
            result = subprocess.run(
                ["git", "show", f"{git_ref}:{old_filename}"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # Save to temporary file
            temp_dir = self.base_dir / "batch_jobs" / "temp_templates"
            temp_dir.mkdir(parents=True, exist_ok=True)

            old_path = temp_dir / f"{metadata_type}_{from_version}_template.yaml"
            with open(old_path, 'w', encoding='utf-8') as f:
                f.write(result.stdout)

            return (old_path, new_path)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Could not retrieve old template from git: {e.stderr}")

    def print_summary(self, scan_results: Dict[str, List[Tuple[Path, str, str]]]):
        """
        Print summary of files needing conversion.

        Args:
            scan_results: Results from scan_all_directories()
        """
        total_files = sum(len(files) for files in scan_results.values())

        if total_files == 0:
            print("✓ All files are up to date!")
            return

        print(f"\nFound {total_files} file(s) needing schema conversion:\n")

        for metadata_type, files in scan_results.items():
            if not files:
                continue

            print(f"{metadata_type.upper()}:")
            # Group by version transition
            version_groups = {}
            for file_path, from_ver, to_ver in files:
                key = f"{from_ver} → {to_ver}"
                if key not in version_groups:
                    version_groups[key] = []
                version_groups[key].append(file_path)

            for transition, file_paths in version_groups.items():
                print(f"  {transition}: {len(file_paths)} files")
                for file_path in file_paths[:5]:  # Show first 5
                    print(f"    - {file_path.name}")
                if len(file_paths) > 5:
                    print(f"    ... and {len(file_paths) - 5} more")

            print()
