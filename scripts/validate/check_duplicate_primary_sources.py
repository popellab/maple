#!/usr/bin/env python3
"""
Check for duplicate primary data sources across extractions.

Validates that primary_data_sources (by DOI) in to-review files are not
already being used in accepted extractions.

Primary data sources should be unique per extraction - if the same
experimental data is being used, it should be added to the existing
extraction rather than creating a new one.

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_duplicate_primary_sources.py \\
        ../qsp-metadata-storage/to-review/test_statistics \\
        output/duplicate_primary_sources.json

    python scripts/validate/check_duplicate_primary_sources.py \\
        ../qsp-metadata-storage/to-review/parameter_estimates \\
        output/duplicate_primary_sources.json
"""
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    ValidationReport
)


class DuplicatePrimarySourceChecker:
    """
    Check for duplicate primary data sources across test statistics.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.main_storage_dir = self._get_main_storage_dir()
        self.doi_to_files = defaultdict(list)  # normalized_doi -> list of (filename, source_tag)

    def _get_main_storage_dir(self) -> Path:
        """
        Determine main storage directory if running on to-review directory.

        If data_dir is qsp-metadata-storage/to-review/test_statistics/,
        returns qsp-metadata-storage/test_statistics/
        """
        # Check if we're in a to-review directory
        if 'to-review' not in str(self.data_dir):
            print("Warning: Not running on to-review directory. This validator is designed for to-review files.")
            return None

        # Get main storage directory by removing to-review from path
        main_dir = Path(str(self.data_dir).replace('/to-review/', '/').replace('/to-review', ''))

        # Verify it's different and exists
        if main_dir != self.data_dir and main_dir.exists():
            return main_dir

        print(f"Warning: Could not find main storage directory at {main_dir}")
        return None

    def normalize_doi(self, doi: str) -> str:
        """
        Normalize DOI for comparison.

        Removes https://doi.org/ prefix and converts to lowercase.

        Args:
            doi: DOI string

        Returns:
            Normalized DOI
        """
        if not doi:
            return ""

        doi_normalized = doi.lower().strip()
        doi_normalized = doi_normalized.replace('https://doi.org/', '')
        doi_normalized = doi_normalized.replace('http://doi.org/', '')

        return doi_normalized

    def extract_primary_dois(self, data: dict) -> list:
        """
        Extract all primary data source DOIs from a YAML file.

        Returns:
            List of (source_tag, doi) tuples
        """
        dois = []

        if 'primary_data_sources' not in data:
            return dois

        pds = data['primary_data_sources']

        # Handle list format
        if isinstance(pds, list):
            for source in pds:
                if isinstance(source, dict):
                    source_tag = source.get('source_tag', 'unknown')
                    doi = source.get('doi', source.get('doi_or_url'))  # Support both fields
                    if doi:
                        dois.append((source_tag, doi))

        # Handle dict format
        elif isinstance(pds, dict):
            for tag, source in pds.items():
                if isinstance(source, dict):
                    doi = source.get('doi', source.get('doi_or_url'))  # Support both fields
                    if doi:
                        dois.append((tag, doi))

        return dois

    def load_accepted_primary_dois(self):
        """
        Load all primary DOIs from accepted test statistics in main storage.

        Builds mapping of normalized_doi -> list of (filename, source_tag)
        """
        if not self.main_storage_dir:
            print("No main storage directory found - skipping duplicate check")
            return

        print(f"Loading accepted test statistics from {self.main_storage_dir}...")
        accepted_files = load_yaml_directory(str(self.main_storage_dir))

        for file_info in accepted_files:
            filename = file_info['filename']
            data = file_info['data']

            # Extract primary DOIs
            primary_dois = self.extract_primary_dois(data)

            for source_tag, doi in primary_dois:
                doi_normalized = self.normalize_doi(doi)
                if doi_normalized:
                    self.doi_to_files[doi_normalized].append((filename, source_tag))

        print(f"  Found {len(self.doi_to_files)} unique primary DOIs in accepted files")

    def validate_file(self, file_info: dict) -> tuple:
        """
        Check for duplicate primary sources in a single to-review file.

        Returns:
            (is_valid, errors) tuple
        """
        data = file_info['data']
        filename = file_info['filename']
        errors = []

        # Extract primary DOIs from this file
        primary_dois = self.extract_primary_dois(data)

        if not primary_dois:
            # No primary sources to check
            return (True, [])

        # Check each DOI for duplicates
        for source_tag, doi in primary_dois:
            doi_normalized = self.normalize_doi(doi)

            if doi_normalized in self.doi_to_files:
                # Found duplicate(s)
                existing_files = self.doi_to_files[doi_normalized]
                existing_desc = "; ".join([f"{f[0]} (source: {f[1]})" for f in existing_files])

                errors.append(
                    f"Primary source '{source_tag}' (DOI: {doi}) is already used in: {existing_desc}"
                )

        is_valid = len(errors) == 0
        return (is_valid, errors)

    def validate_directory(self) -> ValidationReport:
        """Validate all to-review files for duplicate primary sources."""
        # First load accepted files to build DOI index
        self.load_accepted_primary_dois()

        report = ValidationReport("Duplicate Primary Sources Check")

        if not self.main_storage_dir:
            # Can't perform validation without main storage
            print("Skipping duplicate primary source check (no main storage directory)")
            return report

        print(f"\nValidating to-review files in {self.data_dir}...")
        files = load_yaml_directory(str(self.data_dir))

        for file_info in files:
            filename = file_info['filename']

            is_valid, errors = self.validate_file(file_info)

            if is_valid:
                report.add_pass(filename, "No duplicate primary sources")
            else:
                for error in errors:
                    report.add_fail(filename, error)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Check for duplicate primary data sources across test statistics"
    )
    parser.add_argument("data_dir", help="Directory with to-review YAML files to validate")
    parser.add_argument("output", help="Output JSON file for validation report")

    args = parser.parse_args()

    # Run validation
    validator = DuplicatePrimarySourceChecker(args.data_dir)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nDuplicate primary sources report saved to {args.output}")

    # Exit with error code if any validations failed
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
