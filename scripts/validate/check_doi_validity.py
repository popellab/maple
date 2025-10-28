#!/usr/bin/env python3
"""
Validate DOI resolution and metadata matching.

Checks:
- DOIs resolve via doi.org
- Returned metadata matches YAML (title, author, year)

Uses CrossRef API with appropriate rate limiting (1 request/second).

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_doi_validity.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/doi_validation.json
"""
import argparse
import sys
import os
from pathlib import Path
import time
import requests
from difflib import SequenceMatcher

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    ValidationReport
)


class DOIValidator:
    """
    Validate DOI resolution and metadata matching.
    Works for both parameters and test statistics.
    """

    def __init__(self, data_dir: str, rate_limit: float = 1.0):
        self.data_dir = data_dir
        self.rate_limit = rate_limit  # seconds between requests
        self.last_request_time = 0

    def rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def resolve_doi(self, doi: str) -> dict:
        """
        Resolve DOI and get metadata from CrossRef.

        Args:
            doi: DOI string (e.g., "10.1056/NEJMoa1200690")

        Returns:
            Dict with metadata or None if resolution fails
        """
        if not doi:
            return None

        # Enforce rate limiting
        self.rate_limit_wait()

        # Clean DOI (remove https://doi.org/ prefix if present)
        doi_clean = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')

        try:
            # Query CrossRef API
            url = f"https://doi.org/{doi_clean}"
            headers = {
                'Accept': 'application/vnd.citationstyles.csl+json'
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return None

            metadata = response.json()

            # Extract relevant fields
            title = metadata.get('title', '')
            if isinstance(title, list) and len(title) > 0:
                title = title[0]

            # Extract first author last name
            authors = metadata.get('author', [])
            first_author = None
            if authors and len(authors) > 0:
                first_author_obj = authors[0]
                first_author = first_author_obj.get('family', '')

            # Extract year
            date_parts = metadata.get('issued', {}).get('date-parts', [[]])
            year = None
            if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
                year = date_parts[0][0]

            return {
                'title': title,
                'first_author': first_author,
                'year': year,
                'doi': doi_clean
            }

        except Exception as e:
            return None

    def fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy string matching using SequenceMatcher.

        Args:
            str1, str2: Strings to compare
            threshold: Similarity threshold (0-1)

        Returns:
            True if similarity >= threshold
        """
        if not str1 or not str2:
            return False

        # Normalize: lowercase, strip whitespace
        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        similarity = SequenceMatcher(None, s1, s2).ratio()
        return similarity >= threshold

    def collect_sources(self, data: dict) -> list:
        """
        Collect all sources with DOIs.

        Returns:
            List of (source_tag, source_dict) tuples
        """
        sources = []

        # Collect from primary_data_sources
        if 'primary_data_sources' in data:
            pds = data['primary_data_sources']
            if isinstance(pds, list):
                for source in pds:
                    if isinstance(source, dict) and 'source_tag' in source and 'doi' in source:
                        sources.append((source['source_tag'], source))
            elif isinstance(pds, dict):
                for tag, source in pds.items():
                    if isinstance(source, dict) and 'doi' in source:
                        sources.append((tag, source))

        # Collect from secondary_data_sources
        if 'secondary_data_sources' in data:
            sds = data['secondary_data_sources']
            if isinstance(sds, list):
                for source in sds:
                    if isinstance(source, dict) and 'source_tag' in source and 'doi' in source:
                        sources.append((source['source_tag'], source))
            elif isinstance(sds, dict):
                for tag, source in sds.items():
                    if isinstance(source, dict) and 'doi' in source:
                        sources.append((tag, source))

        # Collect from methodological_sources
        if 'methodological_sources' in data:
            ms = data['methodological_sources']
            if isinstance(ms, list):
                for source in ms:
                    if isinstance(source, dict) and 'source_tag' in source and 'doi' in source:
                        sources.append((source['source_tag'], source))
            elif isinstance(ms, dict):
                for tag, source in ms.items():
                    if isinstance(source, dict) and 'doi' in source:
                        sources.append((tag, source))

        return sources

    def validate_source_doi(self, source_tag: str, source_dict: dict) -> tuple:
        """
        Validate a single source's DOI.

        Returns:
            (is_valid, error_msg) tuple
        """
        doi = source_dict.get('doi')
        if not doi:
            return (False, f"Source '{source_tag}': missing DOI")

        # Resolve DOI
        metadata = self.resolve_doi(doi)

        if metadata is None:
            return (False, f"Source '{source_tag}': DOI '{doi}' failed to resolve")

        # Compare metadata
        errors = []

        # Check title (fuzzy match)
        yaml_title = source_dict.get('title', '')
        if not self.fuzzy_match(yaml_title, metadata['title'], threshold=0.7):
            errors.append(
                f"Title mismatch: YAML='{yaml_title[:50]}...', "
                f"CrossRef='{metadata['title'][:50]}...'"
            )

        # Check first author (exact match on last name)
        yaml_author = source_dict.get('first_author', '')
        if metadata['first_author'] and yaml_author.lower() != metadata['first_author'].lower():
            errors.append(
                f"Author mismatch: YAML='{yaml_author}', CrossRef='{metadata['first_author']}'"
            )

        # Check year (exact match)
        yaml_year = source_dict.get('year')
        if yaml_year and metadata['year'] and int(yaml_year) != int(metadata['year']):
            errors.append(
                f"Year mismatch: YAML={yaml_year}, CrossRef={metadata['year']}"
            )

        if errors:
            error_msg = f"Source '{source_tag}': " + "; ".join(errors)
            return (False, error_msg)

        return (True, None)

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate DOIs in a single YAML file.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        data = file_info['data']
        filename = file_info['filename']

        # Collect all sources with DOIs
        sources = self.collect_sources(data)

        if not sources:
            # No sources with DOIs to validate
            return (True, [])

        # Validate each source DOI
        for source_tag, source_dict in sources:
            is_valid, error_msg = self.validate_source_doi(source_tag, source_dict)
            if not is_valid:
                errors.append(error_msg)

        is_valid = len(errors) == 0
        return (is_valid, errors)

    def validate_directory(self) -> ValidationReport:
        """Validate DOIs in all YAML files."""
        report = ValidationReport("DOI Validation")

        print(f"Validating DOIs in {self.data_dir}...")
        print(f"Rate limit: {self.rate_limit}s between requests")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']

            is_valid, errors = self.validate_file(file_info)

            if is_valid:
                report.add_pass(filename)
            else:
                error_msg = "; ".join(errors)
                report.add_fail(filename, error_msg)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate DOI resolution and metadata matching"
    )
    parser.add_argument("data_dir", help="Directory with YAML files to validate")
    parser.add_argument("output", help="Output JSON file for validation report")
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds between DOI resolution requests (default: 1.0)"
    )

    args = parser.parse_args()

    # Run validation
    validator = DOIValidator(args.data_dir, args.rate_limit)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nDOI validation report saved to {args.output}")


if __name__ == "__main__":
    main()
