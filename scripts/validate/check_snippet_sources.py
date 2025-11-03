#!/usr/bin/env python3
"""
Validate that text snippets actually appear in their claimed sources.

Fetches full text from papers and searches for verbatim quotes.

Checks:
- value_snippet appears in full text of referenced source
- units_snippet appears in full text of referenced source

Uses fuzzy matching to handle PDF artifacts (hyphenation, spacing).

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_snippet_sources.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/snippet_sources.json
"""
import argparse
import sys
import os
from pathlib import Path
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    ValidationReport
)
from fulltext_fetcher import FullTextFetcher


class SnippetSourceValidator:
    """
    Validate that text snippets actually appear in their claimed sources.
    Works for both parameters and test statistics.
    """

    def __init__(self, data_dir: str, email: str, cache_dir: str = '.cache/fulltext',
                 proxy_url: str = None, proxy_cookies: dict = None, debug: bool = False):
        self.data_dir = data_dir
        self.debug = debug
        self.fetcher = FullTextFetcher(
            email=email,
            cache_dir=cache_dir,
            proxy_url=proxy_url,
            proxy_cookies=proxy_cookies
        )

    def collect_sources(self, data: dict) -> dict:
        """
        Collect all defined sources from YAML.

        Returns:
            Dict mapping source_tag to source definition
        """
        sources = {}

        # Collect from primary_data_sources
        if 'primary_data_sources' in data:
            pds = data['primary_data_sources']
            if isinstance(pds, list):
                for source in pds:
                    if isinstance(source, dict) and 'source_tag' in source:
                        sources[source['source_tag']] = source
            elif isinstance(pds, dict):
                for tag, source in pds.items():
                    sources[tag] = source if isinstance(source, dict) else {'source_tag': tag}

        # Collect from secondary_data_sources
        if 'secondary_data_sources' in data:
            sds = data['secondary_data_sources']
            if isinstance(sds, list):
                for source in sds:
                    if isinstance(source, dict) and 'source_tag' in source:
                        sources[source['source_tag']] = source
            elif isinstance(sds, dict):
                for tag, source in sds.items():
                    sources[tag] = source if isinstance(source, dict) else {'source_tag': tag}

        # Collect from methodological_sources
        if 'methodological_sources' in data:
            ms = data['methodological_sources']
            if isinstance(ms, list):
                for source in ms:
                    if isinstance(source, dict) and 'source_tag' in source:
                        sources[source['source_tag']] = source
            elif isinstance(ms, dict):
                for tag, source in ms.items():
                    sources[tag] = source if isinstance(source, dict) else {'source_tag': tag}

        return sources

    def extract_inputs_from_yaml(self, data: dict) -> list:
        """
        Extract inputs from either parameter_estimates or test_statistic_estimates.

        Returns:
            List of input dicts
        """
        # Try parameter_estimates first (params)
        if 'parameter_estimates' in data and isinstance(data['parameter_estimates'], dict):
            if 'inputs' in data['parameter_estimates']:
                return data['parameter_estimates']['inputs']

        # Try test_statistic_estimates (test stats)
        if 'test_statistic_estimates' in data and isinstance(data['test_statistic_estimates'], dict):
            if 'inputs' in data['test_statistic_estimates']:
                return data['test_statistic_estimates']['inputs']

        return []

    def get_doi_from_source(self, source: dict) -> str:
        """
        Extract DOI from source definition.

        Args:
            source: Source dict

        Returns:
            DOI string or None
        """
        # Primary sources use 'doi'
        if 'doi' in source:
            return source['doi']

        # Secondary/methodological use 'doi_or_url'
        if 'doi_or_url' in source:
            value = source['doi_or_url']
            # Check if it's a DOI (not a URL)
            if value and (value.startswith('10.') or 'doi.org' in value.lower()):
                return value

        return None

    def fuzzy_search(self, snippet: str, fulltext: str, threshold: float = 0.90) -> tuple:
        """
        Search for snippet in fulltext with fuzzy matching.

        Handles PDF artifacts like hyphenation and spacing variations.

        Args:
            snippet: Text snippet to search for
            fulltext: Full text to search in
            threshold: Similarity threshold (0-1)

        Returns:
            (found, best_match, similarity) tuple
        """
        if not snippet or not fulltext:
            return (False, None, 0.0)

        # Normalize both texts
        snippet_norm = self.fetcher.normalize_text(snippet).lower()
        fulltext_norm = self.fetcher.normalize_text(fulltext).lower()

        # Try exact match first
        if snippet_norm in fulltext_norm:
            return (True, snippet_norm, 1.0)

        # Try with additional spacing variations
        snippet_spaced = re.sub(r'\s+', ' ', snippet_norm)
        if snippet_spaced in fulltext_norm:
            return (True, snippet_spaced, 1.0)

        # Try removing all spaces (for compound words)
        snippet_nospace = snippet_norm.replace(' ', '')
        fulltext_nospace = fulltext_norm.replace(' ', '')
        if snippet_nospace in fulltext_nospace:
            return (True, snippet_nospace, 0.95)

        # Try sliding window fuzzy match for longer snippets
        if len(snippet_norm) > 30:
            best_match = None
            best_ratio = 0.0

            # Split full text into windows roughly the size of snippet
            window_size = len(snippet_norm)
            step = window_size // 4  # Overlap windows

            for i in range(0, len(fulltext_norm) - window_size + 1, step):
                window = fulltext_norm[i:i + window_size]
                ratio = SequenceMatcher(None, snippet_norm, window).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = window

            if best_ratio >= threshold:
                return (True, best_match, best_ratio)

        return (False, None, 0.0)

    def validate_input_snippet(self, input_dict: dict, sources: dict) -> dict:
        """
        Validate snippets for a single input.

        Args:
            input_dict: Input dictionary
            sources: Dict mapping source_tag to source definition

        Returns:
            Validation result dict
        """
        input_name = input_dict.get('name', 'unnamed')
        source_ref = input_dict.get('source_ref')

        result = {
            'input_name': input_name,
            'source_ref': source_ref,
            'value_snippet_status': 'not_checked',
            'units_snippet_status': 'not_checked',
            'errors': []
        }

        # Skip if no source_ref (e.g., assumptions, random seeds)
        if not source_ref:
            result['value_snippet_status'] = 'no_source'
            result['units_snippet_status'] = 'no_source'
            return result

        # Check if source exists
        if source_ref not in sources:
            result['errors'].append(f"Source '{source_ref}' not defined")
            result['value_snippet_status'] = 'source_missing'
            result['units_snippet_status'] = 'source_missing'
            return result

        # Get DOI
        source = sources[source_ref]
        doi = self.get_doi_from_source(source)

        if not doi:
            result['value_snippet_status'] = 'no_doi'
            result['units_snippet_status'] = 'no_doi'
            return result

        # Fetch full text
        if self.debug:
            print(f"\n[DEBUG] Fetching full text for DOI: {doi} (source: {source_ref})")

        fetch_result = self.fetcher.get_full_text(doi)
        fulltext = fetch_result['text']

        # Store fetch details for reporting
        result['fetch_attempts'] = fetch_result['attempts']
        result['fetch_source'] = fetch_result['source']

        # Debug: Print fetch attempts and URLs
        if self.debug:
            print(f"[DEBUG] Fetching {doi}:")
            for attempt in fetch_result['attempts']:
                status_symbol = "✓" if attempt['status'] == 'success' else "✗"
                print(f"  {status_symbol} {attempt['source']}: {attempt['status']}")

                # Show URLs being tried
                if 'pdf_url' in attempt:
                    print(f"      PDF: {attempt['pdf_url']}")
                if 'html_url' in attempt:
                    print(f"      HTML: {attempt['html_url']}")
                if 'landing_url' in attempt:
                    print(f"      Landing: {attempt['landing_url']}")

                # Show brief message if not success
                if attempt['status'] != 'success':
                    print(f"      → {attempt['message']}")

            # Show final result
            if fulltext:
                print(f"  ✓ Success via {fetch_result['source']} ({len(fulltext)} chars)")
            else:
                print(f"  ✗ Failed to fetch full text")
            print()

        if not fulltext:
            result['value_snippet_status'] = 'fulltext_unavailable'
            result['units_snippet_status'] = 'fulltext_unavailable'
            # Format attempts into readable message
            attempts_msg = []
            for attempt in fetch_result['attempts']:
                attempts_msg.append(
                    f"{attempt['source']}: {attempt['status']} - {attempt['message']}"
                )
            result['fetch_details'] = '; '.join(attempts_msg)
            return result

        # Check value_snippet
        value_snippet = input_dict.get('value_snippet')
        if value_snippet:
            found, match, similarity = self.fuzzy_search(value_snippet, fulltext)
            if found:
                result['value_snippet_status'] = 'verified'
                result['value_similarity'] = similarity
            else:
                result['value_snippet_status'] = 'not_found'
                result['errors'].append(
                    f"value_snippet not found in full text: '{value_snippet[:80]}...'"
                )
                # Debug: show normalized snippet for troubleshooting
                if self.debug:
                    snippet_norm = self.fetcher.normalize_text(value_snippet).lower()
                    print(f"[DEBUG] Failed to find value_snippet:")
                    print(f"  Original: {value_snippet}")
                    print(f"  Normalized: {snippet_norm}")
                    print(f"  Snippet length: {len(snippet_norm)}")
                    # Show first 500 chars of fulltext for context
                    print(f"  First 500 chars of fulltext: {fulltext[:500]}")
                    print()

        # Check units_snippet
        units_snippet = input_dict.get('units_snippet')
        if units_snippet:
            found, match, similarity = self.fuzzy_search(units_snippet, fulltext)
            if found:
                result['units_snippet_status'] = 'verified'
                result['units_similarity'] = similarity
            else:
                result['units_snippet_status'] = 'not_found'
                result['errors'].append(
                    f"units_snippet not found in full text: '{units_snippet[:80]}...'"
                )

        return result

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate snippet sources in a single YAML file.

        Returns:
            (is_valid, errors, warnings, results) tuple
        """
        data = file_info['data']
        filename = file_info['filename']

        errors = []
        warnings = []
        results = []

        # Collect sources
        sources = self.collect_sources(data)

        # Extract inputs
        inputs = self.extract_inputs_from_yaml(data)

        if not inputs:
            return (True, [], [], [])  # No inputs to validate

        # Validate each input
        for inp in inputs:
            if not isinstance(inp, dict):
                continue

            result = self.validate_input_snippet(inp, sources)
            results.append(result)

            # Collect errors
            if result['errors']:
                errors.extend(result['errors'])

            # Collect warnings for unavailable full text
            if result['value_snippet_status'] == 'fulltext_unavailable':
                fetch_details = result.get('fetch_details', 'No details available')
                warnings.append(
                    f"Input '{result['input_name']}' (source: {result['source_ref']}): "
                    f"Full text unavailable [{fetch_details}]"
                )

        is_valid = len(errors) == 0

        return (is_valid, errors, warnings, results)

    def validate_directory(self) -> ValidationReport:
        """Validate snippet sources in all YAML files."""
        report = ValidationReport("Snippet Source Verification")

        print(f"Validating snippet sources in {self.data_dir}...")
        print(f"Cache directory: {self.fetcher.cache.directory}")

        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']

            is_valid, errors, warnings, results = self.validate_file(file_info)

            # Count snippets by status
            verified_count = sum(1 for r in results
                                if r['value_snippet_status'] == 'verified'
                                or r['units_snippet_status'] == 'verified')
            unavailable_count = sum(1 for r in results
                                   if r['value_snippet_status'] == 'fulltext_unavailable')
            not_found_count = sum(1 for r in results
                                 if r['value_snippet_status'] == 'not_found'
                                 or r['units_snippet_status'] == 'not_found')

            if is_valid:
                if verified_count > 0:
                    details = f"Verified {verified_count} snippet(s)"
                    if unavailable_count > 0:
                        details += f", {unavailable_count} unavailable"
                    report.add_pass(filename, details)
                elif unavailable_count > 0:
                    report.add_pass(filename, f"Full text unavailable for all {unavailable_count} input(s)")
                else:
                    report.add_pass(filename, "No snippets to verify")
            else:
                error_msg = "\n".join([f"    {e}" for e in errors])
                report.add_fail(filename, f"\n{error_msg}")

            # Add warnings
            for warning in warnings:
                report.add_warning(filename, warning)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate that text snippets appear in their claimed sources"
    )
    parser.add_argument("data_dir", help="Directory with YAML files to validate")
    parser.add_argument("output", help="Output JSON file for validation report")
    parser.add_argument("--email", help="Email for Unpaywall API (or set VALIDATION_EMAIL env var)")
    parser.add_argument("--cache-dir", default=".cache/fulltext",
                       help="Cache directory (default: .cache/fulltext)")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear full-text cache before running")
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information including raw API responses")

    args = parser.parse_args()

    # Get email from args or environment
    email = args.email or os.getenv('VALIDATION_EMAIL')
    if not email:
        print("Error: Email required for full-text fetching")
        print("Set VALIDATION_EMAIL environment variable or use --email flag")
        sys.exit(1)

    # Optional proxy configuration from environment
    proxy_url = os.getenv('HOPKINS_PROXY_URL')
    proxy_cookies_str = os.getenv('HOPKINS_PROXY_COOKIES')
    proxy_cookies = None
    if proxy_cookies_str:
        try:
            import json
            proxy_cookies = json.loads(proxy_cookies_str)
        except:
            print("Warning: Could not parse HOPKINS_PROXY_COOKIES")

    # Run validation
    validator = SnippetSourceValidator(
        args.data_dir,
        email,
        cache_dir=args.cache_dir,
        proxy_url=proxy_url,
        proxy_cookies=proxy_cookies,
        debug=args.debug
    )

    # Clear cache if requested
    if args.clear_cache:
        print("Clearing full-text cache...")
        validator.fetcher.clear_cache()
        print(f"Cache cleared: {args.cache_dir}\n")

    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nSnippet source validation report saved to {args.output}")

    # Print cache stats
    cache_stats = validator.fetcher.get_cache_stats()
    print(f"\nCache stats: {cache_stats['size']} entries in {cache_stats['directory']}")

    # Exit with error code if any validations failed
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
