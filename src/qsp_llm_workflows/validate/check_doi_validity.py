#!/usr/bin/env python3
"""
Validate DOI and URL resolution.

Checks:
- DOIs resolve via doi.org and metadata matches YAML (title, author, year)
- URLs are accessible (HTTP HEAD request)

Uses CrossRef API with appropriate rate limiting (1 request/second).

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_doi_validity.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/doi_validation.json
"""
import time
import requests
from difflib import SequenceMatcher

from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


class DOIValidator:
    """
    Validate DOI and URL resolution.
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

    def is_url(self, value: str) -> bool:
        """
        Check if a value is a URL vs a DOI.

        Args:
            value: String to check

        Returns:
            True if URL, False if DOI
        """
        if not value:
            return False

        value_lower = value.lower().strip()

        # URLs start with http:// or https:// or www.
        if value_lower.startswith(("http://", "https://", "www.")):
            return True

        # DOIs typically start with "10." or contain "doi.org"
        if value_lower.startswith("10.") or "doi.org" in value_lower:
            return False

        # If it contains common URL patterns, treat as URL
        if "://" in value or value.startswith("ftp://"):
            return True

        # Default: treat as DOI
        return False

    def validate_url(self, url: str) -> tuple:
        """
        Validate that a URL is accessible.

        Args:
            url: URL string

        Returns:
            (is_valid, error_msg) tuple
        """
        if not url:
            return (False, "Empty URL")

        # Enforce rate limiting
        self.rate_limit_wait()

        try:
            # Add http:// if missing
            if not url.startswith(("http://", "https://", "ftp://")):
                url = "https://" + url

            # Add User-Agent header to avoid being blocked as a bot (especially by Wikipedia)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Try HEAD request first (faster)
            response = requests.head(url, timeout=10, allow_redirects=True, headers=headers)

            # If HEAD not allowed, try GET
            if response.status_code == 405:
                response = requests.get(
                    url, timeout=10, allow_redirects=True, stream=True, headers=headers
                )

            if response.status_code == 200:
                return (True, None)
            else:
                return (False, f"HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            return (False, "Timeout")
        except requests.exceptions.ConnectionError:
            return (False, "Connection failed")
        except Exception as e:
            return (False, f"Error: {str(e)[:50]}")

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
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        try:
            # Query CrossRef API
            url = f"https://doi.org/{doi_clean}"
            headers = {"Accept": "application/vnd.citationstyles.csl+json"}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return None

            metadata = response.json()

            # Extract relevant fields
            title = metadata.get("title", "")
            if isinstance(title, list) and len(title) > 0:
                title = title[0]

            # Extract first author last name
            authors = metadata.get("author", [])
            first_author = None
            if authors and len(authors) > 0:
                first_author_obj = authors[0]
                first_author = first_author_obj.get("family", "")

            # Extract year
            date_parts = metadata.get("issued", {}).get("date-parts", [[]])
            year = None
            if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
                year = date_parts[0][0]

            return {"title": title, "first_author": first_author, "year": year, "doi": doi_clean}

        except Exception:
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
        Collect all sources with DOIs or URLs.

        Note: Primary sources use 'doi' field, secondary/methodological use 'doi_or_url'

        Returns:
            List of (source_tag, source_dict) tuples
        """
        sources = []

        # Collect from primary_data_sources (uses 'doi' field)
        if "primary_data_sources" in data:
            pds = data["primary_data_sources"]
            if isinstance(pds, list):
                for source in pds:
                    if isinstance(source, dict) and "source_tag" in source:
                        if "doi" in source:
                            sources.append((source["source_tag"], source))
            elif isinstance(pds, dict):
                for tag, source in pds.items():
                    if isinstance(source, dict):
                        if "doi" in source:
                            sources.append((tag, source))

        # Collect from secondary_data_sources (uses 'doi_or_url' field)
        if "secondary_data_sources" in data:
            sds = data["secondary_data_sources"]
            if isinstance(sds, list):
                for source in sds:
                    if isinstance(source, dict) and "source_tag" in source:
                        # Check doi_or_url (new) or doi (backward compatibility)
                        if "doi_or_url" in source or "doi" in source:
                            sources.append((source["source_tag"], source))
            elif isinstance(sds, dict):
                for tag, source in sds.items():
                    if isinstance(source, dict):
                        if "doi_or_url" in source or "doi" in source:
                            sources.append((tag, source))

        # Collect from methodological_sources (uses 'doi_or_url' field)
        if "methodological_sources" in data:
            ms = data["methodological_sources"]
            if isinstance(ms, list):
                for source in ms:
                    if isinstance(source, dict) and "source_tag" in source:
                        # Check doi_or_url (new) or doi (backward compatibility)
                        if "doi_or_url" in source or "doi" in source:
                            sources.append((source["source_tag"], source))
            elif isinstance(ms, dict):
                for tag, source in ms.items():
                    if isinstance(source, dict):
                        if "doi_or_url" in source or "doi" in source:
                            sources.append((tag, source))

        return sources

    def validate_source_doi_or_url(self, source_tag: str, source_dict: dict) -> tuple:
        """
        Validate a single source's DOI or URL.

        Note: Primary sources use 'doi', secondary/methodological use 'doi_or_url'

        Returns:
            (is_valid, error_msg) tuple
        """
        # Check for doi_or_url field first (secondary/methodological), then doi (primary or legacy)
        value = source_dict.get("doi_or_url") or source_dict.get("doi")

        if not value:
            return (False, f"Source '{source_tag}': missing doi or doi_or_url field")

        # Skip validation if explicitly null
        if value is None or str(value).lower() == "null":
            return (True, None)

        # Determine if this is a URL or DOI
        if self.is_url(value):
            # Validate URL
            is_valid, error_msg = self.validate_url(value)
            if not is_valid:
                return (False, f"Source '{source_tag}': URL '{value}' - {error_msg}")
            return (True, None)
        else:
            # Validate DOI
            doi = value
            metadata = self.resolve_doi(doi)

            if metadata is None:
                return (False, f"Source '{source_tag}': DOI '{doi}' failed to resolve")

            # Compare metadata
            errors = []

            # Check title (fuzzy match)
            yaml_title = source_dict.get("title", "")
            if not self.fuzzy_match(yaml_title, metadata["title"], threshold=0.7):
                errors.append(
                    f"Title mismatch: YAML='{yaml_title[:50]}...', "
                    f"CrossRef='{metadata['title'][:50]}...'"
                )

            # Check first author (exact match on last name)
            yaml_author = source_dict.get("first_author", "")
            if metadata["first_author"] and yaml_author.lower() != metadata["first_author"].lower():
                errors.append(
                    f"Author mismatch: YAML='{yaml_author}', CrossRef='{metadata['first_author']}'"
                )

            # Check year (exact match)
            yaml_year = source_dict.get("year")
            if yaml_year and metadata["year"] and int(yaml_year) != int(metadata["year"]):
                errors.append(f"Year mismatch: YAML={yaml_year}, CrossRef={metadata['year']}")

            if errors:
                error_msg = f"Source '{source_tag}': " + "; ".join(errors)
                return (False, error_msg)

            return (True, None)

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate DOIs and URLs in a single YAML file.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        data = file_info["data"]
        file_info["filename"]

        # Collect all sources with DOIs or URLs
        sources = self.collect_sources(data)

        if not sources:
            # No sources with DOIs/URLs to validate
            return (True, [])

        # Validate each source DOI or URL
        for source_tag, source_dict in sources:
            is_valid, error_msg = self.validate_source_doi_or_url(source_tag, source_dict)
            if not is_valid:
                errors.append(error_msg)

        is_valid = len(errors) == 0
        return (is_valid, errors)

    def validate_directory(self) -> ValidationReport:
        """Validate DOIs and URLs in all YAML files."""
        report = ValidationReport("DOI/URL Validation")

        print(f"Validating DOIs and URLs in {self.data_dir}...")
        print(f"Rate limit: {self.rate_limit}s between requests")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info["filename"]
            data = file_info["data"]

            # Collect all sources with DOIs or URLs
            sources = self.collect_sources(data)

            if not sources:
                # No sources to validate - mark file as passed
                report.add_pass(filename, "No sources with DOI/URL fields")
                continue

            # Validate each source individually and report at source level
            for source_tag, source_dict in sources:
                is_valid, error_msg = self.validate_source_doi_or_url(source_tag, source_dict)
                item_name = f"{filename} → {source_tag}"

                if is_valid:
                    report.add_pass(item_name, "Valid")
                else:
                    report.add_fail(item_name, error_msg)

        return report
