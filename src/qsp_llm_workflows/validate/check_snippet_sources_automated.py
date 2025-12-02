#!/usr/bin/env python3
"""
Automated snippet source verification via multiple text sources.

Fetches full-text articles from:
1. NCBI PubMed Central (for papers with PMCID)
2. Publisher websites via Unpaywall (for OA papers not in PMC)
3. Abstracts via Europe PMC (fallback)

Falls back to manual verification only for papers with no available text.

Usage:
    # Called as part of validation suite via run_all_validations.py
"""
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import requests
from bs4 import BeautifulSoup

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


# Reuse collection functions from manual verifier
from qsp_llm_workflows.validate.check_snippet_sources_manual_verify import (
    collect_sources,
    extract_inputs_from_yaml,
    get_doi_from_source,
)

# Reuse value checking logic from text snippet validator
from qsp_llm_workflows.validate.check_text_snippets import TextSnippetValidator


@dataclass
class PaperInfo:
    """Information about a paper from Europe PMC and Unpaywall."""

    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    is_open_access: bool = False
    in_pmc: bool = False
    abstract: Optional[str] = None
    title: Optional[str] = None
    # Unpaywall data
    oa_url: Optional[str] = None  # URL to OA full text (HTML or PDF)
    oa_pdf_url: Optional[str] = None  # Direct PDF URL if available
    oa_status: Optional[str] = None  # gold, green, bronze, hybrid, closed


class AutomatedSnippetVerifier(Validator):
    """
    Automated snippet verification via multiple text sources.

    Text source priority:
    1. NCBI PubMed Central (full text for papers with PMCID)
    2. Publisher websites via Unpaywall (full text for OA papers not in PMC)
    3. Abstracts via Europe PMC (fallback)

    Extracts text with BeautifulSoup and fuzzy-searches for each snippet.
    Falls back to manual verification only when no text is available.
    """

    EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    NCBI_PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles"
    UNPAYWALL_API_URL = "https://api.unpaywall.org/v2"

    # User agent required for NCBI PMC and publisher access
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Email for Unpaywall API (required, but can be generic for research tools)
    UNPAYWALL_EMAIL = "qsp-llm-workflows@research.edu"

    def __init__(
        self,
        data_dir: str,
        rate_limit: float = 0.5,
        fuzzy_threshold: float = 0.8,
        **kwargs,
    ):
        """
        Initialize automated verifier.

        Args:
            data_dir: Directory containing YAML files to validate
            rate_limit: Seconds between API requests (default 0.5)
            fuzzy_threshold: Similarity threshold for fuzzy matching (default 0.8)
            **kwargs: Additional configuration
        """
        super().__init__(data_dir, **kwargs)
        self.rate_limit = rate_limit
        self.fuzzy_threshold = fuzzy_threshold
        self.last_request_time = 0
        self._paper_text_cache: dict[str, Optional[str]] = {}
        # Create a TextSnippetValidator instance for value checking
        self._value_checker = TextSnippetValidator(data_dir)

    @property
    def name(self) -> str:
        return "Automated Snippet Source Verification"

    def rate_limit_wait(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def normalize_doi(self, doi: str) -> str:
        """Normalize DOI by removing URL prefixes."""
        if not doi:
            return ""
        doi_clean = doi.strip()
        doi_clean = doi_clean.replace("https://doi.org/", "")
        doi_clean = doi_clean.replace("http://doi.org/", "")
        return doi_clean

    def get_paper_info_from_doi(self, doi: str) -> Optional[PaperInfo]:
        """
        Query Europe PMC to get paper information from DOI.

        Returns PaperInfo with PMCID, open access status, and abstract.

        Args:
            doi: DOI string (e.g., "10.1056/NEJMoa1200690")

        Returns:
            PaperInfo object or None if not found
        """
        doi_clean = self.normalize_doi(doi)
        if not doi_clean:
            return None

        self.rate_limit_wait()

        try:
            params = {
                "query": f"DOI:{doi_clean}",
                "format": "json",
                "resultType": "core",
            }
            response = requests.get(
                self.EUROPE_PMC_SEARCH_URL,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get("resultList", {}).get("result", [])

            if not results:
                return None

            result = results[0]

            # Strip HTML tags from abstract if present
            abstract = result.get("abstractText", "")
            if abstract:
                # Remove HTML tags like <h4>Background</h4>
                abstract = re.sub(r"<[^>]+>", " ", abstract)
                abstract = re.sub(r"\s+", " ", abstract).strip()

            return PaperInfo(
                pmcid=result.get("pmcid"),
                pmid=result.get("pmid"),
                is_open_access=result.get("isOpenAccess") == "Y",
                in_pmc=result.get("inPMC") == "Y",
                abstract=abstract if abstract else None,
                title=result.get("title"),
            )

        except (requests.RequestException, ValueError, KeyError):
            return None

    def get_unpaywall_info(self, doi: str) -> Optional[dict]:
        """
        Query Unpaywall API for open access information.

        Unpaywall provides OA status and direct URLs to full text even for
        papers not in PMC (e.g., bronze OA on publisher sites).

        Args:
            doi: DOI string (e.g., "10.1038/s41379-019-0291-z")

        Returns:
            Dict with oa_url, oa_pdf_url, oa_status, is_oa, or None if not found
        """
        doi_clean = self.normalize_doi(doi)
        if not doi_clean:
            return None

        self.rate_limit_wait()

        try:
            url = f"{self.UNPAYWALL_API_URL}/{doi_clean}"
            params = {"email": self.UNPAYWALL_EMAIL}
            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return None

            data = response.json()

            # Get best OA location
            best_oa = data.get("best_oa_location")
            if not best_oa and not data.get("is_oa"):
                return None

            result = {
                "is_oa": data.get("is_oa", False),
                "oa_status": data.get("oa_status"),  # gold, green, bronze, hybrid
                "oa_url": None,
                "oa_pdf_url": None,
            }

            if best_oa:
                result["oa_url"] = best_oa.get("url")
                result["oa_pdf_url"] = best_oa.get("url_for_pdf")

            return result

        except (requests.RequestException, ValueError, KeyError):
            return None

    def fetch_pmc_html(self, pmcid: str) -> Optional[str]:
        """
        Fetch full-text HTML from NCBI PMC.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            HTML string or None if not available
        """
        if not pmcid:
            return None

        self.rate_limit_wait()

        try:
            url = f"{self.NCBI_PMC_ARTICLE_URL}/{pmcid}/"
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                return None

            return response.text

        except requests.RequestException:
            return None

    def fetch_publisher_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML from publisher website.

        Works for many open access publishers like Nature, Springer, etc.

        Args:
            url: URL to the article page

        Returns:
            HTML string or None if not available
        """
        if not url:
            return None

        self.rate_limit_wait()

        try:
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)

            if response.status_code != 200:
                return None

            # Check if we got HTML (not PDF or other binary)
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None

            return response.text

        except requests.RequestException:
            return None

    def extract_text_from_publisher_html(self, html_content: str) -> str:
        """
        Extract article text from publisher HTML.

        Different publishers have different HTML structures. This method
        tries common patterns for article content extraction.

        Args:
            html_content: Raw HTML string from publisher

        Returns:
            Plain text string suitable for searching
        """
        if not html_content:
            return ""

        try:
            soup = BeautifulSoup(html_content, "lxml")

            # Remove scripts, styles, nav, header, footer, and common non-content elements
            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside", "figure", "figcaption"]
            ):
                element.decompose()

            # Try to find article content using common publisher patterns
            article_text = ""

            # Pattern 1: Look for article or main content containers
            content_selectors = [
                "article",
                '[role="main"]',
                ".c-article-body",  # Nature/Springer
                ".article-body",
                ".article__body",
                ".fulltext",
                "#article-body",
                ".content-article",
                "main",
            ]

            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    article_text = content.get_text(separator=" ", strip=True)
                    if len(article_text) > 1000:  # Likely found real content
                        break

            # Fallback: get all body text if no article container found
            if not article_text or len(article_text) < 1000:
                body = soup.find("body")
                if body:
                    article_text = body.get_text(separator=" ", strip=True)

            # Normalize whitespace
            article_text = re.sub(r"\s+", " ", article_text)
            return article_text.strip()

        except Exception:
            return ""

    def extract_text_from_html(self, html_content: str) -> str:
        """
        Parse NCBI PMC HTML and extract searchable text.

        Uses BeautifulSoup to extract all text content from the page,
        stripping scripts and styles.

        Args:
            html_content: Raw HTML string

        Returns:
            Plain text string suitable for searching
        """
        if not html_content:
            return ""

        try:
            soup = BeautifulSoup(html_content, "lxml")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get text with space separator
            text = soup.get_text(separator=" ", strip=True)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        except Exception:
            return ""

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching: lowercase, collapse whitespace."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.lower().strip())

    def check_value_in_matched_text(
        self, value, units: Optional[str], matched_text: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if the declared numeric value appears in the matched text from the paper.

        This ensures the specific text location that matched the snippet also
        contains the claimed numeric value, catching cases where the LLM fabricated
        a plausible-looking snippet that doesn't actually contain the value.

        Args:
            value: The declared value (numeric or list of numerics)
            units: Optional units string
            matched_text: The text from the paper that matched the snippet

        Returns:
            (found, matched_pattern) tuple
        """
        if value is None or not matched_text:
            return (True, None)  # No value to check, pass by default

        # Use the TextSnippetValidator's check method
        return self._value_checker.check_snippet_contains_value(matched_text, value, units)

    def normalize_snippet(self, snippet: str) -> str:
        """
        Normalize snippet by removing LaTeX formatting and table artifacts.

        Handles:
        - LaTeX superscripts: ^{+}, ^{-}, ^{2}, etc. -> +, -, 2
        - LaTeX subscripts: _{2}, _{d}, etc. -> 2, d
        - Table separators: |
        - Ellipsis placeholders: ...
        - Multiple whitespace/newlines

        Args:
            snippet: Raw snippet text

        Returns:
            Cleaned snippet suitable for matching
        """
        if not snippet:
            return ""

        text = snippet

        # Remove LaTeX super/subscripts: ^{+} -> +, _{2} -> 2
        text = re.sub(r"\^{([^}]*)}", r"\1", text)
        text = re.sub(r"_{([^}]*)}", r"\1", text)

        # Also handle simple ^ and _ without braces: ^+ -> +
        text = re.sub(r"\^(\S)", r"\1", text)
        text = re.sub(r"_(\S)", r"\1", text)

        # Remove table separators
        text = text.replace("|", " ")

        # Remove ellipsis placeholders
        text = text.replace("...", " ")

        # Collapse whitespace and newlines
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def fuzzy_find_snippet(
        self, snippet: str, full_text: str
    ) -> tuple[bool, float, str, Optional[str]]:
        """
        Search for snippet in full text using fuzzy matching.

        Strategy:
        1. Normalize snippet (remove LaTeX, table formatting)
        2. Normalize both strings (lowercase, collapse whitespace)
        3. Try exact substring match first (fast path)
        4. If not found, use sliding window with SequenceMatcher

        Args:
            snippet: Text snippet to find
            full_text: Full paper text to search in

        Returns:
            (found, best_similarity_score, normalized_snippet, best_match_text) tuple
            best_match_text is the text from the paper that best matched (for near-misses)
        """
        if not snippet or not full_text:
            return (False, 0.0, "", None)

        # First normalize snippet to remove LaTeX/table formatting
        snippet_cleaned = self.normalize_snippet(snippet)
        # Then apply standard text normalization (lowercase, whitespace)
        snippet_norm = self.normalize_text(snippet_cleaned)
        text_norm = self.normalize_text(full_text)

        # Fast path: exact substring match
        if snippet_norm in text_norm:
            # Find the position and extract the matched text
            match_pos = text_norm.find(snippet_norm)
            matched_text = text_norm[match_pos : match_pos + len(snippet_norm)]
            return (True, 1.0, snippet_cleaned, matched_text)

        # Sliding window fuzzy match
        snippet_len = len(snippet_norm)
        if snippet_len > len(text_norm):
            return (False, 0.0, snippet_cleaned, None)

        best_score = 0.0
        best_match_pos = 0
        # Use a larger window to account for minor variations
        window_size = int(snippet_len * 1.2)

        # Sample positions to avoid O(n*m) complexity on long documents
        # Check every 10 characters for documents > 10000 chars
        step = 1 if len(text_norm) < 10000 else max(1, snippet_len // 3)

        for i in range(0, len(text_norm) - snippet_len + 1, step):
            window = text_norm[i : i + window_size]
            score = SequenceMatcher(None, snippet_norm, window).ratio()
            if score > best_score:
                best_score = score
                best_match_pos = i
                if score >= self.fuzzy_threshold:
                    # Return the matched text from the paper
                    matched_text = text_norm[i : i + window_size]
                    return (True, score, snippet_cleaned, matched_text)

        # Extract best matching text for near-misses (score >= 0.5)
        best_match_text = None
        if best_score >= 0.5:
            # Get the matching window from the original (non-normalized) text
            # We need to approximate the position in the original text
            best_match_text = text_norm[best_match_pos : best_match_pos + window_size]

        return (best_score >= self.fuzzy_threshold, best_score, snippet_cleaned, best_match_text)

    def get_paper_text(self, doi: str) -> tuple[Optional[str], Optional[PaperInfo], str]:
        """
        Get text for a paper by DOI.

        Tries sources in order:
        1. Full text from PMC (if available)
        2. Abstract only (fast, already fetched from Europe PMC)

        Note: Unpaywall is NOT called here. It's called separately via
        try_unpaywall_fallback() only when abstract verification fails.

        Args:
            doi: DOI string

        Returns:
            (text, paper_info, status) tuple.
            status is one of:
                - "full_text_open_access": Full text from open access PMC article
                - "full_text_restricted": Full text from restricted PMC article
                - "abstract_only": Only abstract available
                - "cached": Using cached result
                - "no_text_available": No text could be retrieved
        """
        doi_clean = self.normalize_doi(doi)

        # Check cache first
        cache_key = f"text_{doi_clean}"
        info_cache_key = f"info_{doi_clean}"
        if cache_key in self._paper_text_cache:
            cached_text = self._paper_text_cache[cache_key]
            cached_info = self._paper_text_cache.get(info_cache_key)
            if cached_text:
                return (cached_text, cached_info, "cached")
            # If cached as None, we already tried and failed
            return (None, cached_info, "no_text_available")

        # Get paper info from Europe PMC
        paper_info = self.get_paper_info_from_doi(doi_clean)
        if not paper_info:
            # Create minimal paper_info so we can still check Unpaywall
            paper_info = PaperInfo()

        # Cache the paper info
        self._paper_text_cache[info_cache_key] = paper_info

        # Try to get full text from PMC if available
        if paper_info.in_pmc and paper_info.pmcid:
            html_content = self.fetch_pmc_html(paper_info.pmcid)
            if html_content:
                full_text = self.extract_text_from_html(html_content)
                if full_text:
                    self._paper_text_cache[cache_key] = full_text
                    status = (
                        "full_text_open_access"
                        if paper_info.is_open_access
                        else "full_text_restricted"
                    )
                    return (full_text, paper_info, status)

        # Fall back to abstract (don't cache yet - may retry with Unpaywall)
        if paper_info.abstract:
            return (paper_info.abstract, paper_info, "abstract_only")

        # No text available at all
        self._paper_text_cache[cache_key] = None
        return (None, paper_info, "no_text_available")

    def try_unpaywall_fallback(
        self, doi: str, paper_info: Optional[PaperInfo] = None
    ) -> tuple[Optional[str], Optional[PaperInfo], str]:
        """
        Try to get full text from publisher via Unpaywall API.

        Called as a fallback when abstract verification has failures.
        This avoids unnecessary API calls for papers where abstract
        verification is sufficient.

        Args:
            doi: DOI string
            paper_info: Optional existing PaperInfo to update

        Returns:
            (text, paper_info, status) tuple.
            status is "full_text_publisher" on success, "unpaywall_failed" otherwise.
        """
        doi_clean = self.normalize_doi(doi)

        if paper_info is None:
            paper_info = PaperInfo()

        # Query Unpaywall for OA information
        unpaywall_info = self.get_unpaywall_info(doi_clean)
        if not unpaywall_info or not unpaywall_info.get("is_oa"):
            return (None, paper_info, "unpaywall_failed")

        # Update paper_info with Unpaywall data
        paper_info.is_open_access = True
        paper_info.oa_status = unpaywall_info.get("oa_status")
        paper_info.oa_url = unpaywall_info.get("oa_url")
        paper_info.oa_pdf_url = unpaywall_info.get("oa_pdf_url")

        # Build list of URLs to try
        urls_to_try = []
        oa_url = paper_info.oa_url

        if oa_url:
            if oa_url.endswith(".pdf"):
                # Try removing .pdf extension to get HTML page
                html_url = oa_url[:-4]  # Remove ".pdf"
                urls_to_try.append(html_url)
            else:
                urls_to_try.append(oa_url)

        # Also try constructing URL from DOI for common publishers
        # Nature/Springer: https://www.nature.com/articles/{doi_suffix}
        # Elsevier: https://www.sciencedirect.com/science/article/pii/{pii}
        if doi_clean.startswith("10.1038/"):
            # Nature journal
            doi_suffix = doi_clean.replace("10.1038/", "")
            urls_to_try.append(f"https://www.nature.com/articles/{doi_suffix}")

        # Try each URL until we get valid content
        for url in urls_to_try:
            html_content = self.fetch_publisher_html(url)
            if html_content:
                full_text = self.extract_text_from_publisher_html(html_content)
                if full_text and len(full_text) > 1000:
                    # Cache the successful result
                    cache_key = f"text_{doi_clean}"
                    self._paper_text_cache[cache_key] = full_text
                    return (full_text, paper_info, "full_text_publisher")

        return (None, paper_info, "unpaywall_failed")

    def collect_verification_data(self) -> dict:
        """
        Collect all sources and snippets that need verification.

        Returns:
            Dict mapping source_tag to {doi, snippets, inputs}
        """
        files = load_yaml_directory(self.data_dir)

        source_data = defaultdict(lambda: {"doi": None, "snippets": set(), "inputs": []})

        for file_info in files:
            data = file_info["data"]
            filename = file_info["filename"]

            sources = collect_sources(data)
            inputs = extract_inputs_from_yaml(data)

            for inp in inputs:
                if not isinstance(inp, dict):
                    continue

                input_name = inp.get("name", "unnamed")
                source_ref = inp.get("source_ref")

                if not source_ref or source_ref not in sources:
                    continue

                source = sources[source_ref]
                doi = get_doi_from_source(source)

                if not doi:
                    continue

                value_snippet = inp.get("value_snippet")
                units_snippet = inp.get("units_snippet")

                if value_snippet:
                    source_data[source_ref]["snippets"].add(value_snippet)
                if units_snippet:
                    source_data[source_ref]["snippets"].add(units_snippet)

                source_data[source_ref]["doi"] = doi
                source_data[source_ref]["inputs"].append(
                    {
                        "name": input_name,
                        "filename": filename,
                        "value": inp.get("value"),
                        "units": inp.get("units"),
                        "value_snippet": value_snippet,
                        "units_snippet": units_snippet,
                    }
                )

        return source_data

    def print_manual_verification_prompt(
        self, manual_sources: dict, reason: str = "NOT IN PMC"
    ) -> None:
        """Print sources requiring manual verification."""
        print("\n" + "=" * 70)
        print(f"MANUAL VERIFICATION REQUIRED ({reason})")
        print("The following papers require manual verification.")
        print("=" * 70 + "\n")

        for source_tag, info in sorted(manual_sources.items()):
            doi = info["doi"]
            snippets = sorted(info["snippets"])
            filenames = sorted(set(inp["filename"] for inp in info["inputs"]))

            print(f"Source: {source_tag}")
            print(f"DOI: https://doi.org/{self.normalize_doi(doi)}")
            print(f"YAML file(s): {', '.join(filenames)}")
            print(f"\nSnippets to verify ({len(snippets)}):")
            for i, snippet in enumerate(snippets, 1):
                display = snippet if len(snippet) <= 80 else snippet[:77] + "..."
                print(f'  {i}. "{display}"')
            print("-" * 70 + "\n")

        print(f"Total sources requiring manual verification: {len(manual_sources)}")
        print("=" * 70 + "\n")

    def print_abstract_only_verification_prompt(self, abstract_failures: dict) -> None:
        """Print abstract-only sources with failed snippets requiring manual verification."""
        print("\n" + "=" * 70)
        print("MANUAL VERIFICATION REQUIRED (ABSTRACT ONLY - FAILED SNIPPETS)")
        print("The following papers only have abstracts available.")
        print("Some snippets could not be verified - please check the full paper.")
        print("=" * 70 + "\n")

        for source_tag, info in sorted(abstract_failures.items()):
            doi = info["doi"]
            failed_snippets = info.get("failed_snippets", [])
            filenames = sorted(set(inp["filename"] for inp in info["inputs"]))

            print(f"Source: {source_tag}")
            print(f"DOI: https://doi.org/{self.normalize_doi(doi)}")
            print(f"YAML file(s): {', '.join(filenames)}")
            print(f"\nFailed snippets to verify ({len(failed_snippets)}):")
            for i, snippet in enumerate(failed_snippets, 1):
                display = snippet if len(snippet) <= 80 else snippet[:77] + "..."
                print(f'  {i}. "{display}"')
            print("-" * 70 + "\n")

        print(f"Total sources with failed snippets: {len(abstract_failures)}")
        print("=" * 70 + "\n")

    def get_manual_verification(self) -> bool:
        """Prompt user to verify remaining papers manually."""
        print("Please verify the snippets in the papers listed above.")
        print("Click DOI links and use Ctrl+F (Cmd+F) to search.\n")

        while True:
            response = input("Have all snippets been verified? [y/n]: ").lower()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def _verify_snippets(
        self, snippets: set, text: str
    ) -> list[tuple[str, str, bool, float, Optional[str]]]:
        """
        Verify snippets against text and return results.

        Args:
            snippets: Set of snippet strings to verify
            text: Text to search in

        Returns:
            List of (snippet, normalized, found, score, best_match) tuples
        """
        results = []
        for snippet in snippets:
            found, score, normalized, best_match = self.fuzzy_find_snippet(snippet, text)
            results.append((snippet, normalized, found, score, best_match))
        return results

    def _verify_inputs_with_values(
        self, inputs: list, text: str
    ) -> list[tuple[str, str, str, bool, float, Optional[str], bool, Optional[str]]]:
        """
        Verify input value_snippets against text AND check that values are in matched text.

        For each input's value_snippet:
        1. Check if snippet appears in paper text (fuzzy match)
        2. If found, check if the declared value appears in the matched text from the paper

        Args:
            inputs: List of input dicts with name, value, units, value_snippet, units_snippet
            text: Text to search in

        Returns:
            List of tuples:
            (input_name, snippet, normalized, snippet_found, score, best_match,
             value_in_match, value_pattern)

            - snippet_found: True if snippet was found in paper text
            - value_in_match: True if declared value was found in matched text (or no value to check)
            - value_pattern: The pattern that matched the value, or None
        """
        results = []
        for inp in inputs:
            input_name = inp.get("name", "unnamed")
            value = inp.get("value")
            units = inp.get("units")
            value_snippet = inp.get("value_snippet")

            if not value_snippet:
                continue

            # Step 1: Check if snippet appears in paper text
            snippet_found, score, normalized, matched_text = self.fuzzy_find_snippet(
                value_snippet, text
            )

            # Step 2: If snippet found, check if value is in the matched text
            value_in_match = True
            value_pattern = None
            if snippet_found and matched_text and value is not None:
                value_in_match, value_pattern = self.check_value_in_matched_text(
                    value, units, matched_text
                )

            results.append(
                (
                    input_name,
                    value_snippet,
                    normalized,
                    snippet_found,
                    score,
                    matched_text,
                    value_in_match,
                    value_pattern,
                )
            )

        return results

    def _print_snippet_results(
        self,
        snippet_results: list[tuple[str, str, bool, float, Optional[str]]],
        item_name: str,
        text_source: str,
        automated_results: list,
    ) -> int:
        """
        Print snippet verification results with color coding.

        Args:
            snippet_results: List of (snippet, normalized, found, score, best_match) tuples
            item_name: Display name for the item being verified
            text_source: Source of text ("full_text", "abstract", etc.)
            automated_results: List to append results to for report generation

        Returns:
            Count of snippets found
        """
        # Sort: failures first (by score ascending), then successes
        snippet_results = sorted(snippet_results, key=lambda x: (x[2], x[3]))

        # ANSI color codes
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"

        found_count = 0

        for snippet, normalized, found, score, best_match in snippet_results:
            was_normalized = normalized != snippet
            norm_display = normalized if len(normalized) <= 50 else normalized[:47] + "..."

            if found:
                found_count += 1
                automated_results.append((item_name, snippet, True, score, text_source))
                print(f'{GREEN}    ✓ "{norm_display}" (score: {score:.2f}){RESET}')
            else:
                automated_results.append((item_name, snippet, False, score, text_source))

                is_near_miss = 0.6 <= score < self.fuzzy_threshold
                color = YELLOW if is_near_miss else RED

                if was_normalized:
                    snippet_display = snippet if len(snippet) <= 40 else snippet[:37] + "..."
                    print(f'{color}    ✗ "{snippet_display}"{RESET}')
                    print(f'{color}      → normalized: "{norm_display}"{RESET}')
                    print(f"{color}      → best score: {score:.2f}{RESET}")
                else:
                    print(f'{color}    ✗ "{norm_display}" (best score: {score:.2f}){RESET}')

                if is_near_miss and best_match:
                    match_display = best_match if len(best_match) <= 50 else best_match[:47] + "..."
                    print(f'{YELLOW}      → closest match: "{match_display}"{RESET}')

        return found_count

    def _print_input_value_results(
        self,
        input_results: list[tuple[str, str, str, bool, float, Optional[str], bool, Optional[str]]],
        item_name: str,
        text_source: str,
        automated_results: list,
    ) -> tuple[int, int]:
        """
        Print input verification results including value presence checking.

        Args:
            input_results: List of tuples from _verify_inputs_with_values
            item_name: Display name for the item being verified
            text_source: Source of text ("full_text", "abstract", etc.)
            automated_results: List to append results to for report generation

        Returns:
            (snippets_found_count, values_found_count) tuple
        """
        # Sort: failures first (snippet not found, then value not found), then successes
        # Key: (snippet_found, value_in_match, score)
        input_results = sorted(input_results, key=lambda x: (x[3], x[6], x[4]))

        # ANSI color codes
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"

        snippets_found = 0
        values_found = 0

        for (
            input_name,
            snippet,
            normalized,
            snippet_found,
            score,
            matched_text,
            value_in_match,
            value_pattern,
        ) in input_results:
            norm_display = normalized if len(normalized) <= 50 else normalized[:47] + "..."

            if snippet_found and value_in_match:
                # Full success: snippet found AND value in matched text
                snippets_found += 1
                values_found += 1
                automated_results.append((item_name, snippet, True, score, text_source, None))
                value_info = f", value: {value_pattern}" if value_pattern else ""
                print(
                    f'{GREEN}    ✓ [{input_name}] "{norm_display}" '
                    f"(score: {score:.2f}{value_info}){RESET}"
                )
            elif snippet_found and not value_in_match:
                # Snippet found but value NOT in matched text from paper
                snippets_found += 1
                automated_results.append(
                    (item_name, snippet, False, score, text_source, "value_not_in_paper")
                )
                print(
                    f'{RED}    ✗ [{input_name}] "{norm_display}" '
                    f"(snippet found, but VALUE NOT IN PAPER TEXT){RESET}"
                )
                if matched_text:
                    match_display = (
                        matched_text if len(matched_text) <= 60 else matched_text[:57] + "..."
                    )
                    print(f'{RED}      → paper text: "{match_display}"{RESET}')
            else:
                # Snippet not found
                automated_results.append((item_name, snippet, False, score, text_source, None))
                is_near_miss = 0.6 <= score < self.fuzzy_threshold
                color = YELLOW if is_near_miss else RED

                print(
                    f'{color}    ✗ [{input_name}] "{norm_display}" '
                    f"(best score: {score:.2f}){RESET}"
                )
                if is_near_miss and matched_text:
                    match_display = (
                        matched_text if len(matched_text) <= 50 else matched_text[:47] + "..."
                    )
                    print(f'{YELLOW}      → closest match: "{match_display}"{RESET}')

        return (snippets_found, values_found)

    def validate(self) -> ValidationReport:
        """
        Run automated snippet verification.

        Workflow for each source:
        1. Try full text from PMC (if available)
        2. Try abstract (fast, already fetched from Europe PMC)
        3. If abstract has failures, try Unpaywall for publisher full text
        4. Fall back to manual verification only if all automated methods fail

        Returns:
            ValidationReport with results
        """
        report = ValidationReport(self.name)

        print(f"Verifying snippets in {self.data_dir}...")
        print("Using NCBI PubMed Central for full-text, Europe PMC for abstracts")
        print("Unpaywall fallback for OA papers when abstract verification fails")
        print(f"Fuzzy match threshold: {self.fuzzy_threshold}")
        print()

        # Collect all sources and snippets
        source_data = self.collect_verification_data()

        if not source_data:
            print("No sources with snippets found.")
            report.add_pass("All sources", "No snippets to verify")
            return report

        # Track papers needing manual verification
        manual_sources: dict = {}  # Papers with no text available
        abstract_only_failures: dict = (
            {}
        )  # Abstract-only papers with failed snippets (after Unpaywall)
        # Tuple: (item_name, snippet, found, score, source_type, failure_reason)
        automated_results: list[tuple[str, str, bool, float, str, Optional[str]]] = []

        # Process each source
        total_sources = len(source_data)
        for idx, (source_tag, info) in enumerate(sorted(source_data.items()), 1):
            doi = info["doi"]
            filenames = sorted(set(inp["filename"] for inp in info["inputs"]))
            item_name = f"{', '.join(filenames)} → {source_tag}"

            print(f"[{idx}/{total_sources}] Checking {source_tag}...")

            # Try to get text (full text from PMC, or abstract)
            text, paper_info, status = self.get_paper_text(doi)

            if not text:
                if paper_info:
                    print("  ✗ No full text or abstract available - queued for manual verification")
                else:
                    print("  ✗ Paper not found in Europe PMC - queued for manual verification")
                manual_sources[source_tag] = info
                continue

            # Report what we got
            text_len = len(text)
            pmcid = paper_info.pmcid if paper_info else None

            if status == "full_text_open_access":
                print(f"  ✓ Full text from PMC ({pmcid}) - Open Access - {text_len:,} chars")
                text_source = "full_text"
            elif status == "full_text_restricted":
                print(f"  ✓ Full text from PMC ({pmcid}) - Restricted Access - {text_len:,} chars")
                text_source = "full_text"
            elif status == "abstract_only":
                oa_label = "Open Access" if paper_info.is_open_access else "Restricted"
                print(f"  ⚠ Abstract only ({oa_label}) - {text_len:,} chars - not in PMC")
                text_source = "abstract"
            elif status == "cached":
                print(f"  ✓ Using cached text - {text_len:,} chars")
                text_source = "cached"
            else:
                print(f"  ✓ Retrieved text - {text_len:,} chars")
                text_source = "unknown"

            # Verify value_snippets with value presence checking
            inputs = info["inputs"]
            input_results = self._verify_inputs_with_values(inputs, text)
            num_inputs = len(input_results)

            # Count successes (snippet found AND value in matched text)
            full_success_count = sum(
                1 for r in input_results if r[3] and r[6]  # snippet_found and value_in_match
            )

            # If abstract-only with failures, try Unpaywall fallback before printing results
            if text_source == "abstract" and full_success_count < num_inputs:
                failed_inputs = [
                    r for r in input_results if not r[3] or not r[6]
                ]  # snippet or value failed

                print("  → Trying Unpaywall for publisher full text...")
                unpaywall_text, updated_info, unpaywall_status = self.try_unpaywall_fallback(
                    doi, paper_info
                )

                if unpaywall_status == "full_text_publisher" and unpaywall_text:
                    oa_status = updated_info.oa_status if updated_info else "unknown"
                    print(
                        f"  ✓ Full text from publisher via Unpaywall ({oa_status} OA) - "
                        f"{len(unpaywall_text):,} chars"
                    )

                    # Re-verify ALL inputs against full text (not just failed ones)
                    input_results = self._verify_inputs_with_values(inputs, unpaywall_text)
                    snippets_found, values_found = self._print_input_value_results(
                        input_results, item_name, "full_text", automated_results
                    )
                    print(
                        f"  Summary: {snippets_found}/{num_inputs} snippets matched, "
                        f"{values_found}/{num_inputs} values verified"
                    )

                    # If still failures, track for manual verification
                    if values_found < num_inputs:
                        still_failed = [r[1] for r in input_results if not r[3] or not r[6]]
                        abstract_only_failures[source_tag] = {
                            **info,
                            "failed_snippets": still_failed,
                        }
                else:
                    print("  ✗ Unpaywall: No OA full text available")
                    # Print abstract results and track for manual verification
                    snippets_found, values_found = self._print_input_value_results(
                        input_results, item_name, text_source, automated_results
                    )
                    print(
                        f"  Summary: {snippets_found}/{num_inputs} snippets matched, "
                        f"{values_found}/{num_inputs} values verified"
                    )
                    abstract_only_failures[source_tag] = {
                        **info,
                        "failed_snippets": [r[1] for r in failed_inputs],
                    }
            else:
                # Not abstract-only or all inputs verified - just print results
                snippets_found, values_found = self._print_input_value_results(
                    input_results, item_name, text_source, automated_results
                )
                print(
                    f"  Summary: {snippets_found}/{num_inputs} snippets matched, "
                    f"{values_found}/{num_inputs} values verified"
                )

        print()

        # Add automated results to report
        # Tuple format: (item_name, snippet, found, score, source_type, failure_reason)
        for result in automated_results:
            item_name, snippet, found, score, source_type, failure_reason = result
            snippet_display = snippet if len(snippet) <= 50 else snippet[:47] + "..."
            if found:
                if source_type == "abstract":
                    report.add_pass(
                        f'{item_name} → "{snippet_display}"',
                        f"Found in abstract (score: {score:.2f})",
                    )
                else:
                    report.add_pass(
                        f'{item_name} → "{snippet_display}"',
                        f"Found in full text (score: {score:.2f})",
                    )
            else:
                if failure_reason == "value_not_in_paper":
                    report.add_fail(
                        f'{item_name} → "{snippet_display}"',
                        f"Snippet found but VALUE NOT IN PAPER TEXT (score: {score:.2f})",
                    )
                elif source_type == "abstract":
                    report.add_fail(
                        f'{item_name} → "{snippet_display}"',
                        f"Not found in abstract (best score: {score:.2f}) - full text not available",
                    )
                else:
                    report.add_fail(
                        f'{item_name} → "{snippet_display}"',
                        f"Snippet not found in full text (best score: {score:.2f})",
                    )

        # Handle manual verification fallback for papers with no text
        if manual_sources:
            self.print_manual_verification_prompt(manual_sources, "NO TEXT AVAILABLE")
            user_verified = self.get_manual_verification()

            for source_tag, info in manual_sources.items():
                filenames = sorted(set(inp["filename"] for inp in info["inputs"]))
                item_name = f"{', '.join(filenames)} → {source_tag}"

                if user_verified:
                    report.add_warning(
                        item_name,
                        "No text available, verified manually by user",
                    )
                else:
                    report.add_fail(
                        item_name,
                        "No text available, user did not verify",
                    )

        # Handle manual verification for abstract-only papers with failed snippets
        if abstract_only_failures:
            self.print_abstract_only_verification_prompt(abstract_only_failures)
            user_verified = self.get_manual_verification()

            for source_tag, info in abstract_only_failures.items():
                filenames = sorted(set(inp["filename"] for inp in info["inputs"]))
                item_name = f"{', '.join(filenames)} → {source_tag}"
                failed_count = len(info.get("failed_snippets", []))

                # Note: The failures were already added to report above
                # Here we just add a note about manual verification status
                if user_verified:
                    report.add_warning(
                        f"{item_name} (manual verification)",
                        f"{failed_count} snippet(s) not in abstract/full text, verified manually by user",
                    )

        return report
