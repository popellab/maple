#!/usr/bin/env python3
"""
Validation helper functions for calibration targets.

Contains standalone functions for DOI resolution, fuzzy matching, value checking,
and mock data generation. These are used by validators in the calibration target models.
"""

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import requests


# =============================================================================
# MODULE-LEVEL CACHE FOR PAPER TEXTS
# =============================================================================

# Persists across validations to avoid repeated fetches
_paper_text_cache: dict[str, Optional[str]] = {}
_last_request_time: float = 0


def resolve_doi(doi: str) -> Optional[dict]:
    """
    Resolve DOI and get metadata from CrossRef.

    Args:
        doi: DOI string (with or without URL prefix)

    Returns:
        Dict with title, first_author, year, doi or None if resolution fails
    """
    if not doi:
        return None

    # Clean DOI
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

    try:
        # Query CrossRef API
        url = f"https://doi.org/{doi_clean}"
        headers = {"Accept": "application/vnd.citationstyles.csl+json"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        metadata = response.json()

        # Extract title
        title = metadata.get("title", "")
        if isinstance(title, list) and len(title) > 0:
            title = title[0]

        # Extract first author
        authors = metadata.get("author", [])
        first_author = None
        if authors and len(authors) > 0:
            first_author = authors[0].get("family", "")

        # Extract year
        date_parts = metadata.get("issued", {}).get("date-parts", [[]])
        year = None
        if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
            year = date_parts[0][0]

        return {"title": title, "first_author": first_author, "year": year, "doi": doi_clean}

    except Exception:
        return None


def fuzzy_match(str1: str, str2: str, threshold: float = 0.75) -> bool:
    """
    Fuzzy string matching using SequenceMatcher.

    Args:
        str1: First string
        str2: Second string
        threshold: Similarity threshold (0-1)

    Returns:
        True if similarity >= threshold
    """
    if not str1 or not str2:
        return False

    import re as _re

    # Strip HTML tags (CrossRef sometimes returns <sup>, <sub>, etc.)
    s1 = _re.sub(r"<[^>]+>", "", str1).lower().strip()
    s2 = _re.sub(r"<[^>]+>", "", str2).lower().strip()

    similarity = SequenceMatcher(None, s1, s2).ratio()
    return similarity >= threshold


def text_to_number(text: str) -> int | None:
    """
    Convert text-encoded numbers to integers.

    Args:
        text: Text like "fifty-two", "twenty-three", or "one hundred"

    Returns:
        Integer value, or None if not recognized
    """
    ones = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }

    text = text.lower().strip()

    # Handle "hundred" patterns first
    if "hundred" in text:
        parts = text.replace("hundred", "").strip().split()

        result = 0
        i = 0

        if i < len(parts):
            multiplier = ones.get(parts[i], 1)
            result = multiplier * 100
            i += 1
        else:
            result = 100

        remaining = " ".join(parts[i:])
        if remaining:
            if "-" in remaining:
                sub_parts = remaining.split("-")
                if len(sub_parts) == 2:
                    ten_val = tens.get(sub_parts[0], 0)
                    one_val = ones.get(sub_parts[1], 0)
                    result += ten_val + one_val
            elif remaining in ones:
                result += ones[remaining]
            elif remaining in tens:
                result += tens[remaining]
            else:
                remaining_parts = remaining.split()
                if len(remaining_parts) == 2:
                    result += tens.get(remaining_parts[0], 0) + ones.get(remaining_parts[1], 0)
                elif len(remaining_parts) == 1:
                    result += ones.get(remaining_parts[0], 0) + tens.get(remaining_parts[0], 0)

        return result if result > 0 else None

    # Handle single words
    if text in ones:
        return ones[text]
    if text in tens:
        return tens[text]

    # Handle hyphenated numbers like "fifty-two"
    if "-" in text:
        parts = text.split("-")
        if len(parts) == 2:
            ten_val = tens.get(parts[0], 0)
            one_val = ones.get(parts[1], 0)
            if ten_val > 0 or one_val > 0:
                return ten_val + one_val

    # Handle space-separated like "fifty two"
    if " " in text:
        parts = text.split()
        if len(parts) == 2:
            ten_val = tens.get(parts[0], 0)
            one_val = ones.get(parts[1], 0)
            if ten_val > 0 or one_val > 0:
                return ten_val + one_val

    return None


def number_to_text(num: float) -> str | None:
    """
    Convert a number to text representation (limited to 0-999).

    Args:
        num: Numeric value

    Returns:
        Text representation, or None if not in supported range
    """
    if not isinstance(num, (int, float)) or num != int(num):
        return None

    num = int(num)

    ones = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
    }
    tens = {
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }

    if 0 <= num <= 19:
        return ones[num]

    if 20 <= num <= 99:
        tens_val = (num // 10) * 10
        ones_val = num % 10
        if ones_val == 0:
            return tens[tens_val]
        else:
            return f"{tens[tens_val]}-{ones[ones_val]}"

    if 100 <= num <= 999:
        hundreds_val = num // 100
        remainder = num % 100

        result = f"{ones[hundreds_val]} hundred"

        if remainder > 0:
            if 0 <= remainder <= 19:
                result += f" {ones[remainder]}"
            elif 20 <= remainder <= 99:
                tens_val = (remainder // 10) * 10
                ones_val = remainder % 10
                if ones_val == 0:
                    result += f" {tens[tens_val]}"
                else:
                    result += f" {tens[tens_val]}-{ones[ones_val]}"

        return result

    return None


def check_value_in_text(text: str, value: float) -> bool:
    """
    Check if numeric value appears in text.
    Handles different formats: scientific notation, percentages, integers,
    text-encoded numbers (e.g., "forty-five"), etc.

    Args:
        text: Text to search
        value: Numeric value to find

    Returns:
        True if value found in text
    """
    import re

    if not text:
        return False

    text_norm = text.lower().replace(",", "")

    # Generate search patterns for the value
    patterns = []

    # Direct value
    patterns.append(str(value))

    # Integer format (if value is a whole number)
    if value == int(value):
        patterns.append(str(int(value)))

    # Scientific notation variations
    if abs(value) < 0.01 or abs(value) > 1000:
        patterns.append(f"{value:e}")
        patterns.append(f"{value:.2e}")
        patterns.append(f"{value:.3e}")
        patterns.append(f"{value:.1e}")
        patterns.append(f"{value:.0e}")

        # Compact scientific notation (e.g., "1e5" without +/- sign or leading zeros)
        # Handle values like 1e5, 2.5e-3, etc.
        if value != 0:
            import math

            exponent = int(math.floor(math.log10(abs(value))))
            mantissa = value / (10**exponent)

            # Format with minimal notation
            if mantissa == int(mantissa):
                # Integer mantissa: "1e5", "2e-3"
                patterns.append(f"{int(mantissa)}e{exponent}")
            else:
                # Decimal mantissa: "2.5e-3"
                patterns.append(f"{mantissa:.1f}e{exponent}")
                patterns.append(f"{mantissa:.2f}e{exponent}")

            # Also try without exponent sign for positive exponents
            if exponent >= 0:
                if mantissa == int(mantissa):
                    patterns.append(f"{int(mantissa)}e+{exponent}")
                    patterns.append(f"{int(mantissa)}e{exponent:02d}")

    # Percentage format (if value is between 0 and 1)
    if 0 < value < 1:
        pct = value * 100
        patterns.append(f"{pct}%")
        patterns.append(f"{pct:.0f}%")
        patterns.append(f"{pct:.1f}%")
        patterns.append(f"{pct:.2f}%")
        # Also without % symbol
        if pct == int(pct):
            patterns.append(str(int(pct)))

    # Rounded variations
    patterns.append(f"{value:.0f}")
    patterns.append(f"{value:.1f}")
    patterns.append(f"{value:.2f}")
    patterns.append(f"{value:.3f}")

    # Check each pattern
    for pattern in patterns:
        if str(pattern).lower() in text_norm:
            return True

    # Also check for Unicode superscript notation (e.g., "10⁵", "2.5×10⁻³")
    # Convert superscript to regular notation for comparison
    superscript_map = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
    }

    # Check for 10^X notation
    superscript_pattern = r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)"
    for match in re.finditer(superscript_pattern, text):
        exp_super = match.group(1)
        exp_str = "".join(superscript_map.get(c, c) for c in exp_super)
        try:
            exp_val = int(exp_str)
            # Check if this matches our value (10^exp_val)
            if abs(value - 10**exp_val) / abs(value) < 0.01:
                return True
        except ValueError:
            pass

    # Check for coefficient×10^X notation (e.g., "2.5×10⁻³")
    coef_pattern = r"([\d.]+)\s*[×x]\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)"
    for match in re.finditer(coef_pattern, text):
        coef_str = match.group(1)
        exp_super = match.group(2)
        exp_str = "".join(superscript_map.get(c, c) for c in exp_super)
        try:
            coef = float(coef_str)
            exp_val = int(exp_str)
            text_value = coef * (10**exp_val)
            # Check if this matches our value (within 1% tolerance)
            if abs(value - text_value) / abs(value) < 0.01:
                return True
        except ValueError:
            pass

    # Check for text-encoded numbers (e.g., "forty-five" for 45)
    # Only applicable for integer values
    if value == int(value):
        int_value = int(value)

        # Generate text patterns for the value
        text_pattern = number_to_text(value)
        if text_pattern:
            # Check lowercase, capitalized, and title case
            for variant in [text_pattern, text_pattern.capitalize(), text_pattern.title()]:
                if variant.lower() in text_norm:
                    return True

        # Also scan the text for word-form numbers and convert them
        words = re.findall(r"\b[a-z]+(?:-[a-z]+)?\b", text_norm)
        for i, word in enumerate(words):
            text_num = text_to_number(word)
            if text_num == int_value:
                return True
            # Try two-word combinations (e.g., "fifty two")
            if i < len(words) - 1:
                two_word = f"{word} {words[i+1]}"
                text_num = text_to_number(two_word)
                if text_num == int_value:
                    return True

    return False


def get_typical_species_value(unit_str: str) -> float:
    """
    Get a typical biological value for a species based on its unit.

    Returns a reasonable order-of-magnitude value for mock data generation
    in scale validation. These values represent typical magnitudes for
    different biological quantities in QSP models.

    Args:
        unit_str: Pint-parseable unit string (e.g., 'cell', 'nanomolar', 'mg/mL')

    Returns:
        A representative float value for that unit type
    """
    unit_lower = unit_str.lower()

    # Cell counts (tumor typically has 1e6-1e9 cells)
    if "cell" in unit_lower:
        return 1e6

    # Concentrations
    if "nanomolar" in unit_lower or "nm" in unit_lower:
        return 10.0  # 10 nM
    if "micromolar" in unit_lower or "um" in unit_lower:
        return 0.1  # 0.1 μM
    if "molar" in unit_lower:
        return 1e-9  # 1 nM in molar
    if "pg/ml" in unit_lower or "pg / ml" in unit_lower:
        return 100.0  # 100 pg/mL
    if "ng/ml" in unit_lower or "ng / ml" in unit_lower:
        return 10.0  # 10 ng/mL
    if "mg/ml" in unit_lower or "mg / ml" in unit_lower:
        return 1.0  # 1 mg/mL

    # Volumes
    if "ml" in unit_lower or "milliliter" in unit_lower:
        return 1.0  # 1 mL
    if "liter" in unit_lower:
        return 0.001  # 1 mL in liters

    # Masses
    if "mg" in unit_lower or "milligram" in unit_lower:
        return 10.0  # 10 mg
    if "gram" in unit_lower:
        return 0.01  # 10 mg in grams

    # Areas
    if "mm^2" in unit_lower or "mm**2" in unit_lower:
        return 100.0  # 100 mm²
    if "mm^3" in unit_lower or "mm**3" in unit_lower:
        return 500.0  # 500 mm³ (~1 cm diameter tumor)

    # Default: moderate value that won't cause extreme outputs
    return 100.0


def create_mock_species(species_units: dict, ureg, n_timepoints: int = 100) -> dict:
    """
    Create mock species data from species_units dict.

    Args:
        species_units: Dict mapping species names to unit info (str or dict with 'units' key)
        ureg: Pint UnitRegistry
        n_timepoints: Number of timepoints in time series (default 100)

    Returns:
        Dict mapping species names to mock Pint quantities
    """
    mock_species = {}
    for species, unit_info in species_units.items():
        # Handle both old format (string) and new format (dict with 'units' key)
        if isinstance(unit_info, dict):
            unit_str = unit_info.get("units", "dimensionless")
        else:
            unit_str = unit_info

        # Infer reasonable mock values based on unit type
        if "cell" in unit_str:
            value = 1e6
        elif "molarity" in unit_str:
            value = 1.0
        elif "gram" in unit_str:
            value = 100.0
        else:
            value = 1.0
        mock_species[species] = np.ones(n_timepoints) * value * ureg(unit_str)
    return mock_species


# =============================================================================
# SNIPPET-IN-SOURCE VERIFICATION
# =============================================================================


@dataclass
class PaperInfo:
    """Information about a paper from Europe PMC."""

    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    is_open_access: bool = False
    in_pmc: bool = False
    abstract: Optional[str] = None
    title: Optional[str] = None


def _rate_limit_wait(min_interval: float = 0.5):
    """Enforce rate limiting between API requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()


def normalize_doi(doi: str) -> str:
    """Normalize DOI by removing URL prefixes."""
    if not doi:
        return ""
    return doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


# Unpaywall API configuration
UNPAYWALL_API_URL = "https://api.unpaywall.org/v2"
UNPAYWALL_EMAIL = "maple@research.edu"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def get_unpaywall_info(doi: str) -> Optional[dict]:
    """
    Query Unpaywall API for open access information.

    Args:
        doi: DOI string

    Returns:
        Dict with oa_url, is_oa, or None if not found
    """
    doi_clean = normalize_doi(doi)
    if not doi_clean:
        return None

    _rate_limit_wait()

    try:
        url = f"{UNPAYWALL_API_URL}/{doi_clean}"
        params = {"email": UNPAYWALL_EMAIL}
        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            return None

        data = response.json()
        best_oa = data.get("best_oa_location")
        if not best_oa and not data.get("is_oa"):
            return None

        return {
            "is_oa": data.get("is_oa", False),
            "oa_url": best_oa.get("url") if best_oa else None,
        }

    except Exception:
        return None


def fetch_publisher_html(url: str) -> Optional[str]:
    """Fetch HTML from publisher website."""
    if not url:
        return None

    _rate_limit_wait()

    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)

        if response.status_code != 200:
            return None

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None

        return response.text

    except Exception:
        return None


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content."""
    if not pdf_content:
        return ""

    try:
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_content))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = " ".join(text_parts)
        return re.sub(r"\s+", " ", full_text).strip()

    except Exception:
        return ""


def fetch_pdf_and_extract_text(url: str) -> str:
    """Fetch PDF from URL and extract text."""
    if not url:
        return ""

    _rate_limit_wait()

    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)

        if response.status_code != 200:
            return ""

        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
            return ""

        return extract_text_from_pdf(response.content)

    except Exception:
        return ""


def extract_text_from_publisher_html(html_content: str) -> str:
    """Extract article text from publisher HTML."""
    if not html_content:
        return ""

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "lxml")

        # Remove non-content elements
        for element in soup(
            ["script", "style", "nav", "header", "footer", "aside", "figure", "figcaption"]
        ):
            element.decompose()

        # Try common article content selectors
        content_selectors = [
            "article",
            '[role="main"]',
            ".c-article-body",
            ".article-body",
            ".article__body",
            ".fulltext",
            "#article-body",
            ".content-article",
            "main",
        ]

        article_text = ""
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                article_text = content.get_text(separator=" ", strip=True)
                if len(article_text) > 1000:
                    break

        # Fallback to body text
        if not article_text or len(article_text) < 1000:
            body = soup.find("body")
            if body:
                article_text = body.get_text(separator=" ", strip=True)

        return re.sub(r"\s+", " ", article_text).strip()

    except Exception:
        return ""


def get_paper_texts_from_doi(doi: str) -> dict[str, Optional[str]]:
    """
    Fetch all available paper texts from Europe PMC by DOI.

    Returns a dict with both abstract and full_text (if available).
    For snippet validation, we check against BOTH sources since snippets
    may be taken from either the published abstract or preprint/PDF.

    Uses module-level cache to avoid repeated fetches.

    Args:
        doi: DOI string (with or without URL prefix)

    Returns:
        Dict with keys:
        - "abstract": Abstract text (if available)
        - "full_text": Full text from PMC/PDF (if available)
    """
    global _paper_text_cache

    doi_clean = normalize_doi(doi)
    if not doi_clean:
        return {"abstract": None, "full_text": None}

    # Check cache (now stores dict)
    cache_key = f"texts:{doi_clean}"
    if cache_key in _paper_text_cache:
        return _paper_text_cache[cache_key]

    result_texts: dict[str, Optional[str]] = {"abstract": None, "full_text": None}

    _rate_limit_wait()

    try:
        # Query Europe PMC (quotes around DOI required for proper matching)
        params = {
            "query": f'DOI:"{doi_clean}"',
            "format": "json",
            "resultType": "core",
        }
        response = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
            timeout=15,
        )

        if response.status_code != 200:
            _paper_text_cache[cache_key] = result_texts
            return result_texts

        data = response.json()
        results = data.get("resultList", {}).get("result", [])

        if not results:
            _paper_text_cache[cache_key] = result_texts
            return result_texts

        result = results[0]

        # Get abstract
        abstract = result.get("abstractText", "")
        if abstract:
            # Remove HTML tags
            abstract = re.sub(r"<[^>]+>", " ", abstract)
            abstract = re.sub(r"\s+", " ", abstract).strip()
            result_texts["abstract"] = abstract

        # Check if full text available in PMC
        in_pmc = result.get("inPMC") == "Y"
        pmcid = result.get("pmcid")

        if in_pmc and pmcid:
            # Try to fetch full text from PMC
            _rate_limit_wait()
            pmc_response = requests.get(
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            if pmc_response.status_code == 200:
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(pmc_response.text, "lxml")
                    for elem in soup(["script", "style", "nav", "header", "footer"]):
                        elem.decompose()
                    full_text = soup.get_text(separator=" ", strip=True)
                    full_text = re.sub(r"\s+", " ", full_text)
                    if full_text and len(full_text) > 500:
                        result_texts["full_text"] = full_text
                except ImportError:
                    # BeautifulSoup not available
                    pass

        # Try Unpaywall for open access full text (if not already found)
        if not result_texts["full_text"]:
            unpaywall_info = get_unpaywall_info(doi_clean)
            if unpaywall_info and unpaywall_info.get("is_oa") and unpaywall_info.get("oa_url"):
                oa_url = unpaywall_info["oa_url"]

                if oa_url.endswith(".pdf") or "pdf" in oa_url.lower():
                    # Try PDF extraction
                    full_text = fetch_pdf_and_extract_text(oa_url)
                    if full_text and len(full_text) > 1000:
                        result_texts["full_text"] = full_text
                else:
                    # Try HTML extraction
                    html_content = fetch_publisher_html(oa_url)
                    if html_content:
                        full_text = extract_text_from_publisher_html(html_content)
                        if full_text and len(full_text) > 1000:
                            result_texts["full_text"] = full_text

        _paper_text_cache[cache_key] = result_texts
        return result_texts

    except Exception:
        _paper_text_cache[cache_key] = result_texts
        return result_texts


def get_paper_text_from_doi(doi: str) -> tuple[Optional[str], str]:
    """
    Fetch paper text from Europe PMC by DOI (legacy interface).

    Returns the best available text (full_text preferred, abstract as fallback).
    Uses module-level cache to avoid repeated fetches.

    Args:
        doi: DOI string (with or without URL prefix)

    Returns:
        (text, status) tuple where status is one of:
        - "full_text": Full text from PMC/PDF
        - "abstract": Abstract only
        - "no_text": No text available
    """
    texts = get_paper_texts_from_doi(doi)

    if texts["full_text"]:
        return (texts["full_text"], "full_text")
    elif texts["abstract"]:
        return (texts["abstract"], "abstract")
    else:
        return (None, "no_text")


def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy matching by standardizing formatting variations.

    Handles:
    - LaTeX superscripts: ^{+}, ^{-}, ^{2}, etc. -> +, -, 2
    - LaTeX subscripts: _{2}, _{d}, etc. -> 2, d
    - Unicode superscripts: ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ -> 0123456789+-
    - Unicode subscripts: ₀₁₂₃₄₅₆₇₈₉ -> 0123456789
    - Unit-space-number patterns from PDF extraction: "cm 3" -> "cm3"
    - Table separators: |
    - Ellipsis placeholders: ...
    - Multiple whitespace/newlines

    This function is applied to BOTH snippets and source text to ensure
    consistent comparison regardless of formatting differences.

    Args:
        text: Raw text (snippet or source text)

    Returns:
        Cleaned text suitable for matching
    """
    if not text:
        return ""

    result = text

    # Remove LaTeX super/subscripts: ^{+} -> +, _{2} -> 2
    result = re.sub(r"\^{([^}]*)}", r"\1", result)
    result = re.sub(r"_{([^}]*)}", r"\1", result)

    # Also handle simple ^ and _ without braces: ^+ -> +
    result = re.sub(r"\^(\S)", r"\1", result)
    result = re.sub(r"_(\S)", r"\1", result)

    # Convert unicode superscripts to regular characters
    superscript_map = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
    for sup, reg in superscript_map.items():
        result = result.replace(sup, reg)

    # Convert unicode subscripts to regular characters
    subscript_map = {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
    for sub, reg in subscript_map.items():
        result = result.replace(sub, reg)

    # Handle unit-space-number patterns from PDF extraction: "cm 3" -> "cm3"
    # This catches cases where superscripts become space-separated in extracted text
    result = re.sub(r"(cm|mm|m|kg|mg|g|ml|l)\s+(\d)", r"\1\2", result, flags=re.IGNORECASE)

    # Normalize plus-minus variations: "+/-" and "±" both -> "+-"
    result = result.replace("±", "+-")
    result = result.replace("+/-", "+-")
    result = result.replace("+/−", "+-")  # unicode minus

    # Normalize spacing around operators (especially "/" in units)
    # "cm3 /year" -> "cm3/year", "cm3/ year" -> "cm3/year"
    result = re.sub(r"\s*/\s*", "/", result)

    # Remove table separators
    result = result.replace("|", " ")

    # Remove ellipsis placeholders
    result = result.replace("...", " ")

    # Collapse whitespace and newlines
    result = re.sub(r"\s+", " ", result)

    return result.strip()


def normalize_snippet(snippet: str) -> str:
    """
    Normalize snippet by removing LaTeX formatting and table artifacts.

    This is an alias for normalize_text_for_matching for backward compatibility.

    Args:
        snippet: Raw snippet text

    Returns:
        Cleaned snippet suitable for matching
    """
    return normalize_text_for_matching(snippet)


def fuzzy_find_snippet_in_text(
    snippet: str,
    source_text: str,
    threshold: float = 0.8,
) -> tuple[bool, float, Optional[str]]:
    """
    Search for snippet in source text using fuzzy matching.

    Strategy:
    1. Normalize snippet (remove LaTeX, table formatting)
    2. Normalize both strings (lowercase, collapse whitespace)
    3. Try exact substring match first (fast path)
    4. If not found, use sliding window with SequenceMatcher

    Args:
        snippet: Text snippet to find
        source_text: Full text to search in
        threshold: Similarity threshold (0-1), default 0.8

    Returns:
        (found, best_score, matched_text) tuple
        - found: True if snippet found with score >= threshold
        - best_score: Best similarity score achieved
        - matched_text: The text from source that best matched
    """
    if not snippet or not source_text:
        return (False, 0.0, None)

    # Normalize both using the same function for consistent comparison
    snippet_norm = normalize_text_for_matching(snippet).lower()
    text_norm = normalize_text_for_matching(source_text).lower()

    if not snippet_norm or not text_norm:
        return (False, 0.0, None)

    # Fast path: exact substring match
    if snippet_norm in text_norm:
        pos = text_norm.find(snippet_norm)
        return (True, 1.0, text_norm[pos : pos + len(snippet_norm)])

    # Sliding window fuzzy match
    snippet_len = len(snippet_norm)
    if snippet_len > len(text_norm):
        return (False, 0.0, None)

    best_score = 0.0
    best_match = None
    window_size = int(snippet_len * 1.2)

    # Sample positions to avoid O(n*m) complexity on long documents
    step = 1 if len(text_norm) < 10000 else max(1, snippet_len // 3)

    for i in range(0, len(text_norm) - snippet_len + 1, step):
        window = text_norm[i : i + window_size]
        score = SequenceMatcher(None, snippet_norm, window).ratio()
        if score > best_score:
            best_score = score
            best_match = window
            if score >= threshold:
                return (True, score, window)

    return (best_score >= threshold, best_score, best_match)
