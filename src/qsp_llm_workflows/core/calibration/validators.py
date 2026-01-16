#!/usr/bin/env python3
"""
Validation helper functions for calibration targets.

Contains standalone functions for DOI resolution, fuzzy matching, value checking,
and mock data generation. These are used by validators in the calibration target models.
"""

from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import requests


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

    s1 = str1.lower().strip()
    s2 = str2.lower().strip()

    similarity = SequenceMatcher(None, s1, s2).ratio()
    return similarity >= threshold


def check_value_in_text(text: str, value: float) -> bool:
    """
    Check if numeric value appears in text.
    Handles different formats: scientific notation, percentages, integers, etc.

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
