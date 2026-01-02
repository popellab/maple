#!/usr/bin/env python3
"""
Pydantic models for quick calibration target estimates.

Simple batch workflow: CSV in -> single LLM request -> CSV out.

Includes real-time validators:
- DOI resolution and title matching
- Estimate value in snippet
- Snippet in paper full text (fuzzy matching)
"""

from difflib import SequenceMatcher
from typing import List, Optional

import requests
from pydantic import BaseModel, Field, field_validator, model_validator


class QuickTargetEstimate(BaseModel):
    """Single quick estimate for a calibration target."""

    calibration_target_id: str = Field(description="Calibration target ID from input CSV")
    estimate: float = Field(description="Numeric estimate value from paper")
    units: str = Field(
        description="Units of the estimate (Pint-parseable, e.g., 'cell / millimeter**2', 'nanomolarity', 'day')"
    )
    uncertainty: float = Field(
        description="Uncertainty value (e.g., standard error, standard deviation, range width). REQUIRED - extract from paper."
    )
    uncertainty_type: str = Field(
        description="Type of uncertainty: 'se' (standard error), 'sd' (standard deviation), 'ci95' (95% confidence interval), 'range' (min-max range), 'iqr' (interquartile range), or 'other'. REQUIRED."
    )
    value_snippet: str = Field(description="Exact text snippet from paper containing the value")
    paper_name: str = Field(description="Full paper title")
    doi: str = Field(description="Paper DOI")
    threshold_description: str = Field(
        description="Human-readable description of calibration target threshold (e.g., 'samples taken at resection, average tumor volume 500 mm³')"
    )

    @staticmethod
    def resolve_doi(doi: str) -> Optional[dict]:
        """
        Resolve DOI and get metadata from CrossRef.

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

    @staticmethod
    def fuzzy_match(str1: str, str2: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy string matching using SequenceMatcher.

        Returns:
            True if similarity >= threshold
        """
        if not str1 or not str2:
            return False

        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        similarity = SequenceMatcher(None, s1, s2).ratio()
        return similarity >= threshold

    @staticmethod
    def check_value_in_snippet(snippet: str, value: float) -> bool:
        """
        Check if numeric value appears in snippet.
        Handles different formats: scientific notation, percentages, etc.

        Returns:
            True if value found in snippet
        """
        if not snippet:
            return False

        snippet_norm = snippet.lower().replace(",", "")

        # Generate search patterns for the value
        patterns = []

        # Direct value
        patterns.append(str(value))

        # Scientific notation variations
        if abs(value) < 0.01 or abs(value) > 10000:
            # Try e notation
            patterns.append(f"{value:e}")
            patterns.append(f"{value:.2e}")
            patterns.append(f"{value:.3e}")

        # Percentage format (if value is between 0 and 1)
        if 0 < value < 1:
            pct = value * 100
            patterns.append(f"{pct}%")
            patterns.append(f"{pct:.1f}%")
            patterns.append(f"{pct:.2f}%")

        # Rounded variations
        patterns.append(f"{value:.1f}")
        patterns.append(f"{value:.2f}")
        patterns.append(f"{value:.3f}")

        # Check each pattern
        for pattern in patterns:
            if str(pattern).lower() in snippet_norm:
                return True

        return False

    @field_validator("doi")
    @classmethod
    def validate_doi_resolution(cls, doi: str) -> str:
        """Validator 1: Check that DOI resolves via CrossRef (field validator - fails fast)."""
        metadata = cls.resolve_doi(doi)
        if metadata is None:
            raise ValueError(
                f"DOI '{doi}' failed to resolve via CrossRef. "
                "Verify the DOI exists and is correctly formatted (e.g., '10.1234/journal.2023.123'). "
                "Search for the paper on Google Scholar or PubMed to find the correct DOI."
            )
        return doi

    @model_validator(mode="after")
    def validate_title_match(self) -> "QuickTargetEstimate":
        """Validator 2: Check that paper title matches CrossRef metadata."""
        metadata = self.resolve_doi(self.doi)
        if metadata:
            crossref_title = metadata.get("title", "")
            if not self.fuzzy_match(crossref_title, self.paper_name, threshold=0.75):
                raise ValueError(
                    f"Paper title mismatch:\n"
                    f"  CrossRef: '{crossref_title}'\n"
                    f"  Provided: '{self.paper_name}'\n"
                    f"Use the exact title from the DOI. Copy the title from CrossRef or the paper itself."
                )
        return self

    @model_validator(mode="after")
    def validate_estimate_in_snippet(self) -> "QuickTargetEstimate":
        """Validator 3: Check that estimate value appears in value_snippet."""
        if not self.check_value_in_snippet(self.value_snippet, self.estimate):
            raise ValueError(
                f"Estimate value {self.estimate} not found in value_snippet.\n"
                f"Snippet: '{self.value_snippet[:200]}...'\n"
                "Ensure the snippet actually contains the numeric value."
            )
        return self


class QuickEstimateResponse(BaseModel):
    """Response containing estimates for all calibration targets."""

    estimates: List[QuickTargetEstimate] = Field(
        description="List of estimates, one per calibration target from input CSV"
    )
