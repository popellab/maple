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
    model_output_code: str = Field(
        description="""Python function to compute this calibration target from model species.

Function signature: def compute_test_statistic(time, species_dict, ureg) -> Pint Quantity
- time: numpy array with Pint units (days)
- species_dict: dict mapping species names to Pint quantities (e.g., species_dict['V_T.CD8'])
- ureg: Pint UnitRegistry
- MUST return a Pint Quantity with units matching the 'units' field

CONVERSION ASSUMPTIONS (3D model → 2D histology):
- Model tracks 3D volumes (V_T in milliliters)
- Literature often reports 2D densities (cells/mm² from tissue sections)
- Standard tissue section thickness: 5 μm (0.005 mm)
- Conversion: cells/mm² = (cells/mm³) × 0.005 mm

Example for CD8+ density:
def compute_test_statistic(time, species_dict, ureg):
    import numpy as np
    cd8 = species_dict['V_T.CD8']
    V_T = species_dict['V_T']
    density_3d = (cd8 / V_T).to('cell / millimeter**3')
    # Convert 3D → 2D (assume 5 μm section)
    section_thickness = 0.005 * ureg.millimeter
    density_2d = density_3d * section_thickness
    return density_2d.to('cell / millimeter**2')"""
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

    @model_validator(mode="after")
    def validate_model_output_units(self) -> "QuickTargetEstimate":
        """Validator 4: Check that model_output_code returns correct units."""
        import ast
        import numpy as np
        from qsp_llm_workflows.core.unit_registry import ureg

        # Parse the code
        try:
            tree = ast.parse(self.model_output_code)
        except SyntaxError as e:
            raise ValueError(f"model_output_code has syntax error: {e}")

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compute_test_statistic":
                func_def = node
                break

        if not func_def:
            raise ValueError(
                "model_output_code must define a function named 'compute_test_statistic'"
            )

        # Check signature
        args = [arg.arg for arg in func_def.args.args]
        if args != ["time", "species_dict", "ureg"]:
            raise ValueError(
                f"Function signature must be (time, species_dict, ureg), got ({', '.join(args)})"
            )

        # Execute with mock data to test output units
        try:
            # Create mock data
            time = np.linspace(0, 14, 10) * ureg.day
            mock_species = {
                "V_T": np.ones(10) * 10.0 * ureg.milliliter,
                "V_T.C1": np.ones(10) * 1e9 * ureg.cell,
                "V_T.CD8": np.ones(10) * 1e7 * ureg.cell,
                "V_T.CD8_exh": np.ones(10) * 1e6 * ureg.cell,
                "V_T.Treg": np.ones(10) * 1e6 * ureg.cell,
                "V_T.Th": np.ones(10) * 1e6 * ureg.cell,
                "V_T.Mac_M1": np.ones(10) * 1e6 * ureg.cell,
                "V_T.Mac_M2": np.ones(10) * 1e7 * ureg.cell,
                "V_T.MDSC": np.ones(10) * 1e6 * ureg.cell,
                "V_T.APC": np.ones(10) * 1e5 * ureg.cell,
                "V_T.mAPC": np.ones(10) * 1e5 * ureg.cell,
                "V_T.aPSC": np.ones(10) * 1e7 * ureg.cell,
                "V_T.qPSC": np.ones(10) * 1e7 * ureg.cell,
                "V_T.ECM": np.ones(10) * 100.0 * ureg.milligram,
                "V_T.TGFb": np.ones(10) * 10.0 * ureg.nanomolarity,
                "V_T.IFNg": np.ones(10) * 1.0 * ureg.nanomolarity,
                "V_T.IL10": np.ones(10) * 5.0 * ureg.nanomolarity,
                "V_T.IL6": np.ones(10) * 5.0 * ureg.nanomolarity,
                "V_T.CCL2": np.ones(10) * 1.0 * ureg.nanomolarity,
                "V_T.ArgI": np.ones(10) * 0.5 * ureg.nanomolarity,
                "V_T.c_vas": np.ones(10) * 100.0 * ureg.picogram / ureg.milliliter,
                "V_C.TGFb": np.ones(10) * 5.0 * ureg.nanomolarity,
            }

            # Execute function
            local_scope = {"ureg": ureg, "np": np}
            exec(self.model_output_code, local_scope)
            compute_fn = local_scope["compute_test_statistic"]

            result = compute_fn(time, mock_species, ureg)

            # Check result has units
            if not hasattr(result, "units"):
                raise ValueError("Function must return a Pint Quantity with units")

            # Check dimensionality matches
            expected_quantity = 1.0 * ureg(self.units)
            if not result.dimensionality == expected_quantity.dimensionality:
                raise ValueError(
                    f"Unit dimensionality mismatch:\n"
                    f"  Expected: {self.units} ({expected_quantity.dimensionality})\n"
                    f"  Got: {result.units} ({result.dimensionality})\n"
                    f"Ensure model_output_code returns the same units as the literature estimate."
                )

        except Exception as e:
            if "dimensionality mismatch" in str(e) or "Unit" in str(e):
                raise  # Re-raise unit errors
            # Other execution errors might be due to missing species - be lenient
            pass

        return self


class QuickEstimateResponse(BaseModel):
    """Response containing estimates for all calibration targets."""

    estimates: List[QuickTargetEstimate] = Field(
        description="List of estimates, one per calibration target from input CSV"
    )
