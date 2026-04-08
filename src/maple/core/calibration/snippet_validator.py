"""
Validate value_snippets and table_excerpts in SubmodelTarget/CalibrationTarget
YAMLs against source PDFs, with abstract fallback.

Priority: local PDF > Europe PMC abstract > PubMed abstract > skip

This module provides the core validation logic. The CLI entry point is in
``maple.mcp_server`` (via the ``validate_target`` MCP tool) or can be used
directly::

    from maple.core.calibration.snippet_validator import validate_snippets_in_file
    ok, errors, skipped, passed, manual = validate_snippets_in_file(path)
"""

import json
import re
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

from maple.core.calibration.submodel_target import SubmodelTarget
from maple.core.calibration.validators import fuzzy_find_snippet_in_text


# ---------------------------------------------------------------------------
# PDF / abstract text loading
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a local PDF file.

    Prefers ``pdftotext`` (poppler) which handles two-column layouts correctly,
    falling back to pypdf if pdftotext is not installed.
    """
    import shutil
    import subprocess

    # Prefer pdftotext (poppler) — handles multi-column PDFs without interleaving
    if shutil.which("pdftotext"):
        try:
            result = subprocess.run(
                ["pdftotext", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return re.sub(r"\s+", " ", result.stdout).strip()
        except Exception:
            pass  # fall through to pypdf

    # Fallback: pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = " ".join(text_parts)
        return re.sub(r"\s+", " ", full_text).strip()

    except Exception as e:
        print(f"    WARNING: Failed to extract text from {pdf_path.name}: {e}")
        return ""


def fetch_abstract_from_europepmc(doi: str) -> Optional[str]:
    """Fetch abstract text from Europe PMC using a DOI."""
    try:
        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query=DOI:{doi}&format=json&resultType=core"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "maple-validator/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data.get("resultList", {}).get("result", [])
        if results and results[0].get("abstractText"):
            return results[0]["abstractText"]
    except Exception as e:
        print(f"    WARNING: Europe PMC lookup failed for DOI {doi}: {e}")
    return None


def fetch_abstract_from_pubmed(pmid: str) -> Optional[str]:
    """Fetch abstract text from PubMed E-utilities using a PMID."""
    try:
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "maple-validator/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if text.strip():
            return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"    WARNING: PubMed lookup failed for PMID {pmid}: {e}")
    return None


def extract_pmid_from_url(url: Optional[str]) -> Optional[str]:
    """Extract PMID from a PubMed URL."""
    if not url:
        return None
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    return m.group(1) if m else None


def find_paper_pdf(source_tag: str, papers_dir: Path) -> Optional[Path]:
    """Find a PDF matching source_tag in papers_dir/<source_tag>/ or flat in papers_dir/.

    Checks subdirectory first, then falls back to matching the source_tag
    (e.g. "Ganusov2014") against filenames in the flat papers_dir.
    """
    # Prefer subdirectory layout: papers_dir/<source_tag>/*.pdf
    paper_dir = papers_dir / source_tag
    if paper_dir.is_dir():
        pdfs = list(paper_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]

    # Fallback: match source_tag components against flat PDFs in papers_dir
    # Extract author and year from tag like "Ganusov2014" or "denBraber2012"
    m = re.match(r"([A-Za-z]+)(\d{4})", source_tag)
    if m and papers_dir.is_dir():
        author = m.group(1).lower()
        year = m.group(2)
        for pdf in papers_dir.glob("*.pdf"):
            name_lower = pdf.name.lower()
            # Strip spaces (including non-breaking \xa0), hyphens for multi-word authors
            # e.g., "den\xa0Braber" or "Vukmanovic-Stejic" → "denbraber", "vukmanovicstejic"
            name_collapsed = name_lower.replace(" ", "").replace("\xa0", "").replace("-", "")
            if (author in name_lower or author in name_collapsed) and year in name_lower:
                return pdf

    return None


def load_paper_texts(
    source_tags: set[str],
    source_metadata: dict[str, dict],
    papers_dir: Optional[Path] = None,
) -> dict[str, tuple[str, str]]:
    """Load paper text for each source_tag.

    Returns dict of {tag: (text, source_type)} where source_type is
    'pdf', 'abstract', or absent if nothing found.

    Parameters
    ----------
    source_tags : set[str]
        Source tags to look up.
    source_metadata : dict
        Maps tag -> {'doi': ..., 'url': ...} for abstract fallback.
    papers_dir : Path, optional
        Directory containing ``<source_tag>/`` subdirs with PDFs.
        If None, skips local PDF lookup and goes straight to abstract.
    """
    texts: dict[str, tuple[str, str]] = {}
    for tag in source_tags:
        # Try local PDF first
        if papers_dir is not None:
            pdf_path = find_paper_pdf(tag, papers_dir)
            if pdf_path:
                text = extract_text_from_pdf(pdf_path)
                if text:
                    texts[tag] = (text, "pdf")
                    print(f"    Loaded {tag}: {len(text)} chars from {pdf_path.name}")
                    continue
                else:
                    print(f"    WARNING: {tag}: PDF found but no text extracted")

        # Fallback: fetch abstract
        meta = source_metadata.get(tag, {})
        doi = meta.get("doi")
        url = meta.get("url")
        abstract = None

        if doi:
            abstract = fetch_abstract_from_europepmc(doi)

        if not abstract:
            pmid = extract_pmid_from_url(url)
            if pmid:
                abstract = fetch_abstract_from_pubmed(pmid)

        if abstract:
            texts[tag] = (abstract, "abstract")
            print(f"    Loaded {tag}: {len(abstract)} chars from abstract")
        else:
            print(f"    SKIP: {tag}: no PDF or abstract available")
    return texts


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_snippets_in_file(
    yaml_path: Path,
    papers_dir: Optional[Path] = None,
) -> tuple[bool, list[str], list[str], list[str], list[str]]:
    """Validate snippets in a single YAML against source PDFs / abstracts.

    Parameters
    ----------
    yaml_path : Path
        Path to a SubmodelTarget YAML file.
    papers_dir : Path, optional
        Directory containing ``<source_tag>/`` subdirs with PDFs.
        If None, defaults to ``yaml_path.parent / "papers"``.

    Returns
    -------
    (all_passed, errors, skipped, passed, manual_review)
    """
    if papers_dir is None:
        papers_dir = yaml_path.parent / "papers"

    errors = []
    skipped = []
    passed = []

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        target = SubmodelTarget.model_validate(data)
    except Exception as e:
        return (False, [f"Failed to parse: {e}"], [], [], [])

    # Collect all source_tags and their metadata
    source_tags = set()
    source_metadata: dict[str, dict] = {}

    pri = target.primary_data_source
    source_tags.add(pri.source_tag)
    source_metadata[pri.source_tag] = {"doi": pri.doi, "url": None}

    if target.secondary_data_sources:
        for src in target.secondary_data_sources:
            source_tags.add(src.source_tag)
            source_metadata[src.source_tag] = {
                "doi": src.doi,
                "url": getattr(src, "url", None),
            }

    # Load paper texts
    paper_data = load_paper_texts(source_tags, source_metadata, papers_dir)

    # Verify each input's snippet, table_excerpt, and/or figure_excerpt
    manual_review = []
    for inp in target.inputs:
        tag = inp.source_ref
        has_snippet = bool(inp.value_snippet)
        has_table = bool(inp.table_excerpt)
        has_figure = bool(inp.figure_excerpt)

        if has_figure:
            fe = inp.figure_excerpt
            manual_review.append(
                f"{inp.name} figure_excerpt ({fe.figure_id}: "
                f"value={fe.value}) — MANUAL REVIEW REQUIRED"
            )

        # Skip snippet validation for reference_value and derived_arithmetic inputs —
        # these are unit conversions, assumed CVs, etc. where no verbatim quote exists
        if inp.input_type in ("reference_value", "derived_arithmetic", "unit_conversion"):
            skipped.append(f"{inp.name} ({inp.input_type} — snippet validation skipped)")
            continue

        if not has_snippet and not has_table:
            if has_figure:
                skipped.append(f"{inp.name} (figure_excerpt only — flagged for manual review)")
            else:
                skipped.append(f"{inp.name} (no snippet or table_excerpt)")
            continue

        if tag not in paper_data:
            skipped.append(f"{inp.name} (no text for {tag})")
            continue

        text, source_type = paper_data[tag]
        src_label = f"{tag} [{source_type}]"

        # Validate table_excerpt if present
        if has_table:
            if source_type == "abstract":
                skipped.append(f"{inp.name} table_excerpt (table data unlikely in abstract)")
            else:
                te = inp.table_excerpt
                te_errors = []
                for field_name, field_val in [
                    ("table_id", te.table_id),
                    ("column", te.column),
                    ("value", te.value),
                ]:
                    found, score, _ = fuzzy_find_snippet_in_text(field_val, text, threshold=0.7)
                    if found:
                        passed.append(
                            f"{inp.name} table_excerpt.{field_name} "
                            f"(score={score:.2f} in {src_label})"
                        )
                    else:
                        te_errors.append(
                            f"table_excerpt.{field_name}='{field_val}' "
                            f"(best score: {score:.2f})"
                        )

                # Row gets a lower threshold — PDF extraction often mangles
                # row labels with special characters
                found, score, _ = fuzzy_find_snippet_in_text(te.row, text, threshold=0.6)
                if found:
                    passed.append(
                        f"{inp.name} table_excerpt.row " f"(score={score:.2f} in {src_label})"
                    )
                else:
                    te_errors.append(f"table_excerpt.row='{te.row}' (best score: {score:.2f})")

                if te_errors:
                    errors.append(
                        f"{inp.name} table_excerpt in {src_label}:\n"
                        + "\n".join(f"      FAIL: {e}" for e in te_errors)
                    )

        # Validate value_snippet if present
        if has_snippet:
            found, score, matched = fuzzy_find_snippet_in_text(
                inp.value_snippet, text, threshold=0.7
            )

            if found:
                passed.append(f"{inp.name} snippet (score={score:.2f} in {src_label})")
            else:
                snippet_display = (
                    inp.value_snippet[:80] + "..."
                    if len(inp.value_snippet) > 80
                    else inp.value_snippet.strip()
                )
                errors.append(
                    f"{inp.name} snippet (best score: {score:.2f} in "
                    f"{src_label})\n"
                    f"      Snippet: '{snippet_display}'"
                )

    return (len(errors) == 0, errors, skipped, passed, manual_review)


def validate_snippets_in_dir(
    yaml_dir: Path,
    papers_dir: Optional[Path] = None,
    glob_pattern: str = "*.yaml",
) -> tuple[int, int, list[str]]:
    """Validate all YAMLs in a directory.

    Returns (passed_count, total_count, all_errors).
    """
    yaml_files = sorted(yaml_dir.glob(glob_pattern))
    passed_files = 0
    all_errors = []

    for yaml_file in yaml_files:
        print(f"\n--- {yaml_file.name} ---")
        success, errors, skipped, file_passed, manual_review = validate_snippets_in_file(
            yaml_file, papers_dir
        )

        if success:
            passed_files += 1
            print(f"  RESULT: PASSED ({len(file_passed)} verified, {len(skipped)} skipped)")
        else:
            print(f"  RESULT: FAILED ({len(errors)} errors)")
            for err in errors:
                print(f"    FAIL: {err}")
                all_errors.append(f"{yaml_file.name} / {err}")

        for p in file_passed:
            print(f"    OK: {p}")
        for s in skipped:
            print(f"    SKIP: {s}")
        for mr in manual_review:
            print(f"    MANUAL: {mr}")

    return passed_files, len(yaml_files), all_errors
