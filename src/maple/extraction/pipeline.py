# ruff: noqa
# pyrefly: ignore-all-errors
# pyright: reportGeneralIssues=false
"""
Shared schemas, helpers, agents, and stage functions for staged extraction.

Imported by staged_extraction.py — keeps the pipeline script short.
"""

import asyncio
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Literal

import yaml

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, BinaryContent, WebSearchTool
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from maple.core.calibration.submodel_target import SubmodelTarget
from maple.core.model_structure import ModelStructure


# ===========================================================================
# Output schemas
# ===========================================================================

# -- Stage 1: Literature Search --


class LitSearchCandidate(BaseModel):
    """Kept slim — web search only sees abstracts, not full text."""

    rank: int
    doi: str
    title: str
    year: int
    relevance_summary: str = Field(
        description="Why this paper is useful — what quantitative data does the abstract mention?"
    )
    model_role_mapping: str = Field(
        description="How the paper's measurements map to the parameter's model role"
    )
    mapping_clean: bool = Field(description="Whether the data maps cleanly to the model parameter")
    mapping_concerns: str = Field(
        default="", description="Any mismatch between paper measurements and model parameter"
    )
    jointly_constrainable_parameters: list[str] = Field(
        default_factory=list,
        description="Other QSP parameters from the same rate law(s) that this paper's data could also constrain. Check 'Other parameters' in the parameter context section.",
    )
    species: str = Field(default="")
    experimental_system: str = Field(
        default="", description="in_vitro_cell_line | in_vitro_primary | in_vivo_xenograft | ..."
    )

    @model_validator(mode="after")
    def validate_doi_via_crossref(self) -> "LitSearchCandidate":
        """Verify DOI resolves via CrossRef and title matches."""
        from maple.core.calibration.validators import resolve_doi, fuzzy_match

        metadata = resolve_doi(self.doi)
        if metadata is None:
            raise ValueError(f"DOI does not resolve via CrossRef: {self.doi}")

        crossref_title = metadata.get("title", "")
        if crossref_title and not fuzzy_match(crossref_title, self.title, threshold=0.6):
            raise ValueError(
                f"DOI title mismatch for {self.doi}:\n"
                f"  LLM: '{self.title}'\n"
                f"  CrossRef: '{crossref_title}'"
            )

        crossref_year = metadata.get("year")
        if crossref_year and self.year != crossref_year:
            raise ValueError(
                f"DOI year mismatch for {self.doi}: "
                f"LLM says {self.year}, CrossRef says {crossref_year}"
            )

        return self


class LitSearchResult(BaseModel):
    candidates: list[LitSearchCandidate]
    parameters_analyzed: str = Field(
        description="Summary of what the parameter(s) represent mechanistically and what data is needed"
    )
    derivation_strategy: str = Field(
        default="",
        description="For multi-paper derivations: explain the algebraic/steady-state relationship and what each paper contributes. Empty if a single paper suffices.",
    )
    unmappable_notes: str = Field(
        default="", description="If data doesn't map to model parameterization, explain why"
    )


# -- Stage 2: Extraction --


class DataAvailability(BaseModel):
    location: str = Field(
        description="Where in the paper (e.g., 'Table 2', 'Fig 3A', 'Results p.4')"
    )
    data_type: Literal["table", "text", "figure"]
    description: str = Field(description="What quantitative data is available")
    needs_digitization: bool = Field(
        description="True ONLY if data is in a figure AND the figure contains data not available in text or tables"
    )
    digitization_justification: str = Field(
        default="",
        description="What specific added value digitization provides beyond text/table data (e.g., individual data points vs reported mean, time-course resolution, dose-response shape). Empty if needs_digitization is false.",
    )
    digitization_priority: Literal["not_needed", "optional", "helpful", "critical"] = Field(
        description="How important digitization is for constraining the parameter. "
        "'not_needed' = data is available in text/tables, no digitization required. "
        "'optional' = figure confirms text/table values or adds marginal precision. "
        "'helpful' = text/table data exists but figure adds resolution or individual data points. "
        "'critical' = no usable constraint without this figure (e.g., only source of dose-response or time-course data).",
    )
    digitization_hints: str = Field(
        default="",
        description="What to digitize and how (axis labels, units, conditions). Empty if needs_digitization is false.",
    )


class PaperAssessment(BaseModel):
    doi: str
    source_tag: str = Field(description="Author2023 format")
    usable: bool = Field(description="Whether this paper has data that can constrain the parameter")
    role: Literal["standalone", "required_for_derivation", "alternative", "validation_only"] = (
        Field(
            description="This paper's role in the extraction plan. "
            "'standalone' = contains all data needed for a complete derivation by itself. "
            "'required_for_derivation' = contributes one piece of a multi-paper derivation (all required papers are needed). "
            "'alternative' = independent alternative to another paper (pick the better one). "
            "'validation_only' = useful for cross-checking but not part of any derivation.",
        )
    )
    summary: str = Field(
        description="1-2 sentence assessment of what data is available and how it maps to the parameter"
    )
    mapping_concerns: str = Field(
        default="", description="Any mismatch between available data and model parameterization"
    )
    data: list[DataAvailability] = Field(default_factory=list)
    forward_model_suggestion: str = Field(
        default="",
        description="Suggested forward model type: exponential_growth, first_order_decay, algebraic, direct_fit, ...",
    )
    jointly_constrainable_parameters: list[str] = Field(
        default_factory=list,
        description="Other QSP parameters that could be jointly constrained by fitting a forward model to this paper's data. Confirmed after reading the full paper.",
    )


class ExtractionPlan(BaseModel):
    """Minimal set of papers needed for one complete derivation."""

    papers: list[str] = Field(description="DOIs of papers in the plan, in order of use")
    strategy: str = Field(
        description="How these papers combine: what each contributes and how they connect algebraically or via a shared forward model"
    )
    digitizations_needed: list[str] = Field(
        default_factory=list,
        description="source_tag/location pairs that MUST be digitized for this plan to work (e.g., 'Obar2008/Figure 2B')",
    )


class AssessmentResult(BaseModel):
    papers: list[PaperAssessment]
    extraction_plan: ExtractionPlan = Field(
        description="The recommended minimal extraction plan: the smallest set of papers (and their specific figures/tables) "
        "needed for one complete derivation of the target parameter. Prefer standalone papers over multi-paper derivations."
    )
    alternative_plans: list[ExtractionPlan] = Field(
        default_factory=list,
        description="Alternative extraction plans if the primary plan fails or if independent derivations are desirable for validation.",
    )
    overall_notes: str = Field(default="", description="Any cross-paper observations")


# -- Stage 2b: Plan Review --


class PlanReview(BaseModel):
    target: str = Field(description="target_id being reviewed")
    verdict: Literal["proceed", "switch_to_alt", "rerun_lit_search", "defer"] = Field(
        description="'proceed' = plan is good, move to extraction. "
        "'switch_to_alt' = an alternative plan is better (fewer digs, better mapping, larger sample size). "
        "'rerun_lit_search' = no viable plan exists, need different papers. "
        "'defer' = parameter may not be extractable as a submodel target.",
    )
    reason: str = Field(description="1-3 sentence explanation of the verdict")
    replacement_plan: ExtractionPlan | None = Field(
        default=None,
        description="The extraction plan to use instead. Required when verdict is switch_to_alt. "
        "Should match one of the alternative_plans from the assessment.",
    )
    lit_search_notes: str = Field(
        default="",
        description="If rerun_lit_search: what search terms or data types to look for differently. Will be appended to targets.csv notes.",
    )


class PlanReviewResult(BaseModel):
    reviews: list[PlanReview]
    summary: str = Field(
        description="Overall readiness: how many targets are ready, how many need work, recommended next steps"
    )


# -- Stage 3b: Derivation Review --


class DerivationReview(BaseModel):
    target: str = Field(description="target_id being reviewed")
    verdict: Literal["accept", "revise", "reject"] = Field(
        description="'accept' = derivation is scientifically sound, proceed to validation. "
        "'revise' = derivation has fixable issues (e.g., wrong unit conversion, questionable assumption). "
        "'reject' = derivation is fundamentally flawed (circular reasoning, wrong data, nonsensical forward model).",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Specific concerns about the derivation. Be concrete: cite the exact assumption, value, or step that is problematic.",
    )
    suggested_fix: str = Field(
        default="",
        description="For 'revise': what specifically should change. For 'reject': why re-extraction is needed.",
    )


class DerivationReviewResult(BaseModel):
    reviews: list[DerivationReview]
    cross_target_issues: list[str] = Field(
        default_factory=list,
        description="Issues spanning multiple targets: contradictory assumptions, circular dependencies, "
        "inconsistent cell densities or rates, parameters derived from the same data that should be jointly constrained.",
    )
    summary: str = Field(
        description="Overall scientific quality assessment and recommended next steps"
    )


# ===========================================================================
# Helpers
# ===========================================================================


def pdf_to_binary(pdf_path: Path) -> BinaryContent:
    return BinaryContent(
        data=pdf_path.read_bytes(),
        media_type="application/pdf",
        identifier=pdf_path.name,
    )


def build_parameter_context(parameters: str, model_structure_path: Path) -> str:
    from maple.core.model_structure import ModelStructure
    from maple.core.prompt_builder import SubmodelTargetPromptBuilder

    model_structure = ModelStructure.from_json(model_structure_path)
    builder = SubmodelTargetPromptBuilder(base_dir=Path("."))
    return builder.format_parameter_context(parameters, model_structure)


def collect_papers(papers_dir: Path) -> tuple[list[BinaryContent], list[str]]:
    if not papers_dir.exists():
        return [], []

    file_parts = []
    for pdf in sorted(papers_dir.glob("**/*.pdf")):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        if size_mb > 45:
            print(f"    Warning: {pdf.name} is {size_mb:.1f}MB, skipping (>45MB)")
            continue
        file_parts.append(pdf_to_binary(pdf))

    text_parts = []
    for ext in ("*.txt", "*.md"):
        for f in sorted(papers_dir.glob(ext)):
            text_parts.append(f"--- {f.name} ---\n{f.read_text()}\n")

    return file_parts, text_parts


WPD_FORMAT_HEADER = """\
## WPD Export Format
Columns: x, y, point_id, trace_type (mean or mean_plus_error)
mean_plus_error rows give the y-coordinate of the error bar tip, not the error magnitude.
For bar charts, x values cluster around discrete integers (category indices).
"""


def read_digitized_data(digitized_dir: Path) -> str:
    if not digitized_dir.exists():
        return ""
    parts = []
    # Walk source_tag subdirectories
    for subdir in sorted(digitized_dir.iterdir()):
        if not subdir.is_dir():
            continue
        tag = subdir.name
        # README first for context
        readme = subdir / "README.md"
        if readme.exists():
            parts.append(f"--- {tag}/README.md ---\n{readme.read_text()}\n")
        # Then CSVs
        for csv_file in sorted(subdir.glob("*.csv")):
            parts.append(f"--- {tag}/{csv_file.name} ---\n{csv_file.read_text()}\n")
    return "\n\n".join(parts) if parts else ""


def generate_digitization_readme(assessment: dict, digitized_dir: Path) -> None:
    """Generate per-source_tag README templates from assessment digitization hints."""
    digitized_dir.mkdir(parents=True, exist_ok=True)

    for p in assessment.get("papers", []):
        dig_items = [d for d in p.get("data", []) if d.get("needs_digitization")]
        if not dig_items:
            continue

        tag = p["source_tag"]
        tag_dir = digitized_dir / tag
        tag_dir.mkdir(exist_ok=True)

        readme_path = tag_dir / "README.md"
        if readme_path.exists():
            continue  # don't overwrite user edits

        lines = [f"# {tag}\n"]
        lines.append(WPD_FORMAT_HEADER)
        for d in dig_items:
            lines.append(f"## {d['location']}")
            lines.append(f"- {d['description']}")
            if d.get("digitization_hints"):
                lines.append(f"- {d['digitization_hints']}")
            lines.append("")

        readme_path.write_text("\n".join(lines))
        print(f"  wrote {readme_path}")


def validate_prior_derivation(yaml_path: Path, priors_csv: Path) -> tuple[bool, str]:
    """Run MCMC prior derivation on a SubmodelTarget YAML. Returns (success, report)."""
    from maple.core.calibration.yaml_to_prior import format_report, process_yaml

    results = process_yaml(yaml_path, priors_csv=priors_csv)
    errors = [r for r in results if "error" in r]
    successes = [r for r in results if "error" not in r]

    lines = []
    for result in results:
        lines.append(format_report(result))
    report = "\n".join(lines)

    return (len(errors) == 0 and len(successes) > 0), report


def validate_snippets(yaml_path: Path, papers_dir: Path | None) -> tuple[bool, str]:
    """Run snippet-in-paper validation. Returns (success, report)."""
    from maple.core.calibration.snippet_validator import validate_snippets_in_file

    success, errors, skipped, passed, manual_review = validate_snippets_in_file(
        yaml_path, papers_dir
    )

    lines = []
    for p in passed:
        lines.append(f"  OK: {p}")
    for s in skipped:
        lines.append(f"  SKIP: {s}")
    for e in errors:
        lines.append(f"  FAIL: {e}")
    for mr in manual_review:
        lines.append(f"  MANUAL: {mr}")

    return success, "\n".join(lines)


def show_dois(results_path: Path) -> list[dict]:
    """Read lit search results, return candidates."""
    with open(results_path) as f:
        data = json.load(f)
    return data.get("candidates", [])


def write_dois_md(targets: list[dict], lit_results: list[dict], output_path: Path) -> None:
    """Write combined DOIs markdown file from lit search results."""
    lines = ["# Lit Search Results\n"]
    for t, lr in zip(targets, lit_results):
        lines.append(f"## {t['target_id']}\n")
        for c in lr.get("candidates", []):
            clean = "clean" if c.get("mapping_clean") else "needs mapping"
            lines.append(f"- [{c['title']}](https://doi.org/{c['doi']}) ({clean})")
            if c.get("mapping_concerns"):
                lines.append(f"  - {c['mapping_concerns']}")
        lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"  wrote {output_path}")


def summarize_digitizations(work_dir: Path, output_path: Path | None = None) -> None:
    """Print and optionally write a prioritized summary of all pending digitization requests."""
    items = []
    for assessment_path in sorted(work_dir.glob("*/assessment.json")):
        target = assessment_path.parent.name
        with open(assessment_path) as f:
            data = json.load(f)

        # Build DOI->title map from lit search results
        doi_to_title = {}
        lit_path = assessment_path.parent / "lit_search_results.json"
        if lit_path.exists():
            with open(lit_path) as f:
                lit_data = json.load(f)
            for c in lit_data.get("candidates", []):
                doi_to_title[c["doi"]] = c.get("title", "")

        # Determine which papers/digitizations are in the extraction plan
        plan = data.get("extraction_plan", {})
        plan_dois = set(plan.get("papers", []))
        plan_digs = set(plan.get("digitizations_needed", []))
        # Fallback for old-schema assessments
        if not plan_dois:
            best = data.get("best_paper", "")
            if best:
                plan_dois = {best}

        # Find PDF paths for this target
        papers_dir = assessment_path.parent / "papers"

        for p in data.get("papers", []):
            if not p.get("usable", False):
                continue
            in_plan = p["doi"] in plan_dois
            role = p.get("role", "")
            paper_title = doi_to_title.get(p["doi"], "")

            # Find the PDF for this paper by matching source_tag against filename
            pdf_link = ""
            title = p.get("title", "")
            tag = p.get("source_tag", "")
            # Extract author and year from source_tag (e.g., "Miller2019" -> "Miller", "2019")
            tag_match = re.match(r"([A-Za-z]+)(\d{4})", tag)
            tag_author = tag_match.group(1).lower() if tag_match else ""
            tag_year = tag_match.group(2) if tag_match else ""
            for search_dir in [papers_dir / tag, papers_dir]:
                if not search_dir.is_dir():
                    continue
                for pdf in search_dir.glob("*.pdf"):
                    name_lower = pdf.name.lower()
                    if (
                        tag_author
                        and tag_year
                        and tag_author in name_lower
                        and tag_year in name_lower
                    ):
                        pdf_link = str(pdf.resolve())
                        break
                if pdf_link:
                    break

            for d in p.get("data", []):
                if d.get("needs_digitization"):
                    dig_key = f"{p['source_tag']}/{d['location']}"
                    in_plan_dig = dig_key in plan_digs
                    items.append(
                        {
                            "target": target,
                            "source": p["source_tag"],
                            "title": paper_title,
                            "location": d["location"],
                            "desc": d["description"],
                            "priority": d.get("digitization_priority", "optional"),
                            "justification": d.get("digitization_justification", ""),
                            "hints": d.get("digitization_hints", ""),
                            "in_plan": in_plan,
                            "in_plan_dig": in_plan_dig,
                            "role": role,
                            "pdf_link": pdf_link,
                        }
                    )

    priority_order = {"critical": 0, "helpful": 1, "optional": 2}
    items.sort(
        key=lambda x: (
            not x["in_plan_dig"],  # plan-required digitizations first
            not x["in_plan"],  # then plan papers
            priority_order.get(x["priority"], 3),
            x["target"],
        )
    )

    # Split into plan-required vs other
    plan_items = [i for i in items if i["in_plan"]]
    other_items = [i for i in items if not i["in_plan"]]

    lines = []
    lines.append(f"# Digitization Summary\n")
    lines.append(f"**{len(items)} total** ({len(plan_items)} in extraction plan)\n")

    if plan_items:
        lines.append(f"## In Extraction Plan ({len(plan_items)})\n")
        for i in plan_items:
            is_required = i["in_plan_dig"] or i["priority"] == "critical"
            req = " **[REQUIRED]**" if is_required else ""
            title_str = f" — *{i['title']}*" if i["title"] else ""
            lines.append(
                f"- **[{i['target']}]** {i['source']}{title_str} {i['location']}{req} ({i['priority']})"
            )
            lines.append(f"  - {i['desc']}")
            if i["justification"]:
                lines.append(f"  - Why: {i['justification']}")
            if i["hints"]:
                lines.append(f"  - How: {i['hints']}")
            lines.append("")

    if other_items:
        lines.append(f"\n## Alternatives / Validation ({len(other_items)})\n")
        for priority in ["critical", "helpful", "optional"]:
            group = [i for i in other_items if i["priority"] == priority]
            if not group:
                continue
            lines.append(f"### {priority.upper()} ({len(group)})\n")
            for i in group:
                title_str = f" — *{i['title']}*" if i["title"] else ""
                lines.append(
                    f"- **[{i['target']}]** {i['source']}{title_str} ({i['role']}) {i['location']}"
                )
                lines.append(f"  - {i['desc']}")

    output = "\n".join(lines)
    print(output)

    if output_path is None:
        output_path = work_dir / "digitization_summary.md"
    output_path.write_text(output)
    print(f"\n  wrote {output_path}")

    # Collect unique PDFs for required digitizations and offer to open them
    required_items = [i for i in plan_items if i["in_plan_dig"] or i["priority"] == "critical"]
    required_pdfs = list(dict.fromkeys(i["pdf_link"] for i in required_items if i["pdf_link"]))
    if required_pdfs:
        import subprocess

        resp = (
            input(f"\n  Open {len(required_pdfs)} PDFs needing digitization in browser? [y/N] ")
            .strip()
            .lower()
        )
        if resp == "y":
            for pdf in required_pdfs:
                subprocess.run(["open", "-a", "Google Chrome", pdf])


def _zotero_pdf_path(doi: str, zotero_dir: Path) -> Path | None:
    """Look up a PDF path in Zotero's SQLite database by DOI."""
    import sqlite3

    db_path = zotero_dir / "zotero.sqlite"
    if not db_path.exists():
        return None

    query = """
        SELECT att_items.key, ia.path
        FROM itemData id
        JOIN itemDataValues idv ON id.valueID = idv.valueID
        JOIN fields f ON id.fieldID = f.fieldID
        JOIN items i ON id.itemID = i.itemID
        JOIN itemAttachments ia ON i.itemID = ia.parentItemID
        JOIN items att_items ON ia.itemID = att_items.itemID
        WHERE f.fieldName = 'DOI' AND LOWER(idv.value) = LOWER(?)
          AND ia.path LIKE 'storage:%.pdf'
        LIMIT 1
    """

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute(query, (doi,)).fetchone()
        conn.close()
    except sqlite3.Error:
        return None

    if not row:
        return None

    att_key, path_str = row
    filename = path_str.removeprefix("storage:")
    pdf_path = zotero_dir / "storage" / att_key / filename
    return pdf_path if pdf_path.exists() else None


def fetch_pdfs(candidates: list[dict], papers_dir: Path, zotero_storage: Path) -> list[dict]:
    """Copy PDFs from Zotero into papers_dir using DOI lookup. Returns NOT FOUND candidates."""
    papers_dir.mkdir(exist_ok=True)
    not_found = []
    zotero_dir = zotero_storage.parent  # ~/Zotero

    for c in candidates:
        doi = c["doi"]
        title = c.get("title", "")
        tag = c.get("source_tag", "")

        dest_dir = papers_dir / tag if tag else papers_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Look up by DOI in Zotero SQLite
        pdf_path = _zotero_pdf_path(doi, zotero_dir)
        if pdf_path:
            dest = dest_dir / pdf_path.name
            if dest.exists():
                print(f"  [{c['rank']}] SKIP: {pdf_path.name}")
            else:
                shutil.copy2(pdf_path, dest)
                print(f"  [{c['rank']}] COPIED: {pdf_path.name}")
        else:
            print(f"  [{c['rank']}] NOT FOUND: {title[:60]} ({doi})")
            not_found.append(c)

    return not_found


def collect_missing_pdfs(
    targets: list[dict],
    lit_results: list[dict],
    zotero_storage: Path,
) -> None:
    """Interactive PDF collection loop: fetch from Zotero → add by identifier → browser fallback."""
    import subprocess

    # Step 1: Try fetching from Zotero
    print("=" * 60)
    print("Stage 1c: Fetch PDFs from Zotero")
    print("=" * 60)

    all_missing = []
    for t, lr in zip(targets, lit_results):
        missing = fetch_pdfs(lr.get("candidates", []), t["dir"] / "papers", zotero_storage)
        all_missing.extend(missing)

    if not all_missing:
        print("\n  All PDFs found!")
        return

    # Step 2: Copy missing DOIs for "Add by Identifier"
    missing_dois = [c["doi"] for c in all_missing]
    unique_dois = sorted(set(missing_dois))
    print(
        f"\n  {len(unique_dois)} missing PDFs. DOIs copied to clipboard for Zotero 'Add by Identifier'."
    )
    subprocess.run(["pbcopy"], input="\n".join(unique_dois), text=True)

    while True:
        resp = (
            input(
                f"\n  [{len(unique_dois)} missing] Press Enter after adding to Zotero, 'b' to open in browser, or 's' to skip: "
            )
            .strip()
            .lower()
        )

        if resp == "s":
            break

        if resp == "b":
            # Open in browser for manual Zotero Connector download
            urls = [f"https://doi.org/{doi}" for doi in unique_dois]
            print(f"  Opening {len(urls)} DOIs in browser...")
            subprocess.run(["open"] + urls)
            input("  Press Enter after saving with Zotero Connector...")

        # Re-fetch from Zotero
        all_missing = []
        for t, lr in zip(targets, lit_results):
            missing = fetch_pdfs(lr.get("candidates", []), t["dir"] / "papers", zotero_storage)
            all_missing.extend(missing)

        if not all_missing:
            print("\n  All PDFs found!")
            return

        unique_dois = sorted(set(c["doi"] for c in all_missing))
        subprocess.run(["pbcopy"], input="\n".join(unique_dois), text=True)
        print(f"  {len(unique_dois)} still missing. DOIs copied to clipboard.")

    # Print final summary of missing papers
    if all_missing:
        print(f"\n  === {len(set(c['doi'] for c in all_missing))} papers still missing ===")
        seen = set()
        for c in all_missing:
            if c["doi"] not in seen:
                seen.add(c["doi"])
                print(f"  - {c.get('title', 'Unknown')[:80]}")
                print(f"    https://doi.org/{c['doi']}")


def make_model(model_name: str) -> OpenAIResponsesModel:
    return OpenAIResponsesModel(model_name)


def make_settings() -> OpenAIResponsesModelSettings:
    return OpenAIResponsesModelSettings(
        openai_reasoning_summary="concise",
        openai_reasoning_effort="medium",
    )


# ===========================================================================
# Prompts
# ===========================================================================

LIT_SEARCH_PROMPT = """\
You are a scientific literature search assistant for quantitative systems pharmacology (QSP) model calibration.

## Task

Find peer-reviewed papers that contain **quantitative experimental data** suitable for calibrating the following QSP model parameter(s):

**Parameters:** {parameters}

## Model Context

{model_context}

## Parameter Context (from model structure)

{parameter_context}

{notes_section}

## Critical: Search for data that matches the parameter's MODEL ROLE

Before searching, analyze the parameter context above to understand:

1. **What the parameter represents mechanistically** — not just its name, but how it
   appears in reaction rate laws and equations
2. **What units it uses** — standard (nM, cell/mL, 1/day) or model-internal
   (dimensionless fractions, ratios)
3. **What the Hill function / equation input variable is** — e.g., a parameter like
   `DAMP_50` gates on tumor death rate (cell/day), not DAMP concentration (nM).
   Search for data on the gating variable, not the parameter name.
4. **How it interacts with other parameters** — e.g., if the parameter is a fraction
   of another parameter (`ECM_50_APC_mig = f_ECM_50_APC_mig * ECM_max`), you need
   data on the composite quantity, not the fraction in isolation

Search for data that **directly constrains the parameter in its model role**. A parameter
that uses death rate as its Hill input needs death rate data, not concentration data.
A dimensionless fraction needs different literature than an absolute concentration.

## Requirements

1. Search for papers with **quantitative measurements** (not just qualitative descriptions)
2. Prefer **in vitro** or **preclinical** studies with clear experimental conditions
3. Look for papers reporting:
   - Time-course data (proliferation rates, decay curves, dose-response)
   - Summary statistics (mean, SD/SEM, sample size)
   - Figures with scatter plots, bar charts with error bars, or dose-response curves
   - Scatter plots with individual data points are especially valuable
4. Prefer papers where the data is **directly measurable** (not derived from complex modeling)
5. Avoid reviews, meta-analyses, and computational-only papers
6. **Every value must be traceable** — prefer papers where numbers appear in text, tables,
   or figure legends (not buried in inaccessible supplementary material)

## Multi-paper derivations

If no single paper directly reports the parameter value, it is valid to propose a
**multi-paper derivation** where different papers each contribute one piece of data
to an algebraic or steady-state constraint. For example, a trafficking rate might be
derived from an exit rate (paper A) combined with compartment cell counts (paper B)
via a steady-state equation. When proposing this:
- Still prefer a single paper if one exists with all needed data
- Explain the derivation strategy in `parameters_analyzed`
- For each candidate, note what specific piece it contributes (in `model_role_mapping`)
- The downstream extraction will combine inputs from multiple papers into one
  SubmodelTarget using `primary_data_source` and `secondary_data_sources`, each
  with their own `source_relevance` assessment

## Joint parameter constraining

A SubmodelTarget can constrain **multiple QSP parameters simultaneously** from a single
experimental system. For example, a time-course experiment with multiple conditions
(e.g., ±TGFb, ±cytokine) can jointly constrain interconversion rates, EC50s, and
proliferation rates by fitting a forward model ODE to the data. When the target
parameter is hard to measure in isolation, look for experimental systems where it
co-determines an observable alongside other model parameters — these are often more
informative than trying to find a direct single-parameter measurement. For each
candidate, populate the `jointly_constrainable_parameters` field with the names of
other QSP parameters (from the "Other parameters" section in the parameter context)
that the paper's data could also constrain.

## Important

If the literature data does not map cleanly to the model's parameterization, say so
explicitly. These parameters may need a different search strategy or may not be
extractable as direct submodel targets. Use the `unmappable_notes` field for this.

Begin with a brief analysis of what the parameter(s) represent mechanistically and
what data would constrain them, in the `parameters_analyzed` field.

Return 3-5 candidates.

## CRITICAL

You MUST perform web searches BEFORE returning your structured output. Do NOT return
an empty candidates list or a placeholder response. Your structured output is final —
there is no second chance to populate it. If your first searches don't find results,
try different search terms (synonyms, related concepts, broader queries). Only return
an empty candidates list if you have genuinely exhausted search strategies and confirmed
that no suitable quantitative data exists in the literature.
"""


ASSESS_PROMPT = """\
You are a scientific data assessment assistant for QSP model calibration.

## Task

Assess whether the attached paper(s) contain quantitative data suitable for calibrating these parameter(s):

**Parameters:** {parameters}
**Cancer type:** {cancer_type}

## Model Context

{model_context}

## Parameter Context

{parameter_context}

{prior_context_section}

{paper_text_section}

## Instructions

For each paper, determine:

1. **Does it have usable quantitative data?** (values, dose-response curves, time courses, etc.)
2. **Where is the data?** (table, text, or figure — be specific)
3. **Does the data map to the model parameter?** Flag any mismatch between what the paper
   measures and what the parameter represents in the model.
4. **Which specific data within the paper should be used?** Papers often contain data for
   multiple organs, conditions, or cell types. Identify the specific rows, columns, or
   conditions that best map to the model parameter. Use the prior median as a sanity
   check: do a rough back-of-envelope calculation to verify the data would produce a
   value in the right ballpark. If a paper has data for multiple organs/conditions and
   some produce values ~100x away from the prior while others are within ~10x, prefer
   the latter and explain why (e.g., vascular transit vs tissue residence).
5. **Does any figure data warrant digitization?** Only flag `needs_digitization` when a figure
   contains data that is **not available** in text or tables AND digitizing would provide
   meaningful added value (e.g., individual data points vs a reported mean, time-course
   resolution that tables collapse into a single estimate, dose-response shape not captured
   by reported EC50). If text/tables already report the key values with uncertainty, the
   figure is redundant — list it as a data source but do NOT flag it for digitization.
   When you do flag digitization, explain the specific added value in `digitization_justification`.
6. **What forward model would fit?** (exponential_growth, first_order_decay, algebraic, direct_fit, etc.)
7. **What is this paper's role?** Assign each paper a `role`:
   - `standalone`: contains all data needed for a complete derivation by itself
   - `required_for_derivation`: contributes one piece of a multi-paper derivation (all required papers are needed together)
   - `alternative`: independent alternative to another paper (pick the better one)
   - `validation_only`: useful for cross-checking but not part of any derivation

## Extraction Plan

After assessing all papers, construct an `extraction_plan`: the **minimal set of papers**
(and their specific figures/tables) needed for one complete derivation. Prefer:
- A single `standalone` paper over a multi-paper derivation
- Papers where data is in text/tables over those requiring digitization
- The fewest digitizations possible

List only the DOIs that are part of the plan, in the order they should be used, and
explain how they combine. Under `digitizations_needed`, list only the digitizations
that are **required** for this plan (source_tag/location format, e.g., "Obar2008/Figure 2B").

If there is a viable alternative plan (e.g., a different standalone paper, or a different
multi-paper combination), include it in `alternative_plans`.
"""


PLAN_REVIEW_PROMPT = """\
You are reviewing extraction plans for QSP model parameter calibration targets.

## Task

For each target below, evaluate whether the primary extraction plan is the best option,
or whether an alternative plan or a lit search rerun would be better. Consider:

1. **Data-parameter mapping**: Does the plan's data actually measure the target parameter?
   Flag if the plan uses a proxy (e.g., CD8 Ki67 for CD4 Th proliferation) when a direct
   measurement exists in an alternative.
2. **Sample size**: n=2 case reports are weak; prefer cohort studies (n>10).
3. **Cancer type relevance**: PDAC-specific data preferred, but larger non-PDAC datasets
   with direct measurements can be better than tiny PDAC datasets with indirect proxies.
4. **Digitization burden**: If an alt plan needs fewer digitizations for similar data quality,
   prefer it. Zero-dig standalone plans are ideal.
5. **Detection limits**: If the primary plan's key measurement is "below detection" or
   "undetectable," check if an alternative provides actual numeric values.
6. **Empty plans**: If the primary plan has no papers or says "none provide data," recommend
   rerun_lit_search with specific suggestions for what to search differently.

## Targets

{targets_json}

## Digitization Summary

{digitization_summary}

For each target, provide a verdict. Be specific about WHY you're recommending a switch
or rerun — what's wrong with the current plan and what's better about the alternative.
"""


DERIVATION_REVIEW_PROMPT = """\
You are reviewing completed SubmodelTarget derivations for QSP model parameter calibration.

## Task

Review each derivation below for scientific soundness, biological plausibility, and
mechanistic reliability. You are seeing ALL derivations together so you can check for
cross-target consistency.

## What to check per derivation

1. **Forward model appropriateness**: Does the ODE/algebraic model match the experimental
   system? Is it over-parameterized given the available data?
2. **Input data fidelity**: Are the extracted values (means, SDs, sample sizes) consistent
   with what the paper actually reports? Are units converted correctly?
3. **Biological plausibility**: Does the derived parameter value make mechanistic sense?
   Is it consistent with known biology (cell cycle times, diffusion limits, enzyme kinetics)?
4. **Assumptions and proxies**: Are proxy measurements (e.g., CD8 Ki67 for CD4 proliferation,
   dermal data for PDAC) acknowledged and reasonable? Would a different interpretation of
   the same data yield a very different parameter value?
5. **Derivation logic**: Is the algebraic chain from raw data to parameter value correct?
   Any sign of circular reasoning (assuming what you're trying to derive)?
6. **Source relevance**: Are `source_relevance` assessments (direct, analogous, approximate)
   accurate? Is the `translation_sigma` appropriate for the degree of extrapolation?

## What to check across derivations

1. **Contradictory assumptions**: Do two derivations assume different values for the same
   quantity (e.g., cell density, compartment volume)?
2. **Circular dependencies**: Does target A's derivation use target B's parameter as an
   input, and vice versa?
3. **Consistency of cell/tissue densities**: Do the implied cell densities, volumes, and
   concentrations across targets paint a coherent picture of the TME?
4. **Redundant constraints**: Are two targets constraining the same parameter from the
   same underlying data (just different papers reporting the same experiment)?

## Derivations

{derivations_json}

## Extraction Plans (for context)

{plans_json}

Be specific and constructive. For 'revise' verdicts, say exactly what should change.
"""


def get_prior_context(parameters: str, priors_csv: Path) -> str:
    """Look up current prior median and units for target parameters from the priors CSV."""
    param_names = [p.strip() for p in parameters.split(",")]
    rows = []
    with open(priors_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["name"] in param_names:
                rows.append(row)
    if not rows:
        return ""
    lines = [
        "## Current Prior Values (sanity check)\n",
        "The following are the current prior medians from the QSP model. Your derived",
        "posterior should be in the same order of magnitude unless you have strong",
        "evidence otherwise. If your derivation produces values that differ by >10x",
        "from the prior median, pause and verify that you are using the right data",
        "(e.g., tissue residence times vs vascular transit times, correct organ, etc.).\n",
    ]
    for r in rows:
        lines.append(f"- **{r['name']}**: median={r['median']} {r['units']} ({r['distribution']})")
    return "\n".join(lines)


def build_complete_prompt(
    parameters: str,
    cancer_type: str,
    target_id: str,
    model_context: str,
    parameter_context: str,
    assessment_json: str,
    digitized_data_section: str,
    priors_csv: Path,
    plan_review_reason: str = "",
) -> str:
    """Build stage 3 prompt using the canonical maple submodel target prompt."""
    from maple.core.prompts import build_submodel_target_prompt

    # Use maple's canonical extraction prompt as the base
    base_prompt = build_submodel_target_prompt(
        parameters=parameters,
        model_context=model_context,
        parameter_context=parameter_context,
        notes=f"Target ID: {target_id}\nCancer type: {cancer_type}",
    )

    # Extract the extraction plan for a focused instruction
    assessment = json.loads(assessment_json)
    plan = assessment.get("extraction_plan", {})

    extra = ""

    if plan and plan.get("papers"):
        extra += "\n\n## Extraction Plan (FOLLOW THIS)\n\n"
        extra += "You MUST follow this extraction plan. Use the specified papers and strategy.\n"
        extra += "Do NOT substitute different papers or invent a different derivation approach.\n\n"
        extra += f"**Papers:** {', '.join(plan['papers'])}\n"
        extra += f"**Strategy:** {plan.get('strategy', '')}\n"
        if plan.get("digitizations_needed"):
            extra += f"**Required digitizations:** {', '.join(plan['digitizations_needed'])}\n"
        if plan_review_reason:
            extra += f"\n**Why this plan was chosen:** {plan_review_reason}\n"

    # Append prior sanity check, full assessment, and digitized data
    extra += "\n\n" + get_prior_context(parameters, priors_csv)
    extra += f"\n\n## Paper Assessment (from prior stage)\n\n{assessment_json}"
    if digitized_data_section:
        extra += f"\n{digitized_data_section}"

    return base_prompt + extra


# ===========================================================================
# Agents (created once per config, reused across targets)
# ===========================================================================


def make_agents(model_name: str, max_retries: int) -> tuple[Agent, Agent, Agent, Agent, Agent]:
    """Create the five pipeline agents. Returns (lit_search, assess, plan_review, complete, derivation_review)."""
    openai_model = make_model(model_name)
    settings = make_settings()

    lit_search_settings = OpenAIResponsesModelSettings(
        openai_reasoning_summary="concise",
        openai_reasoning_effort="high",
    )

    lit_search_agent = Agent(
        openai_model,
        output_type=LitSearchResult,
        system_prompt=(
            "You are a scientific literature search agent. "
            "You MUST use the web_search tool multiple times to find papers BEFORE returning your final structured output. "
            "Your structured output is FINAL — there is no second call. "
            "NEVER return an empty candidates list or a placeholder. "
            "If your first searches find nothing, try synonyms, broader terms, or related biological concepts."
        ),
        model_settings=lit_search_settings,
        builtin_tools=[WebSearchTool()],
        retries=max_retries,
    )

    assess_agent = Agent(
        openai_model,
        output_type=AssessmentResult,
        model_settings=settings,
        retries=max_retries,
    )

    plan_review_settings = OpenAIResponsesModelSettings(
        openai_reasoning_summary="concise",
        openai_reasoning_effort="high",
    )

    plan_review_agent = Agent(
        openai_model,
        output_type=PlanReviewResult,
        model_settings=plan_review_settings,
        retries=max_retries,
    )

    complete_agent = Agent(
        openai_model,
        output_type=SubmodelTarget,
        model_settings=settings,
        retries=max_retries,
    )

    derivation_review_agent = Agent(
        openai_model,
        output_type=DerivationReviewResult,
        model_settings=plan_review_settings,  # also use high reasoning effort
        retries=max_retries,
    )

    return (
        lit_search_agent,
        assess_agent,
        plan_review_agent,
        complete_agent,
        derivation_review_agent,
    )


# ===========================================================================
# Pipeline orchestration
# ===========================================================================


async def run_stage(stage_fn, targets_list, label):
    """Run an async stage function across targets using as_completed."""
    tasks = {asyncio.ensure_future(stage_fn(t)): t for t in targets_list}
    results = []
    for coro in asyncio.as_completed(tasks.keys()):
        try:
            results.append(await coro)
        except Exception as e:
            # Find which target failed
            failed_target = "unknown"
            for fut, t in tasks.items():
                if fut is coro or (fut.done() and fut.exception() is not None):
                    failed_target = t.get("target_id", "unknown")
                    break
            print(f"  [{failed_target}] ERROR: {type(e).__name__}: {str(e)[:150]}")
            results.append(None)
    return results


async def run_plan_review(
    targets: list[dict],
    work_dir: Path,
    plan_review_agent: Agent,
    targets_csv: Path,
) -> PlanReviewResult | None:
    """Stage 2b: Review all extraction plans and recommend actions."""
    review_path = work_dir / "plan_review.json"

    if review_path.exists():
        print("  CACHED plan review")
        with open(review_path) as f:
            return PlanReviewResult.model_validate(json.load(f))

    # Build per-target summaries for the prompt
    target_summaries = []
    for t in targets:
        assessment_path = t["dir"] / "assessment.json"
        if not assessment_path.exists():
            continue
        with open(assessment_path) as f:
            assessment = json.load(f)

        summary = {
            "target_id": t["target_id"],
            "parameters": t["parameters"],
            "extraction_plan": assessment.get("extraction_plan", {}),
            "alternative_plans": assessment.get("alternative_plans", []),
            "overall_notes": assessment.get("overall_notes", ""),
            "papers": [],
        }
        for p in assessment.get("papers", []):
            summary["papers"].append(
                {
                    "source_tag": p["source_tag"],
                    "doi": p["doi"],
                    "role": p.get("role", ""),
                    "usable": p["usable"],
                    "summary": p["summary"],
                    "mapping_concerns": p.get("mapping_concerns", ""),
                    "digitizations": [
                        {
                            "location": d["location"],
                            "priority": d.get("digitization_priority", "optional"),
                            "description": d["description"][:120],
                        }
                        for d in p.get("data", [])
                        if d.get("needs_digitization")
                    ],
                }
            )
        target_summaries.append(summary)

    if not target_summaries:
        print("  No assessments to review")
        return None

    # Read digitization summary if it exists
    dig_summary_path = work_dir / "digitization_summary.md"
    dig_summary = dig_summary_path.read_text() if dig_summary_path.exists() else ""

    prompt = PLAN_REVIEW_PROMPT.format(
        targets_json=json.dumps(target_summaries, indent=2),
        digitization_summary=dig_summary,
    )

    result = await plan_review_agent.run(prompt)
    review = result.output

    with open(review_path, "w") as f:
        json.dump(review.model_dump(), f, indent=2)

    # Apply actions
    actions = {"proceed": 0, "switch_to_alt": 0, "rerun_lit_search": 0, "defer": 0}
    for r in review.reviews:
        actions[r.verdict] += 1

        if r.verdict == "switch_to_alt" and r.replacement_plan:
            target_dir = work_dir / r.target
            assessment_path = target_dir / "assessment.json"
            if assessment_path.exists():
                with open(assessment_path) as f:
                    assessment = json.load(f)
                old_plan = assessment["extraction_plan"]
                assessment["extraction_plan"] = r.replacement_plan.model_dump()
                # Move old plan to alternatives
                alts = assessment.get("alternative_plans", [])
                alts.append(old_plan)
                assessment["alternative_plans"] = alts
                with open(assessment_path, "w") as f:
                    json.dump(assessment, f, indent=2)
                plan_tags = ", ".join(r.replacement_plan.papers[:3])
                print(f"  [{r.target}] SWITCHED to: {plan_tags}")

        elif r.verdict == "rerun_lit_search":
            # Delete lit search and assessment caches
            target_dir = work_dir / r.target
            for cache_file in ["lit_search_results.json", "assessment.json", "assessment.md"]:
                (target_dir / cache_file).unlink(missing_ok=True)
            # Append notes to targets.csv
            if r.lit_search_notes and targets_csv.exists():
                rows = []
                with open(targets_csv, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    for row in reader:
                        if row["target_id"] == r.target:
                            existing = row.get("notes", "")
                            row["notes"] = (
                                f"{existing} RERUN: {r.lit_search_notes}"
                                if existing
                                else f"RERUN: {r.lit_search_notes}"
                            )
                        rows.append(row)
                with open(targets_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"  [{r.target}] RERUN lit search: {r.lit_search_notes[:80]}")

        elif r.verdict == "defer":
            print(f"  [{r.target}] DEFERRED: {r.reason[:80]}")

    print(
        f"\n  Plan review: {actions['proceed']} proceed, {actions['switch_to_alt']} switched, "
        f"{actions['rerun_lit_search']} rerun, {actions['defer']} deferred"
    )
    print(f"\n  {review.summary}")

    return review


# ===========================================================================
# Stage functions (one target each)
# ===========================================================================


async def run_lit_search(
    t: dict, *, lit_search_agent: Agent, model_context: str, model_structure_path: Path
) -> dict:
    """Stage 1: Literature search for one target."""
    target_dir = t["dir"]
    lit_search_path = target_dir / "lit_search_results.json"

    if lit_search_path.exists():
        print(f"  [{t['target_id']}] CACHED lit search")
        with open(lit_search_path) as f:
            return json.load(f)

    parameter_context = build_parameter_context(t["parameters"], model_structure_path)
    notes_section = f"## Additional Notes\n\n{t['notes']}" if t["notes"] else ""

    prompt = LIT_SEARCH_PROMPT.format(
        parameters=t["parameters"],
        model_context=model_context,
        parameter_context=parameter_context,
        notes_section=notes_section,
    )

    result = await lit_search_agent.run(prompt)
    result_dict = result.output.model_dump()

    with open(lit_search_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    n = len(result_dict.get("candidates", []))
    print(f"  [{t['target_id']}] {n} candidates")
    return result_dict


async def run_assess(
    t: dict,
    *,
    assess_agent: Agent,
    model_context: str,
    model_structure_path: Path,
    priors_csv: Path,
) -> dict | None:
    """Stage 2: Assess papers for one target."""
    target_dir = t["dir"]
    assessment_path = target_dir / "assessment.json"

    if assessment_path.exists():
        print(f"  [{t['target_id']}] CACHED assessment")
        with open(assessment_path) as f:
            return json.load(f)

    file_parts, text_parts = collect_papers(target_dir / "papers")
    if not file_parts and not text_parts:
        print(f"  [{t['target_id']}] no papers, skipping")
        return None

    parameter_context = build_parameter_context(t["parameters"], model_structure_path)
    prior_context_section = get_prior_context(t["parameters"], priors_csv)
    paper_text_section = ""
    if text_parts:
        paper_text_section = "## Paper Content (text)\n\n" + "\n\n".join(text_parts)

    prompt = ASSESS_PROMPT.format(
        parameters=t["parameters"],
        cancer_type=t["cancer_type"],
        model_context=model_context,
        parameter_context=parameter_context,
        prior_context_section=prior_context_section,
        paper_text_section=paper_text_section,
    )

    user_prompt: list = list(file_parts) + [prompt]
    result = await assess_agent.run(user_prompt)
    assessment = result.output.model_dump()

    with open(assessment_path, "w") as f:
        json.dump(assessment, f, indent=2)

    print(f"  [{t['target_id']}] assessed {len(assessment.get('papers', []))} papers")
    return assessment


async def run_complete(
    t: dict,
    *,
    complete_agent: Agent,
    model_context: str,
    model_structure_path: Path,
    priors_csv: Path,
    work_dir: Path,
) -> Path | None:
    """Stage 3: Assemble SubmodelTarget YAML."""
    target_dir = t["dir"]
    output_file = target_dir / f"{t['target_id']}_{t['cancer_type']}_deriv001.yaml"
    assessment_path = target_dir / "assessment.json"

    if output_file.exists():
        print(f"  [{t['target_id']}] CACHED deriv YAML")
        return output_file
    if not assessment_path.exists():
        print(f"  [{t['target_id']}] no assessment, skipping")
        return None

    with open(assessment_path) as f:
        assessment_data = json.load(f)

    # Look up plan review reason for this target
    plan_review_reason = ""
    plan_review_path = work_dir / "plan_review.json"
    if plan_review_path.exists():
        with open(plan_review_path) as f:
            plan_review_data = json.load(f)
        for r in plan_review_data.get("reviews", []):
            if r["target"] == t["target_id"]:
                plan_review_reason = r.get("reason", "")
                break

    parameter_context = build_parameter_context(t["parameters"], model_structure_path)

    digitized_data = read_digitized_data(target_dir / "digitized")
    digitized_section = f"\n## Digitized Figure Data\n\n{digitized_data}" if digitized_data else ""

    # Only send PDFs for papers in the extraction plan to avoid context overflow
    plan = assessment_data.get("extraction_plan", {})
    plan_dois = set(plan.get("papers", []))

    # Build DOI-to-source_tag mapping from assessment
    doi_to_tag = {}
    for p in assessment_data.get("papers", []):
        doi_to_tag[p["doi"]] = p.get("source_tag", "")
    plan_tags = {doi_to_tag.get(doi, "") for doi in plan_dois} - {""}

    # Collect only plan papers (check tag subdirs first, then flat dir)
    papers_dir = target_dir / "papers"
    file_parts = []
    text_parts = []
    if papers_dir.exists():
        for tag in plan_tags:
            tag_dir = papers_dir / tag
            if tag_dir.is_dir():
                parts, texts = collect_papers(tag_dir)
                file_parts.extend(parts)
                text_parts.extend(texts)
        # Also check flat papers dir, matching by source_tag in filename
        for pdf in sorted(papers_dir.glob("*.pdf")):
            name_lower = pdf.name.lower()
            for tag in plan_tags:
                # Extract author and year from tag
                tag_match = re.match(r"([A-Za-z]+)(\d{4})", tag)
                if tag_match:
                    author = tag_match.group(1).lower()
                    year = tag_match.group(2)
                    if author in name_lower and year in name_lower:
                        size_mb = pdf.stat().st_size / (1024 * 1024)
                        if size_mb <= 45:
                            file_parts.append(pdf_to_binary(pdf))
                        break

    paper_text_section = ""
    if text_parts:
        paper_text_section = "\n## Paper Content (text)\n\n" + "\n\n".join(text_parts)

    prompt = build_complete_prompt(
        parameters=t["parameters"],
        cancer_type=t["cancer_type"],
        target_id=t["target_id"],
        model_context=model_context,
        parameter_context=parameter_context,
        assessment_json=json.dumps(assessment_data, indent=2),
        digitized_data_section=digitized_section + paper_text_section,
        priors_csv=priors_csv,
        plan_review_reason=plan_review_reason,
    )

    user_prompt: list = list(file_parts) + [prompt]
    result = await complete_agent.run(user_prompt)

    target_data = result.output.model_dump(mode="json", exclude_none=True)
    ms = ModelStructure.from_json(model_structure_path)
    SubmodelTarget.model_validate(
        target_data, context={"model_structure": ms, "papers_dir": papers_dir}
    )

    with open(output_file, "w") as f:
        yaml.dump(target_data, f, default_flow_style=False, sort_keys=False)

    print(f"  [{t['target_id']}] -> {output_file.name}")
    return output_file


async def run_derivation_review(
    targets: list[dict],
    work_dir: Path,
    derivation_review_agent: Agent,
) -> DerivationReviewResult | None:
    """Stage 3b: Review all completed derivations for scientific soundness."""
    review_path = work_dir / "derivation_review.json"

    if review_path.exists():
        print("  CACHED derivation review")
        with open(review_path) as f:
            return DerivationReviewResult.model_validate(json.load(f))

    # Collect all completed derivations and their extraction plans
    derivations = {}
    plans = {}
    for t in targets:
        target_dir = t["dir"]
        yaml_file = target_dir / f"{t['target_id']}_{t['cancer_type']}_deriv001.yaml"
        if yaml_file.exists():
            with open(yaml_file) as f:
                derivations[t["target_id"]] = yaml.safe_load(f)

        assessment_path = target_dir / "assessment.json"
        if assessment_path.exists():
            with open(assessment_path) as f:
                assessment = json.load(f)
            plans[t["target_id"]] = assessment.get("extraction_plan", {})

    if not derivations:
        print("  No derivations to review")
        return None

    prompt = DERIVATION_REVIEW_PROMPT.format(
        derivations_json=json.dumps(derivations, indent=2),
        plans_json=json.dumps(plans, indent=2),
    )

    result = await derivation_review_agent.run(prompt)
    review = result.output

    with open(review_path, "w") as f:
        json.dump(review.model_dump(), f, indent=2)

    # Print results
    actions = {"accept": 0, "revise": 0, "reject": 0}
    for r in review.reviews:
        actions[r.verdict] += 1
        if r.verdict != "accept":
            print(f"  [{r.target}] {r.verdict.upper()}")
            for c in r.concerns:
                print(f"    - {c[:120]}")
            if r.suggested_fix:
                print(f"    Fix: {r.suggested_fix[:120]}")

    if review.cross_target_issues:
        print(f"\n  Cross-target issues:")
        for issue in review.cross_target_issues:
            print(f"    - {issue[:150]}")

    print(
        f"\n  Derivation review: {actions['accept']} accept, {actions['revise']} revise, {actions['reject']} reject"
    )
    print(f"\n  {review.summary}")

    # Write markdown report
    report_lines = ["# Derivation Review\n"]
    for r in review.reviews:
        icon = {"accept": "pass", "revise": "warn", "reject": "fail"}[r.verdict]
        report_lines.append(f"## [{icon}] {r.target}\n")
        if r.concerns:
            for c in r.concerns:
                report_lines.append(f"- {c}")
        if r.suggested_fix:
            report_lines.append(f"\n**Fix:** {r.suggested_fix}")
        report_lines.append("")

    if review.cross_target_issues:
        report_lines.append("## Cross-Target Issues\n")
        for issue in review.cross_target_issues:
            report_lines.append(f"- {issue}")
        report_lines.append("")

    report_lines.append(f"## Summary\n\n{review.summary}")
    (work_dir / "derivation_review.md").write_text("\n".join(report_lines))
    print(f"  wrote {work_dir / 'derivation_review.md'}")

    return review


def run_validate(t: dict, *, model_structure_path: Path, priors_csv: Path) -> None:
    """Stage 3b: Validate a completed SubmodelTarget (synchronous — MCMC is CPU-bound)."""
    target_dir = t["dir"]
    output_file = target_dir / f"{t['target_id']}_{t['cancer_type']}_deriv001.yaml"

    if not output_file.exists():
        return

    # Skip if already promoted to calibration_targets/
    dest = Path("calibration_targets/submodel_targets") / output_file.name
    if dest.exists():
        print(f"  [{t['target_id']}] SKIP (already in {dest.parent.name}/)")
        return

    # Unit validation
    try:
        ms = ModelStructure.from_json(model_structure_path)
        with open(output_file) as f:
            data = yaml.safe_load(f)
        SubmodelTarget.model_validate(data, context={"model_structure": ms})
        unit_ok = True
    except Exception as e:
        print(f"  [{t['target_id']}] Unit FAIL: {e}")
        unit_ok = False

    # Prior derivation
    try:
        prior_ok, prior_report = validate_prior_derivation(output_file, priors_csv)
    except Exception as e:
        print(f"  [{t['target_id']}] Prior ERROR: {e}")
        prior_ok, prior_report = False, str(e)

    # Snippet validation
    try:
        snippet_ok, snippet_report = validate_snippets(output_file, target_dir / "papers")
    except Exception as e:
        print(f"  [{t['target_id']}] Snippet ERROR: {e}")
        snippet_ok, snippet_report = False, str(e)

    all_pass = unit_ok and prior_ok and snippet_ok
    status = "PASS" if all_pass else "FAIL"
    print(
        f"  [{t['target_id']}] {status} (unit={'OK' if unit_ok else 'FAIL'} prior={'OK' if prior_ok else 'FAIL'} snippet={'OK' if snippet_ok else 'FAIL'})"
    )

    val_report = target_dir / "validation_report.txt"
    with open(val_report, "w") as f:
        f.write(f"Target: {t['target_id']}\nOutput: {output_file}\n\n")
        f.write(f"Unit validation: {'PASS' if unit_ok else 'FAIL'}\n\n")
        f.write(f"Prior derivation: {'PASS' if prior_ok else 'FAIL'}\n{prior_report}\n\n")
        f.write(f"Snippet validation: {'PASS' if snippet_ok else 'FAIL'}\n{snippet_report}\n")

    # Copy passing targets to submodel_targets/
    if all_pass:
        dest_dir = Path("calibration_targets/submodel_targets")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / output_file.name
        shutil.copy2(output_file, dest)
        print(f"  [{t['target_id']}] -> {dest}")


def write_assessment_report(t: dict, assessment: dict) -> None:
    """Write assessment.md and digitization READMEs for one target."""
    target_dir = t["dir"]
    lines = [f"# Paper Assessment: {t['target_id']}\n"]

    # Extraction plan
    plan = assessment.get("extraction_plan", {})
    if plan:
        lines.append("## Extraction Plan\n")
        lines.append(f"**Papers:** {', '.join(plan.get('papers', []))}")
        lines.append(f"**Strategy:** {plan.get('strategy', '')}")
        if plan.get("digitizations_needed"):
            lines.append(f"**Required digitizations:** {', '.join(plan['digitizations_needed'])}")
        lines.append("")

    alt_plans = assessment.get("alternative_plans", [])
    if alt_plans:
        lines.append("## Alternative Plans\n")
        for i, alt in enumerate(alt_plans, 1):
            lines.append(f"**Plan {i}:** {', '.join(alt.get('papers', []))}")
            lines.append(f"- {alt.get('strategy', '')}")
            if alt.get("digitizations_needed"):
                lines.append(f"- Digitizations: {', '.join(alt['digitizations_needed'])}")
        lines.append("")

    # Per-paper details
    for p in assessment["papers"]:
        status = "USABLE" if p["usable"] else "SKIP"
        role = p.get("role", "")
        role_str = f" ({role})" if role else ""
        lines.append(f"## [{status}] {p['source_tag']}{role_str}")
        lines.append(f"- DOI: {p['doi']}")
        lines.append(f"- {p['summary']}")
        if p.get("mapping_concerns"):
            lines.append(f"- **Concerns:** {p['mapping_concerns']}")
        if p.get("forward_model_suggestion"):
            lines.append(f"- **Forward model:** {p['forward_model_suggestion']}")
        if p.get("jointly_constrainable_parameters"):
            lines.append(f"- **Joint params:** {', '.join(p['jointly_constrainable_parameters'])}")
        for d in p.get("data", []):
            if d.get("needs_digitization"):
                pri = d.get("digitization_priority", "")
                pri_str = f" [{pri}]" if pri else ""
                lines.append(f"- **DIGITIZE{pri_str} {d['location']}:** {d['description']}")
                if d.get("digitization_justification"):
                    lines.append(f"  - **Why digitize:** {d['digitization_justification']}")
                if d.get("digitization_hints"):
                    lines.append(f"  - {d['digitization_hints']}")
            else:
                lines.append(f"- {d['location']} ({d['data_type']}): {d['description']}")
        lines.append("")

    # Fallback for old schema
    if not plan and assessment.get("best_paper"):
        lines.append(f"**Best paper:** {assessment['best_paper']}")
    if assessment.get("overall_notes"):
        lines.append(f"\n{assessment['overall_notes']}")

    (target_dir / "assessment.md").write_text("\n".join(lines))
    generate_digitization_readme(assessment, target_dir / "digitized")
