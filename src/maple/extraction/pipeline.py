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
from pydantic_ai.providers.openai import OpenAIProvider

from maple.core.calibration.submodel_target import SubmodelTarget
from maple.core.calibration.calibration_target_models import CalibrationTarget
from maple.core.model_structure import ModelStructure


TargetKind = Literal["submodel", "cal"]


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


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pdftotext (poppler), falling back to pypdf."""
    import shutil
    import subprocess as _sp

    if shutil.which("pdftotext"):
        try:
            result = _sp.run(
                ["pdftotext", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception:
            pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


def collect_papers(
    papers_dir: Path, *, text_only: bool = False
) -> tuple[list[BinaryContent], list[str]]:
    """Collect papers from a directory as binary PDFs and/or extracted text.

    Args:
        papers_dir: Directory containing PDFs (flat or in subdirs).
        text_only: If True, extract text from PDFs instead of sending binary.
            Avoids context overflow with large PDFs. Loses figure images but
            retains tables and captions.
    """
    if not papers_dir.exists():
        return [], []

    file_parts: list[BinaryContent] = []
    text_parts: list[str] = []

    for pdf in sorted(papers_dir.glob("**/*.pdf")):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        if size_mb > 45:
            print(f"    Warning: {pdf.name} is {size_mb:.1f}MB, skipping (>45MB)")
            continue

        if text_only:
            text = _extract_pdf_text(pdf)
            if text.strip():
                text_parts.append(f"--- {pdf.name} ---\n{text}\n")
        else:
            file_parts.append(pdf_to_binary(pdf))

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


def report_digitization_preflight(targets: list[dict]) -> None:
    """Report which targets have pending vs complete digitizations before stage 3.

    Prints a summary of targets missing required digitizations and those ready
    to proceed. Does not block extraction — informational only.
    """
    print("\n" + "=" * 60)
    print("Stage 3 Pre-flight: Digitization Status")
    print("=" * 60)

    missing_digs = []
    ready = []
    for t in targets:
        target_dir = t["dir"]
        assess_path = target_dir / "assessment.json"
        output_file = target_dir / f"{t['target_id']}_{t['cancer_type']}_deriv001.yaml"
        if output_file.exists():
            ready.append((t["target_id"], "cached"))
            continue
        if not assess_path.exists():
            continue
        with open(assess_path) as f:
            assess_data = json.load(f)
        plan = assess_data.get("extraction_plan", {})
        if not plan.get("papers"):
            continue
        required = plan.get("digitizations_needed", [])
        if not required:
            ready.append((t["target_id"], "no digitization needed"))
            continue
        digitized_dir = target_dir / "digitized"
        missing = []
        for dig_spec in required:
            tag = dig_spec.split("/", 1)[0].strip()
            tag_dir = digitized_dir / tag
            if not tag_dir.is_dir() or not list(tag_dir.glob("*.csv")):
                missing.append(dig_spec)
        if missing:
            missing_digs.append((t["target_id"], missing))
        else:
            ready.append((t["target_id"], "digitizations complete"))

    if missing_digs:
        print(f"\n  MISSING DIGITIZATIONS ({len(missing_digs)} targets):")
        for tid, missing in missing_digs:
            print(f"    {tid}: {', '.join(missing)}")

    print(f"\n  READY ({len(ready)} targets)")
    print()


def summarize_digitizations(
    work_dir: Path,
    output_path: Path | None = None,
    target_ids: list[str] | set[str] | None = None,
) -> None:
    """Print and optionally write a prioritized summary of all pending digitization requests.

    If ``target_ids`` is provided, only assessments whose directory name appears
    in that collection are included. Pass the current run's target IDs to skip
    stale work-dir leftovers from prior runs.
    """
    items = []
    promoted_dir = Path("calibration_targets/submodel_targets")
    target_filter = set(target_ids) if target_ids is not None else None
    for assessment_path in sorted(work_dir.glob("*/assessment.json")):
        target = assessment_path.parent.name
        if target_filter is not None and target not in target_filter:
            continue
        # Skip targets already promoted
        if promoted_dir.is_dir() and list(promoted_dir.glob(f"{target}_*_deriv*.yaml")):
            continue
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

            # Check digitized dir for per-figure completion
            digitized_dir = assessment_path.parent / "digitized"
            tag_dig_dir = digitized_dir / tag

            for d in p.get("data", []):
                if d.get("needs_digitization"):
                    dig_key = f"{p['source_tag']}/{d['location']}"
                    in_plan_dig = dig_key in plan_digs

                    # Check if a CSV matching any figure identifier exists
                    done = False
                    if tag_dig_dir.is_dir():
                        # Extract figure IDs from location
                        # e.g. "Figure 7A,B" -> check for "7a" or "7b"
                        # "Fig S1H–S1K" -> check for "s1h" or "s1k"
                        loc = d["location"].lower()
                        for prefix in (
                            "supplementary figure ",
                            "supplementary fig. ",
                            "supplementary fig ",
                            "supplementary ",
                            "figure ",
                            "fig. ",
                            "fig ",
                        ):
                            loc = loc.replace(prefix, "")
                        # Split on delimiters
                        parts = [p.strip() for p in re.split(r"[–\-,;&]\s*", loc) if p.strip()]
                        # Expand single-letter suffixes using the base from the first part
                        # e.g. ["7a", "b"] -> ["7a", "7b"]
                        fig_ids = []
                        base = ""
                        for part in parts:
                            if re.match(r".*\d", part):
                                # Has digits — this is a full ID like "7a" or "s1h"
                                base = re.match(r"(.*\d)", part).group(1)
                                fig_ids.append(part)
                            elif len(part) <= 2 and base:
                                # Single letter suffix like "b" — prepend base
                                fig_ids.append(base + part)
                            else:
                                fig_ids.append(part)

                        csv_names = [f.name.lower() for f in tag_dig_dir.glob("*.csv")]
                        for fig_id in fig_ids:
                            if fig_id and any(fig_id in name for name in csv_names):
                                done = True
                                break

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
                            "done": done,
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

    plan_done = sum(1 for i in plan_items if i.get("done"))
    plan_pending = len(plan_items) - plan_done

    lines = []
    lines.append(f"# Digitization Summary\n")
    lines.append(
        f"**{len(items)} total** ({len(plan_items)} in extraction plan: "
        f"{plan_done} done, {plan_pending} pending)\n"
    )

    if plan_items:
        # Show pending first, then done
        pending = [i for i in plan_items if not i.get("done")]
        done = [i for i in plan_items if i.get("done")]

        if pending:
            critical = [i for i in pending if i["in_plan_dig"] or i["priority"] == "critical"]
            other_pending = [i for i in pending if i not in critical]

            def _fmt_pending(items_list: list) -> None:
                for i in items_list:
                    title_str = f" — *{i['title']}*" if i["title"] else ""
                    lines.append(
                        f"- **[{i['target']}]** {i['source']}{title_str} {i['location']} ({i['priority']})"
                    )
                    lines.append(f"  - {i['desc']}")
                    if i["justification"]:
                        lines.append(f"  - Why: {i['justification']}")
                    if i["hints"]:
                        lines.append(f"  - How: {i['hints']}")
                    lines.append("")

            if critical:
                lines.append(f"## Pending — Critical ({len(critical)})\n")
                _fmt_pending(critical)

            if other_pending:
                lines.append(f"## Pending — Helpful / Optional ({len(other_pending)})\n")
                _fmt_pending(other_pending)

        if done:
            lines.append(f"## Done ({len(done)})\n")
            for i in done:
                title_str = f" — *{i['title']}*" if i["title"] else ""
                lines.append(
                    f"- ~~**[{i['target']}]** {i['source']}{title_str} {i['location']}~~ ✓"
                )
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

    # immutable=1 (not mode=ro): read the DB file directly, ignoring locks.
    # Zotero 7+ holds an aggressive WAL lock while running, so a mode=ro
    # connection stalls on the busy-timeout and then raises "database is
    # locked" — silently caught below and misreported as NOT FOUND. immutable
    # reads the bytes without acquiring a lock (a torn/stale snapshot is
    # acceptable for a "does this DOI have a stored PDF" check).
    try:
        conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
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


#: When False, the browser-open step skips PMC and routes every DOI through
#: doi.org (publisher). PMC's per-IP bot protection 403s the browser after
#: enough article tabs are opened in a short window; the publisher landing page
#: avoids NCBI throttling and the Zotero Connector still saves from it. Flip to
#: True to prefer PMC OA full text when you aren't opening papers in bulk.
PREFER_PMC = False


def _resolve_pmc_urls(dois: list[str]) -> list[str]:
    """Resolve DOIs to PMC full-text URLs where available, else doi.org."""
    import urllib.request

    if not PREFER_PMC:
        return [f"https://doi.org/{doi}" for doi in dois]

    urls = []
    for doi in dois:
        try:
            api_url = (
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                f"?query=DOI:{doi}&format=json&resultType=lite"
            )
            with urllib.request.urlopen(api_url, timeout=5) as resp:
                data = json.loads(resp.read())
            results = data.get("resultList", {}).get("result", [])
            pmcid = results[0].get("pmcid") if results else None
            if pmcid:
                urls.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
                continue
        except Exception:
            pass
        urls.append(f"https://doi.org/{doi}")
    return urls


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

    # Step 2: Walk the missing DOIs in small batches. Each batch is copied to
    # the clipboard for Zotero 'Add by Identifier' AND opened in the browser
    # (PMC full text where available, doi.org otherwise); press Enter to advance
    # to the next batch. Dumping all ~80 DOIs into the clipboard / browser at
    # once is unmanageable — 5 at a time matches how Zotero's add-by-identifier
    # and the Connector are actually used. After a pass, re-fetch from Zotero
    # and repeat on whatever is still missing.
    BATCH_SIZE = 5
    unique_dois = sorted(set(c["doi"] for c in all_missing))

    while True:
        print(f"\n  {len(unique_dois)} missing PDFs — walking in batches of {BATCH_SIZE}.")
        batches = [unique_dois[i : i + BATCH_SIZE] for i in range(0, len(unique_dois), BATCH_SIZE)]
        stopped = False
        for bi, batch in enumerate(batches, 1):
            subprocess.run(["pbcopy"], input="\n".join(batch), text=True)
            urls = _resolve_pmc_urls(batch)
            n_pmc = sum(1 for u in urls if "pmc.ncbi" in u)
            print(
                f"\n  Batch {bi}/{len(batches)} ({len(batch)} DOIs) → clipboard for "
                f"Zotero 'Add by Identifier'; opening browser "
                f"({n_pmc} via PMC, {len(urls) - n_pmc} via doi.org):"
            )
            for d in batch:
                print(f"    - {d}")
            subprocess.run(["open"] + urls)
            resp = input("  Press Enter for next batch, 's' to stop: ").strip().lower()
            if resp == "s":
                stopped = True
                break

        # Re-fetch from Zotero to see what actually landed
        all_missing = []
        for t, lr in zip(targets, lit_results):
            missing = fetch_pdfs(lr.get("candidates", []), t["dir"] / "papers", zotero_storage)
            all_missing.extend(missing)

        if not all_missing:
            print("\n  All PDFs found!")
            return

        unique_dois = sorted(set(c["doi"] for c in all_missing))
        subprocess.run(["pbcopy"], input="\n".join(unique_dois), text=True)
        print(f"\n  {len(unique_dois)} still missing. DOIs copied to clipboard.")
        if stopped:
            break
        again = (
            input("  Re-walk the remaining in batches? Enter to retry, 's' to skip: ")
            .strip()
            .lower()
        )
        if again == "s":
            break

    # Print final summary of missing papers
    if all_missing:
        print(f"\n  === {len(set(c['doi'] for c in all_missing))} papers still missing ===")
        seen = set()
        for c in all_missing:
            if c["doi"] not in seen:
                seen.add(c["doi"])
                print(f"  - {c.get('title', 'Unknown')[:80]}")
                print(f"    https://doi.org/{c['doi']}")


#: Per-HTTP-attempt read timeout for the OpenAI client, in seconds.
#: gpt-5+ with high reasoning effort + tool-call validation feedback can
#: occasionally enter a server-side reasoning rollout that exceeds the
#: openai-python default of 600 s. When that happens, openai-python's two
#: internal retries each silently re-hit the same hang, so a single
#: ``Agent.run()`` call burns ~30 minutes (3 attempts × 600 s) before the
#: ``APITimeoutError`` propagates as ``ModelAPIError("Request timed out.")``.
#: A 5-minute cap aborts those wedges in the same wall-clock window where
#: a healthy gpt-5.5 high-reasoning call returns (typically 30-150 s),
#: making real failures observable to the caller in minutes instead of
#: half-hours. See pydantic-ai #3268 for the underlying long-rollout
#: motivation (background mode is the proper long-term fix).
DEFAULT_OPENAI_HTTP_TIMEOUT_S: float = 300.0

#: openai-python's default ``max_retries`` is 2, which compounds the wedge
#: above (3 HTTP attempts per logical call). pydantic-ai already has its
#: own structural retry loop (validation feedback) above the transport
#: layer, so layering openai-python's blind transport retries on top
#: gains nothing and just multiplies the wall-clock cost of a hang.
DEFAULT_OPENAI_MAX_RETRIES: int = 0


def make_model(
    model_name: str,
    *,
    http_timeout_s: float = DEFAULT_OPENAI_HTTP_TIMEOUT_S,
    openai_max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
) -> OpenAIResponsesModel:
    """Build an ``OpenAIResponsesModel`` with fail-fast HTTP transport settings.

    The defaults override openai-python's stock ``timeout=600`` /
    ``max_retries=2``, which together cap a single hung call at ~30 minutes
    of silent waiting before the ``APITimeoutError`` surfaces. With these
    defaults, a hung call aborts after ``http_timeout_s`` (default 300 s)
    on the first try and propagates immediately to pydantic-ai's
    structural retry layer.
    """
    import openai

    client = openai.AsyncOpenAI(
        timeout=http_timeout_s,
        max_retries=openai_max_retries,
    )
    return OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client))


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
{notes_section}

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


def _build_scenario_section(t: dict) -> str:
    """Render the scenario block for a cal-mode prompt.

    Reads two optional fields off the target dict:
      - ``t['scenario']`` — short scenario name (used as a hint).
      - ``t['scenario_yaml']`` — full scenario YAML contents as a string,
        pre-resolved by the caller. This is the authoritative description
        of the QSP trajectory; the prompts instruct the agent to derive
        required data context from it rather than from hard-coded scenario
        names. When absent, the section degrades to the bucket-level checks.

    Caller responsibility (e.g., pdac-build's staged_extraction.py): resolve
    the scenario name to a YAML string before invoking the stage. The
    pipeline does not assume any directory layout for scenario YAMLs.
    """
    scenario = t.get("scenario") or ""
    scenario_yaml = t.get("scenario_yaml") or ""
    if not scenario and not scenario_yaml:
        return ""
    parts = ["## Scenario"]
    if scenario:
        parts.append(f"**Scenario name:** `{scenario}`")
    if scenario_yaml:
        parts.append(
            "**Scenario YAML** (the authoritative QSP trajectory description "
            "for this target — initial state, interventions, duration, "
            "observable readout timepoint):"
        )
        parts.append("```yaml")
        parts.append(scenario_yaml.rstrip())
        parts.append("```")
    else:
        parts.append("_No scenario YAML provided; fall back to bucket-level checks._")
    return "\n\n".join(parts)


def _format_auxiliary_groups_section(auxiliary_config_path: Path | None) -> str:
    """Render the available compartment/measurement-bridge groups.

    Returned text is injected into cal-mode prompts (lit_search, assess,
    plan_review) so each stage knows which compartment / cross-species /
    measurement-scale gaps are pre-authorized as acceptable via the
    ``observable.auxiliary_parameters`` mechanism (qsp-inference side). When
    no config is supplied, the section explicitly tells the agent that no
    bridging is available — preserving the strict compartment-match rules.
    """
    fallback = (
        "## Available Compartment / Measurement Bridges\n\n"
        "No auxiliary parameter groups are declared for this run. "
        "Compartment mismatches (e.g., serum vs tumor tissue) MUST be "
        "rejected — there is no bridging mechanism available."
    )
    if auxiliary_config_path is None:
        return fallback
    p = Path(auxiliary_config_path)
    if not p.exists():
        return fallback
    import yaml as _yaml

    with open(p) as _f:
        _data = _yaml.safe_load(_f) or {}
    _groups = _data.get("groups", {}) or {}
    if not _groups:
        return fallback
    lines = [
        "## Available Compartment / Measurement Bridges",
        "",
        "The following auxiliary parameter groups are declared in "
        "``auxiliary_config.yaml``. Sources whose compartment / measurement "
        "scale is bridged by one of these groups are ACCEPTABLE — the "
        "bridging factor will be inferred jointly with QSP parameters at "
        "calibration time via ``observable.auxiliary_parameters``.",
        "",
    ]
    for _name, _spec in _groups.items():
        _desc = (_spec.get("description") or "").strip().replace("\n", " ")
        _bp = _spec.get("base_prior") or {}
        _dist = _bp.get("distribution", "?")
        _mu = _bp.get("mu")
        _sigma = _bp.get("sigma")
        _tau = _spec.get("member_deviation_sigma")
        _mu_str = f"{float(_mu):.4g}" if _mu is not None else "?"
        _sigma_str = f"{float(_sigma):.4g}" if _sigma is not None else "?"
        _tau_str = f"{float(_tau):.4g}" if _tau is not None else "?"
        lines.append(
            f"- `{_name}` ({_dist} base, mu={_mu_str}, sigma={_sigma_str}; "
            f"member_deviation_sigma={_tau_str})\n  - {_desc}"
        )
    lines.append("")
    lines.append(
        "When a paper's measurement compartment / scale differs from the "
        "model species but a matching auxiliary group exists, treat the "
        "paper as ACCEPTABLE. The cal-target authoring stage will declare "
        "the relevant ``observable.auxiliary_parameters`` member; the "
        "inference workflow consumes the bridge."
    )
    return "\n".join(lines)


LIT_SEARCH_PROMPT_CAL = """\
You are a scientific literature search assistant for quantitative systems pharmacology (QSP) model calibration.

## Task

Find peer-reviewed papers reporting **direct quantitative observations** of a model species
(observable) that will be used to calibrate the QSP model end-to-end. This is a
*calibration target*, not a parameter-derivation submodel target — the goal is to
find a measurement of the observable in the right indication, compartment,
treatment context, and disease stage, NOT to infer a rate constant from a
mechanistic in-vitro experiment.

**Parameters / observable description:** {parameters}

## Model Context

{model_context}

## Parameter Context (from model structure)

{parameter_context}

{scenario_section}

{notes_section}

{auxiliary_groups_section}

## Critical: Match the model's observation context, not the parameter's mechanism

The notes block above (and the targets CSV) carry the calibration context fields
that MUST match — typically:

- `model_indication` (e.g., PDAC, NSCLC) — strict-match preferred; non-target
  indication is acceptable only as a last resort and must be flagged for an
  explicit translation_sigma.
- `model_compartment` (e.g., V_T tumor tissue, V_C central/serum) — by default,
  compartment mismatches (e.g., a tumor-tissue observable satisfied by a
  serum/plasma measurement) are REJECTED. The exception is when a matching
  auxiliary parameter group is declared in the "Available Compartment /
  Measurement Bridges" section below — those gaps are bridged by an
  auxiliary parameter inferred jointly at calibration time, so sources in
  the bridged compartment are ACCEPTABLE. Reject compartment mismatches
  only when no matching bridge group is available.
- `model_treatment_history` (e.g., treatment-naive at diagnosis, post-neoadjuvant)
  — a measurement at diagnosis is NOT satisfied by a post-treatment biopsy.
- `model_stage_burden` (e.g., resectable vs metastatic) — match where possible.
- `model_system` (human vs mouse vs cell line) — for cal targets, **human is
  strongly preferred**; mouse is acceptable only when no human source exists,
  with explicit translation_sigma.

Search for **observations of the species in the matched bucket**. A "tumor-tissue
TGFβ concentration in human PDAC at diagnosis" is satisfied by a homogenate /
lysate / tissue ELISA in resected human PDAC specimens — NOT by a KPC mouse
model, NOT by serum, NOT by an in-vitro stimulation experiment.

## Scenario semantics (CRITICAL when present)

The `scenario` field above (when set) anchors the data context to a specific
QSP simulation trajectory. The scenario block below — when provided — is the
authoritative specification of what trajectory the simulator runs for this
target. It defines the initial state, any interventions/dosing, the simulated
duration, and the timepoint at which the observable is evaluated. **Use it to
determine what experimental context the paper must report.**

Apply these scenario-derived rules in addition to the bucket-level checks
above:

- If the scenario describes a **pre-treatment / treatment-naive baseline**
  (no on-study interventions before the observable is read out), reject mixed
  cohorts that include any post-neoadjuvant, post-chemo, post-radiation, or
  post-immunotherapy samples — even when the post-treatment fraction is
  small (e.g., 10/36). The aggregate statistic from a mixed cohort is NOT a
  baseline observation. If the paper reports the treatment-naive subset
  separately, that subset is acceptable.

- If the scenario describes an **on-treatment timepoint** (an intervention
  precedes the observable readout), the paper must report data at that
  treatment context and timepoint. Pre-treatment baselines and post-completion
  follow-up samples fail the match. Different regimens fail the match unless
  the target's notes authorize the substitution.

- If the scenario describes **longitudinal disease progression**, the paper's
  reported timepoint must be anchored to the scenario's trajectory.

- If no scenario block is provided, fall back to the bucket-level checks above.

When the scenario disagrees with `model_treatment_history` or
`model_stage_burden`, the scenario takes precedence — it encodes the
simulator state at which the observable will actually be evaluated.

## Requirements

1. Prefer **clinical / surgical / autopsy specimens** in the right indication,
   compartment, and treatment-history bucket. Cohort studies (n>10) preferred
   over case reports.
2. The reported quantity must be **directly comparable** to the model species:
   tumor tissue concentration (pg/mL homogenate, pg/mg protein, or nM tissue),
   absolute cell density (cells/mm^2 or cells/mg), absolute concentration in
   serum, etc. Reject IHC scores or relative RNA expression unless the notes
   explicitly authorize them.
3. Look for papers reporting **summary statistics** (mean ± SD, median + IQR,
   per-patient values) with sample size.
4. Avoid reviews, meta-analyses, and computational-only papers.
5. **Every value must be traceable** to text, tables, or figure legends in the
   primary paper (not buried in inaccessible supplementary material).

## Single-source observations are the norm

A calibration target is typically a single direct measurement, not a multi-paper
algebraic derivation. Do NOT propose multi-paper derivations or joint-parameter
constraining for cal-mode targets unless the notes explicitly request it. If
no single paper provides a direct observation, prefer to return an empty plan
and let plan-review trigger a search rerun, rather than constructing a synthetic
derivation from indirect proxies.

## Important

If no paper meets the indication/compartment/treatment-history match, say so
explicitly in `unmappable_notes`. It is better to flag an unmappable target
than to anchor on a proxy. The downstream pipeline will rerun the search with
revised guidance.

Begin with a brief analysis of what observation context the target requires
(indication, compartment, treatment history, stage), in the
`parameters_analyzed` field. Then return 3-5 candidates that match that
context.

## CRITICAL

You MUST perform web searches BEFORE returning your structured output. Do NOT return
an empty candidates list or a placeholder response. Your structured output is final —
there is no second chance to populate it. If your first searches don't find results,
try different search terms (synonyms, related concepts, broader queries). Only return
an empty candidates list if you have genuinely exhausted search strategies and confirmed
that no suitable observation in the matched indication/compartment/treatment-history
bucket exists in the literature.
"""


ASSESS_PROMPT_CAL = """\
You are a scientific data assessment assistant for QSP model calibration.

## Task

Assess whether the attached paper(s) contain a **direct observation** of the
model species suitable as a calibration target for these parameter(s):

**Parameters / observable description:** {parameters}
**Cancer type:** {cancer_type}
{scenario_section}
{notes_section}

{auxiliary_groups_section}

## Model Context

{model_context}

## Parameter Context

{parameter_context}

{prior_context_section}

{paper_text_section}

## Instructions

For each paper, determine:

1. **Does it report a direct observation of the model species?** (absolute
   concentration, density, count — not IHC score, not relative expression,
   not in-vitro stimulation response).
2. **Does the observation context match the model bucket?** Check, in order:
   - **`scenario` (when set)** — the dominant filter. The scenario block
     (when provided alongside the target) describes the QSP trajectory and
     the timepoint at which the observable is evaluated. Use it to derive
     the required data context (treatment-naive vs on-treatment, timepoint,
     regimen). Reject papers whose cohorts do not match this context. If
     the paper's cohort is mixed and the matching subset is not reported
     separately, the paper fails the scenario match. The scenario takes
     precedence over `model_treatment_history` and `model_stage_burden`
     when they conflict.
   - `model_indication` — strict-match preferred. Non-target indication is a
     proxy and must be flagged.
   - `model_compartment` — by default, a tumor-tissue observable is NOT
     satisfied by a serum/plasma measurement. The exception is when a
     matching auxiliary parameter group is declared in the "Available
     Compartment / Measurement Bridges" section below — those gaps are
     bridged at calibration time and serum sources for tumor observables
     are then ACCEPTABLE. Reject compartment mismatches only when no
     matching bridge group is available.
   - `model_treatment_history` — at-diagnosis is not satisfied by a
     post-treatment biopsy.
   - `model_stage_burden` — match where possible.
   - `model_system` — human strongly preferred for cal targets; mouse only
     when no human source exists.
3. **Where is the data?** (table, text, or figure — be specific).
4. **Which specific values within the paper should be used?** Papers often
   contain multiple cohorts or conditions. Identify the rows/columns/conditions
   that match the model bucket. Do NOT use the "prior median sanity check" to
   pick between mismatched options — pick on indication/compartment/treatment
   match first, then accept whatever value falls out, even if 100x from the
   simulator's current baseline. If all options are mismatched, flag the paper
   as proxy-only and recommend rerun_lit_search.
5. **Does any figure data warrant digitization?** Only flag `needs_digitization`
   when the figure contains data that is **not available** in text or tables AND
   digitizing would provide meaningful added value (e.g., individual data points
   vs a reported mean).
6. **What is this paper's role?** For cal-mode, the typical role is `standalone`
   (single direct observation). Multi-paper derivations and proxy-with-translation
   are exceptions, used only when the notes explicitly authorize them.

## Extraction Plan

After assessing all papers, construct an `extraction_plan`: the **single best
paper** providing a direct observation in the matched bucket. Prefer:

- A paper in the target indication, compartment, and treatment-history bucket
  over a larger cohort in a mismatched bucket.
- Text/table values over digitization.
- A standalone direct observation over any multi-paper combination.

If no paper matches the bucket, set the plan papers to empty and explain the
mismatch in the strategy field — plan-review will then trigger a search rerun.
Include alternative_plans only if there is a genuinely different but viable
direct-observation paper.
"""


PLAN_REVIEW_PROMPT_CAL = """\
You are reviewing extraction plans for QSP model **calibration targets** (direct
observations of model species, used to fit the full simulator end-to-end). This
is NOT a submodel parameter derivation — the standard for acceptance is
indication/compartment/treatment-context match, not mechanistic plausibility.

## Task

For each target below, evaluate whether the primary extraction plan is the best
option, or whether an alternative plan or a lit search rerun would be better.
Consider, in priority order:

1. **Scenario match (when scenario is set)**: The target's `scenario` field
   anchors the data context to a QSP simulation trajectory and is the
   dominant filter. The scenario block (when included with the target)
   describes the trajectory in detail — initial state, interventions,
   duration, and observable readout timepoint. Use it to determine the
   required data context for each target.
   - If the scenario describes a pre-treatment / treatment-naive baseline,
     a paper whose cohort includes ANY post-treatment samples (even a
     minority subset, e.g., 10 of 36) FAILS the scenario match unless the
     paper reports the treatment-naive subset separately. Recommend
     `rerun_lit_search`.
   - If the scenario describes an on-treatment timepoint, the paper must
     report data at the matching regimen and timepoint. Pre-treatment
     baselines and different regimens fail the match unless the target's
     notes authorize the substitution.
   - If the scenario describes longitudinal disease progression, the
     paper's reported timepoint must be anchored to the trajectory.
   - When the scenario disagrees with `model_treatment_history` or
     `model_stage_burden`, the scenario takes precedence.
2. **Indication match**: Does the paper report data in the target's
   `model_indication`? Indication mismatches (e.g., NSCLC paper for a PDAC
   target) are first-class reasons to recommend `rerun_lit_search`, not
   `proceed`. PDAC-specific data is preferred and the "larger non-PDAC cohort
   beats tiny PDAC cohort" trade-off does NOT apply for cal-mode observables —
   indication mismatch dominates uncertainty.
3. **Compartment match**: by default, tumor-tissue observable + serum/plasma
   source = reject and recommend rerun_lit_search. The exception is when a
   matching auxiliary parameter group is declared in the "Available
   Compartment / Measurement Bridges" section below — those gaps are
   bridged via ``observable.auxiliary_parameters`` at calibration time, so
   serum sources for tumor observables are ACCEPTABLE and should usually
   recommend ``proceed`` (assuming other dimensions match). Reject
   compartment mismatches only when no matching bridge group is available.
4. **Treatment-history / stage match**: At-diagnosis + post-neoadjuvant biopsy
   = reject. Recommend rerun_lit_search.
5. **Species / system**: Human preferred. Mouse-PDAC for a human-PDAC observable
   is acceptable only when explicitly authorized by the target's notes column,
   AND only after `rerun_lit_search` has been tried at least once with stricter
   guidance.
6. **Direct vs proxy**: Reject IHC scores, relative RNA expression, or
   in-vitro-stimulation values when an absolute concentration / density exists
   in another paper.
7. **Sample size**: n=2 case reports are weak; prefer cohort studies (n>10)
   when both are in the matched bucket.
8. **Digitization burden**: Lower is better, but never trade an indication
   match for fewer digitizations.
9. **Empty plans / proxy-only plans / scenario-mismatched plans**: If the
   primary plan has no papers, OR has only one paper that is mismatched on
   scenario/indication/compartment/treatment, recommend `rerun_lit_search`
   with concrete suggestions for what to search differently (specific search
   terms, data types, indications, treatment context).

## Targets

{targets_json}

{auxiliary_groups_section}

## Digitization Summary

{digitization_summary}

For each target, provide a verdict. Be specific about WHY you're recommending a
switch or rerun — what bucket dimension is mismatched, and what search terms
would surface a better source.
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


def build_complete_prompt_cal(
    t: dict,
    model_context: str,
    model_structure_path: Path,
    assessment_json: str,
    digitized_data_section: str,
    auxiliary_groups: list[dict] | None = None,
) -> str:
    """Build stage 3 prompt for CalibrationTarget extraction.

    Pulls model-context fields from the target row (populated by the caller from
    its targets CSV): observable_description, model_species, model_indication,
    model_compartment, model_system, model_treatment_history, model_stage_burden,
    relevant_compartments. Falls back to "Not specified" / cancer_type for missing
    values so partially-populated rows still produce a usable prompt.
    """
    from maple.core.prompts import build_calibration_target_prompt

    ms = ModelStructure.from_json(model_structure_path)
    all_species_units = ms.to_species_units()

    relevant_compartments = t.get("relevant_compartments", "") or ""
    if relevant_compartments and all_species_units:
        comps = [c.strip() for c in relevant_compartments.split(",") if c.strip()]
        filtered = {
            sp: u
            for sp, u in all_species_units.items()
            if any(sp.startswith(f"{c}.") for c in comps)
        }
        species_text = (
            "\n".join(f"- {sp}: {u}" for sp, u in sorted(filtered.items()))
            if filtered
            else "Not provided"
        )
    else:
        species_text = (
            "\n".join(f"- {sp}: {u}" for sp, u in sorted(all_species_units.items()))
            if all_species_units
            else "Not provided"
        )

    cancer_type = t.get("cancer_type", "") or "Not specified"
    observable_description = (
        t.get("observable_description") or t.get("description") or t.get("notes") or t["target_id"]
    )

    base_prompt = build_calibration_target_prompt(
        observable_description=observable_description,
        cancer_type=cancer_type,
        model_species=t.get("model_species") or "Not specified",
        model_indication=t.get("model_indication") or cancer_type,
        model_compartment=t.get("model_compartment") or "Not specified",
        model_system=t.get("model_system") or "Not specified",
        model_treatment_history=t.get("model_treatment_history") or "Not specified",
        model_stage_burden=t.get("model_stage_burden") or "Not specified",
        model_species_with_units=species_text,
        used_primary_studies=t.get("used_primary_studies", ""),
        primary_source_title=t.get("primary_source_title", ""),
        auxiliary_groups=auxiliary_groups,
    )

    extra = (
        "\n\n## Model Context\n\n"
        + model_context
        + "\n\n## Paper Assessment (from prior stage)\n\n"
        + assessment_json
    )
    if digitized_data_section:
        extra += f"\n{digitized_data_section}"

    return base_prompt + extra


# ===========================================================================
# Agents (created once per config, reused across targets)
# ===========================================================================


def make_agents(
    model_name: str,
    max_retries: int,
    *,
    target_kind: TargetKind = "submodel",
) -> tuple[Agent, Agent, Agent, Agent, Agent]:
    """Create the five pipeline agents. Returns (lit_search, assess, plan_review, complete, derivation_review).

    target_kind switches the complete_agent's output_type:
      "submodel" -> SubmodelTarget (default)
      "cal"      -> CalibrationTarget
    """
    if target_kind not in ("submodel", "cal"):
        raise ValueError(f"target_kind must be 'submodel' or 'cal', got {target_kind!r}")
    complete_output_type = SubmodelTarget if target_kind == "submodel" else CalibrationTarget
    openai_model = make_model(model_name)
    settings = make_settings()

    # The lit-search agent is the heaviest call in the pipeline (high reasoning
    # effort + WebSearchTool), and gpt-5+ routinely needs >300 s of server-side
    # rollout for hard-to-find parameters. The default 300 s read-timeout cuts
    # those off mid-rollout — a clean httpx read timeout at exactly the cap,
    # not a wedge — and pydantic-ai does NOT retry transport timeouts, so the
    # same targets fail every re-run. Give just this agent a longer read-timeout
    # so long rollouts can land; the other agents stay fail-fast at 300 s.
    lit_search_model = make_model(model_name, http_timeout_s=900.0)

    lit_search_settings = OpenAIResponsesModelSettings(
        openai_reasoning_summary="concise",
        openai_reasoning_effort="high",
    )

    lit_search_agent = Agent(
        lit_search_model,
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
        output_type=complete_output_type,
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
    target_kind: TargetKind = "submodel",
    auxiliary_config_path: Path | None = None,
) -> PlanReviewResult | None:
    """Stage 2b: Review all extraction plans and recommend actions.

    ``auxiliary_config_path`` is consumed only when ``target_kind == 'cal'``
    and is surfaced in the prompt's "Available Compartment / Measurement
    Bridges" section so the reviewer doesn't reject a serum-for-tumor plan
    that the bridging mechanism makes acceptable.
    """
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
            "scenario": t.get("scenario") or "",
            "scenario_yaml": t.get("scenario_yaml") or "",
            # Per-target authoring notes from the targets CSV (e.g., loosened
            # bucket directives that authorize serum sources for tumor
            # observables when bridged by an auxiliary group). Surfaced so
            # plan-review reads the same authoring intent that lit-search /
            # assess saw at earlier stages.
            "notes": t.get("notes", ""),
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

    prompt_template = PLAN_REVIEW_PROMPT_CAL if target_kind == "cal" else PLAN_REVIEW_PROMPT
    fmt_kwargs = dict(
        targets_json=json.dumps(target_summaries, indent=2),
        digitization_summary=dig_summary,
    )
    if target_kind == "cal":
        fmt_kwargs["auxiliary_groups_section"] = _format_auxiliary_groups_section(
            auxiliary_config_path
        )
    prompt = prompt_template.format(**fmt_kwargs)

    result = await plan_review_agent.run(prompt)
    review = result.output

    with open(review_path, "w") as f:
        json.dump(review.model_dump(), f, indent=2)

    actions = {"proceed": 0, "switch_to_alt": 0, "rerun_lit_search": 0, "defer": 0}
    for r in review.reviews:
        actions[r.verdict] += 1

    print(
        f"\n  Plan review: {actions['proceed']} proceed, {actions['switch_to_alt']} switched, "
        f"{actions['rerun_lit_search']} rerun, {actions['defer']} deferred"
    )
    print(f"\n  {review.summary}")

    return review


def apply_plan_swaps(reviews: list, work_dir: Path) -> None:
    """Apply plan swap verdicts: replace primary extraction plan with alternative.

    Modifies assessment.json for each swapped target.
    """
    for r in reviews:
        if r.verdict != "switch_to_alt" or not r.replacement_plan:
            continue
        target_dir = work_dir / r.target
        assessment_path = target_dir / "assessment.json"
        if not assessment_path.exists():
            continue
        with open(assessment_path) as f:
            assessment = json.load(f)
        old_plan = assessment["extraction_plan"]
        assessment["extraction_plan"] = r.replacement_plan.model_dump()
        alts = assessment.get("alternative_plans", [])
        alts.append(old_plan)
        assessment["alternative_plans"] = alts
        with open(assessment_path, "w") as f:
            json.dump(assessment, f, indent=2)
        plan_tags = ", ".join(r.replacement_plan.papers[:3])
        print(f"  [{r.target}] SWITCHED to: {plan_tags}")


def apply_lit_search_reruns(reviews: list, work_dir: Path, targets_csv: Path) -> None:
    """Apply rerun_lit_search verdicts: clear caches and append notes to targets CSV.

    Side effects:
    - Deletes lit_search_results.json and assessment.json for each rerun target
    - Appends RERUN notes to the target's row in targets_csv
    """
    for r in reviews:
        if r.verdict != "rerun_lit_search":
            continue
        # Delete caches
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


# ===========================================================================
# Stage functions (one target each)
# ===========================================================================


async def run_lit_search(
    t: dict,
    *,
    lit_search_agent: Agent,
    model_context: str,
    model_structure_path: Path,
    target_kind: TargetKind = "submodel",
    auxiliary_config_path: Path | None = None,
) -> dict:
    """Stage 1: Literature search for one target.

    ``auxiliary_config_path`` is consumed only when ``target_kind == 'cal'``
    and is rendered into the prompt's "Available Compartment / Measurement
    Bridges" section so the LLM knows which compartment / cross-species /
    measurement-scale gaps are pre-authorized as acceptable via the
    auxiliary-parameter mechanism. Submodel mode ignores it.
    """
    target_dir = t["dir"]
    # Defensive: target dir is created at targets-list build time, but a user
    # may have cleared the cache between then and now. Recreate so the agent's
    # lit_search_results.json write doesn't fail after a 5-min agent call.
    target_dir.mkdir(parents=True, exist_ok=True)
    lit_search_path = target_dir / "lit_search_results.json"

    if lit_search_path.exists():
        print(f"  [{t['target_id']}] CACHED lit search")
        with open(lit_search_path) as f:
            return json.load(f)

    parameter_context = build_parameter_context(t["parameters"], model_structure_path)
    notes_section = f"## Additional Notes\n\n{t['notes']}" if t["notes"] else ""
    scenario_section = _build_scenario_section(t)

    prompt_template = LIT_SEARCH_PROMPT_CAL if target_kind == "cal" else LIT_SEARCH_PROMPT
    fmt_kwargs = dict(
        parameters=t["parameters"],
        model_context=model_context,
        parameter_context=parameter_context,
        notes_section=notes_section,
    )
    if target_kind == "cal":
        fmt_kwargs["scenario_section"] = scenario_section
        fmt_kwargs["auxiliary_groups_section"] = _format_auxiliary_groups_section(
            auxiliary_config_path
        )
    prompt = prompt_template.format(**fmt_kwargs)

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
    target_kind: TargetKind = "submodel",
    auxiliary_config_path: Path | None = None,
) -> dict | None:
    """Stage 2: Assess papers for one target.

    ``auxiliary_config_path`` is consumed only for ``target_kind == 'cal'`` —
    it surfaces the available compartment / measurement-bridge groups so
    the assessor doesn't pre-reject a serum source that the cal-target
    extractor will later bridge via ``observable.auxiliary_parameters``.
    """
    target_dir = t["dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    assessment_path = target_dir / "assessment.json"

    if assessment_path.exists():
        print(f"  [{t['target_id']}] CACHED assessment")
        with open(assessment_path) as f:
            return json.load(f)

    file_parts, text_parts = collect_papers(target_dir / "papers", text_only=True)
    if not file_parts and not text_parts:
        print(f"  [{t['target_id']}] no papers, skipping")
        return None

    parameter_context = build_parameter_context(t["parameters"], model_structure_path)
    prior_context_section = get_prior_context(t["parameters"], priors_csv)
    paper_text_section = ""
    if text_parts:
        paper_text_section = "## Paper Content (text)\n\n" + "\n\n".join(text_parts)

    notes = t.get("notes", "")
    notes_section = f"**Extraction guidance:** {notes}" if notes else ""
    scenario_section = _build_scenario_section(t)

    prompt_template = ASSESS_PROMPT_CAL if target_kind == "cal" else ASSESS_PROMPT
    fmt_kwargs = dict(
        parameters=t["parameters"],
        cancer_type=t["cancer_type"],
        model_context=model_context,
        parameter_context=parameter_context,
        prior_context_section=prior_context_section,
        paper_text_section=paper_text_section,
        notes_section=notes_section,
    )
    if target_kind == "cal":
        fmt_kwargs["scenario_section"] = scenario_section
        fmt_kwargs["auxiliary_groups_section"] = _format_auxiliary_groups_section(
            auxiliary_config_path
        )
    prompt = prompt_template.format(**fmt_kwargs)

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
    target_kind: TargetKind = "submodel",
    auxiliary_config_path: Path | None = None,
) -> Path | None:
    """Stage 3: Assemble target YAML (SubmodelTarget or CalibrationTarget).

    ``auxiliary_config_path`` is optional and only consumed for ``target_kind=="cal"``.
    When provided and the file exists, the loaded group declarations are surfaced
    in the cal-target prompt so the LLM knows which compartment / cross-species /
    measurement-scale bridges are available via ``observable.auxiliary_parameters``.
    """
    target_dir = t["dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / f"{t['target_id']}_{t['cancer_type']}_deriv001.yaml"
    assessment_path = target_dir / "assessment.json"

    if output_file.exists():
        print(f"  [{t['target_id']}] CACHED deriv YAML")
        return output_file
    if not assessment_path.exists():
        print(f"  [{t['target_id']}] no assessment, skipping")
        return None

    # Report missing digitizations but don't block extraction
    with open(assessment_path) as _af:
        _assess = json.load(_af)
    plan = _assess.get("extraction_plan", {})
    required_digs = plan.get("digitizations_needed", [])
    if required_digs:
        digitized_dir = target_dir / "digitized"
        missing_digs = []
        for dig_spec in required_digs:
            parts = dig_spec.split("/", 1)
            tag = parts[0].strip()
            tag_dir = digitized_dir / tag
            if not tag_dir.is_dir() or not list(tag_dir.glob("*.csv")):
                missing_digs.append(dig_spec)
        if missing_digs:
            print(
                f"  [{t['target_id']}] WARNING — {len(missing_digs)} digitization(s) pending: "
                + ", ".join(missing_digs)
            )

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

    if target_kind == "submodel":
        parameter_context = build_parameter_context(t["parameters"], model_structure_path)
    else:
        parameter_context = ""

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

    if target_kind == "cal":
        # Load auxiliary group declarations if a config was supplied.
        # Loaded fresh per-target so the agent loop reflects edits to
        # auxiliary_config.yaml between targets without restarting.
        auxiliary_groups: list[dict] | None = None
        if auxiliary_config_path is not None and Path(auxiliary_config_path).exists():
            import yaml as _yaml

            with open(auxiliary_config_path) as _f:
                _aux_data = _yaml.safe_load(_f) or {}
            _groups = _aux_data.get("groups", {}) or {}
            auxiliary_groups = [
                {
                    "name": _name,
                    "description": _spec.get("description", ""),
                    "base_prior": _spec.get("base_prior", {}),
                    "member_deviation_sigma": _spec.get("member_deviation_sigma"),
                }
                for _name, _spec in _groups.items()
            ]

        prompt = build_complete_prompt_cal(
            t=t,
            model_context=model_context,
            model_structure_path=model_structure_path,
            assessment_json=json.dumps(assessment_data, indent=2),
            digitized_data_section=digitized_section + paper_text_section,
            auxiliary_groups=auxiliary_groups,
        )
        if plan_review_reason:
            prompt += f"\n\n## Plan Review Note\n\n{plan_review_reason}\n"
    else:
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
    if target_kind == "cal":
        CalibrationTarget.model_validate(target_data, context={"model_structure": ms})
    else:
        SubmodelTarget.model_validate(target_data, context={"model_structure": ms})

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


def _promotion_dest_dir(t: dict, target_kind: TargetKind) -> Path:
    """Where a passing target gets copied. Submodel targets always land under
    calibration_targets/submodel_targets/. CalibrationTargets land under
    calibration_targets/<scenario>/, taking the scenario from t['scenario'] or
    t['model_treatment_history'] (fallback: 'general')."""
    if target_kind == "submodel":
        return Path("calibration_targets/submodel_targets")
    scenario = t.get("scenario") or t.get("model_treatment_history") or "general"
    return Path("calibration_targets") / scenario


def run_validate(
    t: dict,
    *,
    model_structure_path: Path,
    priors_csv: Path,
    target_kind: TargetKind = "submodel",
) -> dict:
    """Stage 3b: Validate a completed target (synchronous).

    Returns a status dict: {target_id, state, unit_ok, prior_ok, snippet_ok, promoted_to}.
    state ∈ {'pass', 'fail', 'skip_missing', 'skip_promoted'}.
    """
    target_dir = t["dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / f"{t['target_id']}_{t['cancer_type']}_deriv001.yaml"

    if not output_file.exists():
        return {
            "target_id": t["target_id"],
            "state": "skip_missing",
            "unit_ok": None,
            "prior_ok": None,
            "snippet_ok": None,
            "promoted_to": None,
        }

    # Skip if already promoted to calibration_targets/
    dest_dir_ = _promotion_dest_dir(t, target_kind)
    dest = dest_dir_ / output_file.name
    if dest.exists():
        print(f"  [{t['target_id']}] SKIP (already in {dest.parent.name}/)")
        return {
            "target_id": t["target_id"],
            "state": "skip_promoted",
            "unit_ok": None,
            "prior_ok": None,
            "snippet_ok": None,
            "promoted_to": str(dest),
        }

    # Unit validation
    try:
        ms = ModelStructure.from_json(model_structure_path)
        with open(output_file) as f:
            data = yaml.safe_load(f)
        if target_kind == "cal":
            CalibrationTarget.model_validate(data, context={"model_structure": ms})
        else:
            SubmodelTarget.model_validate(data, context={"model_structure": ms})
        unit_ok = True
    except Exception as e:
        print(f"  [{t['target_id']}] Unit FAIL: {e}")
        unit_ok = False

    prior_ok = True  # Prior derivation moved to qsp-inference

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
        f.write(f"Prior derivation: {'PASS' if prior_ok else 'FAIL'} (moved to qsp-inference)\n\n")
        f.write(f"Snippet validation: {'PASS' if snippet_ok else 'FAIL'}\n{snippet_report}\n")

    # Copy passing targets to the kind-appropriate destination
    promoted_to = None
    if all_pass:
        dest_dir = _promotion_dest_dir(t, target_kind)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / output_file.name
        shutil.copy2(output_file, dest)
        print(f"  [{t['target_id']}] -> {dest}")
        promoted_to = str(dest)

    return {
        "target_id": t["target_id"],
        "state": "pass" if all_pass else "fail",
        "unit_ok": unit_ok,
        "prior_ok": prior_ok,
        "snippet_ok": snippet_ok,
        "promoted_to": promoted_to,
    }


def run_validate_all(
    targets: list,
    *,
    model_structure_path: Path,
    priors_csv: Path,
    target_kind: TargetKind = "submodel",
) -> list:
    """Run run_validate over every target and print a summary table.

    Returns the list of per-target status dicts.
    """
    results = [
        run_validate(
            t,
            model_structure_path=model_structure_path,
            priors_csv=priors_csv,
            target_kind=target_kind,
        )
        for t in targets
    ]

    def _fmt(v):
        if v is None:
            return "  - "
        return " OK " if v else "FAIL"

    print("\n" + "-" * 64)
    print(f"{'target_id':<24} {'state':<14} unit prior snippet promoted")
    print("-" * 64)
    for r in results:
        promoted = "yes" if r.get("promoted_to") else "-"
        print(
            f"{r['target_id']:<24} {r['state']:<14} "
            f"{_fmt(r.get('unit_ok'))} {_fmt(r.get('prior_ok'))}  {_fmt(r.get('snippet_ok'))}     {promoted}"
        )
    counts: dict = {}
    for r in results:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print("-" * 64)
    print(f"Done. {len(targets)} targets processed ({summary}).")
    return results


def write_assessment_report(t: dict, assessment: dict) -> None:
    """Write assessment.md and digitization READMEs for one target."""
    target_dir = t["dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
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
