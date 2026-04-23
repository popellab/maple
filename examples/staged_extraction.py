# ruff: noqa: F704
#!/usr/bin/env python3
"""
Staged submodel target extraction pipeline.

Workflow overview (each stage caches results — re-running skips completed work):

  Stage 1  — Lit search:    Web search for papers matching each parameter
  Stage 1b — DOIs:          Combine DOIs into a single file for Zotero import
  Stage 1c — PDF fetch:     Interactive: fetch from Zotero, open in browser if missing
  Stage 2  — Assess:        Read papers (text-extracted, not binary PDFs) and decide
                            which have usable data, what needs digitization
  Stage 2b — Plan review:   Single LLM call reviews ALL plans together for consistency,
                            can swap plans or request lit search reruns
  ── Human touchpoint ──    Digitize figures flagged in digitization_summary.md
                            (place CSVs in work/<target>/digitized/<source_tag>/)
  Stage 3  — Extract:       Assemble SubmodelTarget YAMLs from papers + digitized data.
                            Snippet-in-PDF validation runs inline (retries on failure).
                            Targets missing digitizations proceed anyway (warning only).
  Stage 3b — Deriv review:  Single LLM call reviews ALL derivations for cross-target
                            consistency, flags issues
  Stage 3c — Validate:      Unit validation, MCMC prior derivation, snippet validation.
                            Passing targets auto-copied to calibration_targets/submodel_targets/

Side effects to be aware of:
  - apply_plan_swaps():         Modifies assessment.json (replaces primary with alt plan)
  - apply_lit_search_reruns():  Deletes cached lit_search + assessment, appends RERUN
                                notes to targets.csv
  - summarize_digitizations():  Writes digitization_summary.md, optionally opens PDFs
  - collect_missing_pdfs():     Interactive — copies DOIs to clipboard, opens browser

Usage:
    # Run in a Jupyter/IPython notebook (needs top-level await):
    exec(open("examples/staged_extraction.py").read())

    # Or run stages interactively in a REPL — copy-paste sections as needed.

Setup:
    1. pip install maple-qsp[inference] pydantic-ai logfire
    2. Set OPENAI_API_KEY in your environment
    3. Edit the Config section below for your project paths
    4. Prepare a targets CSV (see README for format)
"""

import csv
import json
import sys
from functools import partial
from pathlib import Path

# Optional: Logfire instrumentation for pydantic-ai tracing
# import logfire
# logfire.configure()
# logfire.instrument_pydantic_ai()

from maple.extraction import (
    apply_lit_search_reruns,
    apply_plan_swaps,
    collect_missing_pdfs,
    make_agents,
    report_digitization_preflight,
    run_assess,
    run_complete,
    run_derivation_review,
    run_lit_search,
    run_plan_review,
    run_stage,
    run_validate_all,
    summarize_digitizations,
    write_assessment_report,
    write_dois_md,
)


# ===========================================================================
# Config — edit these before running
# ===========================================================================

TARGETS_CSV = Path("notes/targets.csv")  # Parameter targets to extract
MODEL_STRUCTURE = Path("model_structure.json")  # QSP model structure (for unit validation)
MODEL_CONTEXT = Path("model_context.txt")  # Model description for LLM prompts
WORK_DIR = Path("work/staged_extraction")  # All intermediate files go here (gitignored)
ZOTERO_STORAGE = Path("~/Zotero/storage").expanduser()
PRIORS_CSV = Path("parameters/priors.csv")  # Current prior distributions CSV
MODEL = "gpt-5.1"  # LLM model name (OpenAI Responses API)
MAX_RETRIES = 7  # Max pydantic-ai retries per target
TARGET_RANGE = (0, None)  # (start, end) indices — set end=None for all


# ===========================================================================
# 0. Setup — load targets, create agents
# ===========================================================================

if not TARGETS_CSV.exists():
    print(f"Error: {TARGETS_CSV} not found", file=sys.stderr)
    sys.exit(1)

WORK_DIR.mkdir(parents=True, exist_ok=True)

model_context = MODEL_CONTEXT.read_text().strip()

all_targets = []
with open(TARGETS_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        all_targets.append(row)

start, end = TARGET_RANGE
batch = all_targets[start:end]

targets = []
for i, row in enumerate(batch, start=start):
    t = {
        "idx": i,
        "target_id": row["target_id"],
        "parameters": row["parameters"],
        "notes": row.get("notes", ""),
        "cancer_type": row.get("cancer_type", ""),
    }
    t["dir"] = WORK_DIR / t["target_id"]
    t["dir"].mkdir(parents=True, exist_ok=True)
    targets.append(t)
    print(f"  [{i}] {t['target_id']} ({t['parameters']})")
print()

lit_search_agent, assess_agent, plan_review_agent, complete_agent, derivation_review_agent = (
    make_agents(MODEL, MAX_RETRIES)
)


# ===========================================================================
# Stage 1: Literature search
# Each target gets a web search for papers with quantitative data.
# Results cached in work/<target>/lit_search_results.json
# ===========================================================================

print("=" * 60)
print(f"Stage 1: Literature Search ({len(targets)} targets)")
print("=" * 60)

_lit_search = partial(
    run_lit_search,
    lit_search_agent=lit_search_agent,
    model_context=model_context,
    model_structure_path=MODEL_STRUCTURE,
)
await run_stage(_lit_search, targets, "lit search")

# Re-read from disk (as_completed returns in arbitrary order)
lit_results = []
for t in targets:
    with open(t["dir"] / "lit_search_results.json") as f:
        lit_results.append(json.load(f))

# Stage 1b: Combine all DOIs into one file for Zotero import
print("\n" + "=" * 60)
print("Stage 1b: DOIs for Zotero")
print("=" * 60)

write_dois_md(targets, lit_results, WORK_DIR / "dois.md")

# Stage 1c: Fetch PDFs from Zotero storage, with browser fallback (interactive)
# Side effects: copies DOIs to clipboard, opens Europe PMC URLs in browser
collect_missing_pdfs(targets, lit_results, ZOTERO_STORAGE)


# ===========================================================================
# Stage 2: Paper assessment
# Reads papers as extracted text (not binary PDFs — avoids context overflow).
# Decides: which papers are usable, what data they have, what needs digitization.
# Results cached in work/<target>/assessment.json
# ===========================================================================

print("\n" + "=" * 60)
print(f"Stage 2: Assess Papers ({len(targets)} targets)")
print("=" * 60)

_assess = partial(
    run_assess,
    assess_agent=assess_agent,
    model_context=model_context,
    model_structure_path=MODEL_STRUCTURE,
    priors_csv=PRIORS_CSV,
)
await run_stage(_assess, targets, "assess")

# Write human-readable assessment reports + digitization README templates
for t in targets:
    ap = t["dir"] / "assessment.json"
    if ap.exists():
        with open(ap) as f:
            write_assessment_report(t, json.load(f))


# ===========================================================================
# Stage 2b: Plan review
# Single LLM call reviews ALL extraction plans together for cross-target
# consistency. Can recommend: proceed, swap to alternative, rerun lit search.
# Result cached in work/plan_review.json
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 2b: Plan Review")
print("=" * 60)

plan_review_result = await run_plan_review(targets, WORK_DIR, plan_review_agent, TARGETS_CSV)

# Apply plan review side effects (explicitly separated from the review itself):
#   apply_plan_swaps:         Modifies assessment.json for swapped targets
#   apply_lit_search_reruns:  Deletes cached lit/assessment, appends RERUN notes to targets.csv
if plan_review_result:
    apply_plan_swaps(plan_review_result.reviews, WORK_DIR)
    apply_lit_search_reruns(plan_review_result.reviews, WORK_DIR, TARGETS_CSV)


# ===========================================================================
# Digitization summary + pre-flight
# After this point, you should digitize any required figures before proceeding.
# Place CSVs in work/<target>/digitized/<source_tag>/<fig_id>.csv
# Re-run summarize_digitizations() to check progress.
# ===========================================================================

# Writes digitization_summary.md with pending/done status, optionally opens PDFs
summarize_digitizations(WORK_DIR)

# Reports which targets have pending digitizations (informational, does not block)
report_digitization_preflight(targets)


# ===========================================================================
# Stage 3: Assemble SubmodelTarget YAMLs
# Sends binary PDFs (for figure reading) + digitized CSVs to the extraction agent.
# Inline snippet-in-PDF validation catches hallucinated quotes (retries automatically).
# Targets with missing digitizations proceed with a warning.
# Results cached in work/<target>/<target_id>_<cancer_type>_deriv001.yaml
# ===========================================================================

print("=" * 60)
print(f"Stage 3: Assemble SubmodelTarget YAMLs ({len(targets)} targets)")
print("=" * 60)

_complete = partial(
    run_complete,
    complete_agent=complete_agent,
    model_context=model_context,
    model_structure_path=MODEL_STRUCTURE,
    priors_csv=PRIORS_CSV,
    work_dir=WORK_DIR,
)
await run_stage(_complete, targets, "complete")


# ===========================================================================
# Stage 3b: Derivation review
# Single LLM call reviews ALL completed derivations for scientific soundness
# and cross-target consistency (e.g., contradictory assumptions, unit mismatches).
# Result cached in work/derivation_review.json + derivation_review.md
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 3b: Derivation Review")
print("=" * 60)

await run_derivation_review(targets, WORK_DIR, derivation_review_agent)


# ===========================================================================
# Stage 3c: Validate
# Sequential (some checks are CPU-bound). For each target:
#   1. Unit validation (parameter units match QSP model)
#   2. Prior derivation check (delegated to qsp-inference)
#   3. Snippet validation (checks value_snippets against PDF text)
# Passing targets are auto-copied to calibration_targets/submodel_targets/
# Already-promoted targets are skipped. A summary table + state counts prints at end.
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 3c: Validate SubmodelTargets")
print("=" * 60)

run_validate_all(targets, model_structure_path=MODEL_STRUCTURE, priors_csv=PRIORS_CSV)
