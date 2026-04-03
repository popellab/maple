# ruff: noqa: F704
#!/usr/bin/env python3
"""
Staged submodel target extraction pipeline.

Multi-stage workflow with human touchpoints between stages:

  Stage 1   Lit search        Web search for papers per target (parallel)
  Stage 1b  PDF collection    Zotero DOI lookup + interactive fetch loop
  Stage 2   Paper assessment  Read PDFs, assess data quality (parallel)
  Stage 2b  Plan review       Single LLM call reviewing all plans together
            Digitization      Prioritized summary of figures to digitize
  Stage 3   Extract           Assemble SubmodelTarget YAMLs (parallel)
  Stage 3b  Derivation review Single LLM call checking scientific soundness
  Stage 3c  Validate          MCMC prior derivation + unit checks + snippet matching

Each stage caches results per-target. To rerun a stage for a specific target,
delete its cache file (e.g., lit_search_results.json, assessment.json, or the
deriv YAML). Other targets and stages are untouched.

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
from pathlib import Path
from functools import partial

# Optional: Logfire instrumentation for pydantic-ai tracing
# import logfire
# logfire.configure()
# logfire.instrument_pydantic_ai()

from maple.extraction import (
    collect_missing_pdfs,
    make_agents,
    run_assess,
    run_complete,
    run_derivation_review,
    run_lit_search,
    run_plan_review,
    run_stage,
    run_validate,
    summarize_digitizations,
    write_assessment_report,
    write_dois_md,
)


# ===========================================================================
# Config — edit these for your project
# ===========================================================================

TARGETS_CSV = Path("notes/targets.csv")  # CSV with target_id, parameters, cancer_type, notes
MODEL_STRUCTURE = Path("model_structure.json")  # Exported from qsp-export-model
MODEL_CONTEXT = Path("model_context.txt")  # Free-text model description for LLM context
WORK_DIR = Path("work/staged_extraction")  # Working directory (cached results go here)
ZOTERO_STORAGE = Path("~/Zotero/storage").expanduser()  # Zotero local storage path
PRIORS_CSV = Path("parameters/priors.csv")  # Current prior distributions CSV
MODEL = "gpt-5.1"  # LLM model name (OpenAI Responses API)
MAX_RETRIES = 7  # Max pydantic-ai retries per target
TARGET_RANGE = (0, 20)  # (start, end) indices from targets CSV


# ===========================================================================
# Setup
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
print(f"Batch: {len(batch)} targets (indices {start}-{(end or len(all_targets)) - 1})")

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
# Stage 1: Literature search (parallel)
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

# Stage 1b: Write combined DOIs file
print("\n" + "=" * 60)
print("Stage 1b: DOIs for Zotero")
print("=" * 60)

write_dois_md(targets, lit_results, WORK_DIR / "dois.md")

# Stage 1c: Fetch PDFs from Zotero (interactive)
collect_missing_pdfs(targets, lit_results, ZOTERO_STORAGE)


# ===========================================================================
# Stage 2: Assess papers (parallel)
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

# Write assessment reports + digitization READMEs
for t in targets:
    ap = t["dir"] / "assessment.json"
    if ap.exists():
        with open(ap) as f:
            write_assessment_report(t, json.load(f))

# Stage 2b: Plan review (single call, all targets)
print("\n" + "=" * 60)
print("Stage 2b: Plan Review")
print("=" * 60)

await run_plan_review(targets, WORK_DIR, plan_review_agent, TARGETS_CSV)

# Regenerate digitization summary after any plan swaps
summarize_digitizations(WORK_DIR)

# ---- PAUSE HERE ----
# Review digitization_summary.md in WORK_DIR.
# Digitize required figures with WebPlotDigitizer.
# Place CSV exports in work/staged_extraction/{target_id}/digitized/{source_tag}/
# Then continue to Stage 3.


# ===========================================================================
# Stage 3: Extract SubmodelTarget YAMLs (parallel)
# ===========================================================================

print("\n" + "=" * 60)
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

# Stage 3b: Derivation review (single call, all targets)
print("\n" + "=" * 60)
print("Stage 3b: Derivation Review")
print("=" * 60)

await run_derivation_review(targets, WORK_DIR, derivation_review_agent)

# Stage 3c: Validate (sequential — MCMC is CPU-bound)
print("\n" + "=" * 60)
print("Stage 3c: Validate SubmodelTargets")
print("=" * 60)

for t in targets:
    run_validate(t, model_structure_path=MODEL_STRUCTURE, priors_csv=PRIORS_CSV)

print(f"\nDone. {len(targets)} targets processed.")
