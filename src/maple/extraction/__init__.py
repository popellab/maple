"""
Staged batch extraction pipeline for SubmodelTarget derivation.

Provides a multi-stage workflow: lit search → paper assessment → plan review →
extraction → derivation review → validation, with per-target caching at every step.
"""

from maple.extraction.pipeline import (
    # Schemas
    AssessmentResult,
    DataAvailability,
    DerivationReview,
    DerivationReviewResult,
    ExtractionPlan,
    LitSearchCandidate,
    LitSearchResult,
    PaperAssessment,
    PlanReview,
    PlanReviewResult,
    # Agent creation
    make_agents,
    # Stage functions
    run_lit_search,
    run_assess,
    run_plan_review,
    run_complete,
    run_derivation_review,
    run_validate,
    run_validate_all,
    run_stage,
    # Helpers
    apply_lit_search_reruns,  # Clear caches + append RERUN notes to targets CSV
    apply_plan_swaps,  # Replace primary extraction plan with reviewer's alternative
    collect_missing_pdfs,
    report_digitization_preflight,  # Print which targets have pending digitizations
    fetch_pdfs,
    summarize_digitizations,
    write_assessment_report,
    write_dois_md,
)

__all__ = [
    # Schemas
    "AssessmentResult",
    "DataAvailability",
    "DerivationReview",
    "DerivationReviewResult",
    "ExtractionPlan",
    "LitSearchCandidate",
    "LitSearchResult",
    "PaperAssessment",
    "PlanReview",
    "PlanReviewResult",
    # Agent creation
    "make_agents",
    # Stage functions
    "run_lit_search",
    "run_assess",
    "run_plan_review",
    "run_complete",
    "run_derivation_review",
    "run_validate",
    "run_validate_all",
    "run_stage",
    # Helpers — plan review side effects (called explicitly by the pipeline script)
    "apply_lit_search_reruns",
    "apply_plan_swaps",
    # Helpers
    "collect_missing_pdfs",
    "report_digitization_preflight",
    "fetch_pdfs",
    "summarize_digitizations",
    "write_assessment_report",
    "write_dois_md",
]
