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
    run_stage,
    # Helpers
    collect_missing_pdfs,
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
    "run_stage",
    # Helpers
    "collect_missing_pdfs",
    "fetch_pdfs",
    "summarize_digitizations",
    "write_assessment_report",
    "write_dois_md",
]
