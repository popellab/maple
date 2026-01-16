#!/usr/bin/env python3
"""
Scenario models for calibration targets.

Provides models for interventions and experimental scenarios.
Observable definitions are separate (in observable.py for CalibrationTarget,
nested in submodel for IsolatedSystemTarget).
"""

from typing import List

from pydantic import BaseModel, Field


class Intervention(BaseModel):
    """
    Text description of intervention (deferred executable code to later).

    For now, we capture the essential information in text form.
    Later, this can be converted to executable DrugDosing/SurgicalResection/etc.
    """

    intervention_description: str = Field(
        description=(
            "Complete text description of the intervention including:\n"
            "- What: Agent/procedure name\n"
            "- How much: Dose and units\n"
            "- When: Schedule/timing\n"
            "- Additional details: Any relevant context\n\n"
            "Clinical/in vivo examples:\n"
            "- 'Anti-PD-1 antibody 3 mg/kg IV every 2 weeks starting day 0 (patient weight 70 kg)'\n"
            "- 'Surgical resection on day 14, removing 90% of tumor burden'\n"
            "- 'Gemcitabine 1000 mg/m2 on days 0, 7, 14 (patient BSA 1.8 m2)'\n"
            "- 'No intervention (natural disease progression)'\n\n"
            "In vitro examples:\n"
            "- 'Add recombinant IL-2 at 10 ng/mL at t=0'\n"
            "- 'Co-culture tumor cells with CD8+ T cells at E:T ratio 5:1'\n"
            "- 'Stimulate with anti-CD3/CD28 beads (bead:cell ratio 1:1) for 48h'\n"
            "- 'Treat with 1 μM drug X at t=24h'\n"
            "- 'No treatment (unstimulated control)'"
        )
    )


class Scenario(BaseModel):
    """
    Experimental scenario: description and sequence of interventions.

    Defines the experimental setup and any treatments applied.
    Observable computation is defined separately (not in Scenario).
    """

    description: str = Field(
        description="Human-readable description of the scenario (e.g., 'Baseline PDAC tumor at resection, treatment-naive')"
    )
    interventions: List[Intervention] = Field(
        description=(
            "List of interventions applied during the experiment. "
            "May be empty list for natural/untreated state measurements. "
            "Use single entry with 'No intervention (natural disease progression)' for clarity."
        )
    )
