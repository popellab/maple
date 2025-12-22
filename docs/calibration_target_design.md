# Calibration Target Design

## Overview

Calibration targets are raw observables extracted from literature, used to calibrate QSP model parameters via Bayesian inference. Each observable has an **experimental context** that may differ from the **model context**, requiring formal mismatch handling.

---

## Context Dimensions

### 1. Species

```
human
mouse
rat
non_human_primate
other
```

Optional mouse subspecifier: `wild_type`, `immunocompromised`, `transgenic`

---

### 2. Indication

Hierarchical:

```
solid_tumor
├── gi_adenocarcinoma
│   ├── PDAC
│   ├── colorectal
│   ├── gastric
│   └── esophageal
├── hepatobiliary
│   ├── hepatocellular
│   └── cholangiocarcinoma
├── lung
│   ├── lung_adeno
│   ├── lung_squamous
│   └── small_cell_lung
├── immunogenic_solid
│   ├── melanoma
│   ├── renal_cell
│   └── head_and_neck
├── other_solid
│   ├── breast
│   ├── ovarian
│   ├── prostate
│   └── glioblastoma

heme_malignancy
├── lymphoma
├── leukemia
└── myeloma

non_cancer
├── healthy
└── other_disease
```

---

### 3. Compartment

Hierarchical:

```
tumor
├── primary
├── metastatic
└── unspecified

blood
├── whole_blood
├── PBMC
└── plasma_serum

lymphoid
├── tumor_draining_LN
├── other_LN
├── spleen
└── bone_marrow

other_tissue

in_vitro
```

---

### 4. Experimental System

Hierarchical (reflects translatability):

```
clinical
├── biopsy
├── resection
└── liquid_biopsy

ex_vivo
├── fresh
└── cultured

animal_in_vivo
├── orthotopic
├── subcutaneous
├── PDX
├── GEM
└── syngeneic

in_vitro
├── organoid
├── primary_cells
└── cell_line
```

---

### 5. Treatment Context

Multi-select.

**Treatment history** (select all that apply):
```
treatment_naive
prior_chemotherapy
prior_radiation
prior_immunotherapy
prior_targeted_therapy
prior_surgery
```

**Current status** (select one):
```
off_treatment
on_treatment
```

Optional specifier: drug name or class.

---

### 6. Stage

Disease extent and burden (not explicitly modeled, affects biology).

**Extent:**
```
resectable
borderline_resectable
locally_advanced
metastatic
```

**Burden:**
```
low
moderate
high
```

---

## Model Context Specification

The model declares its assumed context:

```yaml
model_context:
  species: human
  indication: PDAC
  compartment: tumor.primary
  system: clinical
  treatment: [treatment_naive]
  stage:
    extent: locally_advanced
    burden: moderate
```

---

## Examples

### Model Context Examples

**PDAC treatment-naive locally advanced model:**
```yaml
model_context:
  species: human
  indication: PDAC
  compartment: tumor.primary
  system: clinical
  treatment: [treatment_naive]
  stage:
    extent: locally_advanced
    burden: moderate
```

**PDAC on-treatment metastatic model:**
```yaml
model_context:
  species: human
  indication: PDAC
  compartment: tumor.primary
  system: clinical
  treatment: [on_treatment]
  treatment_specifier: gemcitabine_nab_paclitaxel
  stage:
    extent: metastatic
    burden: high
```

### Observable Examples

**CD8 TIL density in PDAC resection specimens:**
```yaml
observable:
  class: cell_density
  description: "CD8+ T cell density in tumor"
  value: 150
  uncertainty: 80
  units: "cells/mm^2"

experimental_context:
  species: human
  indication: PDAC
  compartment: tumor.primary
  system: clinical.resection
  treatment: [treatment_naive]
  stage:
    extent: resectable
    burden: low
```
*Context mismatch: Stage (resectable vs locally_advanced).*

---

**Treg:CD8 ratio from mouse KPC model:**
```yaml
observable:
  class: cell_fraction
  description: "Treg to CD8 ratio in tumor"
  value: 0.4
  uncertainty: 0.15
  units: "dimensionless"

experimental_context:
  species: mouse
  mouse_subspecifier: wild_type
  indication: PDAC
  compartment: tumor.primary
  system: animal_in_vivo.orthotopic
  treatment: [treatment_naive]
  stage:
    extent: locally_advanced
    burden: moderate
```
*Context mismatch: Species (mouse vs human), System (orthotopic vs clinical).*

---

**IL-10 concentration from PDAC patient blood:**
```yaml
observable:
  class: concentration
  description: "Plasma IL-10 level"
  value: 12.5
  uncertainty: 8.0
  units: "pg/mL"

experimental_context:
  species: human
  indication: PDAC
  compartment: blood.plasma_serum
  system: clinical.liquid_biopsy
  treatment: [treatment_naive]
  stage:
    extent: metastatic
    burden: high
```
*Context mismatch: Compartment (blood vs tumor), Stage (metastatic/high vs locally_advanced/moderate).*

---

**T cell killing rate from ex vivo co-culture:**
```yaml
observable:
  class: kinetic_rate
  description: "CTL killing rate constant"
  value: 0.15
  uncertainty: 0.05
  units: "1/hour"

experimental_context:
  species: human
  indication: PDAC
  compartment: in_vitro
  system: ex_vivo.cultured
  treatment: [treatment_naive]
  stage:
    extent: resectable
    burden: low
```
*Context mismatch: System (ex vivo vs clinical), Compartment (in vitro vs tumor).*

---

**Tumor volume doubling time from subcutaneous mouse model:**
```yaml
observable:
  class: tumor_measurement
  description: "Tumor volume doubling time"
  value: 5.2
  uncertainty: 1.8
  units: "days"

experimental_context:
  species: mouse
  indication: PDAC
  compartment: tumor.primary
  system: animal_in_vivo.subcutaneous
  treatment: [treatment_naive]
  stage:
    extent: locally_advanced
    burden: moderate
```
*Context mismatch: Species (mouse vs human), System (subcutaneous vs clinical). Allometric scaling may apply.*

---

**CD8 exhaustion (PD-1+TIM-3+) fraction from melanoma TILs:**
```yaml
observable:
  class: cell_fraction
  description: "Fraction of CD8 TILs co-expressing PD-1 and TIM-3"
  value: 0.35
  uncertainty: 0.12
  units: "dimensionless"

experimental_context:
  species: human
  indication: melanoma
  compartment: tumor.primary
  system: clinical.resection
  treatment: [prior_immunotherapy]
  stage:
    extent: metastatic
    burden: high
```
*Context mismatch: Indication (melanoma vs PDAC), Treatment (prior IO vs naive), Stage (metastatic/high vs locally_advanced/moderate).*

---

## Context Mismatch Handling

When observable context differs from model context, apply:

| Adjustment | Description |
|------------|-------------|
| **Bias correction** | Systematic shift (e.g., allometric scaling, compartment ratios) |
| **Variance inflation** | Increased uncertainty due to unknown differences |

### Variance Inflation Composition

Use **geometric mean** across dimensions:

```
total = (∏ σᵢ)^(1/n)
```

### Lookup Tables

Separate tables for bias correction and variance inflation. Default to variance inflation when no bias correction available.

**Bias Corrections:**
```yaml
bias_corrections:
  species:
    allometric:
      mouse_to_human:
        body_weight_ratio: 0.0004  # 30g / 70kg
        exponents:
          tumor_volume: 1.0
          doubling_time: 0.25
          clearance: 0.75
          # Immune kinetics: null (no scaling)

  compartment:
    ratios:
      blood_to_tumor:
        IL10: {factor: 10, uncertainty: 0.5}
        TGFB: {factor: 100, uncertainty: 1.0}
        # Add more as identified
```

**Variance Inflation:**
```yaml
variance_inflation:
  species:
    mouse_to_human:
      default: 1.5
      kinetic_rate: 1.3
      tumor_measurement: 1.2  # After allometric correction

  indication:
    method: similarity_matrix
    scale_factor: 2.0  # inflation = 1 + k*(1-similarity)

  compartment:
    same: 1.0
    related: 1.2
    different: 1.5

  system:
    method: hierarchy_distance
    per_level: 0.2  # inflation = 1 + 0.2 * distance

  treatment:
    same: 1.0
    minor_mismatch: 1.2
    major_mismatch: 1.5

  stage:
    same: 1.0
    adjacent: 1.2  # e.g., resectable vs borderline
    distant: 1.5   # e.g., resectable vs metastatic
```

**Indication Similarity Matrix:**
```yaml
indication_similarity:
  PDAC:
    PDAC: 1.0
    colorectal: 0.7
    gastric: 0.6
    lung_adeno: 0.5
    melanoma: 0.3
    healthy: 0.1
  # Expand as needed
```

---

## Observable Classes

Different observable types have different sensitivity to context mismatches.

### Cell Density / Counts
*Examples: TIL density, Treg count, tumor cell number*

| Dimension | Sensitivity |
|-----------|-------------|
| Species | High |
| Indication | High |
| Compartment | Very High |
| System | High |
| Treatment | Medium |
| Stage | High |

### Cell Fractions / Ratios
*Examples: % PD-1+ of CD8, Treg:CD8 ratio, % Ki67+*

| Dimension | Sensitivity |
|-----------|-------------|
| Species | Medium |
| Indication | High |
| Compartment | Medium |
| System | Medium |
| Treatment | Medium |
| Stage | Medium |

### Concentrations (Soluble Factors)
*Examples: IL-10 level, TGF-β, cytokine concentrations*

| Dimension | Sensitivity |
|-----------|-------------|
| Species | Medium |
| Indication | High |
| Compartment | Very High |
| System | High |
| Treatment | High |
| Stage | High |

### Kinetic Rates
*Examples: Doubling time, proliferation rate, death rate*

| Dimension | Sensitivity |
|-----------|-------------|
| Species | Low |
| Indication | Medium |
| Compartment | Medium |
| System | Medium |
| Treatment | High |
| Stage | Low |

### Functional Readouts
*Examples: % killing, % suppression, cytokine production capacity*

| Dimension | Sensitivity |
|-----------|-------------|
| Species | Medium |
| Indication | Medium |
| Compartment | High |
| System | Very High |
| Treatment | High |
| Stage | Medium |

### Tumor Measurements
*Examples: Tumor volume, diameter, growth rate*

| Dimension | Sensitivity |
|-----------|-------------|
| Species | Very High |
| Indication | High |
| Compartment | N/A |
| System | Very High |
| Treatment | High |
| Stage | Very High |

### Survival / Time-to-Event
*Examples: Overall survival, progression-free survival*

| Dimension | Sensitivity |
|-----------|-------------|
| Species | Very High |
| Indication | Very High |
| Compartment | N/A |
| System | Very High |
| Treatment | Very High |
| Stage | Very High |

---

## Open Questions

1. Hard incompatibilities: when to reject an observable vs inflate heavily?
2. How to handle missing entries in lookup tables? (Current: default to variance inflation)
3. Validation of bias/variance values: literature review vs expert elicitation?
