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

When observable context differs from model context, we use a **meta-analytic framework** that formally handles heterogeneity through:

| Adjustment | Description |
|------------|-------------|
| **Bias correction** | Systematic shift for known relationships (e.g., allometric scaling) |
| **Variance inflation** | Increased uncertainty as a function of context distance |

---

### Meta-Analytic Framework

Each observable provides an estimate of a latent quantity with context-dependent uncertainty:

```
y_ij ~ N(μ_true + bias(ctx_ij, ctx_model) + u_j, σ²_ij + σ²_mismatch_ij)
```

Where:
- `μ_true` = the latent quantity value **in the model's assumed context** (this is what we're estimating)
- `y_ij` = observable `i` from study `j`, with reported uncertainty `σ²_ij`
- `bias(...)` = systematic correction mapping from experimental context to model context (if known), else 0
- `u_j` = study-level random effect (see below)
- `σ²_mismatch` = additional variance due to context distance

**Interpretation:** We estimate the value as it would be observed in the model's context. Bias corrections shift observations toward that context; variance inflation accounts for residual uncertainty in the mapping.

#### Likelihood Assumptions

The Gaussian likelihood is a simplification. Observable types have natural likelihoods:

| Observable Type | Natural Likelihood | Gaussian Approximation |
|-----------------|-------------------|------------------------|
| Cell counts | Poisson / Negative binomial | Valid for large counts (>30) |
| Fractions | Beta | Valid away from 0/1 boundaries |
| Concentrations | Log-normal | Use log-transform |
| Survival times | Weibull / Log-normal | Not recommended |

**Recommendation:** Start with Gaussian for simplicity. For fractions near boundaries or small counts, consider appropriate likelihoods. Log-transform concentrations before modeling.

#### Handling Reported Uncertainty

Literature reports uncertainty inconsistently. Standardize as follows:

| Reported Format | Conversion to σ |
|-----------------|-----------------|
| SD | Use directly |
| SE | `σ = SE × √n` |
| 95% CI | `σ = (upper - lower) / 3.92` |
| IQR | `σ = IQR / 1.35` (assumes Normal) |
| Range | `σ = range / 4` (crude) |
| **Nothing reported** | `σ = 0.3 × |value|` (assume CV = 30%) |

**For missing uncertainty:** The CV = 30% default is conservative for most biological measurements. Flag these observables; they will appropriately have less influence due to larger assumed variance.

**For very small n (n < 5):** Consider inflating reported σ by factor of `√(1 + 2/n)` to account for uncertainty in the variance estimate itself.

#### Study-Level Correlation

Observables from the same study share systematic factors (patients, methods, lab). Model this as:

```
u_j ~ N(0, τ²)
```

Where `τ²` is between-study variance, estimated from data or set via prior.

**Why this matters:** Without study effects, 5 observables from one paper get 5× the influence of a single-observable paper. The random effect appropriately downweights redundant information.

**Implementation options:**
1. **Full hierarchical:** Estimate `τ²` from data (requires sufficient studies)
2. **Fixed τ²:** Set `τ² = 0.1 × median(σ²_obs)` as rule of thumb
3. **Cluster-robust SEs:** Post-hoc adjustment without explicit random effects

---

### Baseline Variance (σ²_base)

The parameter `σ²_base` represents irreducible context-transfer uncertainty even when contexts match perfectly (d ≈ 0). It captures:
- Methodological variation across labs
- Biological variation not in context dimensions
- Residual model-reality mismatch

**Estimation approaches:**

| Approach | Method |
|----------|--------|
| **Empirical** | Estimate from observables with d < 0.1 (near-matched contexts) |
| **Prior-based** | Set `σ²_base = 0.1 × median(σ²_obs)` as fraction of typical measurement error |
| **Hierarchical** | `σ²_base ~ InverseGamma(α, β)` with weakly informative prior |

**Recommendation:** Start with prior-based, validate with residual analysis, upgrade to hierarchical if data permits.

---

### Variance Inflation Functional Form

Context mismatch variance is computed from a **weighted distance function**:

```
σ²_mismatch = σ²_base × f(λ × d(ctx_obs, ctx_model, class))
```

Where `λ` controls sensitivity (can be fixed or learned from data).

#### Why Exponential?

We use `f(x) = exp(x) - 1` (shifted exponential) because:

1. **Bounded at zero:** When d = 0, σ²_mismatch = 0 (no inflation for matched contexts)
2. **Monotonic:** Larger distance → larger variance (always)
3. **Convex:** Marginal penalty increases with distance (compounding uncertainty)
4. **Multiplicative interpretation:** `exp(λd)` corresponds to multiplicative uncertainty factors

#### Alternative Forms

| Form | Formula | Behavior |
|------|---------|----------|
| **Linear** | `λ × d` | Simple; may underweight large mismatches |
| **Quadratic** | `(λ × d)²` | Penalizes large distances more heavily |
| **Exponential** | `exp(λ × d) - 1` | Recommended; unbounded, convex |
| **Logistic** | `L / (1 + exp(-k(d - d₀)))` | Bounded maximum; use if inflation should plateau |

**Recommendation:** Use exponential as default. Include linear and quadratic in model comparison (Validation section) to empirically justify.

---

### Context Distance Function

The distance function uses **observable-class-specific weights** from the sensitivity tables:

```
d(ctx₁, ctx₂, class) = Σ_dim  w[class][dim] × d_dim(ctx₁[dim], ctx₂[dim])
```

**Dimension-specific distances:**

| Dimension | Distance Function |
|-----------|-------------------|
| **Species** | 0 if match, 1 if mismatch |
| **Indication** | `1 - similarity_matrix[ind₁][ind₂]` |
| **Compartment** | Normalized tree distance |
| **System** | Normalized tree distance |
| **Treatment** | Jaccard distance on multi-select sets |
| **Stage** | `|extent_ord₁ - extent_ord₂|/3 + |burden_ord₁ - burden_ord₂|/2` |

**Sensitivity weights** (`w[class][dim]`) come from the Observable Classes section below, mapped as:
- N/A → 0.0
- Low → 0.25
- Medium → 0.5
- High → 0.75
- Very High → 1.0

---

### Bias Corrections

For **known systematic relationships**, apply bias corrections before inference. Default to variance inflation only when no bias correction is available.

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
          # Immune kinetics: null (no scaling, use variance only)

  compartment:
    ratios:
      blood_to_tumor:
        IL10: {factor: 10, uncertainty: 0.5}
        TGFB: {factor: 100, uncertainty: 1.0}
        # Add more as identified from literature
```

---

### Indication Similarity Matrix

Used for `d_indication`. Symmetric; only lower triangle shown.

```yaml
indication_similarity:
  PDAC:
    PDAC: 1.0
  colorectal:
    PDAC: 0.7
    colorectal: 1.0
  gastric:
    PDAC: 0.6
    colorectal: 0.7
    gastric: 1.0
  lung_adeno:
    PDAC: 0.5
    colorectal: 0.5
    gastric: 0.5
    lung_adeno: 1.0
  melanoma:
    PDAC: 0.3
    colorectal: 0.3
    gastric: 0.3
    lung_adeno: 0.4
    melanoma: 1.0
  healthy:
    PDAC: 0.1
    colorectal: 0.1
    gastric: 0.1
    lung_adeno: 0.1
    melanoma: 0.2
    healthy: 1.0
  # Expand as needed
```

---

### Hierarchy Distances

For **Compartment** and **System**, use normalized tree distance:

```
d_tree(node₁, node₂) = (path_length(node₁, node₂)) / max_depth
```

Where `path_length` = number of edges to common ancestor × 2.

*Example:* `tumor.primary` vs `blood.plasma_serum` → common ancestor is root, path = 4, max_depth = 2, d = 1.0

*Example:* `tumor.primary` vs `tumor.metastatic` → common ancestor is `tumor`, path = 2, d = 0.5

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

## Validation Approaches

The context distance structure involves choices (sensitivity weights, similarity matrices, λ) that could be seen as arbitrary. The following validation strategies address this concern.

### 1. Sensitivity Analysis (Required)

Show inference results are **robust to perturbations**:

- Perturb sensitivity weights by ±25%
- Perturb similarity matrix entries by ±0.1
- Vary λ over reasonable range (e.g., 0.5–2.0)

**Success criterion:** Posterior parameter estimates stable within X% across perturbations.

### 2. Model Comparison (Required)

Compare predictive performance across structures using WAIC or LOO-CV:

| Model | Description |
|-------|-------------|
| **Null** | No context adjustment (all observables weighted equally) |
| **Proposed** | Weighted distance with sensitivity tables |
| **Learned** | Hierarchical model estimating weights from data |

**Success criterion:** Proposed structure outperforms Null; comparable to Learned (captures structure without overfitting).

### 3. Residual Calibration (Recommended)

Check that context distance **predicts residual variance**:

1. Fit model ignoring context mismatch
2. Compute residuals for each observable
3. Regress |residual| on context distance
4. Positive slope → distance function captures real heterogeneity

**Success criterion:** Significant positive relationship between distance and residual magnitude.

### 4. Literature Grounding (Required)

Ground each structural choice in published evidence:

| Component | Literature Support |
|-----------|-------------------|
| **Allometric scaling** | West et al. scaling laws; FDA cross-species guidance |
| **Indication similarity** | Tumor molecular profiling; TME clustering studies |
| **System hierarchy** | Translational pharmacology; in vitro-in vivo correlation reviews |
| **Observable sensitivities** | Meta-analyses reporting heterogeneity by context |

### 5. Hierarchical Learning (Ideal)

Put priors on structural parameters and let data inform them:

```
λ ~ HalfNormal(1)
w[class][dim] ~ Beta(α_prior, β_prior)
similarity[i,j] ~ Beta(α_ij, β_ij)
```

**Diagnostic checks:**
- Posteriors concentrated → data informative, structure identifiable
- Posteriors wide → appropriately uncertain
- Prior-to-posterior shift → data updated beliefs (not just prior echo)

### Validation Narrative

For publication, the recommended narrative:

> "We propose a context distance structure grounded in [allometric scaling literature, translational pharmacology]. Sensitivity analysis demonstrates robustness to ±25% perturbations in weights. Model comparison shows the proposed structure outperforms naive pooling (ΔWAIC = X) while avoiding overfitting relative to fully learned alternatives. When structural parameters are learned hierarchically, posteriors are consistent with our prior specification, indicating the structure captures meaningful heterogeneity."

---

## Open Questions

1. **Hard incompatibilities**: When to reject an observable entirely vs. allow heavy variance inflation? (e.g., in_vitro for survival endpoint)
2. **Learning λ**: Fix `λ` from prior knowledge, or estimate from data during inference?
3. **Bias correction coverage**: How to systematically identify when bias corrections exist vs. variance-only?
4. **Similarity matrix calibration**: Literature review vs. expert elicitation vs. data-driven?
5. **Interaction effects**: Should dimension distances be additive, or are there interaction terms (e.g., mouse + cell_line worse than sum)?
6. **Measurement method heterogeneity**: Flow cytometry vs IHC, ELISA vs Luminex, etc. give systematically different values. Currently not captured in context dimensions. Add as optional dimension, or handle via additional variance term?
