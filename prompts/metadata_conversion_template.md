# Parameter Metadata Format Conversion Template

## Goal

You are tasked with converting parameter metadata from an old format to the new consolidated format. The new format consolidates redundant sections into two main metadata sections: `study_context` and `technical_details`.

## Conversion Instructions

### Old Format → New Format Mapping

**Fields to CONSOLIDATE:**
- `study_overview` → Extract biological context and parameter rationale → **`study_context`**
- `disease_context` → Extract relevant disease/system information → **`study_context`**
- `measurement_method` → Extract experimental methods → **`technical_details`** (Experimental Method subsection)
- `study_design` → Extract design details → **`technical_details`** (Study Design subsection)  
- `data_description` → Extract data details → **`technical_details`** (Data Description subsection)
- `other_derivation_notes` → Extract processing steps → **`technical_details`** (Data Processing subsection)

**Fields to MAINTAIN (unchanged):**
- `parameter_name` - Keep exactly as is
- `parameter_units` - Keep exactly as is  
- `parameter_definition` - Keep exactly as is
- `quality_metrics` - Keep exactly as is
- `sources` - Keep but ensure all sources have `citation` field, not `description`

**Fields with UPDATED STRUCTURE:**
- `mathematical_role` → Expand with **Governing Equation(s)**, **Parameter Role**, **Units Interpretation** subsections
- `parameter_estimates` → Use only allowed location types: `mean | median | mode` and uncertainty types: `CI95 | IQR | SD | SE`
- `derivation_process` → Combine old `location_derivation_code` + `scale_derivation_code` into single `location_and_uncertainty_derivation_code` field
- `key_study_limitations` → Format as bullet list with **Category:** descriptions

### Key Requirements for Conversion:

1. **Remove "We" language**: Change "We quantify..." to "This parameter quantifies..." and "The study..." 
2. **Add source attribution**: Include appropriate `(SOURCE_TAG)` references throughout both sections
3. **Eliminate redundancy**: Don't duplicate information between `study_context` and `technical_details`
4. **Use proper LaTeX formatting**: 
   - Use `$...$` delimiters for inline math (e.g., `$\pm$`, `$\geq$`, `$\times$`)
   - Use `$$...$$` for display equations
   - **NEVER use `\(...\)` delimiters** - always use `$...$` instead
   - Replace unicode symbols: ± → `$\pm$`, ≥ → `$\geq$`, × → `$\times$`, etc.
5. **Update mathematical_role**: Include governing equations, parameter role explanation, and units interpretation
6. **Consolidate R code**: Combine separate location/scale derivation into one comprehensive R code block
7. **Source format**: Ensure all sources have `citation` field instead of `description`
8. **Add source tags to limitations**: Include appropriate `(SOURCE_TAG)` references for each limitation category
9. **Maintain completeness**: Ensure no important information is lost during consolidation

### Complete New Format Template:

```yaml
parameter_name: "PARAMETER_NAME"
parameter_units: "UNITS"
parameter_definition: "Clear, concise, one-sentence definition of what this parameter represents biologically. Should specify the biological process, cell type/compartment, and any key modulating factors. Include units and typical range when helpful for interpretation."

mathematical_role: |
  **Governing Equation(s):**
  Provide the specific mathematical equation(s) where this parameter appears, using proper LaTeX formatting:
  $$\frac{d(\text{Species})}{dt} = k_{\text{param}} \cdot \text{[regulatory terms]} \cdot \text{[species concentrations]}$$
  
  **Parameter Role:**
  Describe exactly how this parameter affects the model dynamics. For rate constants, specify what process it controls (synthesis, degradation, transport, etc.). For Hill coefficients, EC50 values, or other modulating parameters, explain how they shape the dose-response relationship.
  
  **Units Interpretation:**
  Briefly explain what the units mean in biological context (e.g., "per day" = turnover rate, "nanomolar" = concentration for half-maximal effect).

study_context: |
  This parameter quantifies [WHAT] using data from [EXPERIMENTAL_SYSTEM]. The study [WHAT_WAS_DONE] (SOURCE_TAG). [MEASUREMENT_APPROACH] was used as [BIOLOGICAL_RATIONALE] (SOURCE_TAG). [HOW_PARAMETER_DERIVED_FROM_DATA] (SOURCE_TAG).
  
  **Context:** [SPECIES] [TISSUE/CELLS] studied [IN_VITRO/IN_VIVO]; [CONDITION_DETAILS].

technical_details: |
  **Experimental Method:**
  - Primary assay/measurement technique: [SPECIFIC_ASSAY_NAME] (SOURCE_TAG)
  - Detection method: [DETECTION_PRINCIPLE, e.g., ELISA, flow cytometry, microscopy] (SOURCE_TAG)
  - Instrumentation: [SPECIFIC_INSTRUMENTS_USED] (SOURCE_TAG)
  - Culture conditions: [MEDIUM, ATMOSPHERE, TEMPERATURE, CO2] for [DURATION] (SOURCE_TAG)
  - Cell seeding: [NUMBER] cells per [VESSEL_TYPE]; confluence at [TIMEPOINT] (SOURCE_TAG)
  - Treatment protocol: [CONCENTRATIONS, TIMING, DELIVERY_METHOD] (SOURCE_TAG)
  
  **Study Design:**
  - Sample size: [TOTAL_N] ([BREAKDOWN_BY_GROUP]) (SOURCE_TAG)
  - Biological replicates: [NUMBER] independent [CELL_PREPARATIONS/ANIMALS/DONORS] (SOURCE_TAG)
  - Technical replicates: [NUMBER] [WELLS/MEASUREMENTS] per biological replicate (SOURCE_TAG)
  - Study duration: [TIME] [UNITS] with measurements at [TIMEPOINTS] (SOURCE_TAG)
  - Experimental design: [RANDOMIZED/PAIRED/CROSSOVER] with [CONTROLS_DESCRIPTION] (SOURCE_TAG)
  - Potential covariates: [DONOR_AGE, PASSAGE_NUMBER, BATCH_EFFECTS, etc.]
  
  **Data Description:**
  Provide detailed description of experimental data used for parameter estimation:
  - **Raw measurements:** [EXACT_READOUTS with UNITS, CONDITIONS, TIMEPOINTS] (SOURCE_TAG)
  - **Sample information:** [SAMPLE_SIZES, REPLICATES, EXPERIMENTAL_GROUPS] (SOURCE_TAG)  
  - **Experimental conditions:** [CELL_TYPES, CULTURE_CONDITIONS, TREATMENTS, TIMING] (SOURCE_TAG)
  - **Measurement details:** [ASSAY_SPECIFICATIONS, DETECTION_LIMITS, CALIBRATION] (SOURCE_TAG)
  - **Quality control:** [EXCLUSION_CRITERIA, OUTLIER_HANDLING, VALIDATION] (SOURCE_TAG)
  
  **Data Processing:**
  - Raw data extraction: [METHOD, e.g., digitized from Figure X, provided in Table Y] (SOURCE_TAG)
  - Data transformations: [NORMALIZATIONS, LOG_TRANSFORMS, UNIT_CONVERSIONS] (SOURCE_TAG)
  - Statistical approach: [POINT_ESTIMATES, UNCERTAINTY_QUANTIFICATION] (SOURCE_TAG)
  - Computational tools: [SOFTWARE_USED, e.g., WebPlotDigitizer, R, GraphPad] (SOURCE_TAG)
  - Assumptions made: [KEY_ASSUMPTIONS_FOR_CALCULATIONS] (SOURCE_TAG)

quality_metrics:
  overall_quality: "HIGH | MEDIUM | LOW"  # Based on design, controls, validation
  relevance_to_target: "HIGH | MEDIUM | LOW"  # How well this matches target context
  
parameter_estimates:
  parameter_location_type: "mean | median | mode"        # MUST BE ONE OF THESE
  parameter_location_value: NUMERIC_OR_NULL               # Numerical value of location parameter
  parameter_uncertainty_type: "CI95 | IQR | SD | SE"     # MUST BE ONE OF THESE
  parameter_uncertainty_value: NUMERIC_OR_NULL            # Numerical value or [lower, upper] for CI95/IQR
  
  derivation_process:
    location_and_uncertainty_derivation_code: |           # Complete R code for parameter calculation:
      ```r
      set.seed(123)  # For reproducibility
      
      # Data input
      pairs <- list(c(1, 70.0, 65.0), c(2, 56.0, 40.0), c(3, 54.0, 51.0))
      x1 <- 1.0; x2 <- 4.0  # mg/mL
      
      # Parameter estimation
      ec50_values <- sapply(pairs, function(p) {
        y1 <- p[2]/100; y2 <- p[3]/100
        H1 <- 1.0 - y1; H2 <- 1.0 - y2
        A1 <- 1.0/H1 - 1.0; A2 <- 1.0/H2 - 1.0
        h <- log(A2/A1) / log(x2/x1)
        x1 * (A1^(1.0/h))
      })
      
      # Unit conversion: nM to mg/mL
      ec50_mgmL <- ec50_values * 450.5e-6  
      
      # Location and uncertainty
      location_value <- median(ec50_mgmL)  # or mean(log10(ec50_mgmL)) for log-transform
      uncertainty_bounds <- quantile(ec50_mgmL, c(0.025, 0.975))  # 95% CI
      
      print(location_value)
      print(uncertainty_bounds)
      ```

key_study_limitations: |
  List the most important limitations that could affect parameter reliability or generalizability:
  - **Experimental Design:** Sample size, controls, replication, statistical power (SOURCE_TAG)
  - **Biological Relevance:** Species differences (human vs. mouse), in vitro vs. in vivo, cell lines vs. primary cells (SOURCE_TAG)
  - **Technical Issues:** Assay limitations, measurement precision, data extraction methods (SOURCE_TAG)
  - **Modeling Assumptions:** Simplifications made during parameter derivation, fixed vs. fitted values (SOURCE_TAG)
  - **Uncertainty Sources:** Missing data, figure digitization, unit conversions, composite calculations (SOURCE_TAG)
  - **Generalizability:** Patient population, disease stage, treatment context differences (SOURCE_TAG)

sources:
  PRIMARY_KEY:
    citation: "FULL_CITATION"
    doi_or_url: "DOI_OR_URL_OR_NA"
    figure_or_table: "SPECIFIC_LOCATION (e.g., Fig. 1A; Table 2; p. 123 lines 15–20)"
    text_snippet: "Exact line of text where values came from, if available"
  SECONDARY_KEY:
    citation: "FULL_CITATION_FOR_SECONDARY_SOURCE"  # NOT description field
    doi_or_url: "DOI_OR_URL_OR_NA"
    figure_or_table: "SPECIFIC_LOCATION"
    text_snippet: "Exact line of text where values came from, if available"
```

## Conversion Process

1. **Read the old metadata carefully** and identify which information belongs in each new section
2. **Extract key content** from old fields without changing the scientific meaning
3. **Rewrite in the new structure** using the templates above
4. **Add source tags** throughout (use existing source keys or add new ones as needed)
5. **Check for redundancy** and eliminate duplicate information
6. **Verify completeness** - ensure no critical information was lost

## Instructions for Use

1. **Input**: Provide the old format parameter metadata below
2. **Process**: Apply the conversion mapping and rewrite according to new structure
3. **Output**: Return the converted metadata in the new consolidated format
4. **Verify**: Ensure no information loss and proper source attribution

---

# OLD PARAMETER METADATA TO CONVERT

[Paste the old format parameter metadata here]
