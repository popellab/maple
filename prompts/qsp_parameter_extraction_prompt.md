# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model. Your task is to create comprehensive, reproducible metadata for a model parameter by carefully analyzing scientific literature and experimental data.

For this parameter, you must:

## Core Parameter Extraction:
1. Extract precise numerical values with appropriate uncertainty measures (prefer CI95 > IQR > SD/SE)
2. Use only the specified uncertainty types: CI95, IQR, SD, SE (no custom or mixed types)
3. Provide complete, executable R code for all derivation calculations
4. Propagate uncertainty using appropriate statistical methods (bootstrap/Monte Carlo preferred)
5. Document all assumptions explicitly, especially for composite parameters

## Experimental Documentation:
6. Document the experimental methodology comprehensively in technical_details section
7. Provide detailed study context with biological rationale in study_context section
8. Capture complete sample information: biological replicates, technical replicates, total N
9. Include experimental conditions: cell types, culture conditions, treatment details, timepoints
10. Document data processing steps: transformations, normalizations, quality control measures

## Data Quality & Validation:
11. Assess data quality and relevance to the target biological system objectively
12. Verify all citations and text snippets are accurate and correspond to real publications
13. Cross-check figure/table references contain the claimed data
14. For digitized data: re-extract values independently to verify accuracy
15. Identify and categorize key limitations systematically (experimental, technical, modeling, generalizability)

## Mathematical Integration:
16. Provide governing equations showing exactly where the parameter appears in the model
17. Account for model modulators (e.g., Hill functions) that affect parameter interpretation
18. Explain parameter role in model dynamics (synthesis, degradation, regulation, etc.)
19. Ensure parameter definition is biologically precise and includes relevant context

## Source Attribution & Formatting:
20. Reference all sources by tag explicitly throughout the documentation
21. Provide doi_or_url fields for all sources (DOI preferred over URL when available)
22. Use proper LaTeX formatting for all mathematical expressions (no unicode characters)
23. Format data descriptions with specific examples and complete experimental details
24. Justify all numerical quantities with appropriate source citations

## Structure & Completeness:
25. Ensure no redundancy between study_context and technical_details sections
26. Verify all required YAML fields are populated appropriately
27. Use consistent terminology and parameter naming throughout
28. Provide sufficient detail for independent replication of the derivation

Focus on scientific rigor, reproducibility, and transparency. Prefer raw data over summary statistics when available, and always show your mathematical work.

Below is the metadata template for your reference:

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
  parameter_location_type: "mean | median | mode"        # Central tendency measure used (MUST BE ONE OF THESE)
  parameter_location_value: NUMERIC_OR_NULL               # Numerical value of location parameter
  parameter_uncertainty_type: "CI95 | IQR | SD | SE"           # Uncertainty measure type (standard deviation, standard error, confidence interval) (MUST BE ONE OF THESE)
  parameter_uncertainty_value: NUMERIC_OR_NULL                 # Numerical value of scale parameter
  
  derivation_process:
    location_and_uncertainty_derivation_code: |                   # Complete R code for parameter calculation:
      ```r
      pairs <- list(c(1, 70.0, 65.0), c(2, 56.0, 40.0), c(3, 54.0, 51.0))
      x1 <- 1.0; x2 <- 4.0  # mg/mL
      ec50_values <- sapply(pairs, function(p) {
        y1 <- p[2]/100; y2 <- p[3]/100
        H1 <- 1.0 - y1; H2 <- 1.0 - y2
        A1 <- 1.0/H1 - 1.0; A2 <- 1.0/H2 - 1.0
        h <- log(A2/A1) / log(x2/x1)
        x1 * (A1^(1.0/h))
      })
      # Convert nM to mg/mL: ec50_values * 450.5e-6  
      ec50_mgmL <- ec50_values * 450.5e-6
      mean(log10(ec50_mgmL))  # Log-transform due to right-skew
      sd(log10(ec50_mgmL)) / sqrt(length(ec50_mgmL))  # SE of log10 values
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
    citation: "WHY_INCLUDED (e.g., assay datasheet, biomarker stoichiometry)"
    doi_or_url: "DOI_OR_URL_OR_NA"
    figure_or_table: "SPECIFIC_LOCATION (e.g., Fig. 1A; Table 2; p. 123 lines 15–20)"
    text_snippet: "Exact line of text where values came from, if available"
```
**Notes**
Always calculate propagate uncertainty via bootstrap or other Monte Carlo uncertainty propagation methods, if at all possible. If raw data available, use non-parametric. Otherwise, assume a reasonable distribution with the other parameters.
Prefer raw data to summary_statistics, if at all possible.
Prefer CI95 to IQR to SD/SE for uncertainty reporting. If raw data is available, this is how summary statistics should be calculated in order of preference. 

# Example

Below is an example of a well constructed metadata:

```yaml
parameter_name: "k_ECM_fib_sec"
parameter_units: "nanomole/cell/day"
parameter_definition: "Maximal per-fibroblast type-I collagen formation rate (via PICP) by normal fibroblasts (human PSCs) in the tumor compartment under TGF-$\\beta$ saturated conditions ($V_{max}$)."

mathematical_role: |
  **Governing Equation:**
  $$\frac{d(\text{ECM})}{dt} = k_{\text{ECM,fib,sec}} \cdot \frac{\text{ECM}_{\max} - \text{ECM}}{\text{ECM}_{\max}} \cdot V_{T,\text{Fib}} \cdot \frac{V_{T,\text{TGF}\beta}}{V_{T,\text{TGF}\beta} + \text{TGF}\beta_{50}}$$

  **Parameter Estimation:**
  
  We first estimate the basal PSC rate $r_0$ from PICP accumulation (normoxia), then set:
  
  $$k_{\text{ECM,fib,sec}} = r_0 \cdot f_{\text{sat}}$$
  
  where:
  - $r_0$ = basal PSC production rate  
  - $f_{\text{sat}}$ = fold-increase under TGF-$\beta$ saturation
  - $f_{\text{sat}} \approx \frac{1}{H_0}$ where $H_0 = \frac{\text{TGF}\beta_0}{\text{TGF}\beta_0 + \text{TGF}\beta_{50}}$
  
  This procedure recovers $V_{max}$ since $r_0 = V_{max} \cdot H_0$.

study_context: |
  This parameter quantifies the maximum collagen secretion rate using data from primary human PSCs (culture-activated, CAF-like) isolated from PDAC resections (MASAMUNE_2008_PSC_PICP). The study cultured PSCs to confluence in 24-well plates with supernatant collection at 24h and 48h (MASAMUNE_2008_PSC_PICP). **PICP** (procollagen I C-peptide) ELISA was used as a direct proxy for collagen synthesis due to 1:1 stoichiometry with collagen formation (PICP_EQUIMOLAR_REVIEW). To estimate $V_{\max}$ under TGF-$\beta$ saturation, basal PICP rates were scaled by fold-increases from independent PSC TGF-$\beta$ stimulation studies (PSC_TGFB_FOLD_HIGH; PSC_TGFB_FOLD_LOWER).
  
  **Context:** Human primary PSCs (culture-activated, CAF-like) studied *in vitro*; baseline production scaled to TGF-$\beta$ saturated conditions.

technical_details: |
  **Experimental Method:**
  - PICP sandwich ELISA on PSC culture supernatants at 24h and 48h timepoints (MASAMUNE_2008_PSC_PICP)
  - Primary human PSCs plated to confluence in 24-well plates; $\geq$3 independent PSC preparations (MASAMUNE_2008_PSC_PICP)
  - TGF-$\beta$ saturation scaling: 170-190% increase (PSC_TGFB_FOLD_HIGH) and 34% increase (PSC_TGFB_FOLD_LOWER)
  
  **Study Design:**
  - Sample size: not reported (panel-specific $n$ not printed for Masamune Fig. 2C) (MASAMUNE_2008_PSC_PICP)
  - Biological replicates: $\geq$3 experiments; PSCs from $\geq$3 independent preparations (MASAMUNE_2008_PSC_PICP)
  - Study duration: 24-48 hours *in vitro* (MASAMUNE_2008_PSC_PICP)
  - Potential covariates: oxygen tension, donor/preparation, passage (3-7), cells/well at confluence, medium volume
  
  **Data Processing:**
  - PICP means $\pm$ SD digitized from Fig. 2C: 24h $\approx$ 40 $\pm$ 8 ng/mL; 48h $\approx$ 95 $\pm$ 12 ng/mL (MASAMUNE_2008_PSC_PICP)
  - Working volume: Uniform[0.38, 0.57] mL for 24-well plates (CLS_AN_209; CORNING_24WELL_SPEC)
  - Growth area: 1.9 cm$^2$ with average $1.9 \times 10^5$ cells/well (CV = 0.30) (CLS_AN_209; CORNING_24WELL_SPEC)
  - PICP molecular weight: 100,000 g/mol with equimolar PICP:collagen-I assumption (PICP_MW_VENDOR; PICP_EQUIMOLAR_REVIEW)
  - TGF-$\beta$ saturation fold: 70% $\times$ Uniform[1.7,1.9] + 30% $\times$ 1.34 (PSC_TGFB_FOLD_HIGH; PSC_TGFB_FOLD_LOWER)
  - Monte Carlo computation: 400k draws; $V_{\max} = r_0 \times f_{\text{sat}}$


quality_metrics:
  overall_quality: "MEDIUM"
  relevance_to_target: "HIGH"

parameter_estimates:
  parameter_location_type: "median"
  parameter_location_value: 1.852e-09
  parameter_uncertainty_type: "CI95" 
  parameter_uncertainty_value: [8.172e-10, 3.889e-09]

  derivation_process:
    location_and_uncertainty_derivation_code: |
      ```r
      set.seed(123)

      means <- c(`24h`=40.0, `48h`=95.0)   # ng/mL  [MASAMUNE_2008_PSC_PICP]
      sds   <- c(`24h`= 8.0, `48h`=12.0)   # ng/mL  [MASAMUNE_2008_PSC_PICP]
      V_low <- 0.38; V_high <- 0.57        # mL     [CLS_AN_209; CORNING_24WELL_SPEC]

      mean_cells <- 1.9e5                  # cells/well  [CLS_AN_209]
      cv_cells <- 0.30                     # prior CV    [SEEDING_CV_GUIDELINE; CELL_COUNTING_VARIABILITY_REVIEW; PSC_METHODS_VARIABILITY_REVIEW]
      sigma_ln <- sqrt(log(1 + cv_cells^2))
      mu_ln    <- log(mean_cells) - 0.5 * sigma_ln^2

      sample_rate <- function(mean_ngml, sd_ngml, t_hours, n, MW_PICP=100000){ # MW 100 kDa [PICP_MW_VENDOR]
        conc  <- pmax(rnorm(n, mean_ngml, sd_ngml), 0.1)   # ng/mL
        V     <- runif(n, V_low, V_high)                   # mL
        cells <- rlnorm(n, meanlog=mu_ln, sdlog=sigma_ln)
        days  <- t_hours / 24
        # nmol/cell/day = (ng/mL * mL * 1e-9 g/ng / MW[g/mol]) / cells / days * 1e9
        (conc * V * 1e-9 / MW_PICP) / cells / days * 1e9
      }

      N <- 400000L
      r24 <- sample_rate(means["24h"], sds["24h"], 24, N %/% 2, 100000)
      r48 <- sample_rate(means["48h"], sds["48h"], 48, N %/% 2, 100000)
      r_base <- c(r24, r48)

      u <- runif(length(r_base))
      f_sat <- ifelse(u < 0.70, runif(length(r_base), 1.7, 1.9), 1.34)  # [PSC_TGFB_FOLD_HIGH; PSC_TGFB_FOLD_LOWER]

      k_vec <- r_base * f_sat

      k_median <- median(k_vec)
      ci <- quantile(k_vec, c(0.025, 0.975), names = FALSE)

      print(k_median)
      print(ci)
      ```
    other_derivation_notes: |
      - Error bars in Masamune Fig. 2C were treated as SD; values were digitized from the PDF [MASAMUNE_2008_PSC_PICP].
      - Plate working volume and cells/well handled via priors (manufacturer guidance and yield table) [CLS_AN_209; CORNING_24WELL_SPEC].
      - **Assumptions**: (i) 1:1 PICP:COL-I stoichiometry [PICP_EQUIMOLAR_REVIEW]; (ii) MW(PICP)=100 kDa (primary) [PICP_MW_VENDOR];
      (iii) basal PSCs may exhibit autocrine TGF-$\beta$ so $r_0 = V_{max} \cdot H_0$ and $f_{\text{sat}} \approx 1/H_0$; (iv) independence among priors.

key_study_limitations: |
  - **Sample Size Reporting:** Panel-specific $n$ not explicitly printed; only "$\geq$3 experiments / $\geq$3 preparations" overall [MASAMUNE_2008_PSC_PICP]
  - **Data Extraction:** Figure digitization introduces small readout error; raw numeric PICP values not listed in text [MASAMUNE_2008_PSC_PICP]
  - **Experimental Parameters:** Cells/well and working volume not stated for this panel; handled via priors [CLS_AN_209; CORNING_24WELL_SPEC]
  - **TGF-$\beta$ Fold Prior:** $f_{\text{sat}}$ is a cross-study mixture prior (PSC data from other labs); dose/time/stiffness conditions may differ [PSC_TGFB_FOLD_HIGH; PSC_TGFB_FOLD_LOWER]
  - **ECM Composition:** Collagen I used as ECM proxy; excludes fibronectin, COL3/IV, and other matrix components

sources:
  MASAMUNE_2008_PSC_PICP:
    citation: "Masamune A, Kikuta K, Watanabe T, et al. Hypoxia stimulates pancreatic stellate cells to induce fibrosis and angiogenesis in pancreas. Am J Physiol Gastrointest Liver Physiol. 2008;295(4):G709–G717."
    doi_or_url: "10.1152/ajpgi.90356.2008"
    figure_or_table: "Fig. 2C; Methods 'Collagen production/ELISA'"
    text_snippet: "Human PSCs were plated in 24-well plates and grown to confluency… PICP level… determined by ELISA (Takara Bio)… measured at 24 and 48 h."
  PSC_TGFB_FOLD_HIGH:
    citation: "Pomianowska E, Sandnes D, Grzyb K, et al. Inhibitory effects of prostaglandin E2 on collagen synthesis and cell proliferation in human stellate cells from pancreatic head adenocarcinoma. BMC Cancer. 2014;14:413."
    doi_or_url: "10.1186/1471-2407-14-413"
    figure_or_table: "Fig. 6A (collagen synthesis % of control; n$\approx$5 experiments)"
    text_snippet: "TGF-$\\beta_1$ increased collagen synthesis to $\\sim$170–190% of control."
  PSC_TGFB_FOLD_LOWER:
    citation: "Shek FWT, Benyon RC, Walker FM, et al. Expression of transforming growth factor-$\beta_1$ by pancreatic stellate cells and its implications for matrix secretion and turnover in chronic pancreatitis. Am J Pathol. 2002;160(5):1787–1798."
    doi_or_url: "10.1016/S0002-9440(10)61125-X"
    figure_or_table: "Results text"
    text_snippet: "Exogenous TGF-$\\beta_1$ (10 ng/mL) increased collagen protein synthesis by $\\sim$34%."
  CLS_AN_209:
    citation: "Corning Application Note CLS-AN-209. Surface Areas and Guide for Recommended Medium Volumes for Corning Cell Culture Vessels. Rev3 (6/2023)."
    doi_or_url: "https://www.corning.com/catalog/cls/documents/application-notes/CLS-AN-209.pdf"
    figure_or_table: "Tables for multiwell plates"
    text_snippet: "24-well: growth area 1.9 cm$^2$; working volume 0.38–0.57 mL; average cell yield $1.9 \\times 10^5$ cells/well; guidance that $\\geq 1 \\times 10^5$ cells/cm$^2$ can be produced in attached monolayers."
  CORNING_24WELL_SPEC:
    citation: "Corning Costar 24-well Clear TC-treated Multiple Well Plate, Product 3524."
    doi_or_url: "https://ecatalog.corning.com/life-sciences/b2c/US/en/Microplates/Assay-Microplates/96-Well-Microplates/Costar%C2%AE-Multiple-Well-Cell-Culture-Plates/p/3524"
    figure_or_table: "Specifications"
    text_snippet: "Cell Growth Area $\\approx$ 1.9 cm$^2$; Recommended Medium Well Volume 0.38–0.57 mL."
  PICP_EQUIMOLAR_REVIEW:
    citation: "Gillett MJ, Vasikaran SD, Inderjeeth CA. The Role of PINP in Diagnosis and Management of Metabolic Bone Disease. Clin Biochem Rev. 2021;42(1):3–10."
    doi_or_url: "10.33176/AACB-20-0001"
    figure_or_table: "Review text"
    text_snippet: "PINP and PICP molecules are produced in equimolar amounts with collagen-I."
  PICP_MW_VENDOR:
    citation: "ThermoFisher datasheet: Procollagen Type I C-Peptide (PIP) antibody (clone PC5-6)."
    doi_or_url: "https://www.thermofisher.com/antibody/product/Procollagen-Type-I-C-Peptide-PIP-Antibody-clone-PC5-6-Monoclonal/42024-100UG"
    figure_or_table: "Product datasheet"
    text_snippet: "Procollagen type I C-terminal propeptide (PICP) has a molecular weight of approximately 100 kDa."
  SEEDING_CV_GUIDELINE:
    citation: "Roper SJ, et al. Establishing an In Vitro 3D Spheroid Model to Study Drug Response. Curr Protoc. 2022;2(1):e357."
    doi_or_url: "10.1002/cpz1.357"
    figure_or_table: "Methods/guidance"
    text_snippet: "CV values >20% indicate inconsistencies in cell seeding (used to justify conservative CV prior)."
  CELL_COUNTING_VARIABILITY_REVIEW:
    citation: "Richards C, Sarkar S, Kandell J, Snyder R, Lakshmipathy U. Assessing the suitability of cell counting methods during different stages of a cell processing workflow using an ISO 20391-2 guided study design and analysis. Front Bioeng Biotechnol. 2023;11:1223227."
    doi_or_url: "10.3389/fbioe.2023.1223227"
    figure_or_table: "Methods comparison/variability"
    text_snippet: "Quantifies variability/precision across counting methods; supports inclusion of counting variability in CV prior."
  PSC_METHODS_VARIABILITY_REVIEW:
    citation: "Erkan M, Adler G, Apte MV, et al. StellaTUM: current consensus and discussion on pancreatic stellate cell research. Gut. 2012;61(2):172–178."
    doi_or_url: "10.1136/gutjnl-2011-301220"
    figure_or_table: "Review text"
    text_snippet: "PSC methodology varies across laboratories; donor/preparation variability noted."
```
# PARAMETER INFORMATION

## PARAMETER_TO_SEARCH:
[Parameter name, units, and definition will be provided]

## MODEL_CONTEXT:
[Mathematical role and biological context will be provided]

Fill out the provided YAML metadata template given this parameter information.
