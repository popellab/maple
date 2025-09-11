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

{{TEMPLATE}}

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

{{OLD_METADATA}}
