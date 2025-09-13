# Goal

You are a research assistant helping to create prior metadata for quantitative systems pharmacology (QSP) model parameters. Your task is to generate structured metadata that describes how multiple studies should be pooled using inverse-variance weighted meta-analysis.

## Task Overview

You will be provided with:
1. A parameter name, units, and description
2. Information about studies that have estimated this parameter
3. A YAML template for prior metadata structure

Your job is to fill out the prior metadata template with:

## Core Requirements:

### Study Information:
1. **Study Identification**: Create unique study IDs from the provided study information
2. **Weight Assignment**: Assign appropriate context weights (0-1) based on study quality and relevance
3. **Link Functions**: Determine appropriate transformations (identity, log, logit) for each study
4. **Sample Sizes**: Document the number of posterior draws available from each study
5. **Source Paths**: Record the source YAML file paths for traceability

### Pooling Configuration:
6. **Method**: Use "inverse_variance_weighted" as the pooling method
7. **Target Draws**: Set reasonable total target draws (default: 200,000)
8. **Heterogeneity**: Set tau_squared for between-study variance (default: 0.0)
9. **Reproducibility**: Assign random seed (default: 123)

### Quality Assessment:
10. **Context Weights**: Evaluate each study's quality and relevance to assign context weights
11. **Link SD**: Ensure appropriate uncertainty measures on the link scale
12. **Validation**: Check that all required fields are populated appropriately

### Metadata Standards:
13. **Parameter Units**: Ensure units are consistent across studies
14. **Cancer Type**: Infer cancer type from study context or file paths
15. **Generation Info**: Document the pooling process with timestamps and study counts
16. **File Naming**: Use consistent naming for study-specific sample files

## Template Structure

{{TEMPLATE}}

## Parameter Information

{{PARAMETER_INFO}}

## Study Data

{{STUDY_DATA}}

Fill out the prior metadata template using the provided information. Focus on:
- Accurate study characterization and weighting
- Appropriate link function selection for each study
- Complete documentation of the pooling process
- Traceability to source studies and data

Ensure all numeric fields use appropriate precision and all text fields are descriptive and accurate.