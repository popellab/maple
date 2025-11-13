# Complete parameter definition and extraction workflow
# Extracts parameters to central qsp-metadata-storage repository

# STEP 1: Create parameter extraction requests
python ./scripts/prepare/create_parameter_batch.py ./batch_jobs/input_data/core_extraction_input_7bd60509.csv

python ./scripts/debug/inspect_jsonl.py batch_jobs/parameter_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/debug/extract_prompt.py batch_jobs/parameter_requests.jsonl 0

# Upload parameter extraction requests (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/run/upload_batch.py ./batch_jobs/parameter_requests.jsonl
python ./scripts/run/batch_monitor.py batch_{batch_id}
# Next step: Create checklist batch for QA review (see STEP 2 below)

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/run/upload_immediate.py ./batch_jobs/parameter_requests.jsonl
# Next step: Create checklist batch for QA review (see STEP 2 below)

# STEP 2: Quality assurance review - Checklist batch on raw JSON responses
# This catches packing errors early by validating raw LLM responses before unpacking
python ./scripts/prepare/create_checklist_from_json_batch.py ./batch_jobs/batch_{batch_id}_results.jsonl ./batch_jobs/input_data/core_extraction_input.csv

python ./scripts/debug/inspect_jsonl.py batch_jobs/checklist_from_json_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/debug/extract_prompt.py batch_jobs/checklist_from_json_requests.jsonl 0

# Upload checklist batch (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/run/upload_batch.py ./batch_jobs/checklist_from_json_requests.jsonl
python ./scripts/run/batch_monitor.py batch_{batch_id}
# Next step: Unpack reviewed results (see STEP 3 below)

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/run/upload_immediate.py ./batch_jobs/checklist_from_json_requests.jsonl
# Next step: Unpack reviewed results (see STEP 3 below)

# STEP 3: Unpack reviewed results to central metadata storage
# This unpacks the AI-reviewed JSON to YAML with header fields and 'ai-reviewed' tag
# Also generates a checklist review summary markdown file in scratch/
# Note: Add schema template path as 5th arg to specify schema version (determines header fields and filename format)
# Use corresponding results file based on upload method chosen above:
python ./scripts/process/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-metadata-storage/parameter_estimates ./batch_jobs/input_data/core_extraction_input.csv "" templates/parameter_metadata_template.yaml
# OR (if using immediate processing):
python ./scripts/process/unpack_results.py ./batch_jobs/checklist_from_json_requests_immediate_results.jsonl ../qsp-metadata-storage/parameter_estimates ./batch_jobs/input_data/core_extraction_input.csv "" templates/parameter_metadata_template.yaml

# Review the checklist summaries in scratch/checklist_reviews_*.md
# The YAML files are now in ../qsp-metadata-storage/parameter_estimates/ with 'ai-reviewed' tag

# ============================================================================
# ALTERNATIVE WORKFLOWS
# ============================================================================

# ALTERNATIVE: Lightweight JSON validation (quick structure check without deep review)
# This validates JSON structure and required fields, fixes syntax errors, but skips content validation
python ./scripts/prepare/create_json_validation_batch.py ./batch_jobs/batch_{batch_id}_results.jsonl ./batch_jobs/input_data/core_extraction_input.csv

python ./scripts/debug/inspect_jsonl.py batch_jobs/json_validation_requests.jsonl 1

# Upload validation batch (choose one method):
python ./scripts/run/upload_batch.py ./batch_jobs/json_validation_requests.jsonl
python ./scripts/run/batch_monitor.py batch_{batch_id}
# OR:
python ./scripts/run/upload_immediate.py ./batch_jobs/json_validation_requests.jsonl

# Unpack validated results with 'ai-validated' tag:
python ./scripts/process/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-metadata-storage/parameter_estimates ./batch_jobs/input_data/core_extraction_input.csv "" templates/parameter_metadata_template.yaml
# OR (if using immediate processing):
python ./scripts/process/unpack_results.py ./batch_jobs/json_validation_requests_immediate_results.jsonl ../qsp-metadata-storage/parameter_estimates ./batch_jobs/input_data/core_extraction_input.csv "" templates/parameter_metadata_template.yaml

# ALTERNATIVE: Unpack parameter extractions WITHOUT any validation
# (Skip STEP 2 and unpack raw results directly - not recommended)
python ./scripts/process/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-metadata-storage/parameter_estimates ./batch_jobs/input_data/core_extraction_input.csv "" templates/parameter_metadata_template.yaml
# OR (if using immediate processing):
python ./scripts/process/unpack_results.py ./batch_jobs/parameter_requests_immediate_results.jsonl ../qsp-metadata-storage/parameter_estimates ./batch_jobs/input_data/core_extraction_input.csv "" templates/parameter_metadata_template.yaml

# ALTERNATIVE: Checklist from unpacked YAML files (post-hoc review)
# (For reviewing parameter extractions already in ../qsp-metadata-storage/parameter_estimates/)
python ./scripts/prepare/create_checklist_batch.py ./batch_jobs/pdac_parameters_test.csv

python ./scripts/debug/inspect_jsonl.py batch_jobs/checklist_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/debug/extract_prompt.py batch_jobs/checklist_requests.jsonl 0

# Upload checklist batch (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/run/upload_batch.py ./batch_jobs/checklist_requests.jsonl
python ./scripts/run/batch_monitor.py batch_{batch_id}

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/run/upload_immediate.py ./batch_jobs/checklist_requests.jsonl

# Process checklist results manually - these are quality audit reports for review
# batch_jobs/batch_{batch_id}_results.jsonl (for batch processing)
# batch_jobs/checklist_requests_immediate_results.jsonl (for immediate processing)

# ============================================================================
# TEST STATISTIC WORKFLOW
# ============================================================================
# Generate test statistics for model validation from literature
# (Quantifies expected distributions of model-derived quantities based on experimental data)

# Create sample test statistics CSV (example format):
# test_statistic_id,model_context,scenario_context,species_formula
# tumor_volume_envelope,"QSPIO_PDAC model with tumor compartment V_T...","Nivolumab monotherapy in NSCLC patients...","V_T.TumorVolume"
# cd8_treg_ratio_peak,"QSPIO_PDAC model with immune cell populations...","GVAX + entinostat combination therapy...","V_T.T_eff / V_T.T_reg"
# See scratch/test_statistic_input_example.csv for complete examples

# Create test statistic batch requests
python ./scripts/prepare/create_test_statistic_batch.py ./batch_jobs/pdac_pretreatment_metrics.csv

# Optional: Include model context CSV with variable descriptions
# python ./scripts/prepare/create_test_statistic_batch.py ./batch_jobs/test_statistic_examples.csv ./data/model_variables.csv

python ./scripts/debug/inspect_jsonl.py batch_jobs/test_statistic_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/debug/extract_prompt.py batch_jobs/test_statistic_requests.jsonl 0

# Upload test statistic requests (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/run/upload_batch.py ./batch_jobs/test_statistic_requests.jsonl
python ./scripts/run/batch_monitor.py batch_{batch_id}

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/run/upload_immediate.py ./batch_jobs/test_statistic_requests.jsonl

# Unpack test statistic results - creates YAML test statistic definitions
# Use corresponding results file based on upload method chosen above:
python ./scripts/process/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-metadata-storage/test_statistics ./batch_jobs/pdac_pretreatment_metrics.csv "" templates/test_statistic_template.yaml
# OR (if using immediate processing):
python ./scripts/process/unpack_results.py ./batch_jobs/test_statistic_requests_immediate_results.jsonl ../qsp-metadata-storage/test_statistics ./batch_jobs/pdac_pretreatment_metrics.csv "" templates/test_statistic_template.yaml

# Note: Test statistic results are YAML files with statistical distributions and R code
# that can be integrated into your model validation framework
