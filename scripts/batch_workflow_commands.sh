# Complete parameter definition and extraction workflow
# Extracts parameters to central qsp-metadata-storage repository

# STEP 1: Create parameter definitions with canonical scales (run first)
# Note: Uses simbio_parameters.csv for Name/Units, loads definitions from data/parameter_definitions.csv
python ./scripts/create_parameter_definition_batch.py ./batch_jobs/pdac_parameters_modules.csv ./data/simbio_parameters.csv ./data/model_context.csv

python ./scripts/inspect_jsonl.py batch_jobs/parameter_definition_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/extract_prompt.py batch_jobs/parameter_definition_requests.jsonl 0

# Upload parameter definitions batch (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/upload_batch.py ./batch_jobs/parameter_definition_requests.jsonl
python ./scripts/batch_monitor.py batch_{batch_id}

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/upload_immediate.py ./batch_jobs/parameter_definition_requests.jsonl

# Unpack parameter definitions to storage (creates canonical scales)
# Use corresponding results file based on upload method chosen above:
python ./scripts/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-metadata-storage/parameter_estimates
# OR (if using immediate processing):
python ./scripts/unpack_results.py ./batch_jobs/parameter_definition_requests_immediate_results.jsonl ../qsp-metadata-storage/parameter_estimates

# STEP 2: Create parameter extraction requests (uses parameter definitions from step 1)
python ./scripts/create_parameter_batch.py ./batch_jobs/input_data/core_extraction_input_7bd60509.csv

python ./scripts/inspect_jsonl.py batch_jobs/parameter_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/extract_prompt.py batch_jobs/parameter_requests.jsonl 0

# Upload parameter extraction requests (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/upload_batch.py ./batch_jobs/parameter_requests.jsonl
python ./scripts/batch_monitor.py batch_{batch_id}

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/upload_immediate.py ./batch_jobs/parameter_requests.jsonl

# Unpack results to central metadata storage
# Use corresponding results file based on upload method chosen above:
python ./scripts/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-metadata-storage/parameter_estimates
# OR (if using immediate processing):
python ./scripts/unpack_results.py ./batch_jobs/parameter_requests_immediate_results.jsonl ../qsp-metadata-storage/parameter_estimates

# Optional: Create checklist batch requests for quality assurance auditing
# (Audits parameter extractions in ../qsp-metadata-storage/parameter_estimates/to-review/)
python ./scripts/create_checklist_batch.py ./batch_jobs/pdac_parameters_test.csv

python ./scripts/inspect_jsonl.py batch_jobs/checklist_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/extract_prompt.py batch_jobs/checklist_requests.jsonl 0

# Upload checklist batch (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/upload_batch.py ./batch_jobs/checklist_requests.jsonl
python ./scripts/batch_monitor.py batch_{batch_id}

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/upload_immediate.py ./batch_jobs/checklist_requests.jsonl

# Process checklist results manually - these are quality audit reports for review
# Use corresponding results file based on upload method chosen above:
# batch_jobs/batch_{batch_id}_results.jsonl (for batch processing)
# batch_jobs/checklist_requests_immediate_results.jsonl (for immediate processing)

# QUICK ESTIMATION WORKFLOW: Fast ballpark estimates with sources
# (Alternative to full parameter extraction when you need quick estimates)
python ./scripts/create_quick_estimate_batch.py ./batch_jobs/pd1_50.csv

python ./scripts/inspect_jsonl.py batch_jobs/quick_estimate_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/extract_prompt.py batch_jobs/quick_estimate_requests.jsonl 0

# Upload quick estimate requests (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/upload_batch.py ./batch_jobs/quick_estimate_requests.jsonl
python ./scripts/batch_monitor.py batch_{batch_id}

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/upload_immediate.py ./batch_jobs/quick_estimate_requests.jsonl

# Unpack quick estimate results to central metadata storage
# Use corresponding results file based on upload method chosen above:
python ./scripts/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-metadata-storage/parameter_estimates
# OR (if using immediate processing):
python ./scripts/unpack_results.py ./batch_jobs/quick_estimate_requests_immediate_results.jsonl ../qsp-metadata-storage/parameter_estimates

# TEST STATISTIC WORKFLOW: Generate test statistics for model validation from literature
# (Quantifies expected distributions of model-derived quantities based on experimental data)

# Create sample test statistics CSV (example format):
# test_statistic_id,model_context,scenario_context,species_formula
# tumor_volume_envelope,"QSPIO_PDAC model with tumor compartment V_T...","Nivolumab monotherapy in NSCLC patients...","V_T.TumorVolume"
# cd8_treg_ratio_peak,"QSPIO_PDAC model with immune cell populations...","GVAX + entinostat combination therapy...","V_T.T_eff / V_T.T_reg"
# See scratch/test_statistic_input_example.csv for complete examples

# Create test statistic batch requests
python ./scripts/create_test_statistic_batch.py ./batch_jobs/test_statistic_input_example.csv

# Optional: Include model context CSV with variable descriptions
# python ./scripts/create_test_statistic_batch.py ./batch_jobs/test_statistic_examples.csv ./data/model_variables.csv

python ./scripts/inspect_jsonl.py batch_jobs/test_statistic_requests.jsonl 1
# Optional: Extract prompt to examine more easily
python ./scripts/extract_prompt.py batch_jobs/test_statistic_requests.jsonl 0

# Upload test statistic requests (choose one method):
# Option 1: Batch processing (slower, handles large volumes)
python ./scripts/upload_batch.py ./batch_jobs/test_statistic_requests.jsonl
python ./scripts/batch_monitor.py batch_{batch_id}

# Option 2: Immediate processing (faster feedback, good for testing)
python ./scripts/upload_immediate.py ./batch_jobs/test_statistic_requests.jsonl

# Process test statistic results manually - these are YAML test statistic definitions
# Use corresponding results file based on upload method chosen above:
# batch_jobs/batch_{batch_id}_results.jsonl (for batch processing)
# batch_jobs/test_statistic_requests_immediate_results.jsonl (for immediate processing)

# Note: Test statistic results are YAML files with statistical distributions and R code
# that can be integrated into your model validation framework
