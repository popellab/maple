# Main parameter extraction workflow
# Replace ../qsp-project-name with the actual path to your QSP project

# Create batch requests from parameter CSV
python ./scripts/create_batch.py ./batch_jobs/red_parameters.csv

# Upload to OpenAI batch API
python ./scripts/upload_batch.py ./batch_jobs/batch_requests.jsonl

# Monitor batch progress and download when complete  
python ./scripts/batch_monitor.py batch_{batch_id} --download

# Unpack results to target QSP project
python ./scripts/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-project-name

# Generate parameters in target project (if applicable)
# cd ../qsp-project-name && ./scripts/generate_parameters.sh --review

# Optional: Create batch requests for missing pooling metadata
python ./scripts/create_pooling_metadata_batch.py ../qsp-project-name/to-review

# Upload and monitor pooling metadata batch
python ./scripts/upload_batch.py ./batch_jobs/pooling_metadata_requests.jsonl
python ./scripts/batch_monitor.py batch_{batch_id} --download

# Unpack pooling metadata results
python ./scripts/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-project-name
