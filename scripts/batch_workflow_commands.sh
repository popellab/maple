# Main parameter extraction workflow
# Extracts parameters to central qsp-parameter-storage repository

# Create batch requests from parameter CSV
python ./scripts/create_batch.py ./examples/input_format.csv

# Upload to OpenAI batch API
python ./scripts/upload_batch.py ./batch_jobs/batch_requests.jsonl

# Monitor batch progress and download when complete  
python ./scripts/batch_monitor.py batch_{batch_id}

# Unpack results to central parameter storage
python ./scripts/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-parameter-storage

# Optional: Create batch requests for missing pooling metadata
python ./scripts/create_pooling_metadata_batch.py ../qsp-parameter-storage/to-review

# Upload and monitor pooling metadata batch
python ./scripts/upload_batch.py ./batch_jobs/pooling_metadata_requests.jsonl
python ./scripts/batch_monitor.py batch_{batch_id}

# Unpack pooling metadata results
python ./scripts/unpack_results.py ./batch_jobs/batch_{batch_id}_results.jsonl ../qsp-parameter-storage
