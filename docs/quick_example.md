  # 1. Export model definitions (replace with your actual model path)
  qsp-export-model \
    --matlab-model ../qspio-pdac/scripts/immune_oncology_model_PDAC.m \
    --output batch_jobs/input_data/model_definitions.json

  # 2. Enrich the example CSV
  qsp-enrich-csv parameter \
    docs/example_parameter_input.csv \
    batch_jobs/input_data/model_definitions.json \
    PDAC \
    -o batch_jobs/input_data/test_enriched.csv

  # 3. Run extraction with immediate mode
  qsp-extract batch_jobs/input_data/test_enriched.csv --type parameter --immediate

  # 4. Check the results (results go to timestamped directory)
  cd ../qsp-metadata-storage/to-review
  ls  # You'll see a timestamped directory like: 20251123_143022_parameter_immediate/
  ls 20251123_*_parameter_immediate/  # List files in the extraction directory
  cat 20251123_*_parameter_immediate/<filename>.yaml  # Pick any file to inspect
