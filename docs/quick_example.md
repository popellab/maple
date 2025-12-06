  # 1. Export model definitions (replace with your actual model path)
  qsp-export-model \
    --matlab-model ../qspio-pdac/scripts/immune_oncology_model_PDAC.m \
    --output batch_jobs/input_data/model_definitions.json

  # 1b. Or use a saved simbiology project
  qsp-export-model \
    --simbiology-project batch_jobs/input_data/tnbc_model.sbproj \
    --output batch_jobs/input_data/model_definitions.json

  # 2. Enrich the example CSV
  qsp-enrich-csv parameter \
    docs/example_parameter_input.csv \
    batch_jobs/input_data/model_definitions.json \
    PDAC \
    -o batch_jobs/input_data/test_enriched.csv

  # 3. Run extraction with immediate mode (specify output directory)
  qsp-extract batch_jobs/input_data/test_enriched.csv --type parameter --output-dir metadata-storage --immediate --reasoning-effort low

  # 4. Check the results (results go to timestamped directory)
  ls metadata-storage/to-review  # You'll see a timestamped directory like: 20251123_143022_parameter_immediate/
  ls metadata-storage/to-review/20251123_*_parameter_immediate/  # List files in the extraction directory
  cat metadata-storage/to-review/20251123_*_parameter_immediate/<filename>.yaml  # Pick any file to inspect
