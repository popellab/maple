  # 1. Export model definitions (replace with your actual model path)
  qsp-export-model \
    --matlab-model ../qspio-pdac/scripts/immune_oncology_model_PDAC.m \
    --output jobs/input_data/model_definitions.json

  # 1b. Or use a saved simbiology project
  qsp-export-model \
    --simbiology-project jobs/input_data/tnbc_model.sbproj \
    --output jobs/input_data/model_definitions.json

  # 2. Enrich the example CSV
  qsp-enrich-csv parameter \
    docs/example_parameter_input.csv \
    jobs/input_data/model_definitions.json \
    PDAC \
    -o jobs/input_data/test_enriched.csv

  # 3. Run extraction (specify output directory)
  qsp-extract jobs/input_data/test_enriched.csv --type parameter --output-dir metadata-storage --reasoning-effort low

  # 4. Check the results (results go to timestamped directory)
  ls metadata-storage/to-review  # You'll see a timestamped directory like: 20251123_143022_parameter_estimates/
  ls metadata-storage/to-review/20251123_*_parameter_estimates/  # List files in the extraction directory
  cat metadata-storage/to-review/20251123_*_parameter_estimates/<filename>.yaml  # Pick any file to inspect
