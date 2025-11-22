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

  # 3. Run extraction with immediate mode (no push for testing)



  # 4. Check the results
  cd ../qsp-metadata-storage
  ls to-review/
  cat to-review/<filename>.yaml  # Pick any file to inspect
