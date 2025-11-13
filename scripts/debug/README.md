# Debug Tools

This directory contains debugging utilities for inspecting batch requests, responses, and intermediate data files during workflow development.

## Scripts

### `inspect_jsonl.py`

Pretty print specific lines from JSONL batch files for debugging.

**Usage:**
```bash
python scripts/debug/inspect_jsonl.py <file.jsonl> [line_number]
```

**Features:**
- Pretty prints JSON structure with 2-space indentation
- Shows prompt preview (first 500 characters)
- Extracts and displays YAML content blocks from prompts
- Line numbers default to 1 if not specified

**Examples:**
```bash
# Inspect first request
python scripts/debug/inspect_jsonl.py batch_jobs/parameter_requests.jsonl

# Inspect 5th request
python scripts/debug/inspect_jsonl.py batch_jobs/parameter_requests.jsonl 5
```

### `extract_prompt.py`

Extract and save prompts from batch JSONL files for detailed examination.

**Usage:**
```bash
python scripts/debug/extract_prompt.py <jsonl_file> [request_index]
```

**Features:**
- Extracts prompt text from `body.input` field
- Saves to markdown file in `scratch/` directory
- Includes custom_id in filename and file header
- Prints prompt to stdout (for piping to other tools)
- Prints metadata to stderr (file path, custom_id, character count)

**Examples:**
```bash
# Extract first prompt to scratch/
python scripts/debug/extract_prompt.py batch_jobs/parameter_requests.jsonl

# Extract third prompt
python scripts/debug/extract_prompt.py batch_jobs/parameter_requests.jsonl 2

# Pipe to less for viewing
python scripts/debug/extract_prompt.py batch_jobs/parameter_requests.jsonl | less
```

### `pretty_print_csv.py`

Format CSV files with long text fields in a readable table format.

**Usage:**
```bash
python scripts/debug/pretty_print_csv.py <csv_file> [--width WIDTH]
```

**Features:**
- Text wrapping for long fields (default 70 characters)
- Numbered entries for easy reference
- Field names formatted as "Title Case"
- Shows empty fields explicitly

**Examples:**
```bash
# Print with default width
python scripts/debug/pretty_print_csv.py batch_jobs/input_data/test_stats.csv

# Custom wrap width
python scripts/debug/pretty_print_csv.py batch_jobs/input_data/test_stats.csv --width 100
```

## Common Use Cases

**Verify batch request structure:**
```bash
python scripts/debug/inspect_jsonl.py batch_jobs/parameter_requests.jsonl 1
```

**Debug prompt assembly issues:**
```bash
python scripts/debug/extract_prompt.py batch_jobs/parameter_requests.jsonl 0
```

**Review enriched CSV input:**
```bash
python scripts/debug/pretty_print_csv.py batch_jobs/input_data/pdac_extraction_input.csv
```

## Output Locations

- `inspect_jsonl.py`: Prints to stdout
- `extract_prompt.py`: Saves to `scratch/` directory, prints prompt to stdout
- `pretty_print_csv.py`: Prints to stdout
