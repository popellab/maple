"""
MCP server for maple — exposes extraction guidance and validation tools.

Run with:
    python -m maple.mcp_server

Or configure in claude_desktop_config.json / .claude/settings.json.
"""

from pathlib import Path

from fastmcp import FastMCP

from maple.core.calibration.enums import (
    ExperimentalSystem,
    IndicationMatch,
    MeasurementDirectness,
    PerturbationType,
    SourceQuality,
    TemporalResolution,
    TMECompatibility,
)
from maple.core.resource_utils import read_prompt

mcp = FastMCP("maple")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enum_table(enum_cls, heading: str) -> str:
    """Render an enum as a markdown table with docstrings."""
    lines = [f"### {heading}\n", "| Value | Description |", "|-------|-------------|"]
    for member in enum_cls:
        doc = (member.__doc__ or "").strip().split("\n")[0]
        lines.append(f"| `{member.value}` | {doc} |")
    return "\n".join(lines)


def _build_enum_reference() -> str:
    """Build a complete enum reference from the actual Pydantic schema."""
    sections = [
        _enum_table(IndicationMatch, "indication_match"),
        _enum_table(SourceQuality, "source_quality"),
        _enum_table(PerturbationType, "perturbation_type"),
        _enum_table(TMECompatibility, "tme_compatibility"),
        _enum_table(MeasurementDirectness, "measurement_directness"),
        _enum_table(TemporalResolution, "temporal_resolution"),
        _enum_table(ExperimentalSystem, "experimental_system"),
    ]
    return "\n\n".join(sections)


def _build_extraction_workflow() -> str:
    """Standard workflow for multi-parameter extraction sessions."""
    return """\
## Extraction Session Workflow

When the user asks you to extract submodel targets for a set of parameters,
follow this sequence. Do NOT skip steps or reorder them.

### Step 0: Call `extract_target`

Load this guide before doing anything else. It contains the YAML schema,
valid enum values, and hard rules that validators enforce.

**Important:** SubmodelTarget and CalibrationTarget are distinct schemas.
SubmodelTargets constrain individual parameters via in vitro/ex vivo data
and submodel equations (forward model + error model). CalibrationTargets
are full-model observables (IHC densities, clinical outcomes) compared
against simulation output. There is no "IsolatedSystemTarget" — that name
is deprecated and should not be used.

### Step 1: Investigate parameters in model code

Before any literature search, look up each parameter in the model to understand:

- **What it represents mechanistically** (not just its name)
- **What units it uses** and whether they are standard (nM, cell/mL) or
  model-internal (dimensionless fractions, ratios, rates)
- **What the Hill function / equation input variable is** — e.g., DAMP_50
  gates on tumor death rate (cell/day), not DAMP concentration (nM)
- **How the parameter interacts with other parameters** — e.g.,
  `ECM_50_APC_mig = f_ECM_50_APC_mig * ECM_max`

Where to look:
- `parameters/parameters_PDAC.m` — parameter definitions, default values,
  units, and `.Notes` / `.derivedFrom` metadata
- `core/modules/*.m` — module code where the parameter appears in reactions,
  rules, or Hill functions
- `parameters/pdac_priors.csv` — current prior (median, σ, bounds)
- Existing submodel targets in `calibration_targets/submodel_targets/` for
  similar parameters

This step determines **what data you actually need** from the literature.
A parameter that uses death rate as its Hill input needs death rate data,
not concentration data. A dimensionless fraction needs different literature
than an absolute concentration.

### Step 2: Literature search

Now that you know what each parameter means and what data would constrain it:

**Launch one subagent per parameter** to search in parallel. Each subagent
should receive the parameter's mechanistic context from Step 1 and search
for papers with quantitative data that maps to the parameter's actual model
role (not just its name). This prevents the main context window from being
overwhelmed with search results and reduces the chance of missing important
details.

Each subagent should:
1. **Web search** for papers with quantitative dose-response, density, or
   measurement data that directly constrains the parameter
2. **Return** a summary with: parameter name, current prior, model role,
   and for each candidate paper: title, DOI, key quantitative data, and
   whether it maps cleanly to the model parameter

After all subagents return, **write consolidated results** to a notes file
(e.g., `notes/calibration/submodel_target_extraction_roundN.md`).

Flag parameters where the literature data does not map to the model's
parameterization — these need a different search strategy or may not
be extractable as direct submodel targets.

### Step 3: User obtains PDFs

The user will get the PDFs into `to-review/papers/<source_tag>/` directories
(one subfolder per source, named like `Smith2020`). Create the directory
structure for them if needed.

Wait for the user to confirm PDFs are in place before proceeding.

### Step 4: Extract targets one by one

For each parameter with good literature data:

1. Call `extract_target` if not already loaded
2. Read the PDF via the papers directory
3. Write the SubmodelTarget YAML to `to-review/<parameter_name>.yaml`
4. Call `validate_target` to check schema, prior derivation, and snippets
5. Fix any validation errors
6. Move to the next parameter

Work through parameters one at a time — do not batch-write YAMLs without
validating each one.
"""


def _build_extraction_rules() -> str:
    """Hard rules that LLMs frequently violate during extraction."""
    return """\
## Rules for SubmodelTarget extraction

These are hard constraints enforced by validators. Violating them wastes
a validation round-trip. Read before writing a YAML.

### 1. No invented uncertainties

Every numeric value in `observation_code` must trace to a named `input`.
The only allowed bare constants are: `0`, `1`, `2`, `1.96`.

**Wrong:** `sigma_log = 0.7` hardcoded in code, or an input with
`input_type: unit_conversion` whose value is an assumed CV / sigma.

**Right:** Derive spread from the data. If a paper gives subtype-level
values, bootstrap over them. If it gives mean ± SD, propagate the SD.
If uncertainty truly cannot be derived, the target may not be extractable.

### 2. Table data → table_excerpt

Values from paper tables MUST use `table_excerpt` with fields:
`table_id`, `column`, `row`, `value`, `context`.

Use `value_snippet` only for running text passages. The snippet validator
scores table_excerpt fields individually against the PDF (higher precision).

### 3. source_ref must match a source_tag

Every input's `source_ref` must be the `source_tag` of either the
`primary_data_source` or one of the `secondary_data_sources`.
There is no special `reference_db` source_ref.

For reference constants (cell diameters, section volumes), attribute
them to the paper whose methods define the relevant value (e.g., section
thickness) and use `input_type: unit_conversion` with a `rationale`.

### 4. input_type enum

Only three values: `direct_measurement`, `unit_conversion`, `reference_value`.

- `unit_conversion` requires a `rationale` field.
- Do NOT use `derived`, `reference_constant`, or any other value.

### 5. Observation code must use all data

Take full advantage of the extracted data rather than collapsing it into
a single point estimate. For example:
- If the paper reports values per subtype, bootstrap over subtypes with
  their reported prevalences.
- If the paper reports individual patient data or scatter plots, use the
  actual distribution.
- If the paper reports mean ± SD from n patients, use those directly in
  the bootstrap.

The goal is to let the data speak — the observation_code should be a
faithful statistical model of how the reported numbers were generated,
not a simplification that discards information.

### 6. Forward model must be JAX-traceable

`submodel_inference.py` evaluates the forward model inside NumPyro MCMC.
The code must use only `np.*` functions (mapped to `jax.numpy` at inference
time). No `scipy`, no `ureg`, no branching on parameter values.
"""


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("maple://prompts/submodel_target")
def submodel_target_prompt() -> str:
    """The full SubmodelTarget extraction prompt template from maple."""
    return read_prompt("submodel_target_prompt.md")


@mcp.resource("maple://prompts/calibration_target")
def calibration_target_prompt() -> str:
    """The full CalibrationTarget extraction prompt template from maple."""
    return read_prompt("calibration_target_prompt.md")


@mcp.resource("maple://schema/enums")
def schema_enums() -> str:
    """All valid enum values for SubmodelTarget and CalibrationTarget fields."""
    return _build_enum_reference()


@mcp.resource("maple://workflow/extraction")
def extraction_workflow() -> str:
    """Standard workflow for multi-parameter extraction sessions."""
    return _build_extraction_workflow()


@mcp.resource("maple://rules/extraction")
def extraction_rules() -> str:
    """Hard rules for writing SubmodelTarget YAMLs that LLMs frequently violate."""
    return _build_extraction_rules()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def extract_target(target_type: str = "submodel_target") -> str:
    """Get the complete extraction guide: workflow + prompt + enum reference + rules.

    Call this as soon as the user asks to extract, calibrate, or constrain
    model parameters from literature — BEFORE doing any literature search
    or writing any YAML. The guide includes the full multi-step workflow
    (model investigation → literature search → PDF collection → YAML
    extraction → validation), the prompt template, valid enum values,
    and hard rules that validators enforce.

    Args:
        target_type: "submodel_target" or "calibration_target"
    """
    if target_type == "submodel_target":
        prompt = read_prompt("submodel_target_prompt.md")
    elif target_type == "calibration_target":
        prompt = read_prompt("calibration_target_prompt.md")
    else:
        return f"Unknown target_type: {target_type}. Use 'submodel_target' or 'calibration_target'."

    return "\n\n---\n\n".join(
        [
            f"# {target_type} Extraction Guide",
            "# Workflow\n\n" + _build_extraction_workflow(),
            "# Extraction Prompt\n\n" + prompt,
            "# Valid Enum Values\n\n" + _build_enum_reference(),
            "# Hard Rules\n\n" + _build_extraction_rules(),
        ]
    )


@mcp.tool()
def validate_target(
    yaml_path: str,
    priors_csv: str,
    papers_dir: str | None = None,
) -> str:
    """Validate a SubmodelTarget or CalibrationTarget YAML file.

    Runs schema validation (Pydantic), prior derivation via NumPyro MCMC
    (bootstrap + forward model + distribution fitting + translation sigma),
    and snippet verification against source PDFs.

    Args:
        yaml_path: Absolute path to the YAML file to validate.
        priors_csv: Path to priors CSV (e.g., pdac_priors.csv).
            Required for SubmodelTarget prior derivation.
        papers_dir: Optional directory containing source PDFs in
            subdirectories named by source_tag (e.g., papers/Smith2020/).
            If not provided, defaults to ``<yaml_dir>/papers/``.

    Returns:
        Validation report with pass/fail status and details.
    """
    import yaml

    path = Path(yaml_path)
    if not path.exists():
        return f"File not found: {yaml_path}"

    with open(path) as f:
        data = yaml.safe_load(f)

    report_lines = [f"# Validation Report: {path.name}\n"]

    # ── Detect target type ──
    is_submodel = "target_id" in data and "calibration" in data
    is_calibration = "calibration_target_id" in data or "observable" in data

    if is_submodel:
        report_lines.append("**Type:** SubmodelTarget\n")
        report_lines.extend(_validate_submodel_full(path, priors_csv=Path(priors_csv)))
    elif is_calibration:
        report_lines.append("**Type:** CalibrationTarget\n")
        report_lines.extend(_validate_calibration(data))
    else:
        report_lines.append(
            "**Type:** Unknown (could not detect target_id or calibration_target_id)\n"
        )
        report_lines.append("FAIL: Cannot determine target type from YAML structure.")
        return "\n".join(report_lines)

    # ── Snippet validation (real validator) ──
    if is_submodel:
        report_lines.append("\n## Snippet Validation\n")
        pd = Path(papers_dir) if papers_dir else None
        report_lines.extend(_validate_snippets_real(path, pd))

    return "\n".join(report_lines)


def _validate_submodel_full(yaml_path: Path, priors_csv: Path) -> list[str]:
    """Run SubmodelTarget schema validation + MCMC prior derivation."""
    lines = []

    # Schema validation
    lines.append("## Schema Validation\n")
    try:
        from maple.core.calibration.submodel_target import SubmodelTarget

        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        target = SubmodelTarget.model_validate(data)
        lines.append("PASS: Schema validation succeeded.\n")
        lines.append(f"- target_id: `{target.target_id}`")
        for p in target.calibration.parameters:
            lines.append(f"- parameter: `{p.name}` ({p.units})")
        lines.append("")
    except Exception as e:
        lines.append(f"FAIL: Schema validation failed.\n```\n{e}\n```\n")
        return lines

    # Full prior derivation via MCMC
    lines.append("## Prior Derivation\n")
    try:
        from maple.core.calibration.yaml_to_prior import format_report, process_yaml

        results = process_yaml(yaml_path, priors_csv=priors_csv)
        errors = [r for r in results if "error" in r]
        successes = [r for r in results if "error" not in r]
        if errors and not successes:
            lines.append(f"FAIL: {errors[0]['error']}\n")
        else:
            lines.append("PASS: Prior derivation succeeded.\n")
            lines.append("```")
            for result in results:
                lines.append(format_report(result))
                lines.append("")
            lines.append("```\n")
    except Exception as e:
        lines.append(f"WARNING: Prior derivation failed: {e}\n")

    return lines


def _validate_calibration(data: dict) -> list[str]:
    """Run CalibrationTarget schema validation."""
    lines = []
    lines.append("## Schema Validation\n")
    try:
        from maple.core.calibration.calibration_target_models import CalibrationTarget

        target = CalibrationTarget.model_validate(data)
        lines.append("PASS: Schema validation succeeded.\n")
        lines.append(f"- calibration_target_id: `{target.calibration_target_id}`")
        lines.append("")
    except Exception as e:
        lines.append(f"FAIL: Schema validation failed.\n```\n{e}\n```\n")
    return lines


def _validate_snippets_real(yaml_path: Path, papers_dir: Path | None) -> list[str]:
    """Run the real snippet validator from maple.core.calibration."""
    lines = []
    try:
        from maple.core.calibration.snippet_validator import validate_snippets_in_file

        success, errors, skipped, passed, manual_review = validate_snippets_in_file(
            yaml_path, papers_dir
        )

        if success:
            lines.append(
                f"PASS: All snippets verified ({len(passed)} passed, {len(skipped)} skipped).\n"
            )
        else:
            lines.append(f"FAIL: {len(errors)} snippet errors.\n")

        for p in passed:
            lines.append(f"  OK: {p}")
        for s in skipped:
            lines.append(f"  SKIP: {s}")
        for e in errors:
            lines.append(f"  FAIL: {e}")
        for mr in manual_review:
            lines.append(f"  MANUAL: {mr}")

    except ImportError as e:
        lines.append(f"SKIP: Missing dependency for snippet validation: {e}")
    except Exception as e:
        lines.append(f"WARNING: Snippet validation failed: {e}")

    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run()


if __name__ == "__main__":
    main()
