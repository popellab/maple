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


def _build_schema_overview() -> str:
    """Generate the SubmodelTarget schema overview from the actual Pydantic models.

    This ensures the extraction prompt always reflects the current schema,
    avoiding drift between the prompt and the code.
    """
    from maple.core.calibration.submodel_target import (
        ExperimentalContext,
        InputType,
    )
    from maple.core.calibration.shared_models import SourceRelevanceAssessment

    def _field_names(model_cls) -> list[str]:
        return list(model_cls.model_fields.keys())

    def _enum_values(enum_cls) -> str:
        return " | ".join(m.value for m in enum_cls)

    input_type_vals = _enum_values(InputType)

    # Build SourceRelevanceAssessment field summary
    sra_fields = _field_names(SourceRelevanceAssessment)

    # Build ExperimentalContext field summary
    ec_fields = _field_names(ExperimentalContext)

    # Get forward model types from the ForwardModel union
    from maple.core.calibration.submodel_target import ForwardModel
    import typing

    fm_types = []
    for t in typing.get_args(typing.get_args(ForwardModel)[0]):
        type_field = t.model_fields.get("type")
        if type_field and type_field.default:
            fm_types.append(type_field.default)

    fm_type_str = " | ".join(fm_types)

    lines = [
        "```",
        "SubmodelTarget",
        "├── target_id: str",
        "├── inputs: List[Input]                    # Extracted values with provenance",
        "│   ├── name, value, units",
        f"│   ├── input_type: {input_type_vals}",
        "│   ├── rationale (required for unit_conversion, reference_value, derived_arithmetic)",
        "│   ├── source_inputs, formula (required for derived_arithmetic)",
        "│   ├── source_ref, source_location",
        "│   └── value_snippet | table_excerpt | figure_excerpt  # provenance (at least one required)",
        "├── calibration",
        "│   ├── parameters: [{name, units, nuisance?, prior?}]  # QSP priors from CSV; nuisance: inline prior",
        "│   ├── forward_model: ForwardModel        # Physics/math: params → predictions",
        f"│   │   ├── type: {fm_type_str}",
        "│   │   ├── state_variables: [{name, units, initial_condition}]  # For ODE models",
        "│   │   ├── independent_variable: {name, units, span}            # For ODE models",
        "│   │   └── (type-specific ParameterRole fields, or code for algebraic/custom_ode)",
        "│   ├── error_model: List[ErrorModel]      # Statistics: predictions + data → likelihood",
        "│   │   ├── name, units, uses_inputs",
        "│   │   ├── x_input: str                   # For direct_fit / power_law (independent variable input)",
        "│   │   ├── evaluation_points: [float]     # For ODE models only",
        "│   │   ├── sample_size_input: str          # Name of input providing sample size",
        "│   │   ├── observation_code: str           # def derive_observation(inputs, sample_size, rng, n_bootstrap) -> np.ndarray",
        "│   │   ├── n_bootstrap: int (default 10000)",
        "│   │   └── observable: Observable          # For ODE models (identity or custom transform)",
        "│   └── identifiability_notes: str",
        f"├── experimental_context: {{{', '.join(ec_fields)}}}",
        "├── source_relevance: SourceRelevanceAssessment  # REQUIRED - see below",
        f"│   └── {{{', '.join(sra_fields)}}}",
        "├── study_interpretation, key_assumptions, key_study_limitations",
        "├── primary_data_source: {doi, source_tag, title, authors, year}",
        "├── secondary_data_sources: [{doi|url, source_tag, title, ...}]",
        "└── tags, extraction_model  # optional metadata",
        "```",
    ]
    return "\n".join(lines)


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

### Step 2b: Verify DOIs and create directories

Call `verify_dois` with the candidate DOIs and source tags. This:
1. Resolves each DOI via CrossRef (catches hallucinated/wrong DOIs)
2. Reports title, authors, year, journal for confirmation
3. Creates `papers/<source_tag>/` directories

### Step 3: User obtains PDFs via Zotero

The user adds the papers to Zotero (via browser connector, DOI import, etc.).
Then call `fetch_papers_from_zotero` with the source tags — this copies
PDFs from Zotero's local storage into the paper directories automatically.

If fetch_papers_from_zotero can't find a PDF, the user may need to
download it manually into the `papers/<source_tag>/` directory.

### Step 4: Extract targets one by one

For each parameter with good literature data:

1. Call `extract_target` if not already loaded
2. Read the PDF via the papers directory
3. **Check figures for richer data.** If the paper has scatter plots with
   individual data points or dose-response curves with error bars, prefer
   digitizing those over using text-reported summary statistics. Describe
   the figure to the user (axes, scale, what to capture) and ask them to
   digitize with WebPlotDigitizer (WPD). Read the resulting CSV.
4. Write the SubmodelTarget YAML to `to-review/<parameter_name>.yaml`
5. Call `validate_target` to check schema, prior derivation, and snippets
6. Fix any validation errors
7. Move to the next parameter

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

### 4. input_type enum — strongly prefer `direct_measurement`

Four values: `direct_measurement`, `unit_conversion`, `reference_value`, `derived_arithmetic`.

**Strong preference for `direct_measurement`.** Every numeric value should trace
to a specific location in the paper with a checkable snippet or table excerpt.
Only `direct_measurement` and `derived_arithmetic` get full validation.
`reference_value` and `unit_conversion` bypass snippet checking and should be
used sparingly — only for genuine physical constants or unit conversions, never
for assumed CVs, uncertainty factors, or modeling assumptions. If you find
yourself reaching for `reference_value`, ask whether the data can be digitized
from a figure instead.

- `unit_conversion` and `reference_value` require a `rationale` field.
- `derived_arithmetic` requires `formula`, `source_inputs`, and `rationale` fields.
  The validator evaluates the formula against source input values and checks it
  matches the declared value (within 1%). Use this for deterministic conversions
  like `E = 3 * G'` or `n_obs = n_ROIs * n_gels`.
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
        # Inject auto-generated schema overview from Pydantic models
        prompt = prompt.replace("{{SCHEMA_OVERVIEW}}", _build_schema_overview())
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


@mcp.tool()
def run_joint_inference(
    priors_csv: str,
    submodel_dir: str,
    output_dir: str | None = None,
    glob_pattern: str = "*_PDAC_deriv*.yaml",
    num_warmup: int = 1000,
    num_samples: int = 5000,
    num_chains: int = 4,
) -> str:
    """Run joint MCMC inference across all SubmodelTarget YAMLs in a directory.

    Loads starting priors from the CSV (read-only — does NOT modify it),
    runs NumPyro NUTS across all targets jointly, fits marginal distributions,
    computes Gaussian copula for correlated parameters, and returns a
    diagnostic report.

    Outputs submodel_priors.yaml to output_dir if provided.

    Args:
        priors_csv: Path to base priors CSV (e.g., parameters/pdac_priors.csv).
            Read-only — never modified.
        submodel_dir: Directory containing SubmodelTarget YAML files.
        output_dir: Optional directory for submodel_priors.yaml output.
        glob_pattern: Glob pattern to match YAML files in submodel_dir.
        num_warmup: NUTS warmup iterations per chain.
        num_samples: Post-warmup samples per chain.
        num_chains: Number of MCMC chains.

    Returns:
        Diagnostic report with per-parameter marginals, contraction,
        copula correlations, and MCMC convergence diagnostics.
    """
    from maple.core.calibration.yaml_to_prior import process_targets

    yaml_dir = Path(submodel_dir)
    if not yaml_dir.exists():
        return f"Directory not found: {submodel_dir}"

    yamls = sorted(yaml_dir.glob(glob_pattern))
    if not yamls:
        return f"No YAML files matching '{glob_pattern}' in {submodel_dir}"

    csv_path = Path(priors_csv)
    if not csv_path.exists():
        return f"Priors CSV not found: {priors_csv}"

    out = Path(output_dir) if output_dir else None

    result = process_targets(
        priors_csv=csv_path,
        yaml_paths=yamls,
        output_dir=out,
        plot=False,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    return _format_joint_report(result, yamls)


def _format_joint_report(result: dict, yaml_paths: list[Path]) -> str:
    """Format joint inference result as a markdown report."""
    lines = ["# Joint Inference Report\n"]

    # Metadata
    meta = result.get("metadata", {})
    lines.append(f"**Targets:** {meta.get('n_targets', '?')}")
    lines.append(f"**Parameters:** {meta.get('n_parameters', '?')}")
    lines.append(f"**Samples:** {meta.get('n_samples', '?')}")
    lines.append(
        f"**MCMC:** {meta.get('num_warmup', '?')} warmup, "
        f"{meta.get('num_samples', '?')} samples, "
        f"{meta.get('num_chains', '?')} chains"
    )
    lines.append("")

    # Target files
    lines.append("## Targets\n")
    for p in yaml_paths:
        lines.append(f"- `{p.name}`")
    lines.append("")

    # Per-parameter marginals
    lines.append("## Parameter Marginals\n")
    lines.append(
        f"| {'Parameter':<30} | {'Distribution':<12} | "
        f"{'Median':>10} | {'CV':>6} | {'Source Targets'} |"
    )
    lines.append(f"|{'-'*30}:|{'-'*12}:|{'-'*10}:|{'-'*6}:|{'-'*40}|")

    for p in result.get("parameters", []):
        m = p["marginal"]
        sources = ", ".join(p.get("source_targets", []))
        lines.append(
            f"| `{p['name']:<28}` | {m['distribution']:<12} | "
            f"{m['median']:>10.4g} | {m['cv']:>6.2f} | {sources} |"
        )
    lines.append("")

    # Translation sigmas
    trans = result.get("translation_sigma", {})
    if trans:
        lines.append("## Translation Sigmas\n")
        for target_id, info in trans.items():
            breakdown = info.get("breakdown", {})
            breakdown_str = ", ".join(f"{k}=+{v:.2f}" for k, v in breakdown.items() if v > 0)
            lines.append(f"- **{target_id}**: total={info['total']:.3f} ({breakdown_str})")
        lines.append("")

    # Copula
    copula = result.get("copula")
    if copula:
        lines.append("## Gaussian Copula (significant correlations)\n")
        params = copula["parameters"]
        corr = copula["correlation"]
        lines.append(f"**Participants:** {len(params)} parameters\n")
        # Show off-diagonal entries above threshold
        pairs = []
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                r = corr[i][j]
                if abs(r) > 0.05:
                    pairs.append((params[i], params[j], r))
        if pairs:
            pairs.sort(key=lambda x: -abs(x[2]))
            lines.append(f"| {'Param A':<25} | {'Param B':<25} | {'r':>6} |")
            lines.append(f"|{'-'*25}:|{'-'*25}:|{'-'*6}:|")
            for a, b, r in pairs:
                lines.append(f"| `{a:<23}` | `{b:<23}` | {r:>+.3f} |")
        else:
            lines.append("No significant pairwise correlations above threshold.")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def compare_inference(
    priors_csv: str,
    submodel_dir: str,
    glob_pattern: str = "*_PDAC_deriv*.yaml",
    num_samples: int = 4000,
) -> str:
    """Run component-wise NPE inference across all SubmodelTargets.

    Finds connected components of targets, runs NPE on each via scipy
    simulation + sbi neural posterior estimation. Produces a comparison
    report with per-parameter contraction, tension detection, and
    diagnostics.

    Args:
        priors_csv: Path to base priors CSV (read-only).
        submodel_dir: Directory containing SubmodelTarget YAML files.
        glob_pattern: Glob pattern for YAML files.
        num_samples: Number of posterior samples per component.

    Returns:
        Markdown comparison report.
    """
    from maple.core.calibration.inference_comparison import run_comparison

    return run_comparison(
        priors_csv=priors_csv,
        submodel_dir=submodel_dir,
        glob_pattern=glob_pattern,
        num_samples=num_samples,
    )


# ---------------------------------------------------------------------------
# Paper management tools
# ---------------------------------------------------------------------------


@mcp.tool()
def verify_dois(
    dois: str,
    source_tags: str,
    papers_dir: str,
) -> str:
    """Verify DOIs via CrossRef and create paper directories.

    Call this after the literature search agent returns candidate papers.
    Verifies each DOI resolves, reports title/author/year, and creates
    ``papers/<source_tag>/`` directories for PDF placement.

    Args:
        dois: Comma-separated DOIs (e.g., "10.3390/ijms19103043,10.1155/2014/590654").
        source_tags: Comma-separated source tags (e.g., "Saga2018,Kawka2014").
            Must be same count as dois.
        papers_dir: Base directory for paper folders
            (e.g., "calibration_targets/submodel_targets/papers").

    Returns:
        Markdown report with verification results and created directories.
    """
    import json
    from pathlib import Path
    from urllib.error import URLError
    from urllib.request import urlopen

    dois = [d.strip() for d in dois.split(",")]
    source_tags = [t.strip() for t in source_tags.split(",")]

    if len(dois) != len(source_tags):
        return f"ERROR: dois ({len(dois)}) and source_tags ({len(source_tags)}) must be same length"

    papers_path = Path(papers_dir)
    lines = ["# DOI Verification Report\n"]

    for doi, tag in zip(dois, source_tags):
        lines.append(f"## {tag} — `{doi}`\n")

        try:
            url = f"https://api.crossref.org/works/{doi}"
            with urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())["message"]

            title = data.get("title", ["Unknown"])[0]
            authors = [a.get("family", "?") for a in data.get("author", [])]
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"

            year = "?"
            for date_field in ["published-print", "published-online", "created"]:
                if date_field in data:
                    year = data[date_field]["date-parts"][0][0]
                    break

            journal = data.get("container-title", ["?"])[0]

            lines.append(f"- **Title:** {title}")
            lines.append(f"- **Authors:** {author_str}")
            lines.append(f"- **Year:** {year}")
            lines.append(f"- **Journal:** {journal}")
            lines.append("- **Status:** VERIFIED\n")

        except (URLError, KeyError, json.JSONDecodeError) as e:
            lines.append(f"- **Status:** FAILED — {e}\n")

        # Create directory
        dir_path = papers_path / tag
        dir_path.mkdir(parents=True, exist_ok=True)
        lines.append(f"- **Directory:** `{dir_path}` (created)\n")

    lines.append("---\n")
    lines.append(
        "Add PDFs to Zotero, then call `fetch_papers_from_zotero` to pull them into these directories."
    )

    return "\n".join(lines)


@mcp.tool()
def fetch_papers_from_zotero(
    source_tags: str,
    papers_dir: str,
    zotero_storage: str = "~/Zotero/storage",
) -> str:
    """Fetch PDFs from local Zotero storage into paper directories.

    For each source_tag, searches Zotero for a matching item, finds the
    PDF attachment, and copies it from Zotero's local storage to
    ``papers/<source_tag>/``.

    Args:
        source_tags: Comma-separated source tags (e.g., "Saga2018,Magni2021").
        papers_dir: Base directory for paper folders.
        zotero_storage: Path to Zotero local storage (default: ~/Zotero/storage).

    Returns:
        Report of which PDFs were found and copied.
    """
    import shutil
    from pathlib import Path

    source_tags = [t.strip() for t in source_tags.split(",")]

    storage = Path(zotero_storage).expanduser()
    papers_path = Path(papers_dir)
    lines = ["# Zotero PDF Fetch Report\n"]

    if not storage.exists():
        return f"ERROR: Zotero storage not found at {storage}"

    for tag in source_tags:
        dest_dir = papers_path / tag
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Check if PDF already exists
        existing = list(dest_dir.glob("*.pdf"))
        if existing:
            lines.append(f"## {tag} — SKIP (already has {existing[0].name})\n")
            continue

        # Scan all Zotero storage folders for a PDF matching the tag
        # (e.g., "Saga" in filename for Saga2018)
        author = tag.rstrip("0123456789")  # Strip year digits
        found = False

        for key_dir in storage.iterdir():
            if not key_dir.is_dir():
                continue
            for pdf in key_dir.glob("*.pdf"):
                if author.lower() in pdf.name.lower():
                    shutil.copy2(pdf, dest_dir / pdf.name)
                    lines.append(f"## {tag} — COPIED\n")
                    lines.append(f"- **Source:** `{pdf}`")
                    lines.append(f"- **Dest:** `{dest_dir / pdf.name}`\n")
                    found = True
                    break
            if found:
                break

        if not found:
            lines.append(f"## {tag} — NOT FOUND\n")
            lines.append(f"- No PDF matching '{author}' found in Zotero storage.")
            lines.append("- Ensure the paper is in Zotero with a downloaded PDF.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run()


if __name__ == "__main__":
    main()
