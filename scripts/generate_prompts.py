#!/usr/bin/env python3
# build_prompts_v3.py
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# ------------------------- Utilities -------------------------

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", str(text).strip())
    text = re.sub(r"_{2,}", "_", text)
    return (text.strip("_") or "parameter")[:80]

def clean_template(text: str) -> str:
    # Remove placeholder lines like "[... will be provided]"
    cleaned = re.sub(r"(?im)^.*will be provided.*\n?", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse >2 blank lines
    return cleaned

def load_inputs(params_path: Path, reactions_path: Path, template_path: Path):
    params_df = pd.read_csv(params_path)
    reactions_df = pd.read_csv(reactions_path)
    template_text = template_path.read_text()
    need_params = {"Name", "Units", "Definition"}
    need_rxns = {"Parameter", "Reaction", "ReactionRate", "OtherParameters", "OtherSpeciesWithNotes"}
    miss_p = need_params - set(params_df.columns)
    miss_r = need_rxns - set(reactions_df.columns)
    if miss_p:
        raise ValueError(f"Parameters CSV missing columns: {sorted(miss_p)}")
    if miss_r:
        raise ValueError(f"Reactions CSV missing columns: {sorted(miss_r)}")
    return params_df, reactions_df, template_text

def index_param_info(params_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    return (
        params_df.assign(Name=params_df["Name"].astype(str))
                 .set_index("Name")[["Units", "Definition", "References"]]
                 .to_dict(orient="index")
    )

def parse_other_params_list(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    # Extract names from a string like "['k1','k2']"
    return re.findall(r"'([^']+)'", s)

# ------------------------- Rendering Blocks -------------------------

def render_parameter_to_search(name: str, units: str, definition: str) -> str:
    """Explain fields, then show the parameter descriptor line."""
    explanation = (
        "**Field meaning:**\n"
        "- **Name**: The model parameter identifier to focus on.\n"
        "- **Units**: The unit system used for this parameter.\n"
        "- **Definition**: Short description of what this parameter represents.\n"
    )
    descriptor = f"**Parameter:** {name} [{units}] — {definition}".strip(" —")
    return explanation + "\n" + descriptor

def render_other_params_details(other_params: List[str], param_info: Dict[str, Dict[str, str]]) -> str:
    if not other_params:
        return "  - **Other parameters details:** —"
    lines = []
    for nm in sorted(set(other_params)):
        info = param_info.get(nm, {}) or {}
        units = (info.get("Units") or "").strip()
        definition = (info.get("Definition") or "").strip()
        if units and definition:
            lines.append(f"    - **{nm}** [{units}] — {definition}")
        elif definition:
            lines.append(f"    - **{nm}** — {definition}")
        elif units:
            lines.append(f"    - **{nm}** [{units}]")
        else:
            lines.append(f"    - **{nm}**")
    return "  - **Other parameters details:**\n" + "\n".join(lines)

def render_species_comp_details(species_json_str: str) -> str:
    items: List[Dict[str, Any]] = []
    if isinstance(species_json_str, str) and species_json_str.strip():
        try:
            parsed = json.loads(species_json_str)
            if isinstance(parsed, list):
                items = [x for x in parsed if isinstance(x, dict)]
        except Exception:
            items = []
    if not items:
        return "  - **Other species/compartment details:** —"

    # Deduplicate by (name, notes) while preserving order
    seen = set()
    lines = []
    for it in items:
        name = str(it.get("name", "")).strip()
        notes = str(it.get("notes", "")).strip()
        if not name:
            continue
        key = (name, notes)
        if key in seen:
            continue
        seen.add(key)
        if notes:
            lines.append(f"    - **{name}** — {notes}")
        else:
            lines.append(f"    - **{name}**")
    return "  - **Other species/compartment details:**\n" + "\n".join(lines) if lines else \
           "  - **Other species/compartment details:** —"

TARGET_BIO_CONTEXT = (
    "## Target Biological Context\n\n"
    "- **Priority:** Human > (if absent) closest human in vitro/ex vivo > (if still absent) relevant animal in vivo.\n"
    "- Use therapy-induced/withdrawal/stress datasets only as bounds; do not center priors on those unless explicitly requested.\n"
)

def build_model_context(param_name: str, rxns: pd.DataFrame, param_info: Dict[str, Dict[str, str]]) -> str:
    if rxns.empty:
        body = (f"{param_name} is currently not referenced in any reactions "
                f"according to the provided mapping table.")
        return body + "\n\n" + TARGET_BIO_CONTEXT

    bullets = []
    for _, row in rxns.iterrows():
        other_params_str = row.get("OtherParameters", "")
        other_params = parse_other_params_list(other_params_str)
        other_params_details = render_other_params_details(other_params, param_info)
        species_details = render_species_comp_details(row.get("OtherSpeciesWithNotes", ""))

        bullets.append(
            "- **Reaction:** `{}`\n"
            "  - **Rate:** `{}`\n"
            "  - **Other parameters in rate:** {}\n"
            "{}\n"
            "{}".format(
                row.get("Reaction", ""),
                row.get("ReactionRate", ""),
                other_params_str if (isinstance(other_params_str, str) and other_params_str.strip()) else "[]",
                other_params_details,
                species_details
            )
        )

    # Add your fixed biological context block beneath the reaction info
    header = "Mathematical role and biological context for this parameter based on the model:\n"
    return header + "\n".join(bullets) + "\n\n" + TARGET_BIO_CONTEXT

def fill_template(template_text: str, parameter_block: str, model_context_block: str) -> str:
    out = template_text
    out = out.replace("## PARAMETER_TO_SEARCH:", "## PARAMETER_TO_SEARCH:\n" + parameter_block)
    out = out.replace("## MODEL_CONTEXT:", "## MODEL_CONTEXT:\n" + model_context_block)
    return out

def maybe_zip(outdir: Path, zip_path: Path):
    import zipfile, os
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(outdir):
            for f in files:
                p = Path(root) / f
                arcname = p.relative_to(outdir.parent)
                zf.write(p, arcname.as_posix())

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Build per-parameter metadata prompts from SimBiology tables.")
    ap.add_argument("--params", required=True, type=Path, help="Path to simbio_parameters.csv")
    ap.add_argument("--reactions", required=True, type=Path, help="Path to model_context.csv")
    ap.add_argument("--template", required=True, type=Path, help="Path to metadata_template.md")
    ap.add_argument("--outdir", default=Path("generated_prompts"), type=Path, help="Output directory")
    ap.add_argument("--zip", dest="zip_output", action="store_true", help="Also create a zip archive of outputs")
    ap.add_argument("--zipname", default="generated_prompts.zip", help="Zip filename (if --zip is used)")
    args = ap.parse_args()

    params_df, reactions_df, template_text = load_inputs(args.params, args.reactions, args.template)
    param_info = index_param_info(params_df)
    cleaned_template = clean_template(template_text)
    args.outdir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for _, prow in params_df.iterrows():
        name = str(prow.get("Name", "")).strip()
        units = str(prow.get("Units", "")).strip()
        definition = str(prow.get("Definition", "")).strip()

        parameter_block = render_parameter_to_search(name, units, definition)

        rxns = reactions_df[reactions_df["Parameter"].astype(str) == name]
        model_context_block = build_model_context(name, rxns, param_info)

        filled = fill_template(cleaned_template, parameter_block, model_context_block)
        outfile = args.outdir / f"{slugify(name)}_metadata_prompt.md"
        outfile.write_text(filled)
        n_written += 1

    if args.zip_output:
        zip_path = args.outdir.parent / args.zipname
        maybe_zip(args.outdir, zip_path)
        print(f"Zipped outputs → {zip_path.resolve()}")

    print(f"Wrote {n_written} prompt file(s) to: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
