"""
Interactive target selection for qsp-extract.

Presents a checklist UI in the terminal so users can toggle individual
CSV rows on/off before extraction begins.  When a subset is selected,
a filtered temporary CSV is written and its path returned.
"""
import csv
import sys
import tempfile
from pathlib import Path


# Column used as the display ID for each workflow type
_ID_COLUMNS = {
    "parameter": "parameter_name",
    "test_statistic": "test_statistic_id",
    "calibration_target": "calibration_target_id",
    "isolated_system_target": "target_id",
    "submodel_target": "target_id",
}

# Secondary column used for the description suffix
_DESC_COLUMNS = {
    "parameter": ("cancer_type", "description"),
    "test_statistic": ("derived_species_description",),
    "calibration_target": ("observable_description",),
    "isolated_system_target": ("parameters",),
    "submodel_target": ("parameters",),
}


def _get_display_label(row: dict, workflow_type: str, index: int) -> str:
    """Build a human-readable label for a CSV row."""
    id_col = _ID_COLUMNS.get(workflow_type)
    row_id = row.get(id_col, f"row_{index + 1}") if id_col else f"row_{index + 1}"

    parts = []
    for col in _DESC_COLUMNS.get(workflow_type, ()):
        val = row.get(col, "")
        if val:
            parts.append(val[:60])

    suffix = " - ".join(parts)
    if suffix:
        return f"{row_id} ({suffix})"
    return row_id


def _parse_selection(text: str, n_items: int) -> list[int] | None:
    """Parse a toggle specification like '1-5,8,10-12' into 0-based indices.

    Returns None if the input is invalid.
    """
    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                lo, hi = int(bounds[0]), int(bounds[1])
            except ValueError:
                return None
            if lo < 1 or hi > n_items or lo > hi:
                return None
            indices.extend(range(lo - 1, hi))  # convert to 0-based
        else:
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 1 or idx > n_items:
                return None
            indices.append(idx - 1)
    return indices


def _print_checklist(labels: list[str], selected: list[bool]):
    """Print the current checklist state."""
    n_selected = sum(selected)
    print(f"\n  {n_selected}/{len(labels)} targets selected\n")
    for i, (label, sel) in enumerate(zip(labels, selected)):
        marker = "x" if sel else " "
        print(f"  [{marker}] {i + 1:>3}. {label}")
    print()


def interactive_select(input_csv: Path, workflow_type: str) -> Path | None:
    """Run interactive selection and return the (possibly filtered) CSV path.

    Returns:
        Path to the original or a temporary filtered CSV, or None if cancelled.
    """
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        print("No rows found in input CSV.", file=sys.stderr)
        return None

    labels = [_get_display_label(row, workflow_type, i) for i, row in enumerate(rows)]
    selected = [True] * len(rows)

    _print_checklist(labels, selected)
    print("  Commands: <numbers> toggle (e.g. 1-5,8)  a=all  n=none  Enter=confirm  q=cancel")

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw == "q":
            return None

        if raw == "":
            # Confirm
            break

        if raw == "a":
            selected = [True] * len(rows)
            _print_checklist(labels, selected)
            continue

        if raw == "n":
            selected = [False] * len(rows)
            _print_checklist(labels, selected)
            continue

        indices = _parse_selection(raw, len(rows))
        if indices is None:
            print(f"  Invalid input: {raw!r}. Use numbers like 1-5,8 or commands a/n/q.")
            continue

        for idx in indices:
            selected[idx] = not selected[idx]
        _print_checklist(labels, selected)

    # Determine result
    if all(selected):
        return input_csv

    chosen = [row for row, sel in zip(rows, selected) if sel]
    if not chosen:
        print("No targets selected.", file=sys.stderr)
        return None

    # Write filtered CSV to a temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="qsp_extract_", delete=False, newline=""
    )
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(chosen)
    tmp.close()

    n_total = len(rows)
    n_selected = len(chosen)
    print(f"  Proceeding with {n_selected}/{n_total} targets.")
    return Path(tmp.name)
