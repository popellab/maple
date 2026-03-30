#!/usr/bin/env python3
"""
Export model structure from SimBiology model.

This module exports the complete model structure (species, compartments,
parameters, reactions) as JSON for use with ModelStructure query tools.
"""

import json
import subprocess
from pathlib import Path


class ModelStructureExporter:
    """Exports model structure from SimBiology models as JSON."""

    def __init__(self, model_file: str, model_type: str = "matlab_script"):
        """
        Initialize the exporter.

        Args:
            model_file: Path to model file (.m script or .sbproj)
            model_type: "matlab_script" or "simbiology_project"
        """
        self.model_file = Path(model_file).resolve()
        self.model_type = model_type

        if model_type not in ["matlab_script", "simbiology_project"]:
            raise ValueError(f"Invalid model_type: {model_type}")

        if not self.model_file.exists():
            raise ValueError(f"Model file does not exist: {model_file}")

    def export_to_json(self, output_file: str) -> None:
        """
        Export model structure to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        output_path = Path(output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Exporting model structure from {self.model_file}")
        self._run_matlab_export(output_path)
        print(f"Model structure exported to {output_path}")

    def _run_matlab_export(self, output_path: Path) -> None:
        """Run MATLAB to export model structure."""
        scripts_dir = Path(__file__).parent.parent / "matlab"
        export_script = scripts_dir / "export_model_structure.m"

        if not export_script.exists():
            raise FileNotFoundError(f"MATLAB export script not found: {export_script}")

        model_dir = self.model_file.parent

        # Find project root
        project_root = model_dir
        for _ in range(3):
            if project_root.parent == project_root:
                break
            project_root = project_root.parent
            if any((project_root / marker).exists() for marker in [".git", "README.md"]):
                break

        matlab_cmd = (
            f"addpath('{scripts_dir}'); "
            f"addpath('{model_dir}'); "
            f"addpath(genpath('{project_root}')); "
            f"export_model_structure('{self.model_file}', '{output_path}', '{self.model_type}')"
        )

        result = subprocess.run(
            ["matlab", "-batch", matlab_cmd],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        if result.returncode != 0:
            raise RuntimeError(f"MATLAB export failed: {result.stderr}")

    @staticmethod
    def export_species_units(model_structure_file: str, output_file: str) -> None:
        """
        Derive species_units.json from an exported model_structure.json.

        Maps every species and parameter to its unit string and description,
        producing the flat lookup used by qsp-hpc for Pint conversions.
        """
        ms_path = Path(model_structure_file)
        out_path = Path(output_file)

        with open(ms_path, "r") as f:
            ms = json.load(f)

        species_units: dict[str, dict[str, str]] = {}

        for s in ms.get("species", []):
            species_units[s["name"]] = {
                "units": s.get("units", "dimensionless"),
                "description": s.get("description", ""),
            }

        for p in ms.get("parameters", []):
            name = p["name"]
            if name not in species_units:
                species_units[name] = {
                    "units": p.get("units", "dimensionless"),
                    "description": p.get("description", ""),
                }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(species_units, f, indent=2)
            f.write("\n")
