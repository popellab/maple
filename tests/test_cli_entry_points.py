"""
Test that CLI entry points are properly configured and accessible.
"""

import subprocess


def test_qsp_extract_help():
    """Test that qsp-extract command is available and shows help."""
    result = subprocess.run(["qsp-extract", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "qsp-extract" in result.stdout or "extract" in result.stdout.lower()


def test_qsp_export_model_help():
    """Test that qsp-export-model command is available and shows help."""
    result = subprocess.run(["qsp-export-model", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "export" in result.stdout.lower() or "model" in result.stdout.lower()
