"""
Test that CLI entry points are properly configured and accessible.
"""

import subprocess
import sys


def test_qsp_extract_help():
    """Test that qsp-extract command is available and shows help."""
    result = subprocess.run(
        [sys.executable, "-m", "maple.cli.extract", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "extract" in result.stdout.lower()


def test_qsp_export_model_help():
    """Test that qsp-export-model command is available and shows help."""
    result = subprocess.run(
        [sys.executable, "-m", "maple.cli.export_model", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "export" in result.stdout.lower() or "model" in result.stdout.lower()
