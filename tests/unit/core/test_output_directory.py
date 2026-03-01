"""
Unit tests for output directory management.

Tests the unique directory generation for extraction results.
"""

from pathlib import Path
from freezegun import freeze_time
import time

from maple.core.output_directory import (
    generate_output_directory_name,
    create_unique_output_directory,
)


class TestGenerateOutputDirectoryName:
    """Test output directory name generation."""

    @freeze_time("2025-11-23 14:30:22")
    def test_generate_name_parameter(self):
        """Test generating directory name for parameter workflow."""
        name = generate_output_directory_name(
            workflow_type="parameter",
        )

        assert name == "20251123_143022_parameter"

    @freeze_time("2025-11-23 15:45:10")
    def test_generate_name_test_statistic(self):
        """Test generating directory name for test_statistic workflow."""
        name = generate_output_directory_name(
            workflow_type="test_statistic",
        )

        assert name == "20251123_154510_test_statistic"

    @freeze_time("2025-11-23 16:00:00")
    def test_generate_name_calibration_target(self):
        """Test generating directory name for calibration_target workflow."""
        name = generate_output_directory_name(
            workflow_type="calibration_target",
        )

        assert name == "20251123_160000_calibration_target"

    def test_name_format_components(self):
        """Test that name contains all expected components."""
        name = generate_output_directory_name(
            workflow_type="parameter",
        )

        # Should have format: YYYYMMDD_HHMMSS_type
        parts = name.split("_")
        assert len(parts) == 3

        # First part is date (8 digits)
        assert len(parts[0]) == 8
        assert parts[0].isdigit()

        # Second part is time (6 digits)
        assert len(parts[1]) == 6
        assert parts[1].isdigit()

        # Third part is workflow type
        assert parts[2] in ["parameter", "test_statistic", "calibration_target"]

    def test_different_timestamps_produce_different_names(self):
        """Test that names generated at different times are unique."""
        with freeze_time("2025-11-23 14:30:00"):
            name1 = generate_output_directory_name("parameter")

        with freeze_time("2025-11-23 14:30:01"):
            name2 = generate_output_directory_name("parameter")

        assert name1 != name2


class TestCreateUniqueOutputDirectory:
    """Test unique output directory creation."""

    def test_create_directory(self, tmp_path):
        """Test creating unique output directory."""
        base_dir = tmp_path / "to-review"

        output_dir = create_unique_output_directory(
            base_dir=base_dir,
            workflow_type="parameter",
        )

        # Directory should exist
        assert output_dir.exists()
        assert output_dir.is_dir()

        # Directory should be under base_dir
        assert output_dir.parent == base_dir

        # Name should match expected format
        assert "parameter" in output_dir.name
        assert output_dir.name.endswith("_parameter")

    def test_create_directory_test_statistic(self, tmp_path):
        """Test creating directory for test statistic workflow."""
        base_dir = tmp_path / "to-review"

        output_dir = create_unique_output_directory(
            base_dir=base_dir,
            workflow_type="test_statistic",
        )

        assert output_dir.exists()
        assert "test_statistic" in output_dir.name
        assert output_dir.name.endswith("_test_statistic")

    def test_creates_base_directory_if_not_exists(self, tmp_path):
        """Test that base directory is created if it doesn't exist."""
        base_dir = tmp_path / "nonexistent" / "to-review"

        # Base dir doesn't exist yet
        assert not base_dir.exists()

        output_dir = create_unique_output_directory(
            base_dir=base_dir,
            workflow_type="parameter",
        )

        # Both base_dir and output_dir should exist
        assert base_dir.exists()
        assert output_dir.exists()

    def test_multiple_calls_create_unique_directories(self, tmp_path):
        """Test that multiple calls create unique directories."""
        base_dir = tmp_path / "to-review"

        # Create three directories with small delays to ensure unique timestamps
        dir1 = create_unique_output_directory(base_dir, "parameter")
        time.sleep(1.1)  # Sleep slightly over 1 second
        dir2 = create_unique_output_directory(base_dir, "parameter")
        time.sleep(1.1)
        dir3 = create_unique_output_directory(base_dir, "parameter")

        # All should be unique
        assert dir1 != dir2
        assert dir2 != dir3
        assert dir1 != dir3

        # All should exist
        assert dir1.exists()
        assert dir2.exists()
        assert dir3.exists()

    def test_returns_path_object(self, tmp_path):
        """Test that function returns Path object."""
        base_dir = tmp_path / "to-review"

        output_dir = create_unique_output_directory(
            base_dir=base_dir,
            workflow_type="parameter",
        )

        assert isinstance(output_dir, Path)
