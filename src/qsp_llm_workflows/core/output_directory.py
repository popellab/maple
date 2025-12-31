"""
Output directory management for extraction workflows.

Provides utilities for creating unique timestamped directories for extraction results.
"""

from pathlib import Path
from datetime import datetime


def generate_output_directory_name(workflow_type: str) -> str:
    """
    Generate unique directory name for extraction output.

    Format: {timestamp}_{workflow_type}

    Examples:
        - 20251123_143022_parameter
        - 20251123_150000_test_statistic
        - 20251123_160000_calibration_target

    Args:
        workflow_type: Type of workflow (parameter, test_statistic, calibration_target)

    Returns:
        Directory name string
    """
    # Generate timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build directory name
    return f"{timestamp}_{workflow_type}"


def create_unique_output_directory(
    base_dir: Path,
    workflow_type: str,
) -> Path:
    """
    Create unique output directory for extraction results.

    Creates a timestamped directory under base_dir with format:
        {base_dir}/{timestamp}_{workflow_type}/

    Args:
        base_dir: Base directory (e.g., to-review/)
        workflow_type: Type of workflow (parameter, test_statistic, calibration_target)

    Returns:
        Path to created directory

    Example:
        >>> create_unique_output_directory(
        ...     Path("to-review"),
        ...     "parameter"
        ... )
        PosixPath('to-review/20251123_143022_parameter')
    """
    # Ensure base directory exists
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique directory name
    dir_name = generate_output_directory_name(workflow_type)

    # Create output directory
    output_dir = base_dir / dir_name
    output_dir.mkdir(exist_ok=True)

    return output_dir
