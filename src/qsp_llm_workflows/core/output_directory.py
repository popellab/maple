"""
Output directory management for extraction workflows.

Provides utilities for creating unique timestamped directories for extraction results.
"""
from pathlib import Path
from datetime import datetime
from typing import Optional


def generate_output_directory_name(
    workflow_type: str,
    immediate: bool,
    batch_id: Optional[str] = None,
) -> str:
    """
    Generate unique directory name for extraction output.

    Format: {timestamp}_{workflow_type}_{mode}[_{batch_id}]

    Examples:
        - 20251123_143022_parameter_immediate
        - 20251123_150000_parameter_batch_abc123
        - 20251123_160000_test_statistic_immediate

    Args:
        workflow_type: Type of workflow (parameter, test_statistic)
        immediate: True for immediate mode, False for batch mode
        batch_id: Optional batch ID for batch mode (e.g., "batch_abc123xyz")

    Returns:
        Directory name string
    """
    # Generate timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build directory name
    parts = [timestamp, workflow_type]

    # Add mode/batch_id
    if immediate:
        parts.append("immediate")
    elif batch_id:
        # Batch ID already contains "batch_" prefix typically, so just use it
        parts.append(batch_id)
    else:
        parts.append("batch")

    return "_".join(parts)


def create_unique_output_directory(
    base_dir: Path,
    workflow_type: str,
    immediate: bool,
    batch_id: Optional[str] = None,
) -> Path:
    """
    Create unique output directory for extraction results.

    Creates a timestamped directory under base_dir with format:
        {base_dir}/{timestamp}_{workflow_type}_{mode}[_{batch_id}]/

    Args:
        base_dir: Base directory (e.g., to-review/)
        workflow_type: Type of workflow (parameter, test_statistic)
        immediate: True for immediate mode, False for batch mode
        batch_id: Optional batch ID for batch mode

    Returns:
        Path to created directory

    Example:
        >>> create_unique_output_directory(
        ...     Path("to-review"),
        ...     "parameter",
        ...     immediate=True
        ... )
        PosixPath('to-review/20251123_143022_parameter_immediate')
    """
    # Ensure base directory exists
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique directory name
    dir_name = generate_output_directory_name(workflow_type, immediate, batch_id)

    # Create output directory
    output_dir = base_dir / dir_name
    output_dir.mkdir(exist_ok=True)

    return output_dir
