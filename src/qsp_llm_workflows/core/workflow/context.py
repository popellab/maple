"""
Workflow context that holds state between workflow steps.

The context is passed through the chain of workflow steps, with each step
reading from and writing to the context.
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """
    Context object that holds workflow state.

    This object is passed through the chain of workflow steps,
    allowing each step to read inputs and write outputs.
    """

    # Input parameters (immutable)
    input_csv: Path
    workflow_type: str

    # Configuration
    config: Any  # WorkflowConfig (avoiding circular import)
    progress_callback: Optional[Callable[[str], None]] = None

    # State accumulated during workflow (mutable)
    batch_file: Optional[Path] = None  # Used for preview mode output file
    results_file: Optional[Path] = None
    output_directory: Optional[Path] = None
    file_count: int = 0

    # Metadata
    metadata: dict = field(default_factory=dict)

    def report_progress(self, message: str) -> None:
        """
        Report progress to logger and optional callback.

        Always logs the message. If a progress callback is provided,
        also calls the callback for user-facing output.

        Args:
            message: Progress message to report
        """
        # Always log progress messages
        logger.info(message)

        # Also call user callback if provided
        if self.progress_callback:
            self.progress_callback(message)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)
