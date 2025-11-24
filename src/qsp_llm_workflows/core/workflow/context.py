"""
Workflow context that holds state between workflow steps.

The context is passed through the chain of workflow steps, with each step
reading from and writing to the context.
"""
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field


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
    immediate: bool

    # Configuration
    config: Any  # WorkflowConfig (avoiding circular import)
    progress_callback: Optional[Callable[[str], None]] = None

    # State accumulated during workflow (mutable)
    batch_file: Optional[Path] = None
    batch_id: Optional[str] = None
    results_file: Optional[Path] = None
    output_directory: Optional[Path] = None
    file_count: int = 0

    # Metadata
    metadata: dict = field(default_factory=dict)

    def report_progress(self, message: str) -> None:
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(message)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)
