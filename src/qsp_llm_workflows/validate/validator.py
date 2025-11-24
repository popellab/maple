"""
Base validator class for validation workflow.

Provides consistent interface for all validators using abstract base class pattern.
"""

from abc import ABC, abstractmethod
from qsp_llm_workflows.validate.validation_utils import ValidationReport


class Validator(ABC):
    """
    Base class for all validators.

    Provides consistent interface for validation workflow orchestration.
    Each validator performs a specific validation check and returns a ValidationReport.
    """

    def __init__(self, data_dir: str, **kwargs):
        """
        Initialize validator.

        Args:
            data_dir: Directory containing YAML files to validate
            **kwargs: Additional validator-specific configuration
        """
        self.data_dir = data_dir
        self.config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable validator name.

        Returns:
            Name of this validator (e.g., "Schema Compliance Validation")
        """
        pass

    @abstractmethod
    def validate(self) -> ValidationReport:
        """
        Run validation and return report.

        Returns:
            ValidationReport with results
        """
        pass

    def __repr__(self) -> str:
        """String representation of validator."""
        return f"{self.__class__.__name__}(data_dir={self.data_dir})"
