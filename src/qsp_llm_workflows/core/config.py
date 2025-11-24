"""
Configuration management for QSP LLM workflows.

Provides centralized configuration with validation and environment loading.
"""
import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class WorkflowConfig(BaseModel):
    """
    Configuration for workflow execution.

    Centralizes all workflow settings with validation and type safety.
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    # Required fields
    base_dir: Path = Field(
        ...,
        description="Base directory of qsp-llm-workflows repository",
    )
    storage_dir: Path = Field(
        ...,
        description="Path to qsp-metadata-storage repository",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (loaded from env if not provided)",
    )

    # API settings
    openai_model: str = Field(
        default="gpt-5",
        description="OpenAI model to use for requests",
    )
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="high",
        description="Reasoning effort level for API requests",
    )
    batch_completion_window: str = Field(
        default="24h",
        description="Completion window for batch API requests",
    )

    # Timeout settings
    batch_timeout: int = Field(
        default=3600,
        ge=0,
        description="Maximum seconds to wait for batch completion",
    )
    poll_interval: int = Field(
        default=30,
        ge=1,
        description="Seconds between batch status checks",
    )

    @field_validator("base_dir", "storage_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def batch_jobs_dir(self) -> Path:
        """Get batch jobs directory path."""
        return self.base_dir / "batch_jobs"

    @property
    def to_review_dir(self) -> Path:
        """Get to-review directory path."""
        return self.storage_dir / "to-review"

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "WorkflowConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            OPENAI_API_KEY: OpenAI API key (required)
            QSP_BASE_DIR: Base directory (defaults to current directory)
            QSP_STORAGE_DIR: Storage directory (defaults to ../qsp-metadata-storage)
            QSP_MODEL: OpenAI model (optional)
            QSP_REASONING_EFFORT: Reasoning effort level (optional)
            QSP_BATCH_TIMEOUT: Batch timeout in seconds (optional)
            QSP_POLL_INTERVAL: Poll interval in seconds (optional)

        Args:
            env_file: Optional path to .env file (defaults to .env in current directory)

        Returns:
            WorkflowConfig instance

        Raises:
            ValueError: If required environment variables are missing
        """
        # Load from .env file if it exists
        if env_file is None:
            env_file = Path(".env")

        if env_file.exists():
            _load_env_file(env_file)

        # Load API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables or .env file"
            )

        # Load directory settings
        base_dir = os.getenv("QSP_BASE_DIR", ".")
        storage_dir = os.getenv("QSP_STORAGE_DIR")

        if not storage_dir:
            # Default to sibling directory
            storage_dir = str(Path(base_dir).parent / "qsp-metadata-storage")

        # Load optional settings
        config_dict = {
            "base_dir": base_dir,
            "storage_dir": storage_dir,
            "openai_api_key": api_key,
        }

        # Optional overrides
        if model := os.getenv("QSP_MODEL"):
            config_dict["openai_model"] = model

        if effort := os.getenv("QSP_REASONING_EFFORT"):
            config_dict["reasoning_effort"] = effort

        if timeout := os.getenv("QSP_BATCH_TIMEOUT"):
            config_dict["batch_timeout"] = int(timeout)

        if interval := os.getenv("QSP_POLL_INTERVAL"):
            config_dict["poll_interval"] = int(interval)

        return cls(**config_dict)


def _load_env_file(env_file: Path) -> None:
    """
    Load environment variables from .env file.

    Simple implementation that reads KEY=VALUE pairs.

    Args:
        env_file: Path to .env file
    """
    with open(env_file) as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Set environment variable
                os.environ[key] = value
