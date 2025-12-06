"""
Unit tests for WorkflowConfig.

Tests the configuration object that centralizes workflow settings.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from qsp_llm_workflows.core.config import WorkflowConfig
from qsp_llm_workflows.core.exceptions import ConfigurationError


class TestWorkflowConfig:
    """Test WorkflowConfig creation and validation."""

    def test_create_config_with_defaults(self, tmp_path):
        """Test creating config with required fields only."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        config = WorkflowConfig(
            base_dir=base_dir,
            storage_dir=storage_dir,
        )

        # Check required fields
        assert config.base_dir == base_dir
        assert config.storage_dir == storage_dir

        # Check defaults
        assert config.openai_model == "gpt-5"
        assert config.reasoning_effort == "high"
        assert config.batch_completion_window == "24h"
        assert config.batch_timeout == 3600
        assert config.poll_interval == 30

    def test_create_config_with_overrides(self, tmp_path):
        """Test creating config with custom values."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        config = WorkflowConfig(
            base_dir=base_dir,
            storage_dir=storage_dir,
            openai_model="gpt-4",
            reasoning_effort="medium",
            batch_timeout=7200,
        )

        assert config.openai_model == "gpt-4"
        assert config.reasoning_effort == "medium"
        assert config.batch_timeout == 7200

    def test_paths_converted_to_path_objects(self, tmp_path):
        """Test that string paths are converted to Path objects."""
        base_dir = str(tmp_path / "workflows")
        storage_dir = str(tmp_path / "storage")

        config = WorkflowConfig(
            base_dir=base_dir,
            storage_dir=storage_dir,
        )

        assert isinstance(config.base_dir, Path)
        assert isinstance(config.storage_dir, Path)

    def test_batch_jobs_dir_property(self, tmp_path):
        """Test batch_jobs_dir computed property."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        config = WorkflowConfig(
            base_dir=base_dir,
            storage_dir=storage_dir,
        )

        expected = base_dir / "batch_jobs"
        assert config.batch_jobs_dir == expected

    def test_to_review_dir_property(self, tmp_path):
        """Test to_review_dir computed property."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        config = WorkflowConfig(
            base_dir=base_dir,
            storage_dir=storage_dir,
        )

        expected = storage_dir / "to-review"
        assert config.to_review_dir == expected

    def test_invalid_reasoning_effort(self, tmp_path):
        """Test validation of reasoning_effort field."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        with pytest.raises(ValidationError) as exc_info:
            WorkflowConfig(
                base_dir=base_dir,
                storage_dir=storage_dir,
                reasoning_effort="invalid",
            )

        assert "reasoning_effort" in str(exc_info.value)

    def test_negative_timeout(self, tmp_path):
        """Test validation of timeout values."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        with pytest.raises(ValidationError) as exc_info:
            WorkflowConfig(
                base_dir=base_dir,
                storage_dir=storage_dir,
                batch_timeout=-100,
            )

        assert "batch_timeout" in str(exc_info.value)

    def test_config_immutable(self, tmp_path):
        """Test that config is immutable after creation."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        config = WorkflowConfig(
            base_dir=base_dir,
            storage_dir=storage_dir,
        )

        # Pydantic models are mutable by default, but we can make them frozen
        # If we add frozen=True to the model config
        with pytest.raises(ValidationError):
            config.openai_model = "gpt-3"

    def test_from_env_missing_api_key(self, tmp_path, monkeypatch):
        """Test loading from environment without API key."""
        # Clear any existing API key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Change to tmp directory (no .env file there)
        monkeypatch.chdir(tmp_path)

        # Set environment variables
        monkeypatch.setenv("QSP_BASE_DIR", str(tmp_path / "workflows"))
        monkeypatch.setenv("QSP_STORAGE_DIR", str(tmp_path / "storage"))

        with pytest.raises(ConfigurationError) as exc_info:
            WorkflowConfig.from_env()

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_from_env_missing_storage_dir(self, tmp_path, monkeypatch):
        """Test loading from environment without storage directory raises error."""
        # Set API key but not storage dir
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("QSP_BASE_DIR", str(tmp_path / "workflows"))
        monkeypatch.delenv("QSP_STORAGE_DIR", raising=False)

        # Change to tmp directory
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            WorkflowConfig.from_env()

        assert "Storage directory not specified" in str(exc_info.value)

    def test_from_env_with_dotenv_file(self, tmp_path, monkeypatch):
        """Test loading from .env file."""
        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key-123\n")

        # Change to tmp directory
        monkeypatch.chdir(tmp_path)

        # Set required dirs
        monkeypatch.setenv("QSP_BASE_DIR", str(tmp_path / "workflows"))
        monkeypatch.setenv("QSP_STORAGE_DIR", str(tmp_path / "storage"))

        config = WorkflowConfig.from_env()

        assert config.openai_api_key == "test-key-123"

    def test_from_env_with_explicit_storage_dir(self, tmp_path, monkeypatch):
        """Test loading from env with explicit storage_dir parameter."""
        # Change to tmp_path to avoid reading .env from project root
        monkeypatch.chdir(tmp_path)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("QSP_BASE_DIR", str(tmp_path / "workflows"))
        # Don't set QSP_STORAGE_DIR - will use parameter instead

        storage_path = tmp_path / "my-metadata-storage"

        config = WorkflowConfig.from_env(storage_dir=storage_path)

        assert config.storage_dir == storage_path
        assert config.openai_api_key == "test-key"

    def test_from_env_explicit_storage_dir_overrides_env(self, tmp_path, monkeypatch):
        """Test that explicit storage_dir parameter overrides environment variable."""
        # Change to tmp_path to avoid reading .env from project root
        monkeypatch.chdir(tmp_path)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("QSP_BASE_DIR", str(tmp_path / "workflows"))
        monkeypatch.setenv(
            "QSP_STORAGE_DIR", str(tmp_path / "env-storage")
        )  # This should be overridden

        explicit_storage = tmp_path / "explicit-storage"

        config = WorkflowConfig.from_env(storage_dir=explicit_storage)

        assert config.storage_dir == explicit_storage  # Parameter takes precedence

    def test_from_env_with_overrides(self, tmp_path, monkeypatch):
        """Test loading from environment with custom settings."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("QSP_BASE_DIR", str(tmp_path / "workflows"))
        monkeypatch.setenv("QSP_STORAGE_DIR", str(tmp_path / "storage"))
        monkeypatch.setenv("QSP_MODEL", "gpt-4")
        monkeypatch.setenv("QSP_REASONING_EFFORT", "low")
        monkeypatch.setenv("QSP_BATCH_TIMEOUT", "7200")

        config = WorkflowConfig.from_env()

        assert config.openai_model == "gpt-4"
        assert config.reasoning_effort == "low"
        assert config.batch_timeout == 7200

    def test_to_dict(self, tmp_path):
        """Test converting config to dictionary."""
        base_dir = tmp_path / "workflows"
        storage_dir = tmp_path / "storage"

        config = WorkflowConfig(
            base_dir=base_dir,
            storage_dir=storage_dir,
        )

        config_dict = config.model_dump()

        assert config_dict["openai_model"] == "gpt-5"
        assert config_dict["batch_timeout"] == 3600
        # Paths should be serialized
        assert "base_dir" in config_dict
