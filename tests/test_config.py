"""Tests for configuration loading and validation."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.config.loader import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    load_config,
    load_config_from_dict,
)
from src.config.schemas import (
    AssistantConfig,
    ContextTransferMode,
    HandoffTarget,
    HandoffType,
    MultiAssistantConfig,
)
from src.config.validator import (
    SemanticValidationError,
    ValidationResult,
    validate_config,
    validate_config_strict,
)


class TestConfigurationLoading:
    """Test JSON loading and parsing."""

    def test_load_valid_config(self, valid_medical_config_path: Path) -> None:
        """Load a valid configuration file."""
        if not valid_medical_config_path.exists():
            pytest.skip("Medical config file not found")

        config = load_config(valid_medical_config_path)
        assert config.metadata.name == "Medical Office Triage System"
        assert len(config.assistants) == 4

    def test_load_nonexistent_file(self) -> None:
        """Raise ConfigFileNotFoundError for missing config."""
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            load_config("nonexistent/config.json")

        assert "not found" in str(exc_info.value)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Raise ConfigParseError for malformed JSON."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{invalid json content")

        with pytest.raises(ConfigParseError) as exc_info:
            load_config(invalid_json)

        assert "Failed to parse" in str(exc_info.value)

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Handle empty configuration file."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")

        with pytest.raises(ConfigParseError):
            load_config(empty_file)


class TestPydanticValidation:
    """Test Pydantic schema validation."""

    def test_valid_minimal_config(self, minimal_valid_config: dict[str, Any]) -> None:
        """Minimal config with required fields passes validation."""
        config = load_config_from_dict(minimal_valid_config)
        assert config.metadata.name == "Test Config"
        assert len(config.assistants) == 2

    def test_valid_medical_triage_config(
        self, valid_medical_config_path: Path
    ) -> None:
        """Full medical triage config passes validation."""
        if not valid_medical_config_path.exists():
            pytest.skip("Medical config file not found")

        config = load_config(valid_medical_config_path)
        assert config.orchestration.entry_point == "receptionist"
        assert config.orchestration.handoff_type == HandoffType.TOOL_BASED

    def test_missing_required_field(self) -> None:
        """Raise ValidationError for missing metadata."""
        invalid_config = {
            "assistants": [{"id": "test", "name": "Test", "instructions": "Test"}],
            "orchestration": {"entry_point": "test"},
        }

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config_from_dict(invalid_config)

        assert "metadata" in str(exc_info.value).lower()

    def test_invalid_handoff_type_enum(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Reject invalid handoff_type values."""
        minimal_valid_config["orchestration"]["handoff_type"] = "invalid_type"

        with pytest.raises(ConfigValidationError):
            load_config_from_dict(minimal_valid_config)

    def test_duplicate_assistant_ids(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Reject duplicate assistant IDs."""
        minimal_valid_config["assistants"] = [
            {"id": "same_id", "name": "First", "instructions": "First"},
            {"id": "same_id", "name": "Second", "instructions": "Second"},
        ]

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config_from_dict(minimal_valid_config)

        assert "duplicate" in str(exc_info.value).lower()

    def test_invalid_assistant_id_format(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Reject IDs with special characters."""
        minimal_valid_config["assistants"][0]["id"] = "invalid@id!"

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config_from_dict(minimal_valid_config)

        assert "alphanumeric" in str(exc_info.value).lower()

    def test_empty_assistants_list(self) -> None:
        """Reject config with no assistants."""
        invalid_config = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "assistants": [],
            "orchestration": {"entry_point": "test"},
        }

        with pytest.raises(ConfigValidationError):
            load_config_from_dict(invalid_config)

    def test_model_override_partial(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Allow partial model overrides."""
        minimal_valid_config["assistants"][0]["model_overrides"] = {
            "llm": "openai/gpt-4o"
            # Only specifying LLM, not STT or TTS
        }

        config = load_config_from_dict(minimal_valid_config)
        assert config.assistants[0].model_overrides is not None
        assert config.assistants[0].model_overrides.llm == "openai/gpt-4o"


class TestSemanticValidation:
    """Test business logic validation."""

    def test_entry_point_exists(self, minimal_valid_config: dict[str, Any]) -> None:
        """Entry point references existing assistant."""
        config = load_config_from_dict(minimal_valid_config)
        result = validate_config(config)
        assert result.is_valid

    def test_entry_point_not_found(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Raise error for nonexistent entry point."""
        minimal_valid_config["orchestration"]["entry_point"] = "nonexistent"
        config = load_config_from_dict(minimal_valid_config)

        result = validate_config(config)
        assert not result.is_valid
        assert any("INVALID_ENTRY_POINT" in e.code for e in result.errors)

    def test_handoff_targets_exist(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """All handoff targets reference existing assistants."""
        minimal_valid_config["assistants"][0]["handoff_targets"] = [
            {"assistant_id": "specialist", "description": "Transfer to specialist"}
        ]

        config = load_config_from_dict(minimal_valid_config)
        result = validate_config(config)
        assert result.is_valid

    def test_invalid_handoff_target(
        self, config_invalid_handoff_target: dict[str, Any]
    ) -> None:
        """Detect invalid handoff target reference."""
        config = load_config_from_dict(config_invalid_handoff_target)
        result = validate_config(config)

        assert not result.is_valid
        assert any("INVALID_HANDOFF_TARGET" in e.code for e in result.errors)

    def test_orphaned_assistant_warning(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Warn about assistants with no inbound handoffs."""
        # Add an orphaned assistant
        minimal_valid_config["assistants"].append(
            {"id": "orphan", "name": "Orphan", "instructions": "Orphan assistant"}
        )

        config = load_config_from_dict(minimal_valid_config)
        result = validate_config(config)

        # Should still be valid (it's a warning)
        assert result.is_valid
        assert any("ORPHANED_ASSISTANT" in w.code for w in result.warnings)

    def test_fallback_assistant_exists(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Fallback assistant references existing assistant."""
        minimal_valid_config["orchestration"]["fallback_assistant"] = "entry"

        config = load_config_from_dict(minimal_valid_config)
        result = validate_config(config)
        assert result.is_valid

    def test_invalid_fallback_assistant(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Detect invalid fallback assistant reference."""
        minimal_valid_config["orchestration"]["fallback_assistant"] = "nonexistent"

        config = load_config_from_dict(minimal_valid_config)
        result = validate_config(config)

        assert not result.is_valid
        assert any("INVALID_FALLBACK_ASSISTANT" in e.code for e in result.errors)

    def test_validate_config_strict_raises(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """validate_config_strict raises on errors."""
        minimal_valid_config["orchestration"]["entry_point"] = "nonexistent"
        config = load_config_from_dict(minimal_valid_config)

        with pytest.raises(SemanticValidationError):
            validate_config_strict(config)

    def test_validate_config_strict_passes(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """validate_config_strict returns config when valid."""
        config = load_config_from_dict(minimal_valid_config)
        result = validate_config_strict(config)
        assert result is config


class TestSchemaModels:
    """Test individual schema model functionality."""

    def test_assistant_config_get_by_id(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Test get_assistant_by_id method."""
        config = load_config_from_dict(minimal_valid_config)

        entry = config.get_assistant_by_id("entry")
        assert entry is not None
        assert entry.name == "Entry Assistant"

        nonexistent = config.get_assistant_by_id("nonexistent")
        assert nonexistent is None

    def test_assistant_config_get_all_ids(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Test get_all_assistant_ids method."""
        config = load_config_from_dict(minimal_valid_config)

        ids = config.get_all_assistant_ids()
        assert ids == {"entry", "specialist"}

    def test_handoff_target_context_modes(self) -> None:
        """Test context transfer mode enum values.

        Only FULL mode is implemented (includes narrative reframing,
        smart compaction, and session memory injection).
        """
        assert ContextTransferMode.FULL == "full"
        # Only FULL mode exists - SUMMARY and USERDATA_ONLY were removed as unimplemented

    def test_handoff_type_enum_values(self) -> None:
        """Test handoff type enum values."""
        assert HandoffType.TOOL_BASED == "tool_based"
        assert HandoffType.RULE_BASED == "rule_based"
        assert HandoffType.STEP_BASED == "step_based"
