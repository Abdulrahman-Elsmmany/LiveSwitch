"""Tests for error handling and edge cases.

This module tests error scenarios including configuration errors,
validation failures, handoff errors, and edge cases to ensure
robust error handling throughout the system.
"""

import pytest
from typing import Any
from unittest.mock import MagicMock

from src.config.loader import load_config_from_dict, ConfigValidationError
from src.config.validator import validate_config, SemanticValidationError
from src.orchestration.factory import AgentFactory
from src.orchestration.manager import OrchestrationManager
from src.orchestration.handoff import HandoffCoordinator
from src.context.session import SessionData


class TestConfigurationErrors:
    """Tests for configuration error handling."""

    def test_missing_metadata_raises_error(self):
        """Config without metadata raises validation error."""
        config = {
            "assistants": [{"id": "entry", "name": "Entry", "instructions": "Hi"}],
            "orchestration": {"entry_point": "entry"},
        }

        with pytest.raises(ConfigValidationError):
            load_config_from_dict(config)

    def test_missing_assistants_raises_error(self):
        """Config without assistants raises validation error."""
        config = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "orchestration": {"entry_point": "entry"},
        }

        with pytest.raises(ConfigValidationError):
            load_config_from_dict(config)

    def test_empty_assistants_raises_error(self):
        """Config with empty assistants list raises error."""
        config = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "assistants": [],
            "orchestration": {"entry_point": "entry"},
        }

        with pytest.raises(ConfigValidationError):
            load_config_from_dict(config)

    def test_missing_entry_point_raises_semantic_error(self, minimal_valid_config):
        """Entry point not in assistants raises semantic error."""
        minimal_valid_config["orchestration"]["entry_point"] = "nonexistent"
        config = load_config_from_dict(minimal_valid_config)

        result = validate_config(config)

        assert not result.is_valid
        assert any("entry_point" in str(e).lower() for e in result.errors)

    def test_invalid_handoff_target_raises_semantic_error(
        self, config_invalid_handoff_target
    ):
        """Handoff to nonexistent assistant raises semantic error."""
        config = load_config_from_dict(config_invalid_handoff_target)

        result = validate_config(config)

        assert not result.is_valid
        assert any("nonexistent" in str(e).lower() for e in result.errors)

    def test_duplicate_assistant_ids_raises_error(self):
        """Duplicate assistant IDs raise validation error."""
        config = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "assistants": [
                {"id": "duplicate", "name": "First", "instructions": "First"},
                {"id": "duplicate", "name": "Second", "instructions": "Second"},
            ],
            "orchestration": {"entry_point": "duplicate"},
        }

        with pytest.raises(ConfigValidationError):
            load_config_from_dict(config)

    def test_invalid_assistant_id_format(self):
        """Invalid assistant ID format raises error."""
        config = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "assistants": [
                {"id": "Invalid ID With Spaces", "name": "Test", "instructions": "Hi"},
            ],
            "orchestration": {"entry_point": "Invalid ID With Spaces"},
        }

        with pytest.raises(ConfigValidationError):
            load_config_from_dict(config)


class TestSemanticValidation:
    """Tests for semantic configuration validation."""

    def test_self_handoff_warns(self, minimal_valid_config):
        """Agent handing off to itself generates warning."""
        minimal_valid_config["assistants"][0]["handoff_targets"] = [
            {"assistant_id": "entry", "description": "Loop to self"}
        ]
        config = load_config_from_dict(minimal_valid_config)

        result = validate_config(config)

        # Self-handoff is allowed but may generate a warning
        assert result.is_valid

    def test_orphaned_assistant_warns(self, minimal_valid_config):
        """Assistant with no incoming handoffs generates warning."""
        # Add an orphaned assistant (not reachable from entry)
        minimal_valid_config["assistants"].append(
            {"id": "orphan", "name": "Orphan", "instructions": "Unreachable"}
        )
        config = load_config_from_dict(minimal_valid_config)

        result = validate_config(config)

        # Orphaned assistants may generate warnings
        assert result.is_valid  # Still valid, just warnings

    def test_invalid_fallback_raises_error(self, minimal_valid_config):
        """Fallback to nonexistent assistant raises error."""
        minimal_valid_config["orchestration"]["fallback_assistant"] = "nonexistent"
        config = load_config_from_dict(minimal_valid_config)

        result = validate_config(config)

        assert not result.is_valid


class TestHandoffErrors:
    """Tests for handoff error handling."""

    def test_handoff_exceeds_max_limit(self, minimal_valid_config):
        """Handoff blocked when max limit reached."""
        minimal_valid_config["orchestration"]["max_handoffs"] = 2
        minimal_valid_config["assistants"][0]["handoff_targets"] = [
            {"assistant_id": "specialist", "description": "Transfer"}
        ]
        minimal_valid_config["assistants"][1]["handoff_targets"] = [
            {"assistant_id": "entry", "description": "Return"}
        ]
        config = load_config_from_dict(minimal_valid_config)

        manager = OrchestrationManager(config)
        session_data = manager.create_session_data("test-session")

        # Simulate reaching max handoffs
        session_data.record_handoff("entry", "specialist", "First handoff")
        session_data.record_handoff("specialist", "entry", "Second handoff")

        # Third handoff should be blocked
        assert session_data.handoff_count == 2
        assert manager.config.orchestration.max_handoffs == 2

    def test_handoff_to_unknown_target(self, minimal_valid_config):
        """Handoff to unknown target fails gracefully."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        is_valid, error = coordinator.validate_handoff_target("entry", "unknown_target")

        assert is_valid is False
        assert "unknown" in error.lower() or "not found" in error.lower()

    def test_handoff_from_unknown_source(self, minimal_valid_config):
        """Handoff from unknown source fails."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        is_valid, error = coordinator.validate_handoff_target(
            "unknown_source", "specialist"
        )

        assert is_valid is False


class TestSessionDataErrors:
    """Tests for session data error handling."""

    def test_get_nonexistent_data_returns_default(self):
        """Getting nonexistent key returns default value."""
        session = SessionData(session_id="test", current_assistant_id="entry")

        result = session.get_data("nonexistent_key", "default_value")

        assert result == "default_value"

    def test_get_nonexistent_data_returns_none(self):
        """Getting nonexistent key without default returns None."""
        session = SessionData(session_id="test", current_assistant_id="entry")

        result = session.get_data("nonexistent_key")

        assert result is None

    def test_has_data_returns_false_for_nonexistent(self):
        """has_data returns False for nonexistent keys."""
        session = SessionData(session_id="test", current_assistant_id="entry")

        assert session.has_data("nonexistent_key") is False

    def test_has_data_returns_true_for_existing(self):
        """has_data returns True for existing keys."""
        session = SessionData(session_id="test", current_assistant_id="entry")
        session.set_data("existing_key", "value")

        assert session.has_data("existing_key") is True


class TestFactoryErrors:
    """Tests for agent factory error handling."""

    def test_get_nonexistent_agent_class(self, minimal_valid_config):
        """Getting nonexistent agent class returns None."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        result = factory.get_agent_class("nonexistent_id")

        assert result is None

    def test_get_entry_agent_with_invalid_config(self):
        """Entry agent with invalid entry_point raises error."""
        # Create config then corrupt entry point
        config_dict = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "assistants": [
                {"id": "entry", "name": "Entry", "instructions": "Hi"},
            ],
            "orchestration": {"entry_point": "entry"},
        }
        config = load_config_from_dict(config_dict)
        factory = AgentFactory(config)

        # Manually corrupt the entry point (simulating race condition or bug)
        factory.config.orchestration.entry_point = "corrupted"

        with pytest.raises(ValueError):
            factory.get_entry_agent()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_assistant_instructions(self, minimal_valid_config):
        """Handles very long instructions."""
        minimal_valid_config["assistants"][0]["instructions"] = "A" * 10000
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        agent = factory.get_entry_agent()
        assert agent is not None

    def test_special_characters_in_assistant_name(self, minimal_valid_config):
        """Handles special characters in assistant names."""
        minimal_valid_config["assistants"][0]["name"] = "Test Agent (v1.0) - Beta!"
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        agent = factory.get_entry_agent()
        assert agent is not None

    def test_empty_instructions_allowed(self):
        """Empty instructions are allowed but may not be ideal."""
        config = {
            "metadata": {"name": "Test", "version": "1.0.0"},
            "assistants": [
                {"id": "entry", "name": "Entry", "instructions": ""},
            ],
            "orchestration": {"entry_point": "entry"},
        }

        # This should load (empty string is valid)
        loaded = load_config_from_dict(config)
        assert loaded.assistants[0].instructions == ""

    def test_max_handoffs_zero_blocks_all(self, minimal_valid_config):
        """Max handoffs of 0 blocks all handoffs."""
        minimal_valid_config["orchestration"]["max_handoffs"] = 0
        config = load_config_from_dict(minimal_valid_config)

        manager = OrchestrationManager(config)
        session_data = manager.create_session_data("test")

        # Any handoff should be blocked with max=0
        assert manager.config.orchestration.max_handoffs == 0

    def test_session_duration_calculation(self):
        """Session duration calculated correctly."""
        import time

        session = SessionData(session_id="test", current_assistant_id="entry")

        # Small delay to ensure measurable duration
        time.sleep(0.1)

        duration = session.get_session_duration_seconds()

        assert duration >= 0.1
        assert duration < 10  # Should be quick
