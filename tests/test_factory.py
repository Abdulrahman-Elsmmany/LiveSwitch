"""Tests for agent factory and dynamic agent generation."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config.loader import load_config_from_dict
from src.context.session import SessionData, create_session_data
from src.orchestration.factory import AgentFactory
from src.orchestration.handoff import HandoffCoordinator
from src.orchestration.manager import OrchestrationManager


class TestAgentFactory:
    """Test dynamic agent class generation."""

    def test_create_single_agent(self, minimal_valid_config: dict[str, Any]) -> None:
        """Generate agent class from config."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        assert "entry" in factory.agent_registry
        assert "specialist" in factory.agent_registry

    def test_agent_has_correct_instructions(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Generated agent has config instructions."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        agent_class = factory.agent_registry["entry"]
        agent = agent_class()

        assert agent.instructions == "Greet the user warmly."
        assert agent.assistant_id == "entry"
        assert agent.assistant_name == "Entry Assistant"

    def test_agent_has_handoff_tools(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Agent class has handoff function_tools when configured."""
        # Add handoff target to entry assistant
        minimal_valid_config["assistants"][0]["handoff_targets"] = [
            {"assistant_id": "specialist", "description": "Transfer to specialist"}
        ]

        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        agent_class = factory.agent_registry["entry"]

        # Check that the handoff tool was added
        assert hasattr(agent_class, "transfer_to_specialist")

    def test_handoff_tool_count_matches_targets(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Number of tools matches handoff_targets."""
        # Add multiple handoff targets
        minimal_valid_config["assistants"][0]["handoff_targets"] = [
            {"assistant_id": "specialist", "description": "Transfer to specialist"}
        ]

        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        agent_class = factory.agent_registry["entry"]

        # Count handoff tools (methods starting with transfer_to_)
        handoff_tools = [
            name for name in dir(agent_class) if name.startswith("transfer_to_")
        ]
        assert len(handoff_tools) == 1

    def test_agent_registry_population(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """All assistants registered in factory."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        assert len(factory.agent_registry) == len(config.assistants)
        for assistant in config.assistants:
            assert assistant.id in factory.agent_registry

    def test_get_entry_agent(self, minimal_valid_config: dict[str, Any]) -> None:
        """Factory returns instantiated entry agent."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        entry_agent = factory.get_entry_agent()
        assert entry_agent is not None
        assert entry_agent.assistant_id == "entry"

    def test_get_agent_class(self, minimal_valid_config: dict[str, Any]) -> None:
        """Factory returns agent class by ID."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        agent_class = factory.get_agent_class("specialist")
        assert agent_class is not None

        nonexistent = factory.get_agent_class("nonexistent")
        assert nonexistent is None

    def test_list_assistants(self, minimal_valid_config: dict[str, Any]) -> None:
        """Factory lists all assistant IDs."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)

        assistants = factory.list_assistants()
        assert set(assistants) == {"entry", "specialist"}


class TestHandoffCoordinator:
    """Test handoff coordination logic."""

    def test_can_handoff_within_limit(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Handoff allowed within limit."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        assert coordinator.can_handoff()

    def test_cannot_handoff_at_limit(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Handoff blocked when limit reached."""
        minimal_valid_config["orchestration"]["max_handoffs"] = 1
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        # Simulate reaching the limit
        coordinator.state.handoff_count = 1

        assert not coordinator.can_handoff()

    def test_validate_valid_handoff(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Valid handoff passes validation."""
        minimal_valid_config["assistants"][0]["handoff_targets"] = [
            {"assistant_id": "specialist", "description": "Transfer"}
        ]

        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        is_valid, error = coordinator.validate_handoff_target("entry", "specialist")
        assert is_valid
        assert error is None

    def test_validate_invalid_target(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Invalid handoff target fails validation."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        # entry doesn't have specialist as a handoff target
        is_valid, error = coordinator.validate_handoff_target("entry", "specialist")
        assert not is_valid
        assert "cannot hand off" in error.lower()

    def test_validate_nonexistent_target(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Nonexistent target fails validation."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        is_valid, error = coordinator.validate_handoff_target("entry", "nonexistent")
        assert not is_valid
        assert "unknown" in error.lower()

    def test_get_valid_handoff_targets(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Get list of valid targets for assistant."""
        minimal_valid_config["assistants"][0]["handoff_targets"] = [
            {"assistant_id": "specialist", "description": "Transfer"}
        ]

        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        targets = coordinator.get_valid_handoff_targets("entry")
        assert targets == ["specialist"]

    def test_record_handoff(self, minimal_valid_config: dict[str, Any]) -> None:
        """Record handoff updates state."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)
        session_data = create_session_data("test-session", "entry")

        coordinator.record_handoff("entry", "specialist", session_data)

        assert coordinator.state.current_assistant_id == "specialist"
        assert coordinator.state.handoff_count == 1
        assert "entry" in coordinator.state.previous_assistant_ids

    def test_reset_coordinator(self, minimal_valid_config: dict[str, Any]) -> None:
        """Reset returns to initial state."""
        config = load_config_from_dict(minimal_valid_config)
        factory = AgentFactory(config)
        coordinator = HandoffCoordinator(config, factory)

        # Make some changes
        coordinator.state.handoff_count = 5
        coordinator.state.current_assistant_id = "other"

        coordinator.reset()

        assert coordinator.state.handoff_count == 0
        assert coordinator.state.current_assistant_id == "entry"


class TestOrchestrationManager:
    """Test orchestration manager."""

    def test_create_from_config_dict(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Create manager from config dictionary."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        assert manager.config.metadata.name == "Test Config"
        assert len(manager.list_assistants()) == 2

    def test_create_session_data(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Create session data with entry assistant."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        session_data = manager.create_session_data("test-session")
        assert session_data.session_id == "test-session"
        assert session_data.current_assistant_id == "entry"

    def test_get_entry_agent(self, minimal_valid_config: dict[str, Any]) -> None:
        """Get entry point agent."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        agent = manager.get_entry_agent()
        assert agent.assistant_id == "entry"

    def test_get_fallback_agent(self, minimal_valid_config: dict[str, Any]) -> None:
        """Get fallback agent when configured."""
        minimal_valid_config["orchestration"]["fallback_assistant"] = "specialist"
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        fallback = manager.get_fallback_agent()
        assert fallback is not None
        assert fallback.assistant_id == "specialist"

    def test_get_fallback_agent_not_configured(
        self, minimal_valid_config: dict[str, Any]
    ) -> None:
        """Get None when fallback not configured."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        fallback = manager.get_fallback_agent()
        assert fallback is None

    def test_get_assistant_info(self, minimal_valid_config: dict[str, Any]) -> None:
        """Get assistant information."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        info = manager.get_assistant_info("entry")
        assert info is not None
        assert info["id"] == "entry"
        assert info["name"] == "Entry Assistant"

        nonexistent = manager.get_assistant_info("nonexistent")
        assert nonexistent is None

    def test_reset_manager(self, minimal_valid_config: dict[str, Any]) -> None:
        """Reset manager state."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        # Make some changes through coordinator
        manager.coordinator.state.handoff_count = 5

        manager.reset()

        assert manager.coordinator.state.handoff_count == 0
