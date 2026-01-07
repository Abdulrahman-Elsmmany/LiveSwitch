"""Integration tests for multi-assistant handoff scenarios."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.loader import load_config, load_config_from_dict
from src.context.session import SessionData, create_session_data
from src.orchestration.factory import AgentFactory
from src.orchestration.handoff import HandoffCoordinator
from src.orchestration.manager import OrchestrationManager


class TestMedicalTriageIntegration:
    """Integration tests using the medical triage configuration."""

    @pytest.fixture
    def medical_config(self, valid_medical_config_path: Path):
        """Load the medical triage configuration."""
        if not valid_medical_config_path.exists():
            pytest.skip("Medical config file not found")
        return load_config(valid_medical_config_path)

    @pytest.fixture
    def medical_manager(self, medical_config):
        """Create manager with medical config."""
        return OrchestrationManager(medical_config)

    def test_receptionist_is_entry_point(self, medical_manager):
        """Receptionist should be the entry point."""
        entry_agent = medical_manager.get_entry_agent()
        assert entry_agent.assistant_id == "receptionist"

    def test_receptionist_can_handoff_to_nurse(self, medical_config):
        """Receptionist can hand off to nurse triage."""
        factory = AgentFactory(medical_config)
        coordinator = HandoffCoordinator(medical_config, factory)

        is_valid, error = coordinator.validate_handoff_target(
            "receptionist", "nurse_triage"
        )
        assert is_valid, f"Handoff should be valid: {error}"

    def test_receptionist_can_handoff_to_scheduling(self, medical_config):
        """Receptionist can hand off to scheduling."""
        factory = AgentFactory(medical_config)
        coordinator = HandoffCoordinator(medical_config, factory)

        is_valid, error = coordinator.validate_handoff_target(
            "receptionist", "scheduling"
        )
        assert is_valid, f"Handoff should be valid: {error}"

    def test_receptionist_can_handoff_to_pharmacy(self, medical_config):
        """Receptionist can hand off to pharmacy."""
        factory = AgentFactory(medical_config)
        coordinator = HandoffCoordinator(medical_config, factory)

        is_valid, error = coordinator.validate_handoff_target(
            "receptionist", "pharmacy"
        )
        assert is_valid, f"Handoff should be valid: {error}"

    def test_nurse_can_handoff_to_scheduling(self, medical_config):
        """Nurse can hand off to scheduling."""
        factory = AgentFactory(medical_config)
        coordinator = HandoffCoordinator(medical_config, factory)

        is_valid, error = coordinator.validate_handoff_target(
            "nurse_triage", "scheduling"
        )
        assert is_valid, f"Handoff should be valid: {error}"

    def test_nurse_can_handoff_back_to_receptionist(self, medical_config):
        """Nurse can hand off back to receptionist."""
        factory = AgentFactory(medical_config)
        coordinator = HandoffCoordinator(medical_config, factory)

        is_valid, error = coordinator.validate_handoff_target(
            "nurse_triage", "receptionist"
        )
        assert is_valid, f"Handoff should be valid: {error}"

    def test_circular_handoff_allowed(self, medical_config):
        """Agent can hand back to previous agent (receptionist)."""
        factory = AgentFactory(medical_config)
        coordinator = HandoffCoordinator(medical_config, factory)

        # Nurse can hand back to receptionist
        is_valid, error = coordinator.validate_handoff_target(
            "nurse_triage", "receptionist"
        )
        assert is_valid, f"Handoff back should be valid: {error}"

    def test_context_preserved_through_chain(self, medical_manager):
        """Data persists through handoff chain."""
        session_data = medical_manager.create_session_data("test-session")

        # Set some data
        session_data.set_data("patient_name", "John Doe")
        session_data.set_data("chief_complaint", "Headache")

        # Simulate handoffs
        session_data.record_handoff("receptionist", "nurse_triage", "Symptoms reported")
        session_data.record_handoff("nurse_triage", "scheduling", "Routine appointment")

        # Verify data persisted
        assert session_data.get_data("patient_name") == "John Doe"
        assert session_data.get_data("chief_complaint") == "Headache"
        assert session_data.handoff_count == 2
        assert session_data.current_assistant_id == "scheduling"

    def test_max_handoff_limit_enforced(self, medical_config):
        """Stop after max_handoffs reached."""
        factory = AgentFactory(medical_config)
        coordinator = HandoffCoordinator(medical_config, factory)

        # Simulate reaching max handoffs (6 in medical config)
        for i in range(6):
            coordinator.state.handoff_count = i
            assert coordinator.can_handoff()

        coordinator.state.handoff_count = 6
        assert not coordinator.can_handoff()

    def test_all_assistants_have_correct_handoff_tools(self, medical_config):
        """All assistants have the expected handoff tools."""
        factory = AgentFactory(medical_config)

        for assistant in medical_config.assistants:
            agent_class = factory.agent_registry[assistant.id]

            # Check each expected handoff tool exists
            for target in assistant.handoff_targets:
                tool_name = f"transfer_to_{target.assistant_id}"
                assert hasattr(agent_class, tool_name), (
                    f"{assistant.id} missing tool {tool_name}"
                )


class TestHandoffScenarios:
    """Test specific handoff scenarios."""

    @pytest.fixture
    def handoff_config(self) -> dict[str, Any]:
        """Create a config with specific handoff scenarios."""
        return {
            "metadata": {"name": "Handoff Test", "version": "1.0.0"},
            "assistants": [
                {
                    "id": "greeter",
                    "name": "Greeter",
                    "instructions": "Greet users",
                    "handoff_targets": [
                        {"assistant_id": "sales", "description": "Sales inquiry"},
                        {"assistant_id": "support", "description": "Support request"},
                    ],
                },
                {
                    "id": "sales",
                    "name": "Sales",
                    "instructions": "Handle sales",
                    "handoff_targets": [
                        {"assistant_id": "greeter", "description": "Back to greeter"},
                    ],
                },
                {
                    "id": "support",
                    "name": "Support",
                    "instructions": "Handle support",
                    "handoff_targets": [
                        {"assistant_id": "greeter", "description": "Back to greeter"},
                    ],
                },
            ],
            "orchestration": {
                "entry_point": "greeter",
                "handoff_type": "tool_based",
                "max_handoffs": 5,
            },
        }

    def test_greeter_to_sales_handoff(self, handoff_config: dict[str, Any]):
        """Test greeter to sales handoff."""
        config = load_config_from_dict(handoff_config)
        manager = OrchestrationManager(config)

        session_data = manager.create_session_data("test")
        session_data.record_handoff("greeter", "sales", "Sales inquiry")

        assert session_data.current_assistant_id == "sales"
        assert session_data.handoff_count == 1

    def test_greeter_to_support_handoff(self, handoff_config: dict[str, Any]):
        """Test greeter to support handoff."""
        config = load_config_from_dict(handoff_config)
        manager = OrchestrationManager(config)

        session_data = manager.create_session_data("test")
        session_data.record_handoff("greeter", "support", "Support request")

        assert session_data.current_assistant_id == "support"
        assert session_data.handoff_count == 1

    def test_round_trip_handoff(self, handoff_config: dict[str, Any]):
        """Test handoff from greeter to sales and back."""
        config = load_config_from_dict(handoff_config)
        manager = OrchestrationManager(config)

        session_data = manager.create_session_data("test")

        # Greeter -> Sales
        session_data.record_handoff("greeter", "sales", "Sales inquiry")
        assert session_data.current_assistant_id == "sales"

        # Sales -> Greeter
        session_data.record_handoff("sales", "greeter", "Transfer back")
        assert session_data.current_assistant_id == "greeter"
        assert session_data.handoff_count == 2

    def test_handoff_history_tracking(self, handoff_config: dict[str, Any]):
        """Test that handoff history is properly tracked."""
        config = load_config_from_dict(handoff_config)
        manager = OrchestrationManager(config)

        session_data = manager.create_session_data("test")

        # Multiple handoffs
        session_data.record_handoff("greeter", "sales", "Sales inquiry")
        session_data.record_handoff("sales", "greeter", "Back to greeter")
        session_data.record_handoff("greeter", "support", "Support request")

        assert len(session_data.handoff_history) == 3
        assert session_data.handoff_history[0].from_assistant == "greeter"
        assert session_data.handoff_history[0].to_assistant == "sales"
        assert session_data.handoff_history[1].from_assistant == "sales"
        assert session_data.handoff_history[1].to_assistant == "greeter"
        assert session_data.handoff_history[2].from_assistant == "greeter"
        assert session_data.handoff_history[2].to_assistant == "support"


class TestSessionLifecycle:
    """Test full session lifecycle."""

    def test_session_data_to_dict(self, minimal_valid_config: dict[str, Any]):
        """Session data converts to dictionary correctly."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        session_data = manager.create_session_data("test-session-123")
        session_data.set_data("user_name", "Test User")
        session_data.record_handoff("entry", "specialist", "Transfer")

        result = session_data.to_dict()

        assert result["session_id"] == "test-session-123"
        assert result["handoff_count"] == 1
        assert result["collected_data"]["user_name"] == "Test User"
        assert len(result["handoff_history"]) == 1

    def test_session_duration_tracking(self, minimal_valid_config: dict[str, Any]):
        """Session tracks duration correctly."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        session_data = manager.create_session_data("test")

        # Duration should be positive
        duration = session_data.get_session_duration_seconds()
        assert duration >= 0

    def test_session_with_initial_data(self, minimal_valid_config: dict[str, Any]):
        """Session can be created with initial data."""
        config = load_config_from_dict(minimal_valid_config)
        manager = OrchestrationManager(config)

        initial_data = {"source": "web", "language": "en"}
        session_data = manager.create_session_data("test", initial_data)

        assert session_data.get_data("source") == "web"
        assert session_data.get_data("language") == "en"
