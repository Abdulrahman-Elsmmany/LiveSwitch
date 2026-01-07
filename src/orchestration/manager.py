"""Orchestration manager for multi-assistant system.

This module provides the OrchestrationManager class that ties together
the agent factory and handoff coordinator to provide a unified
interface for managing the multi-assistant conversation.
"""


from typing import Any

from livekit.agents import Agent
from livekit.agents.llm import ChatContext

from src.config.loader import load_config
from src.config.schemas import MultiAssistantConfig
from src.config.validator import validate_config_strict
from src.context.session import SessionData, create_session_data
from src.orchestration.factory import AgentFactory
from src.orchestration.handoff import HandoffCoordinator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrchestrationManager:
    """Manager for multi-assistant orchestration.

    This is the main entry point for using the multi-assistant system.
    It handles configuration loading, validation, and provides access
    to the agent factory and handoff coordinator.

    Attributes:
        config: The validated configuration.
        factory: Agent factory for creating agents.
        coordinator: Handoff coordinator for managing handoffs.
    """

    def __init__(self, config: MultiAssistantConfig) -> None:
        """Initialize the orchestration manager.

        Args:
            config: Validated multi-assistant configuration.
        """
        self.config = config
        self.factory = AgentFactory(config)
        self.coordinator = HandoffCoordinator(config, self.factory)

        logger.config(
            f"OrchestrationManager initialized: {config.metadata.name} "
            f"with {len(config.assistants)} assistants"
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "OrchestrationManager":
        """Create an OrchestrationManager from a config file path.

        This is the recommended way to create an OrchestrationManager.
        It handles loading, structural validation, and semantic validation.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            Initialized OrchestrationManager.

        Raises:
            ConfigurationError: If file cannot be loaded or parsed.
            SemanticValidationError: If configuration fails semantic validation.
        """
        logger.config(f"Loading configuration from: {config_path}")

        # Load and structurally validate
        config = load_config(config_path)

        # Perform semantic validation
        config = validate_config_strict(config)

        return cls(config)

    def create_session_data(
        self,
        session_id: str,
        initial_data: dict[str, Any] | None = None,
    ) -> SessionData:
        """Create a new session data instance.

        Args:
            session_id: Unique session identifier.
            initial_data: Optional initial collected data.

        Returns:
            Initialized SessionData instance.
        """
        return create_session_data(
            session_id=session_id,
            entry_assistant_id=self.config.orchestration.entry_point,
            initial_data=initial_data,
        )

    def get_entry_agent(self, chat_ctx: ChatContext | None = None) -> Agent:
        """Get the entry point agent.

        Args:
            chat_ctx: Optional initial chat context.

        Returns:
            An instance of the entry point agent.
        """
        return self.factory.get_entry_agent(chat_ctx)

    def get_fallback_agent(self, chat_ctx: ChatContext | None = None) -> Agent | None:
        """Get the fallback agent if configured.

        Args:
            chat_ctx: Optional chat context.

        Returns:
            An instance of the fallback agent, or None if not configured.
        """
        return self.factory.get_fallback_agent(chat_ctx)

    def get_agent(
        self,
        assistant_id: str,
        chat_ctx: ChatContext | None = None,
    ) -> Agent | None:
        """Get an agent by assistant ID.

        Args:
            assistant_id: The assistant ID.
            chat_ctx: Optional chat context.

        Returns:
            An instance of the agent, or None if not found.
        """
        agent_class = self.factory.get_agent_class(assistant_id)
        if agent_class is None:
            return None
        return agent_class(chat_ctx=chat_ctx)  # type: ignore[call-arg]

    def list_assistants(self) -> list[str]:
        """List all available assistant IDs.

        Returns:
            List of assistant ID strings.
        """
        return self.factory.list_assistants()

    def get_assistant_info(self, assistant_id: str) -> dict[str, Any] | None:
        """Get information about an assistant.

        Args:
            assistant_id: The assistant ID.

        Returns:
            Dictionary with assistant info, or None if not found.
        """
        config = self.config.get_assistant_by_id(assistant_id)
        if config is None:
            return None

        return {
            "id": config.id,
            "name": config.name,
            "handoff_targets": [t.assistant_id for t in config.handoff_targets],
            "has_on_enter": config.on_enter_instructions is not None,
            "tools_count": len(config.tools),
        }

    def reset(self) -> None:
        """Reset the orchestration state.

        Useful for starting a new session.
        """
        self.coordinator.reset()
        logger.session("Orchestration state reset")
