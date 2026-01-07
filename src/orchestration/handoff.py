"""Handoff coordination for multi-assistant system.

This module provides the HandoffCoordinator class that manages
handoff state, validation, and coordination across assistants.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.config.schemas import HandoffType, MultiAssistantConfig
from src.context.session import SessionData
from src.orchestration.factory import AgentFactory

logger = logging.getLogger(__name__)


@dataclass
class HandoffState:
    """Current state of the handoff system.

    Tracks the active assistant and handoff history for
    monitoring and debugging purposes.
    """

    current_assistant_id: str
    previous_assistant_ids: list[str] = field(default_factory=list)
    handoff_count: int = 0
    last_handoff_time: datetime | None = None

    def record_handoff(self, new_assistant_id: str) -> None:
        """Record a handoff to a new assistant.

        Args:
            new_assistant_id: ID of the new active assistant.
        """
        self.previous_assistant_ids.append(self.current_assistant_id)
        self.current_assistant_id = new_assistant_id
        self.handoff_count += 1
        self.last_handoff_time = datetime.utcnow()


class HandoffCoordinator:
    """Coordinator for managing handoffs between assistants.

    This class manages the state and validation of handoffs,
    ensuring handoffs are valid and within limits.

    Attributes:
        config: The multi-assistant configuration.
        factory: The agent factory for creating agent instances.
        state: Current handoff state.
    """

    def __init__(
        self,
        config: MultiAssistantConfig,
        factory: AgentFactory,
    ) -> None:
        """Initialize the handoff coordinator.

        Args:
            config: Validated multi-assistant configuration.
            factory: Agent factory for creating agents.
        """
        self.config = config
        self.factory = factory
        self.state = HandoffState(
            current_assistant_id=config.orchestration.entry_point
        )

    @property
    def max_handoffs(self) -> int:
        """Maximum allowed handoffs per session."""
        return self.config.orchestration.max_handoffs

    @property
    def handoff_type(self) -> HandoffType:
        """Type of handoff mechanism in use."""
        return self.config.orchestration.handoff_type

    def can_handoff(self) -> bool:
        """Check if a handoff is currently allowed.

        Returns:
            True if handoff is allowed, False if limit reached.
        """
        return self.state.handoff_count < self.max_handoffs

    def validate_handoff_target(
        self,
        source_id: str,
        target_id: str,
    ) -> tuple[bool, str | None]:
        """Validate a proposed handoff.

        Args:
            source_id: ID of the source assistant.
            target_id: ID of the target assistant.

        Returns:
            Tuple of (is_valid, error_message).
            error_message is None if valid.
        """
        # Check handoff limit
        if not self.can_handoff():
            return False, f"Maximum handoffs ({self.max_handoffs}) reached"

        # Check source exists
        source_config = self.config.get_assistant_by_id(source_id)
        if source_config is None:
            return False, f"Unknown source assistant: {source_id}"

        # Check target exists
        target_config = self.config.get_assistant_by_id(target_id)
        if target_config is None:
            return False, f"Unknown target assistant: {target_id}"

        # Check source has target in handoff_targets
        valid_targets = {t.assistant_id for t in source_config.handoff_targets}
        if target_id not in valid_targets:
            return (
                False,
                f"Assistant '{source_id}' cannot hand off to '{target_id}'. "
                f"Valid targets: {sorted(valid_targets)}",
            )

        return True, None

    def get_valid_handoff_targets(self, assistant_id: str) -> list[str]:
        """Get list of valid handoff targets for an assistant.

        Args:
            assistant_id: ID of the assistant.

        Returns:
            List of valid target assistant IDs.
        """
        config = self.config.get_assistant_by_id(assistant_id)
        if config is None:
            return []
        return [t.assistant_id for t in config.handoff_targets]

    def record_handoff(
        self,
        source_id: str,
        target_id: str,
        session_data: SessionData,
    ) -> None:
        """Record a successful handoff.

        Updates both the coordinator state and session data.

        Args:
            source_id: ID of the source assistant.
            target_id: ID of the target assistant.
            session_data: Session data to update.
        """
        self.state.record_handoff(target_id)
        logger.info(
            f"Handoff recorded: {source_id} -> {target_id} "
            f"(total: {self.state.handoff_count})"
        )

    def get_current_assistant_id(self) -> str:
        """Get the ID of the currently active assistant.

        Returns:
            ID of the current assistant.
        """
        return self.state.current_assistant_id

    def get_handoff_history(self) -> list[str]:
        """Get the history of assistant IDs.

        Returns:
            List of previous assistant IDs (not including current).
        """
        return self.state.previous_assistant_ids.copy()

    def reset(self) -> None:
        """Reset the handoff state to initial.

        Useful for starting a new session.
        """
        self.state = HandoffState(
            current_assistant_id=self.config.orchestration.entry_point
        )
        logger.info("Handoff state reset")
