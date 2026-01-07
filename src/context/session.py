"""Session data management for multi-assistant system.

This module provides the SessionData class that stores shared state
across all assistants during a conversation session.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class HandoffRecord:
    """Record of a single handoff event.

    Attributes:
        from_assistant: ID of the source assistant.
        to_assistant: ID of the target assistant.
        reason: Reason for the handoff.
        timestamp: When the handoff occurred.
    """

    from_assistant: str
    to_assistant: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionData:
    """Shared state across all assistants in a session.

    This dataclass is passed to AgentSession[SessionData].userdata and
    persists across all handoffs during a conversation.

    The fields here are common across use cases. For domain-specific
    fields (like patient_name for medical), the configuration's
    shared_context schema can be used to extend this.

    Attributes:
        session_id: Unique session identifier.
        start_time: When the session started.
        current_assistant_id: ID of the currently active assistant.
        handoff_count: Number of handoffs that have occurred.
        handoff_history: List of handoff records.
        collected_data: Dictionary of data collected during session.
        metadata: Additional metadata about the session.
    """

    # Session identification
    session_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)

    # Current state
    current_assistant_id: str = ""
    handoff_count: int = 0

    # Handoff tracking
    handoff_history: list[HandoffRecord] = field(default_factory=list)

    # Collected data (from tools and conversation)
    collected_data: dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_handoff(
        self,
        from_assistant: str,
        to_assistant: str,
        reason: str,
    ) -> None:
        """Record a handoff event.

        Args:
            from_assistant: ID of the source assistant.
            to_assistant: ID of the target assistant.
            reason: Reason for the handoff.
        """
        self.handoff_history.append(
            HandoffRecord(
                from_assistant=from_assistant,
                to_assistant=to_assistant,
                reason=reason,
            )
        )
        self.handoff_count += 1
        self.current_assistant_id = to_assistant

    def set_data(self, key: str, value: Any) -> None:
        """Store a value in collected data.

        Args:
            key: Data key.
            value: Data value.
        """
        self.collected_data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from collected data.

        Args:
            key: Data key.
            default: Default value if key not found.

        Returns:
            The stored value or default.
        """
        return self.collected_data.get(key, default)

    def has_data(self, key: str) -> bool:
        """Check if a key exists in collected data.

        Args:
            key: Data key to check.

        Returns:
            True if key exists, False otherwise.
        """
        return key in self.collected_data

    def get_session_duration_seconds(self) -> float:
        """Get the session duration in seconds.

        Returns:
            Duration in seconds since session start.
        """
        return (datetime.utcnow() - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert session data to dictionary.

        Useful for serialization and logging.

        Returns:
            Dictionary representation of session data.
        """
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "current_assistant_id": self.current_assistant_id,
            "handoff_count": self.handoff_count,
            "handoff_history": [
                {
                    "from": h.from_assistant,
                    "to": h.to_assistant,
                    "reason": h.reason,
                    "timestamp": h.timestamp.isoformat(),
                }
                for h in self.handoff_history
            ],
            "collected_data": self.collected_data,
            "metadata": self.metadata,
            "duration_seconds": self.get_session_duration_seconds(),
        }


def create_session_data(
    session_id: str,
    entry_assistant_id: str,
    initial_data: dict[str, Any] | None = None,
) -> SessionData:
    """Factory function to create a new SessionData instance.

    Args:
        session_id: Unique session identifier.
        entry_assistant_id: ID of the entry point assistant.
        initial_data: Optional initial collected data.

    Returns:
        Initialized SessionData instance.
    """
    return SessionData(
        session_id=session_id,
        current_assistant_id=entry_assistant_id,
        collected_data=initial_data or {},
    )
