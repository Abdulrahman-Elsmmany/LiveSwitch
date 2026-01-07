"""Chat history utilities for multi-assistant system.

This module provides utilities for managing and transferring
chat history between assistants during handoffs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class TranscriptEntry:
    """A single entry in the conversation transcript.

    Attributes:
        role: Speaker role (user, assistant, system).
        content: Message content.
        assistant_id: ID of the assistant (for assistant messages).
        timestamp: When the message was created.
    """

    role: str  # "user", "assistant", "system"
    content: str
    assistant_id: str | None = None
    timestamp: datetime = None  # type: ignore

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with role, content, assistant_id, and timestamp.
        """
        return {
            "role": self.role,
            "content": self.content,
            "assistant_id": self.assistant_id,
            "timestamp": self.timestamp.isoformat(),
        }


class TranscriptLogger:
    """Logger for conversation transcripts.

    Maintains a log of all conversation messages across assistants.
    """

    def __init__(self) -> None:
        """Initialize an empty transcript."""
        self._entries: list[TranscriptEntry] = []

    def log_user_message(self, content: str) -> None:
        """Log a user message.

        Args:
            content: The user's message content.
        """
        self._entries.append(TranscriptEntry(role="user", content=content))

    def log_assistant_message(self, content: str, assistant_id: str) -> None:
        """Log an assistant message.

        Args:
            content: The assistant's message content.
            assistant_id: ID of the assistant that sent the message.
        """
        self._entries.append(
            TranscriptEntry(
                role="assistant",
                content=content,
                assistant_id=assistant_id,
            )
        )

    def log_system_message(self, content: str) -> None:
        """Log a system message.

        Args:
            content: The system message content.
        """
        self._entries.append(TranscriptEntry(role="system", content=content))

    def log_handoff(self, from_assistant: str, to_assistant: str, reason: str) -> None:
        """Log a handoff event as a system message.

        Args:
            from_assistant: ID of the source assistant.
            to_assistant: ID of the target assistant.
            reason: Reason for the handoff.
        """
        self.log_system_message(
            f"[HANDOFF] {from_assistant} -> {to_assistant}: {reason}"
        )

    def get_entries(self) -> list[TranscriptEntry]:
        """Get all transcript entries.

        Returns:
            List of all transcript entries.
        """
        return self._entries.copy()

    def get_entries_as_dicts(self) -> list[dict[str, Any]]:
        """Get all transcript entries as dictionaries.

        Returns:
            List of dictionaries representing transcript entries.
        """
        return [entry.to_dict() for entry in self._entries]

    def get_last_n_entries(self, n: int) -> list[TranscriptEntry]:
        """Get the last N transcript entries.

        Args:
            n: Number of entries to return.

        Returns:
            List of the last N transcript entries.
        """
        return self._entries[-n:] if n < len(self._entries) else self._entries.copy()

    def clear(self) -> None:
        """Clear all transcript entries."""
        self._entries.clear()

    def __len__(self) -> int:
        """Return the number of transcript entries.

        Returns:
            Number of entries in the transcript.
        """
        return len(self._entries)


def format_transcript_for_summary(entries: list[TranscriptEntry]) -> str:
    """Format transcript entries for LLM summary generation.

    Creates a formatted string suitable for asking an LLM to summarize
    the conversation so far.

    Args:
        entries: List of transcript entries.

    Returns:
        Formatted string representation of the transcript.
    """
    lines = []
    for entry in entries:
        if entry.role == "user":
            lines.append(f"User: {entry.content}")
        elif entry.role == "assistant":
            assistant_label = entry.assistant_id or "Assistant"
            lines.append(f"{assistant_label}: {entry.content}")
        elif entry.role == "system":
            lines.append(f"[System: {entry.content}]")
    return "\n".join(lines)
