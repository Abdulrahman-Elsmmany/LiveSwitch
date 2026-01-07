"""Context module for multi-assistant system.

This module handles:
- Session data management and typed userdata
- Chat history utilities for context transfer
"""

from src.context.session import SessionData

__all__ = [
    "SessionData",
]
