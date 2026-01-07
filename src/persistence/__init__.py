"""Persistence module for SQLite database operations.

This module provides cross-call patient memory using SQLite.
When a patient verifies their identity (name+DOB), the system
looks up their history from previous calls and injects it into
the conversation context.
"""

from src.persistence.database import get_connection, init_database
from src.persistence.repository import PatientRepository, SessionRepository

__all__ = [
    "get_connection",
    "init_database",
    "PatientRepository",
    "SessionRepository",
]
