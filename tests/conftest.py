"""Shared pytest fixtures for all tests.

This module provides common test fixtures including:
- Temporary database setup and teardown
- Sample data for patients, sessions, assessments
- Mock SessionData for tool testing
- Configuration fixtures for config/factory/integration tests
"""

import os
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from src.persistence.database import get_connection, init_database


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing.

    Yields:
        Path to temporary database file.

    Cleanup:
        Removes the temporary file after test completes.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def test_db(temp_db_path: Path) -> Generator[Path, None, None]:
    """Initialize a test database with schema.

    Sets DATABASE_PATH environment variable to use temp database,
    initializes schema, and cleans up after test.

    Args:
        temp_db_path: Path from temp_db_path fixture.

    Yields:
        Path to initialized test database.
    """
    # Save original and set test path
    original_path = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = str(temp_db_path)

    # Initialize schema
    init_database()

    yield temp_db_path

    # Restore original
    if original_path:
        os.environ["DATABASE_PATH"] = original_path
    elif "DATABASE_PATH" in os.environ:
        del os.environ["DATABASE_PATH"]


@pytest.fixture
def sample_patient() -> dict[str, str]:
    """Sample patient data for testing.

    Returns:
        Dict with full_name and date_of_birth.
    """
    return {
        "full_name": "John Smith",
        "date_of_birth": "01/15/1980",
    }


@pytest.fixture
def sample_patient_2() -> dict[str, str]:
    """Second sample patient for multi-patient tests.

    Returns:
        Dict with full_name and date_of_birth.
    """
    return {
        "full_name": "Jane Doe",
        "date_of_birth": "03/22/1992",
    }


@pytest.fixture
def sample_assessment() -> dict[str, Any]:
    """Sample medical assessment data.

    Returns:
        Dict with assessment fields.
    """
    return {
        "chief_complaint": "Persistent headache for 3 days",
        "severity": 7,
        "urgency_level": "next_day",
        "emergency_flagged": False,
    }


@pytest.fixture
def sample_appointment() -> dict[str, Any]:
    """Sample appointment data.

    Returns:
        Dict with appointment fields.
    """
    return {
        "confirmation_number": "APT-12345",
        "department": "General Medicine",
        "date_time": "Monday at 10:00 AM",
    }


@pytest.fixture
def sample_pharmacy_request() -> dict[str, Any]:
    """Sample pharmacy refill request data.

    Returns:
        Dict with pharmacy request fields.
    """
    return {
        "medication_name": "Lisinopril 10mg",
        "request_id": "RX-123456",
    }


@pytest.fixture
def test_session_id(test_db: Path) -> Generator[str, None, None]:
    """Create a test session in the database.

    Creates a session record and returns its ID for use in tests
    that need a valid session_id for database operations.

    Args:
        test_db: Initialized test database fixture.

    Yields:
        Session ID string (UUID).
    """
    from src.persistence.repository import SessionRepository

    session_id = str(uuid.uuid4())
    repo = SessionRepository()
    repo.create(session_id)

    yield session_id


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def valid_medical_config_path() -> Path:
    """Path to the medical triage configuration file.

    Returns:
        Path to config/medical_triage.json.
    """
    # Get the project root (parent of tests directory)
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    return project_root / "config" / "medical_triage.json"


@pytest.fixture
def minimal_valid_config() -> dict[str, Any]:
    """Minimal valid configuration for testing.

    Returns a config with two assistants (entry and specialist)
    that passes all validation. Handoff targets are NOT included
    by default - tests that need them should add them explicitly.

    Returns:
        Dictionary containing minimal valid config.
    """
    return {
        "metadata": {
            "name": "Test Config",
            "version": "1.0.0",
            "description": "Test configuration",
        },
        "assistants": [
            {
                "id": "entry",
                "name": "Entry Assistant",
                "instructions": "Greet the user warmly.",
                "handoff_targets": [],
            },
            {
                "id": "specialist",
                "name": "Specialist Assistant",
                "instructions": "Handle specialized requests.",
                "handoff_targets": [],
            },
        ],
        "orchestration": {
            "entry_point": "entry",
            "handoff_type": "tool_based",
            "max_handoffs": 5,
        },
    }


@pytest.fixture
def config_invalid_handoff_target() -> dict[str, Any]:
    """Configuration with an invalid handoff target.

    Returns a config where an assistant references a
    nonexistent handoff target.

    Returns:
        Dictionary containing config with invalid handoff target.
    """
    return {
        "metadata": {
            "name": "Invalid Handoff Config",
            "version": "1.0.0",
        },
        "assistants": [
            {
                "id": "entry",
                "name": "Entry Assistant",
                "instructions": "Greet the user.",
                "handoff_targets": [
                    {
                        "assistant_id": "nonexistent",
                        "description": "Transfer to nonexistent assistant",
                    }
                ],
            },
        ],
        "orchestration": {
            "entry_point": "entry",
            "handoff_type": "tool_based",
            "max_handoffs": 5,
        },
    }
