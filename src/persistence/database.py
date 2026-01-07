"""SQLite database initialization and connection management.

This module provides the database schema and connection management
for persisting patient data across multiple calls. Uses Python's
built-in sqlite3 module - no external dependencies required.
"""

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default database path (configurable via environment variable)
DEFAULT_DATABASE_PATH = "data/sessions.db"


def get_db_path() -> Path:
    """Get database path from environment or default.

    Creates the parent directory if it doesn't exist.

    Returns:
        Path to the SQLite database file.
    """
    db_path = os.environ.get("DATABASE_PATH", DEFAULT_DATABASE_PATH)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections.

    Provides automatic commit on success and connection cleanup.
    Uses Row factory for dict-like access to query results.

    Yields:
        SQLite connection with Row factory enabled.
    """
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # Dict-like access to rows
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Database schema definition - V2 with patient_id in all tables
# Key improvements:
# - patient_id in all child tables for direct queries
# - updated_at in all tables for audit trail
# - UNIQUE constraints for UPSERT support
# - NOT NULL constraints where appropriate
SCHEMA = """
-- ============================================
-- PATIENTS (master data, survives across calls)
-- ============================================
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(full_name, date_of_birth)
);

-- ============================================
-- SESSIONS (one per call)
-- ============================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    patient_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    start_time TEXT NOT NULL,
    end_time TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- ============================================
-- MEDICAL_ASSESSMENTS (one per session - UPSERT enabled)
-- ============================================
CREATE TABLE IF NOT EXISTS medical_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL,
    chief_complaint TEXT NOT NULL,
    severity INTEGER,
    urgency_level TEXT,
    emergency_flagged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- ============================================
-- APPOINTMENTS (one per session - UPSERT enabled)
-- ============================================
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL,
    confirmation_number TEXT NOT NULL UNIQUE,
    department TEXT NOT NULL,
    appointment_datetime TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- ============================================
-- PHARMACY_REQUESTS (one per medication per session)
-- ============================================
CREATE TABLE IF NOT EXISTS pharmacy_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    medication_name TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    UNIQUE(session_id, medication_name)
);

-- ============================================
-- INDEXES for fast lookups
-- ============================================
CREATE INDEX IF NOT EXISTS idx_patients_lookup
    ON patients(full_name, date_of_birth);
CREATE INDEX IF NOT EXISTS idx_sessions_patient
    ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status
    ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_assessments_session
    ON medical_assessments(session_id);
CREATE INDEX IF NOT EXISTS idx_assessments_patient
    ON medical_assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_appointments_session
    ON appointments(session_id);
CREATE INDEX IF NOT EXISTS idx_appointments_patient
    ON appointments(patient_id);
CREATE INDEX IF NOT EXISTS idx_pharmacy_session
    ON pharmacy_requests(session_id);
CREATE INDEX IF NOT EXISTS idx_pharmacy_patient
    ON pharmacy_requests(patient_id);
"""


def init_database() -> None:
    """Initialize database schema.

    Creates all tables if they don't exist. Safe to call multiple times.
    """
    db_path = get_db_path()
    logger.config(f"Initializing database at: {db_path}")

    with get_connection() as conn:
        conn.executescript(SCHEMA)

    logger.config("Database schema initialized successfully")


def reset_database() -> None:
    """Reset database by dropping all tables.

    WARNING: This deletes all data. Use only for testing.
    """
    db_path = get_db_path()
    if db_path.exists():
        db_path.unlink()
        logger.warning(f"Database deleted: {db_path}")

    init_database()


def migrate_to_v2() -> None:
    """Migrate existing database to V2 schema.

    This migration adds:
    - patient_id column to medical_assessments, appointments, pharmacy_requests
    - updated_at column to all tables
    - UNIQUE constraints for UPSERT support
    - Removes duplicate records before adding constraints

    Safe to run multiple times - checks for existing columns/indexes.
    """
    logger.config("Starting database migration to V2...")

    with get_connection() as conn:
        # Check if migration is needed by looking for patient_id in medical_assessments
        cursor = conn.execute("PRAGMA table_info(medical_assessments)")
        columns = [row[1] for row in cursor.fetchall()]

        if "patient_id" in columns:
            logger.config("Database already migrated to V2, skipping")
            return

        logger.config("Migrating medical_assessments table...")

        # 1. Add patient_id to medical_assessments
        conn.execute(
            "ALTER TABLE medical_assessments ADD COLUMN patient_id TEXT"
        )

        # 2. Populate patient_id from sessions
        conn.execute("""
            UPDATE medical_assessments
            SET patient_id = (
                SELECT patient_id FROM sessions
                WHERE sessions.session_id = medical_assessments.session_id
            )
        """)

        # 3. Add updated_at to medical_assessments
        conn.execute(
            "ALTER TABLE medical_assessments "
            "ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))"
        )

        # 4. Remove duplicate assessments (keep most recent per session)
        conn.execute("""
            DELETE FROM medical_assessments
            WHERE id NOT IN (
                SELECT MAX(id) FROM medical_assessments GROUP BY session_id
            )
        """)

        logger.config("Migrating appointments table...")

        # 5. Add patient_id to appointments
        conn.execute("ALTER TABLE appointments ADD COLUMN patient_id TEXT")
        conn.execute("""
            UPDATE appointments
            SET patient_id = (
                SELECT patient_id FROM sessions
                WHERE sessions.session_id = appointments.session_id
            )
        """)
        conn.execute(
            "ALTER TABLE appointments "
            "ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))"
        )

        logger.config("Migrating pharmacy_requests table...")

        # 6. Add patient_id to pharmacy_requests
        conn.execute("ALTER TABLE pharmacy_requests ADD COLUMN patient_id TEXT")
        conn.execute("""
            UPDATE pharmacy_requests
            SET patient_id = (
                SELECT patient_id FROM sessions
                WHERE sessions.session_id = pharmacy_requests.session_id
            )
        """)
        conn.execute(
            "ALTER TABLE pharmacy_requests "
            "ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))"
        )

        logger.config("Migrating patients and sessions tables...")

        # 7. Add updated_at to patients (rename last_call_at)
        conn.execute(
            "ALTER TABLE patients "
            "ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))"
        )
        conn.execute("""
            UPDATE patients SET updated_at = last_call_at
            WHERE last_call_at IS NOT NULL
        """)

        # 8. Add created_at and updated_at to sessions
        conn.execute(
            "ALTER TABLE sessions "
            "ADD COLUMN created_at TEXT DEFAULT (datetime('now'))"
        )
        conn.execute("UPDATE sessions SET created_at = start_time")
        conn.execute(
            "ALTER TABLE sessions "
            "ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))"
        )

        logger.config("Creating new indexes...")

        # 9. Create new indexes for patient_id in child tables
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_assessments_patient
            ON medical_assessments(patient_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_appointments_patient
            ON appointments(patient_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pharmacy_patient
            ON pharmacy_requests(patient_id)
        """)

        # Note: UNIQUE constraints cannot be added via ALTER TABLE in SQLite
        # New databases will have them; existing ones will enforce via code

    logger.config("Database migration to V2 completed successfully")
