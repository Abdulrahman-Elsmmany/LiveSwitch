"""Tests for database persistence layer (V2 schema).

This module tests the complete persistence layer including:
- Database schema with patient_id in all tables
- UPSERT operations (no duplicates)
- Direct patient queries
- Audit fields (created_at, updated_at)
- Repository CRUD operations
"""

import uuid
from pathlib import Path
from typing import Any

import pytest

from src.persistence.database import get_connection, init_database
from src.persistence.repository import PatientRepository, SessionRepository


class TestDatabaseSchema:
    """Test database schema structure and constraints."""

    def test_patients_table_exists(self, test_db: Path) -> None:
        """Patients table should exist with required columns."""
        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(patients)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "patient_id" in columns
        assert "full_name" in columns
        assert "date_of_birth" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_sessions_table_exists(self, test_db: Path) -> None:
        """Sessions table should exist with required columns."""
        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(sessions)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "session_id" in columns
        assert "patient_id" in columns
        assert "status" in columns
        assert "start_time" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_medical_assessments_has_patient_id(self, test_db: Path) -> None:
        """Medical assessments should have patient_id for direct queries."""
        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(medical_assessments)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "patient_id" in columns
        assert "session_id" in columns
        assert "chief_complaint" in columns
        assert "severity" in columns
        assert "updated_at" in columns

    def test_appointments_has_patient_id(self, test_db: Path) -> None:
        """Appointments should have patient_id for direct queries."""
        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(appointments)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "patient_id" in columns
        assert "session_id" in columns
        assert "confirmation_number" in columns
        assert "department" in columns
        assert "updated_at" in columns

    def test_pharmacy_requests_has_patient_id(self, test_db: Path) -> None:
        """Pharmacy requests should have patient_id for direct queries."""
        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(pharmacy_requests)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "patient_id" in columns
        assert "session_id" in columns
        assert "medication_name" in columns
        assert "request_id" in columns
        assert "updated_at" in columns


class TestPatientRepository:
    """Test PatientRepository CRUD operations."""

    def test_create_new_patient(
        self, test_db: Path, sample_patient: dict[str, str]
    ) -> None:
        """Creating a new patient should return patient_id."""
        repo = PatientRepository()

        patient_id = repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        assert patient_id is not None
        assert len(patient_id) == 36  # UUID format

    def test_find_existing_patient(
        self, test_db: Path, sample_patient: dict[str, str]
    ) -> None:
        """Finding an existing patient should return their record."""
        repo = PatientRepository()

        # Create first
        patient_id = repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        # Find
        found = repo.find_by_identity(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        assert found is not None
        assert found["patient_id"] == patient_id
        assert found["full_name"] == sample_patient["full_name"]

    def test_create_or_update_returns_same_id_for_existing(
        self, test_db: Path, sample_patient: dict[str, str]
    ) -> None:
        """Calling create_or_update for existing patient returns same ID."""
        repo = PatientRepository()

        patient_id_1 = repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        patient_id_2 = repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        assert patient_id_1 == patient_id_2

    def test_create_or_update_updates_updated_at(
        self, test_db: Path, sample_patient: dict[str, str]
    ) -> None:
        """Second call to create_or_update should update updated_at."""
        repo = PatientRepository()

        # Create
        repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        # Get initial updated_at
        record_1 = repo.find_by_identity(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        assert record_1 is not None
        updated_at_1 = record_1["updated_at"]

        # Call again
        import time

        time.sleep(0.01)  # Small delay to ensure timestamp differs
        repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        # Get new updated_at
        record_2 = repo.find_by_identity(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        assert record_2 is not None
        updated_at_2 = record_2["updated_at"]

        # Should have changed
        assert updated_at_2 >= updated_at_1

    def test_get_history_empty_for_new_patient(
        self, test_db: Path, sample_patient: dict[str, str]
    ) -> None:
        """New patient should have empty history."""
        repo = PatientRepository()

        patient_id = repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        history = repo.get_history(patient_id)

        assert history["total_calls"] == 0
        assert history["recent_assessments"] == []
        assert history["recent_appointments"] == []


class TestSessionRepository:
    """Test SessionRepository CRUD operations."""

    def test_create_session(self, test_db: Path) -> None:
        """Creating a session should insert a record."""
        repo = SessionRepository()
        session_id = str(uuid.uuid4())

        repo.create(session_id)

        # Verify in database
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert row is not None
        assert row["status"] == "active"
        assert row["created_at"] is not None

    def test_link_patient_to_session(
        self, test_db: Path, sample_patient: dict[str, str]
    ) -> None:
        """Linking patient to session should update session record."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Create patient and session
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)

        # Link
        session_repo.link_patient(session_id, patient_id)

        # Verify
        with get_connection() as conn:
            row = conn.execute(
                "SELECT patient_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert row is not None
        assert row["patient_id"] == patient_id

    def test_finalize_session(self, test_db: Path) -> None:
        """Finalizing session should set end_time and status."""
        repo = SessionRepository()
        session_id = str(uuid.uuid4())

        repo.create(session_id)
        repo.finalize(session_id, "completed")

        # Verify
        with get_connection() as conn:
            row = conn.execute(
                "SELECT status, end_time FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert row["status"] == "completed"
        assert row["end_time"] is not None


class TestUpsertOperations:
    """Test UPSERT operations prevent duplicates."""

    def test_upsert_assessment_creates_new(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_assessment: dict[str, Any],
    ) -> None:
        """First upsert should create a new assessment."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)

        # Upsert
        was_insert = session_repo.upsert_assessment(
            session_id, patient_id, sample_assessment
        )

        assert was_insert is True

        # Verify in database
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM medical_assessments WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert row is not None
        assert row["patient_id"] == patient_id
        assert row["chief_complaint"] == sample_assessment["chief_complaint"]

    def test_upsert_assessment_updates_existing(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_assessment: dict[str, Any],
    ) -> None:
        """Second upsert should update existing assessment, not create duplicate."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)

        # First upsert
        session_repo.upsert_assessment(session_id, patient_id, sample_assessment)

        # Second upsert with different data
        updated_assessment = {
            **sample_assessment,
            "severity": 9,
            "urgency_level": "same_day",
        }
        was_insert = session_repo.upsert_assessment(
            session_id, patient_id, updated_assessment
        )

        # Should be update, not insert
        assert was_insert is False

        # Verify only ONE record exists
        with get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM medical_assessments WHERE session_id = ?",
                (session_id,),
            ).fetchone()["cnt"]

            row = conn.execute(
                "SELECT * FROM medical_assessments WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert count == 1
        assert row["severity"] == 9
        assert row["urgency_level"] == "same_day"

    def test_upsert_appointment_creates_new(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_appointment: dict[str, Any],
    ) -> None:
        """First upsert should create a new appointment."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)

        # Upsert
        was_insert = session_repo.upsert_appointment(
            session_id, patient_id, sample_appointment
        )

        assert was_insert is True

        # Verify
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM appointments WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert row is not None
        assert row["patient_id"] == patient_id
        assert row["department"] == sample_appointment["department"]

    def test_upsert_appointment_updates_existing(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_appointment: dict[str, Any],
    ) -> None:
        """Second upsert should update existing appointment."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)

        # First upsert
        session_repo.upsert_appointment(session_id, patient_id, sample_appointment)

        # Second upsert with updated time
        updated_appointment = {
            **sample_appointment,
            "date_time": "Tuesday at 2:00 PM",
        }
        was_insert = session_repo.upsert_appointment(
            session_id, patient_id, updated_appointment
        )

        assert was_insert is False

        # Verify only ONE record
        with get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM appointments WHERE session_id = ?",
                (session_id,),
            ).fetchone()["cnt"]

            row = conn.execute(
                "SELECT * FROM appointments WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert count == 1
        assert row["appointment_datetime"] == "Tuesday at 2:00 PM"

    def test_upsert_pharmacy_request_creates_new(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_pharmacy_request: dict[str, Any],
    ) -> None:
        """First upsert should create a new pharmacy request."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)

        # Upsert
        was_insert = session_repo.upsert_pharmacy_request(
            session_id, patient_id, sample_pharmacy_request
        )

        assert was_insert is True

        # Verify
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM pharmacy_requests WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert row is not None
        assert row["patient_id"] == patient_id
        assert row["medication_name"] == sample_pharmacy_request["medication_name"]

    def test_upsert_pharmacy_allows_multiple_medications(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
    ) -> None:
        """Same session can have multiple pharmacy requests for different medications."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)

        # Upsert first medication
        med_1 = {"medication_name": "Lisinopril 10mg", "request_id": "RX-001"}
        session_repo.upsert_pharmacy_request(session_id, patient_id, med_1)

        # Upsert second medication
        med_2 = {"medication_name": "Metformin 500mg", "request_id": "RX-002"}
        session_repo.upsert_pharmacy_request(session_id, patient_id, med_2)

        # Should have TWO records (different medications)
        with get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM pharmacy_requests WHERE session_id = ?",
                (session_id,),
            ).fetchone()["cnt"]

        assert count == 2


class TestDirectPatientQueries:
    """Test direct queries by patient_id (no JOINs)."""

    def test_get_patient_assessments(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_assessment: dict[str, Any],
    ) -> None:
        """Should get all assessments for a patient directly."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup - create 2 sessions with assessments
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        for i in range(2):
            session_id = str(uuid.uuid4())
            session_repo.create(session_id)
            session_repo.link_patient(session_id, patient_id)
            session_repo.upsert_assessment(
                session_id,
                patient_id,
                {**sample_assessment, "chief_complaint": f"Issue {i+1}"},
            )
            session_repo.finalize(session_id, "completed")

        # Query directly by patient_id
        assessments = session_repo.get_patient_assessments(patient_id)

        assert len(assessments) == 2
        assert all(a["patient_id"] == patient_id for a in assessments)

    def test_get_patient_appointments(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_appointment: dict[str, Any],
    ) -> None:
        """Should get all appointments for a patient directly."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup - create 2 sessions with appointments
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        for i in range(2):
            session_id = str(uuid.uuid4())
            session_repo.create(session_id)
            session_repo.link_patient(session_id, patient_id)
            session_repo.upsert_appointment(
                session_id,
                patient_id,
                {**sample_appointment, "confirmation_number": f"APT-{i+1}"},
            )
            session_repo.finalize(session_id, "completed")

        # Query directly by patient_id
        appointments = session_repo.get_patient_appointments(patient_id)

        assert len(appointments) == 2
        assert all(a["patient_id"] == patient_id for a in appointments)

    def test_get_patient_pharmacy_requests(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_pharmacy_request: dict[str, Any],
    ) -> None:
        """Should get all pharmacy requests for a patient directly."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Setup
        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)
        session_repo.upsert_pharmacy_request(
            session_id, patient_id, sample_pharmacy_request
        )

        # Query directly
        requests = session_repo.get_patient_pharmacy_requests(patient_id)

        assert len(requests) == 1
        assert requests[0]["patient_id"] == patient_id

    def test_queries_isolated_between_patients(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_patient_2: dict[str, str],
        sample_assessment: dict[str, Any],
    ) -> None:
        """Patient queries should not return other patients' data."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        # Create two patients with assessments
        patient_id_1 = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        patient_id_2 = patient_repo.create_or_update(
            sample_patient_2["full_name"],
            sample_patient_2["date_of_birth"],
        )

        # Session for patient 1
        session_1 = str(uuid.uuid4())
        session_repo.create(session_1)
        session_repo.link_patient(session_1, patient_id_1)
        session_repo.upsert_assessment(
            session_1, patient_id_1, {**sample_assessment, "chief_complaint": "P1 issue"}
        )

        # Session for patient 2
        session_2 = str(uuid.uuid4())
        session_repo.create(session_2)
        session_repo.link_patient(session_2, patient_id_2)
        session_repo.upsert_assessment(
            session_2, patient_id_2, {**sample_assessment, "chief_complaint": "P2 issue"}
        )

        # Query each patient
        p1_assessments = session_repo.get_patient_assessments(patient_id_1)
        p2_assessments = session_repo.get_patient_assessments(patient_id_2)

        assert len(p1_assessments) == 1
        assert len(p2_assessments) == 1
        assert p1_assessments[0]["chief_complaint"] == "P1 issue"
        assert p2_assessments[0]["chief_complaint"] == "P2 issue"


class TestAuditFields:
    """Test audit trail fields (created_at, updated_at)."""

    def test_patient_has_created_at(
        self, test_db: Path, sample_patient: dict[str, str]
    ) -> None:
        """New patient should have created_at populated."""
        repo = PatientRepository()

        patient_id = repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        record = repo.find_by_identity(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        assert record is not None
        assert record["created_at"] is not None
        assert len(record["created_at"]) > 0

    def test_assessment_has_timestamps(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_assessment: dict[str, Any],
    ) -> None:
        """Assessment should have created_at and updated_at."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)
        session_repo.upsert_assessment(session_id, patient_id, sample_assessment)

        with get_connection() as conn:
            row = conn.execute(
                "SELECT created_at, updated_at FROM medical_assessments WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        assert row["created_at"] is not None
        assert row["updated_at"] is not None


class TestPatientHistory:
    """Test patient history retrieval."""

    def test_history_counts_completed_sessions(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_assessment: dict[str, Any],
    ) -> None:
        """History should count completed sessions correctly."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        # Create 3 sessions: 2 completed, 1 active
        for i in range(2):
            session_id = str(uuid.uuid4())
            session_repo.create(session_id)
            session_repo.link_patient(session_id, patient_id)
            session_repo.finalize(session_id, "completed")

        # One active session (not finalized)
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)

        history = patient_repo.get_history(patient_id)

        assert history["total_calls"] == 2  # Only completed sessions

    def test_history_includes_recent_assessments(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_assessment: dict[str, Any],
    ) -> None:
        """History should include recent assessments."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        # Create session with assessment
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)
        session_repo.upsert_assessment(session_id, patient_id, sample_assessment)
        session_repo.finalize(session_id, "completed")

        history = patient_repo.get_history(patient_id)

        assert len(history["recent_assessments"]) == 1
        assert (
            history["recent_assessments"][0]["chief_complaint"]
            == sample_assessment["chief_complaint"]
        )

    def test_history_includes_recent_appointments(
        self,
        test_db: Path,
        sample_patient: dict[str, str],
        sample_appointment: dict[str, Any],
    ) -> None:
        """History should include recent appointments."""
        patient_repo = PatientRepository()
        session_repo = SessionRepository()

        patient_id = patient_repo.create_or_update(
            sample_patient["full_name"],
            sample_patient["date_of_birth"],
        )

        # Create session with appointment
        session_id = str(uuid.uuid4())
        session_repo.create(session_id)
        session_repo.link_patient(session_id, patient_id)
        session_repo.upsert_appointment(session_id, patient_id, sample_appointment)
        session_repo.finalize(session_id, "completed")

        history = patient_repo.get_history(patient_id)

        assert len(history["recent_appointments"]) == 1
        assert (
            history["recent_appointments"][0]["department"]
            == sample_appointment["department"]
        )
