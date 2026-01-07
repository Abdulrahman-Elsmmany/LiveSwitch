"""Data access layer for patient and session operations.

Provides repository classes with CRUD operations for:
- Patients: Master data that persists across calls
- Sessions: Individual call records
- Medical assessments, appointments, pharmacy requests
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from src.persistence.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _utcnow() -> str:
    """Get current UTC time as ISO format string."""
    return datetime.now(UTC).isoformat()


class PatientRepository:
    """CRUD operations for patient data.

    Patients are identified by name+DOB combination.
    When a patient calls again, we recognize them and load their history.
    """

    def find_by_identity(self, name: str, dob: str) -> dict[str, Any] | None:
        """Find patient by name and date of birth.

        Args:
            name: Patient's full name (case-sensitive).
            dob: Date of birth in MM/DD/YYYY format.

        Returns:
            Patient record as dict, or None if not found.
        """
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE full_name = ? AND date_of_birth = ?",
                (name, dob),
            ).fetchone()
            return dict(row) if row else None

    def create_or_update(self, name: str, dob: str) -> str:
        """Create new patient or update existing one's updated_at.

        This is the main entry point when a patient verifies their identity.
        If they exist, we update their updated_at timestamp.
        If new, we create a fresh patient record.

        Args:
            name: Patient's full name.
            dob: Date of birth in MM/DD/YYYY format.

        Returns:
            The patient_id (existing or newly created).
        """
        existing = self.find_by_identity(name, dob)

        with get_connection() as conn:
            if existing:
                # Returning patient - update updated_at
                conn.execute(
                    "UPDATE patients SET updated_at = ? WHERE patient_id = ?",
                    (_utcnow(), existing["patient_id"]),
                )
                logger.config(
                    f"Returning patient recognized: {name} "
                    f"(patient_id={existing['patient_id'][:8]}...)"
                )
                return str(existing["patient_id"])
            else:
                # New patient - create record
                patient_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO patients
                    (patient_id, full_name, date_of_birth, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)""",
                    (patient_id, name, dob, _utcnow(), _utcnow()),
                )
                logger.config(
                    f"New patient created: {name} (patient_id={patient_id[:8]}...)"
                )
                return patient_id

    def get_history(self, patient_id: str) -> dict[str, Any]:
        """Get patient's call history for context injection.

        Retrieves previous sessions, medical assessments, and appointments
        to provide context when the patient calls again.

        Uses direct patient_id queries (no JOINs needed in V2 schema).

        Args:
            patient_id: The patient's unique identifier.

        Returns:
            Dict with total_calls, recent_assessments, recent_appointments.
        """
        with get_connection() as conn:
            # Count completed sessions
            count_result = conn.execute(
                """SELECT COUNT(*) as count FROM sessions
                WHERE patient_id = ? AND status = 'completed'""",
                (patient_id,),
            ).fetchone()
            total_calls = count_result["count"] if count_result else 0

            # Previous medical assessments (direct query by patient_id)
            assessments = conn.execute(
                """SELECT chief_complaint, severity, urgency_level, created_at
                FROM medical_assessments
                WHERE patient_id = ?
                ORDER BY created_at DESC LIMIT 3""",
                (patient_id,),
            ).fetchall()

            # Previous appointments (direct query by patient_id)
            appointments = conn.execute(
                """SELECT department, appointment_datetime, confirmation_number
                FROM appointments
                WHERE patient_id = ?
                ORDER BY created_at DESC LIMIT 3""",
                (patient_id,),
            ).fetchall()

            history = {
                "total_calls": total_calls,
                "recent_assessments": [dict(a) for a in assessments],
                "recent_appointments": [dict(a) for a in appointments],
            }

            if total_calls > 0:
                logger.config(
                    f"Loaded patient history: {total_calls} previous calls, "
                    f"{len(assessments)} assessments, {len(appointments)} appointments"
                )

            return history


class SessionRepository:
    """CRUD operations for session data.

    Each call creates a new session. Sessions track:
    - Start/end times
    - Link to patient (after verification)
    - Status (active/completed/error)

    V2 schema includes patient_id in all child tables for direct queries.
    """

    def create(self, session_id: str) -> None:
        """Create new session record for an incoming call.

        Args:
            session_id: Unique session identifier (UUID).
        """
        now = _utcnow()
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO sessions
                (session_id, start_time, created_at, updated_at)
                VALUES (?, ?, ?, ?)""",
                (session_id, now, now, now),
            )
        logger.config(f"Session created in DB: {session_id[:8]}...")

    def link_patient(self, session_id: str, patient_id: str) -> None:
        """Link session to patient after verification.

        Called when verify_patient() tool succeeds.

        Args:
            session_id: The current session's ID.
            patient_id: The verified patient's ID.
        """
        with get_connection() as conn:
            conn.execute(
                """UPDATE sessions
                SET patient_id = ?, updated_at = ?
                WHERE session_id = ?""",
                (patient_id, _utcnow(), session_id),
            )
        logger.config(
            f"Session {session_id[:8]}... linked to patient {patient_id[:8]}..."
        )

    def finalize(self, session_id: str, status: str = "completed") -> None:
        """Mark session as completed or errored.

        Called in the finally block when a call ends.

        Args:
            session_id: The session to finalize.
            status: Final status ('completed' or 'error').
        """
        now = _utcnow()
        with get_connection() as conn:
            conn.execute(
                """UPDATE sessions
                SET end_time = ?, status = ?, updated_at = ?
                WHERE session_id = ?""",
                (now, status, now, session_id),
            )
        logger.config(f"Session {session_id[:8]}... finalized with status: {status}")

    def upsert_assessment(
        self, session_id: str, patient_id: str, data: dict[str, Any]
    ) -> bool:
        """Insert or update medical assessment for a session.

        Uses UPSERT pattern to prevent duplicates. If an assessment
        already exists for this session, it updates the existing row.

        Args:
            session_id: The current session's ID.
            patient_id: The verified patient's ID.
            data: Dict with chief_complaint, severity, urgency_level, emergency_flagged.

        Returns:
            True if new row inserted, False if existing row updated.
        """
        now = _utcnow()
        with get_connection() as conn:
            # Check if record exists to determine insert vs update
            existing = conn.execute(
                "SELECT 1 FROM medical_assessments WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            was_insert = existing is None

            conn.execute(
                """INSERT INTO medical_assessments
                (session_id, patient_id, chief_complaint, severity,
                 urgency_level, emergency_flagged, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    chief_complaint = excluded.chief_complaint,
                    severity = excluded.severity,
                    urgency_level = excluded.urgency_level,
                    emergency_flagged = excluded.emergency_flagged,
                    updated_at = excluded.updated_at""",
                (
                    session_id,
                    patient_id,
                    data.get("chief_complaint"),
                    data.get("severity"),
                    data.get("urgency_level"),
                    1 if data.get("emergency_flagged", False) else 0,
                    now,
                    now,
                ),
            )

        action = "saved" if was_insert else "updated"
        logger.config(
            f"Medical assessment {action}: {data.get('chief_complaint')} "
            f"(severity={data.get('severity')})"
        )
        return was_insert

    def upsert_appointment(
        self, session_id: str, patient_id: str, data: dict[str, Any]
    ) -> bool:
        """Insert or update appointment for a session.

        Uses UPSERT pattern to prevent duplicates.

        Args:
            session_id: The current session's ID.
            patient_id: The verified patient's ID.
            data: Dict with confirmation_number, department, date_time.

        Returns:
            True if new row inserted, False if existing row updated.
        """
        now = _utcnow()
        with get_connection() as conn:
            # Check if record exists to determine insert vs update
            existing = conn.execute(
                "SELECT 1 FROM appointments WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            was_insert = existing is None

            conn.execute(
                """INSERT INTO appointments
                (session_id, patient_id, confirmation_number, department,
                 appointment_datetime, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    confirmation_number = excluded.confirmation_number,
                    department = excluded.department,
                    appointment_datetime = excluded.appointment_datetime,
                    updated_at = excluded.updated_at""",
                (
                    session_id,
                    patient_id,
                    data.get("confirmation_number"),
                    data.get("department"),
                    data.get("date_time"),
                    now,
                    now,
                ),
            )

        action = "saved" if was_insert else "updated"
        logger.config(
            f"Appointment {action}: {data.get('department')} on {data.get('date_time')}"
        )
        return was_insert

    def upsert_pharmacy_request(
        self, session_id: str, patient_id: str, data: dict[str, Any]
    ) -> bool:
        """Insert or update pharmacy refill request.

        Uses UPSERT pattern - unique on (session_id, medication_name).

        Args:
            session_id: The current session's ID.
            patient_id: The verified patient's ID.
            data: Dict with medication_name, request_id.

        Returns:
            True if new row inserted, False if existing row updated.
        """
        now = _utcnow()
        medication_name = data.get("medication_name")
        with get_connection() as conn:
            # Check if record exists to determine insert vs update
            existing = conn.execute(
                "SELECT 1 FROM pharmacy_requests WHERE session_id = ? AND medication_name = ?",
                (session_id, medication_name),
            ).fetchone()
            was_insert = existing is None

            conn.execute(
                """INSERT INTO pharmacy_requests
                (session_id, patient_id, medication_name, request_id,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, medication_name) DO UPDATE SET
                    request_id = excluded.request_id,
                    updated_at = excluded.updated_at""",
                (
                    session_id,
                    patient_id,
                    medication_name,
                    data.get("request_id"),
                    now,
                    now,
                ),
            )

        action = "saved" if was_insert else "updated"
        logger.config(f"Pharmacy request {action}: {medication_name}")
        return was_insert

    # ========================================
    # Direct patient query methods (V2 schema)
    # ========================================

    def get_patient_assessments(self, patient_id: str) -> list[dict[str, Any]]:
        """Get all medical assessments for a patient.

        Direct query by patient_id - no JOIN needed.

        Args:
            patient_id: The patient's unique identifier.

        Returns:
            List of assessment records as dicts.
        """
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM medical_assessments
                WHERE patient_id = ?
                ORDER BY created_at DESC""",
                (patient_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_patient_appointments(self, patient_id: str) -> list[dict[str, Any]]:
        """Get all appointments for a patient.

        Direct query by patient_id - no JOIN needed.

        Args:
            patient_id: The patient's unique identifier.

        Returns:
            List of appointment records as dicts.
        """
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM appointments
                WHERE patient_id = ?
                ORDER BY created_at DESC""",
                (patient_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_patient_pharmacy_requests(self, patient_id: str) -> list[dict[str, Any]]:
        """Get all pharmacy requests for a patient.

        Direct query by patient_id - no JOIN needed.

        Args:
            patient_id: The patient's unique identifier.

        Returns:
            List of pharmacy request records as dicts.
        """
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM pharmacy_requests
                WHERE patient_id = ?
                ORDER BY created_at DESC""",
                (patient_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # ========================================
    # Legacy methods (for backwards compatibility)
    # ========================================

    def save_assessment(self, session_id: str, data: dict[str, Any]) -> None:
        """Legacy method - use upsert_assessment instead.

        Kept for backwards compatibility with existing code.
        Note: This won't include patient_id in V2 schema.
        """
        logger.warning(
            "save_assessment is deprecated, use upsert_assessment with patient_id"
        )
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO medical_assessments
                (session_id, patient_id, chief_complaint, severity,
                 urgency_level, emergency_flagged)
                VALUES (?, '', ?, ?, ?, ?)""",
                (
                    session_id,
                    data.get("chief_complaint"),
                    data.get("severity"),
                    data.get("urgency_level"),
                    1 if data.get("emergency_flagged", False) else 0,
                ),
            )

    def save_appointment(self, session_id: str, data: dict[str, Any]) -> None:
        """Legacy method - use upsert_appointment instead."""
        logger.warning(
            "save_appointment is deprecated, use upsert_appointment with patient_id"
        )
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO appointments
                (session_id, patient_id, confirmation_number, department,
                 appointment_datetime)
                VALUES (?, '', ?, ?, ?)""",
                (
                    session_id,
                    data.get("confirmation_number"),
                    data.get("department"),
                    data.get("date_time"),
                ),
            )

    def save_pharmacy_request(self, session_id: str, data: dict[str, Any]) -> None:
        """Legacy method - use upsert_pharmacy_request instead."""
        logger.warning(
            "save_pharmacy_request is deprecated, "
            "use upsert_pharmacy_request with patient_id"
        )
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO pharmacy_requests
                (session_id, patient_id, medication_name, request_id)
                VALUES (?, '', ?, ?)""",
                (
                    session_id,
                    data.get("medication_name"),
                    data.get("request_id"),
                ),
            )
