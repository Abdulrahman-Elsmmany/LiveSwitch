"""Session memory utilities for persistent context across handoffs.

This module provides functions to format SessionData as a structured
memory context that survives chat context compaction.

Inspired by Claude Code's auto-compact approach which preserves:
- Key decisions and current objectives
- Patterns and configuration established
- Recent changes and outcomes

The SESSION MEMORY is injected as a system message at the start of
each agent's context, ensuring critical patient information persists
even when conversation history is summarized.
"""

from src.context.session import SessionData


def format_session_memory(session_data: SessionData) -> str:
    """Format collected session data as persistent memory context.

    This creates a structured summary of critical information that
    survives chat context compaction. The formatted memory is designed
    to be injected as a system message.

    Args:
        session_data: The session data containing collected information.

    Returns:
        Formatted memory string, or empty string if no data to include.
    """
    sections: list[str] = []

    # === Patient Identity (Critical - ALWAYS preserve) ===
    patient_name = session_data.get_data("patient_name")
    patient_dob = session_data.get_data("patient_dob")
    patient_verified = session_data.get_data("patient_verified", False)

    if patient_verified and patient_name:
        sections.append("## VERIFIED PATIENT")
        sections.append(f"- Name: {patient_name}")
        if patient_dob:
            sections.append(f"- DOB: {patient_dob}")
        sections.append("- Status: VERIFIED")

    # === Patient History (from SQLite - previous calls) ===
    patient_history = session_data.get_data("patient_history")
    if patient_history and patient_history.get("total_calls", 0) > 0:
        sections.append("\n## RETURNING PATIENT HISTORY")
        sections.append(f"- Previous calls: {patient_history['total_calls']}")

        # Recent medical assessments from previous calls
        for assessment in patient_history.get("recent_assessments", [])[:2]:
            complaint = assessment.get("chief_complaint", "Unknown")
            recorded_at = assessment.get("recorded_at", "")
            # Extract just the date part if it's an ISO timestamp
            date_str = recorded_at[:10] if recorded_at else "Unknown date"
            sections.append(f"- Previous complaint: {complaint} ({date_str})")

        # Recent appointments from previous calls
        for appt in patient_history.get("recent_appointments", [])[:2]:
            dept = appt.get("department", "Unknown")
            dt = appt.get("appointment_datetime", "Unknown")
            sections.append(f"- Had appointment: {dept} on {dt}")

    # === Medical Information ===
    chief_complaint = session_data.get_data("chief_complaint")
    symptoms = session_data.get_data("symptoms")
    urgency = session_data.get_data("urgency_level")
    emergency = session_data.get_data("emergency_flagged", False)

    if chief_complaint or symptoms:
        sections.append("\n## MEDICAL ASSESSMENT")
        if chief_complaint:
            sections.append(f"- Chief Complaint: {chief_complaint}")
        if symptoms and isinstance(symptoms, dict):
            severity = symptoms.get("severity")
            onset = symptoms.get("onset")
            if severity is not None:
                sections.append(f"- Severity: {severity}/10")
            if onset:
                sections.append(f"- Onset: {onset}")
        if urgency:
            sections.append(f"- Urgency: {urgency.upper()}")
        if emergency:
            sections.append("- EMERGENCY FLAGGED")

    # === Appointment Information ===
    appointment = session_data.get_data("appointment")
    confirmation = session_data.get_data("confirmation_number")

    if appointment or confirmation:
        sections.append("\n## APPOINTMENT BOOKED")
        if confirmation:
            sections.append(f"- Confirmation: {confirmation}")
        if appointment and isinstance(appointment, dict):
            date_time = appointment.get("date_time", "N/A")
            department = appointment.get("department", "N/A")
            sections.append(f"- Date/Time: {date_time}")
            sections.append(f"- Department: {department}")

    # === Pharmacy/Refill Information ===
    refill_request = session_data.get_data("refill_request")
    medication_name = session_data.get_data("medication_name")

    if refill_request or medication_name:
        sections.append("\n## PHARMACY")
        if medication_name:
            sections.append(f"- Medication: {medication_name}")
        if refill_request and isinstance(refill_request, dict):
            request_id = refill_request.get("request_id")
            if request_id:
                sections.append(f"- Refill Request: {request_id}")

    # === Call Journey ===
    if session_data.handoff_count > 0:
        sections.append("\n## CALL JOURNEY")

        # Build journey path
        if session_data.handoff_history:
            journey_parts = [session_data.handoff_history[0].from_assistant]
            for h in session_data.handoff_history:
                journey_parts.append(h.to_assistant)
            journey = " -> ".join(journey_parts)
            sections.append(f"- Path: {journey}")

        sections.append(f"- Handoffs: {session_data.handoff_count}")

    # Build final memory block
    if sections:
        header = "# SESSION MEMORY (Persistent Data - Do Not Ignore)\n"
        header += "This information was collected earlier in this call.\n"
        return header + "\n".join(sections)

    return ""


def is_returning_patient(session_data: SessionData) -> bool:
    """Check if patient is returning from another agent.

    A returning patient is one who:
    1. Has been verified earlier in the call
    2. Has been transferred at least once

    Args:
        session_data: The session data to check.

    Returns:
        True if this is a returning (previously verified) patient.
    """
    return (
        session_data.handoff_count > 0
        and session_data.get_data("patient_verified", False)
    )


def get_patient_name(session_data: SessionData) -> str | None:
    """Get verified patient name if available.

    Args:
        session_data: The session data to check.

    Returns:
        Patient name if verified, None otherwise.
    """
    if session_data.get_data("patient_verified", False):
        name = session_data.get_data("patient_name")
        return str(name) if name is not None else None
    return None


def get_memory_summary(session_data: SessionData) -> dict[str, bool]:
    """Get a summary of what data is available in session memory.

    Useful for debugging and logging.

    Args:
        session_data: The session data to check.

    Returns:
        Dictionary indicating which data types are present.
    """
    return {
        "has_patient": bool(session_data.get_data("patient_name")),
        "patient_verified": session_data.get_data("patient_verified", False),
        "has_symptoms": bool(session_data.get_data("chief_complaint")),
        "has_appointment": bool(session_data.get_data("appointment")),
        "has_refill": bool(session_data.get_data("refill_request")),
        "handoff_count": session_data.handoff_count,
    }
