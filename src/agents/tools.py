"""Tool handlers for multi-assistant system.

This module implements actual logic for tools declared in configuration.
Each tool validates input, performs the action, and updates session data.

Tools are designed to work with the LiveKit Agents framework and
integrate with SessionData for state persistence across handoffs.

SQLite Integration:
- Patient verification triggers DB lookup for returning patients
- Symptoms, appointments, and refills are persisted to DB
- Patient history is loaded and injected into session memory
"""

import random
import re
from datetime import datetime, timedelta
from typing import Any

from src.context.session import SessionData
from src.persistence.repository import PatientRepository, SessionRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize repositories for database operations
_patient_repo = PatientRepository()
_session_repo = SessionRepository()


# =============================================================================
# Patient Verification Tool
# =============================================================================


async def verify_patient(
    userdata: SessionData,
    full_name: str,
    date_of_birth: str,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify patient identity using name and date of birth.

    Validates the DOB format and stores patient information in session data.
    This is typically called by the receptionist at the start of a call.

    Args:
        userdata: Session data for storing verified patient info.
        full_name: Patient's full name.
        date_of_birth: Patient's DOB in MM/DD/YYYY format.
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with verification result:
        - verified: bool - Whether verification succeeded
        - message: str - Human-readable result message
        - error: str (optional) - Error description if failed
    """
    # Get config or use defaults (currently unused but available for future extensions)
    _ = tool_config or {}
    # Validate DOB format (MM/DD/YYYY)
    dob_pattern = r"^\d{2}/\d{2}/\d{4}$"
    if not re.match(dob_pattern, date_of_birth):
        logger.tool(f"Patient verification failed: invalid DOB format '{date_of_birth}'")
        return {
            "verified": False,
            "error": "Invalid date format. Please provide date of birth as MM/DD/YYYY.",
        }

    # Validate DOB is a real date
    try:
        month, day, year = date_of_birth.split("/")
        dob_date = datetime(int(year), int(month), int(day))

        # Check if DOB is in the future
        if dob_date > datetime.now():
            return {
                "verified": False,
                "error": "Date of birth cannot be in the future.",
            }

        # Check if age is reasonable (0-120 years)
        age = (datetime.now() - dob_date).days // 365
        if age > 120:
            return {
                "verified": False,
                "error": "Please verify the date of birth is correct.",
            }

    except ValueError:
        return {
            "verified": False,
            "error": "Invalid date. Please check month, day, and year values.",
        }

    # Clean up name
    cleaned_name = " ".join(full_name.strip().split())

    # Store in session data
    userdata.set_data("patient_name", cleaned_name)
    userdata.set_data("patient_dob", date_of_birth)
    userdata.set_data("patient_verified", True)

    # === SQLite Persistence ===
    # Create or update patient in DB, get patient_id
    patient_id = _patient_repo.create_or_update(cleaned_name, date_of_birth)
    userdata.set_data("patient_id", patient_id)

    # Link current session to this patient
    _session_repo.link_patient(userdata.session_id, patient_id)

    # Load patient history from previous calls (for context injection)
    history = _patient_repo.get_history(patient_id)
    userdata.set_data("patient_history", history)

    logger.tool(f"Patient verified: {cleaned_name} (DOB: {date_of_birth})")

    # Include history in response for returning patients
    if history.get("total_calls", 0) > 0:
        return {
            "verified": True,
            "message": f"Welcome back, {cleaned_name}. I see you've called us before.",
            "patient_name": cleaned_name,
            "patient_dob": date_of_birth,
            "returning_patient": True,
            "previous_calls": history["total_calls"],
        }

    return {
        "verified": True,
        "message": f"Thank you, {cleaned_name}. I've verified your identity.",
        "patient_name": cleaned_name,
        "patient_dob": date_of_birth,
        "returning_patient": False,
    }


# =============================================================================
# Symptom Recording Tool
# =============================================================================


async def record_symptoms(
    userdata: SessionData,
    chief_complaint: str,
    symptom_onset: str | None = None,
    severity_1_to_10: int | None = None,
    associated_symptoms: list[str] | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record patient symptoms for clinical documentation.

    Captures the chief complaint and related symptom details,
    storing them in session data for use by clinical staff.

    Args:
        userdata: Session data for storing symptom info.
        chief_complaint: Primary symptom or reason for call.
        symptom_onset: When symptoms started (e.g., "3 days ago").
        severity_1_to_10: Pain/discomfort severity on 1-10 scale.
        associated_symptoms: List of additional symptoms.
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with recording confirmation:
        - recorded: bool - Whether recording succeeded
        - message: str - Confirmation message
        - summary: dict - Summary of recorded symptoms
    """
    # Get config or use defaults (currently unused but available for future extensions)
    _ = tool_config or {}
    # Validate severity if provided
    if severity_1_to_10 is not None:
        if severity_1_to_10 < 1 or severity_1_to_10 > 10:
            severity_1_to_10 = max(1, min(10, severity_1_to_10))

    symptoms = {
        "chief_complaint": chief_complaint.strip(),
        "onset": symptom_onset or "Not specified",
        "severity": severity_1_to_10,
        "associated_symptoms": associated_symptoms or [],
        "recorded_at": datetime.utcnow().isoformat(),
        "recorded_by": userdata.current_assistant_id,
    }

    # Store in session data
    userdata.set_data("symptoms", symptoms)
    userdata.set_data("chief_complaint", chief_complaint.strip())

    # Determine initial urgency based on severity
    if severity_1_to_10 is not None and severity_1_to_10 >= 8:
        userdata.set_data("urgency_level", "same_day")
    elif severity_1_to_10 is not None and severity_1_to_10 >= 5:
        userdata.set_data("urgency_level", "next_day")
    else:
        userdata.set_data("urgency_level", "routine")

    # === SQLite Persistence (V2: upsert with patient_id) ===
    patient_id = userdata.get_data("patient_id")
    if patient_id:
        _session_repo.upsert_assessment(
            session_id=userdata.session_id,
            patient_id=patient_id,
            data={
                "chief_complaint": chief_complaint.strip(),
                "severity": severity_1_to_10,
                "urgency_level": userdata.get_data("urgency_level"),
                "emergency_flagged": False,
            },
        )

    logger.tool(f"Symptoms recorded: {chief_complaint} (severity: {severity_1_to_10})")

    return {
        "recorded": True,
        "message": "I've documented your symptoms.",
        "summary": symptoms,
    }


# =============================================================================
# Emergency Flag Tool
# =============================================================================


async def flag_emergency(
    userdata: SessionData,
    reason: str,
    recommended_action: str,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flag call as potential emergency requiring immediate action.

    Sets emergency flags in session data and provides appropriate
    guidance based on the severity and type of emergency.

    Args:
        userdata: Session data for storing emergency info.
        reason: Clinical reason for flagging emergency.
        recommended_action: One of "call_911", "go_to_er", "urgent_care".
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with emergency flag confirmation:
        - flagged: bool - Whether flag was set
        - action: str - Recommended action
        - message: str - Instructions for the patient
    """
    valid_actions = {"call_911", "go_to_er", "urgent_care"}
    if recommended_action not in valid_actions:
        recommended_action = "call_911"  # Default to safest option

    # Store emergency information
    userdata.set_data("emergency_flagged", True)
    userdata.set_data("emergency_reason", reason)
    userdata.set_data("emergency_action", recommended_action)
    userdata.set_data("urgency_level", "emergency")
    userdata.set_data("emergency_flagged_at", datetime.utcnow().isoformat())

    logger.tool(f"EMERGENCY FLAGGED: {reason} -> {recommended_action}")

    # Get action messages from config or use defaults
    config = tool_config or {}
    default_messages = {
        "call_911": (
            "Based on what you've described, this could be a medical emergency. "
            "Please hang up and call 911 immediately, or have someone drive you "
            "to the nearest emergency room. Do not drive yourself."
        ),
        "go_to_er": (
            "Based on your symptoms, you should go to the emergency room as soon "
            "as possible. If your symptoms worsen or you feel unsafe, call 911."
        ),
        "urgent_care": (
            "Your symptoms need prompt attention. Please visit an urgent care "
            "center within the next hour. If symptoms worsen, go to the ER."
        ),
    }
    action_messages = config.get("action_messages", default_messages)

    return {
        "flagged": True,
        "action": recommended_action,
        "reason": reason,
        "message": action_messages.get(recommended_action, default_messages[recommended_action]),
    }


# =============================================================================
# Availability Check Tool
# =============================================================================


async def check_availability(
    userdata: SessionData,
    department: str,
    urgency: str,
    preferred_time: str | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check appointment availability for a department.

    Queries (simulated) scheduling system for available slots
    based on department, urgency level, and time preference.

    Args:
        userdata: Session data for storing request info.
        department: Medical department (e.g., "General Medicine").
        urgency: One of "same_day", "next_day", "this_week", "routine".
        preferred_time: Patient's preferred time (e.g., "morning").
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with availability:
        - available: bool - Whether slots are available
        - department: str - Requested department
        - urgency: str - Urgency level
        - slots: list[str] - Available time slots
        - message: str - Human-readable summary
    """
    valid_urgencies = {"same_day", "next_day", "this_week", "routine"}
    if urgency not in valid_urgencies:
        urgency = "routine"

    # Get scheduling config from tool_config or use defaults
    config = tool_config or {}
    business_hours = config.get("business_hours", {"open": 9, "close": 17})
    slot_interval = config.get("slot_interval_hours", 2)
    min_hours_ahead = config.get("min_hours_ahead", 2)
    next_day_slots_config = config.get("next_day_slots", ["9:00 AM", "11:30 AM", "2:00 PM", "4:00 PM"])

    # Generate realistic mock availability
    now = datetime.now()
    open_hour = business_hours.get("open", 9)
    close_hour = business_hours.get("close", 17)

    if urgency == "same_day":
        # Available slots today
        base_hour = max(open_hour, now.hour + min_hours_ahead)
        slots = []
        for h in range(base_hour, close_hour, slot_interval):
            slot_time = now.replace(hour=h, minute=0)
            if slot_time > now:
                slots.append(slot_time.strftime("Today at %I:%M %p"))
        if not slots:
            slots = ["Tomorrow at 9:00 AM (earliest available)"]

    elif urgency == "next_day":
        tomorrow = now + timedelta(days=1)
        slots = [
            f"{tomorrow.strftime('%A')} at {slot_time}"
            for slot_time in next_day_slots_config
        ]

    elif urgency == "this_week":
        slots = []
        for days_ahead in range(2, 6):
            future = now + timedelta(days=days_ahead)
            if future.weekday() < 5:  # Weekdays only
                slots.append(f"{future.strftime('%A, %B %d')} at 10:00 AM")
        slots = slots[:4]  # Max 4 options

    else:  # routine
        slots = []
        for weeks_ahead in range(1, 3):
            future = now + timedelta(weeks=weeks_ahead)
            # Find next Monday
            days_until_monday = (7 - future.weekday()) % 7
            monday = future + timedelta(days=days_until_monday)
            slots.append(f"{monday.strftime('%A, %B %d')} at 9:00 AM")
            wednesday = monday + timedelta(days=2)
            slots.append(f"{wednesday.strftime('%A, %B %d')} at 2:00 PM")
        slots = slots[:4]

    # Filter by preference if provided
    if preferred_time:
        pref_lower = preferred_time.lower()
        if "morning" in pref_lower:
            slots = [s for s in slots if "AM" in s or "9:" in s or "10:" in s or "11:" in s] or slots
        elif "afternoon" in pref_lower:
            slots = [s for s in slots if "PM" in s] or slots

    # Store request in session
    userdata.set_data("requested_department", department)
    userdata.set_data("requested_urgency", urgency)
    userdata.set_data("available_slots", slots)

    logger.tool(f"Availability checked: {department} / {urgency} - {len(slots)} slots")

    urgency_labels = {
        "same_day": "same-day",
        "next_day": "next-day",
        "this_week": "this week",
        "routine": "routine",
    }

    return {
        "available": len(slots) > 0,
        "department": department,
        "urgency": urgency,
        "slots": slots,
        "message": (
            f"I found {len(slots)} {urgency_labels[urgency]} openings for {department}."
        ),
    }


# =============================================================================
# Book Appointment Tool
# =============================================================================


async def book_appointment(
    userdata: SessionData,
    department: str,
    date_time: str,
    appointment_type: str,
    provider_preference: str | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Book an appointment for the patient.

    Creates an appointment record with confirmation number
    and stores details in session data.

    Args:
        userdata: Session data for storing appointment info.
        department: Medical department for appointment.
        date_time: Selected appointment date/time.
        appointment_type: Type of visit (e.g., "Follow-up", "New Patient").
        provider_preference: Preferred provider name (optional).
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with booking confirmation:
        - booked: bool - Whether booking succeeded
        - confirmation: str - Confirmation number
        - details: dict - Full appointment details
        - message: str - Confirmation message for patient
    """
    # Get config or use defaults (currently unused but available for future extensions)
    _ = tool_config or {}
    # Generate confirmation number
    confirmation = f"APT-{random.randint(10000, 99999)}"

    # Get patient name from session
    patient_name = userdata.get_data("patient_name", "Patient")

    appointment = {
        "confirmation_number": confirmation,
        "patient_name": patient_name,
        "department": department,
        "date_time": date_time,
        "type": appointment_type,
        "provider": provider_preference or "Next available provider",
        "booked_at": datetime.utcnow().isoformat(),
        "booked_by": userdata.current_assistant_id,
    }

    # Store in session
    userdata.set_data("appointment", appointment)
    userdata.set_data("appointment_booked", True)
    userdata.set_data("confirmation_number", confirmation)

    # === SQLite Persistence (V2: upsert with patient_id) ===
    patient_id = userdata.get_data("patient_id")
    if patient_id:
        _session_repo.upsert_appointment(
            session_id=userdata.session_id,
            patient_id=patient_id,
            data={
                "confirmation_number": confirmation,
                "department": department,
                "date_time": date_time,
            },
        )

    logger.tool(f"Appointment booked: {confirmation} for {patient_name}")

    return {
        "booked": True,
        "confirmation": confirmation,
        "details": appointment,
        "message": (
            f"Your appointment is confirmed. "
            f"Confirmation number: {confirmation}. "
            f"{department} - {date_time}. "
            f"Please arrive 15 minutes early."
        ),
    }


# =============================================================================
# Pharmacy Tools
# =============================================================================


async def check_refill_status(
    userdata: SessionData,
    medication_name: str,
    patient_dob: str | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check the status of a prescription refill.

    Queries (simulated) pharmacy system for refill eligibility
    and remaining refills.

    Args:
        userdata: Session data for storing refill info.
        medication_name: Name of medication to check.
        patient_dob: Patient DOB for verification (optional if already verified).
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with refill status:
        - found: bool - Whether prescription was found
        - medication: str - Medication name
        - refills_remaining: int - Number of refills left
        - last_filled: str - Date of last fill
        - status: str - Current status
        - message: str - Status summary for patient
    """
    # Get config or use defaults
    config = tool_config or {}
    controlled_medications = config.get(
        "controlled_medications",
        ["adderall", "ritalin", "xanax", "valium", "oxycodone", "hydrocodone"]
    )
    common_rx_defaults = config.get(
        "common_rx_defaults",
        {"refills_range": [1, 5], "days_since_filled_range": [20, 45]}
    )
    controlled_rx_defaults = config.get(
        "controlled_rx_defaults",
        {"refills_remaining": 0, "days_since_filled_range": [25, 30]}
    )

    # Simulate prescription lookup
    med_lower = medication_name.lower()

    # Generate simulated prescription data based on config
    common_refills = common_rx_defaults.get("refills_range", [1, 5])
    common_days = common_rx_defaults.get("days_since_filled_range", [20, 45])
    controlled_days = controlled_rx_defaults.get("days_since_filled_range", [25, 30])

    simulated_prescriptions = {
        "common": {
            "refills_remaining": random.randint(common_refills[0], common_refills[1]),
            "last_filled": (datetime.now() - timedelta(days=random.randint(common_days[0], common_days[1]))).strftime("%m/%d/%Y"),
            "status": "ready_for_refill",
        },
        "controlled": {
            "refills_remaining": controlled_rx_defaults.get("refills_remaining", 0),
            "last_filled": (datetime.now() - timedelta(days=random.randint(controlled_days[0], controlled_days[1]))).strftime("%m/%d/%Y"),
            "status": "requires_new_prescription",
        },
    }

    # Determine prescription type using config list
    is_controlled = any(kw in med_lower for kw in controlled_medications)

    rx_data: dict[str, Any] = simulated_prescriptions["controlled" if is_controlled else "common"]

    # Store in session
    userdata.set_data("medication_name", medication_name)
    userdata.set_data("refill_status", rx_data["status"])
    userdata.set_data("refills_remaining", rx_data["refills_remaining"])

    logger.tool(f"Refill status checked: {medication_name} - {rx_data['status']}")

    status_messages = {
        "ready_for_refill": (
            f"Your {medication_name} prescription has {rx_data['refills_remaining']} "
            f"refills remaining. Last filled on {rx_data['last_filled']}."
        ),
        "requires_new_prescription": (
            f"Your {medication_name} prescription requires a new authorization from your provider. "
            "I can transfer you to schedule an appointment, or leave a message for your doctor."
        ),
    }

    return {
        "found": True,
        "medication": medication_name,
        "refills_remaining": rx_data["refills_remaining"],
        "last_filled": rx_data["last_filled"],
        "status": rx_data["status"],
        "message": status_messages[rx_data["status"]],
    }


async def request_refill(
    userdata: SessionData,
    medication_name: str,
    pharmacy_location: str | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a prescription refill request.

    Creates a refill request and provides estimated pickup time.

    Args:
        userdata: Session data for storing request info.
        medication_name: Name of medication to refill.
        pharmacy_location: Preferred pharmacy location (optional).
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with refill request confirmation:
        - submitted: bool - Whether request was submitted
        - medication: str - Medication name
        - request_id: str - Request tracking number
        - estimated_ready: str - Estimated pickup time
        - message: str - Confirmation for patient
    """
    # Get config or use defaults
    config = tool_config or {}
    business_hours = config.get("business_hours", {"open": 9, "close": 18})
    processing_hours = config.get("processing_hours", {"min": 2, "max": 4})

    close_hour = business_hours.get("close", 18)
    min_processing = processing_hours.get("min", 2)
    max_processing = processing_hours.get("max", 4)

    # Generate request ID
    request_id = f"RX-{random.randint(100000, 999999)}"

    # Estimate ready time based on config
    now = datetime.now()
    hours_to_add = random.randint(min_processing, max_processing)

    if now.hour >= close_hour - 1:  # After close - 1 hour
        ready_time = (now + timedelta(days=1)).replace(hour=10, minute=0)
        estimated_ready = ready_time.strftime("Tomorrow by %I:%M %p")
    elif now.hour + hours_to_add >= close_hour:
        estimated_ready = f"Today before close ({close_hour}:00 PM)"
    else:
        ready_time = now + timedelta(hours=hours_to_add)
        estimated_ready = ready_time.strftime("Today by %I:%M %p")

    # Store in session
    refill_request = {
        "request_id": request_id,
        "medication": medication_name,
        "pharmacy_location": pharmacy_location or "Main pharmacy",
        "estimated_ready": estimated_ready,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    userdata.set_data("refill_request", refill_request)
    userdata.set_data("refill_submitted", True)

    # === SQLite Persistence (V2: upsert with patient_id) ===
    patient_id = userdata.get_data("patient_id")
    if patient_id:
        _session_repo.upsert_pharmacy_request(
            session_id=userdata.session_id,
            patient_id=patient_id,
            data={
                "medication_name": medication_name,
                "request_id": request_id,
            },
        )

    logger.tool(f"Refill requested: {request_id} for {medication_name}")

    return {
        "submitted": True,
        "medication": medication_name,
        "request_id": request_id,
        "estimated_ready": estimated_ready,
        "message": (
            f"Your refill request for {medication_name} has been submitted. "
            f"Request number: {request_id}. "
            f"Estimated ready: {estimated_ready}. "
            f"We'll send a text when it's ready for pickup."
        ),
    }


# =============================================================================
# Cardiac History Tool (for cardiology intake if needed)
# =============================================================================


async def record_cardiac_history(
    userdata: SessionData,
    prior_conditions: list[str] | None = None,
    current_medications: list[str] | None = None,
    family_history: str | None = None,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record patient cardiac history for cardiology assessment.

    Args:
        userdata: Session data for storing cardiac history.
        prior_conditions: List of prior cardiac conditions.
        current_medications: List of current cardiac medications.
        family_history: Description of family cardiac history.
        tool_config: Tool-specific configuration from JSON config.

    Returns:
        Dictionary with recording confirmation.
    """
    # Get config or use defaults (currently unused but available for future extensions)
    _ = tool_config or {}
    cardiac_history = {
        "prior_conditions": prior_conditions or [],
        "current_medications": current_medications or [],
        "family_history": family_history or "Not provided",
        "recorded_at": datetime.utcnow().isoformat(),
    }

    userdata.set_data("cardiac_history", cardiac_history)

    logger.tool(f"Cardiac history recorded: {len(prior_conditions or [])} conditions")

    return {
        "recorded": True,
        "message": "I've documented your cardiac history.",
        "summary": cardiac_history,
    }


# =============================================================================
# Tool Registry
# =============================================================================

# Registry mapping tool names to handler functions
# Used by the agent factory to inject tools into dynamic agents
TOOL_REGISTRY: dict[str, Any] = {
    "verify_patient": verify_patient,
    "record_symptoms": record_symptoms,
    "flag_emergency": flag_emergency,
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "check_refill_status": check_refill_status,
    "request_refill": request_refill,
    "record_cardiac_history": record_cardiac_history,
}
