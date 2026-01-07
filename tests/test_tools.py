"""Tests for tool handlers.

This module tests all tool implementations in src/agents/tools.py
to ensure they validate input, store data correctly, and return
expected responses.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.agents.tools import (
    verify_patient,
    record_symptoms,
    flag_emergency,
    check_availability,
    book_appointment,
    check_refill_status,
    request_refill,
    TOOL_REGISTRY,
)
from src.context.session import SessionData


@pytest.fixture
def mock_userdata(test_session_id: str) -> MagicMock:
    """Create a mock SessionData for testing.

    Uses test_session_id fixture to ensure a valid session exists in DB.
    """
    userdata = MagicMock(spec=SessionData)
    userdata.current_assistant_id = "test_assistant"
    userdata.session_id = test_session_id  # Add real session_id for DB operations
    userdata.collected_data = {}

    def set_data(key, value):
        userdata.collected_data[key] = value

    def get_data(key, default=None):
        return userdata.collected_data.get(key, default)

    userdata.set_data = MagicMock(side_effect=set_data)
    userdata.get_data = MagicMock(side_effect=get_data)
    return userdata


class TestToolRegistry:
    """Tests for the TOOL_REGISTRY."""

    def test_registry_contains_all_tools(self):
        """All expected tools are registered."""
        expected_tools = {
            "verify_patient",
            "record_symptoms",
            "flag_emergency",
            "check_availability",
            "book_appointment",
            "check_refill_status",
            "request_refill",
            "record_cardiac_history",
        }
        assert set(TOOL_REGISTRY.keys()) == expected_tools

    def test_registry_values_are_callable(self):
        """All registry values are callable functions."""
        for name, handler in TOOL_REGISTRY.items():
            assert callable(handler), f"Handler for '{name}' is not callable"


class TestVerifyPatient:
    """Tests for patient verification tool."""

    @pytest.mark.asyncio
    async def test_valid_verification(self, mock_userdata):
        """Valid name and DOB verifies successfully."""
        result = await verify_patient(
            userdata=mock_userdata,
            full_name="John Smith",
            date_of_birth="01/15/1985",
        )

        assert result["verified"] is True
        assert "John Smith" in result["message"]
        mock_userdata.set_data.assert_any_call("patient_name", "John Smith")
        mock_userdata.set_data.assert_any_call("patient_dob", "01/15/1985")
        mock_userdata.set_data.assert_any_call("patient_verified", True)

    @pytest.mark.asyncio
    async def test_invalid_dob_format_yyyy_mm_dd(self, mock_userdata):
        """Wrong date format (YYYY-MM-DD) returns error."""
        result = await verify_patient(
            userdata=mock_userdata,
            full_name="John Smith",
            date_of_birth="1985-01-15",
        )

        assert result["verified"] is False
        assert "Invalid date format" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_dob_format_text(self, mock_userdata):
        """Text date format returns error."""
        result = await verify_patient(
            userdata=mock_userdata,
            full_name="John Smith",
            date_of_birth="January 15, 1985",
        )

        assert result["verified"] is False
        assert "Invalid date format" in result["error"]

    @pytest.mark.asyncio
    async def test_future_dob_rejected(self, mock_userdata):
        """Future DOB is rejected."""
        result = await verify_patient(
            userdata=mock_userdata,
            full_name="John Smith",
            date_of_birth="12/25/2099",
        )

        assert result["verified"] is False
        assert "future" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_date_values(self, mock_userdata):
        """Invalid date values (month 13) return error."""
        result = await verify_patient(
            userdata=mock_userdata,
            full_name="John Smith",
            date_of_birth="13/01/1985",
        )

        assert result["verified"] is False
        assert "Invalid date" in result["error"]

    @pytest.mark.asyncio
    async def test_name_whitespace_cleaned(self, mock_userdata):
        """Extra whitespace in name is cleaned."""
        result = await verify_patient(
            userdata=mock_userdata,
            full_name="  John   Smith  ",
            date_of_birth="01/15/1985",
        )

        assert result["verified"] is True
        mock_userdata.set_data.assert_any_call("patient_name", "John Smith")


class TestRecordSymptoms:
    """Tests for symptom recording tool."""

    @pytest.mark.asyncio
    async def test_record_basic_symptoms(self, mock_userdata):
        """Records basic symptoms successfully."""
        result = await record_symptoms(
            userdata=mock_userdata,
            chief_complaint="Headache",
        )

        assert result["recorded"] is True
        assert "documented" in result["message"].lower()
        mock_userdata.set_data.assert_any_call("chief_complaint", "Headache")

    @pytest.mark.asyncio
    async def test_record_full_symptom_details(self, mock_userdata):
        """Records all symptom details."""
        result = await record_symptoms(
            userdata=mock_userdata,
            chief_complaint="Chest pain",
            symptom_onset="2 hours ago",
            severity_1_to_10=7,
            associated_symptoms=["shortness of breath", "sweating"],
        )

        assert result["recorded"] is True
        assert result["summary"]["chief_complaint"] == "Chest pain"
        assert result["summary"]["onset"] == "2 hours ago"
        assert result["summary"]["severity"] == 7
        assert "shortness of breath" in result["summary"]["associated_symptoms"]

    @pytest.mark.asyncio
    async def test_high_severity_sets_same_day_urgency(self, mock_userdata):
        """Severity >= 8 sets same_day urgency."""
        await record_symptoms(
            userdata=mock_userdata,
            chief_complaint="Severe pain",
            severity_1_to_10=9,
        )

        mock_userdata.set_data.assert_any_call("urgency_level", "same_day")

    @pytest.mark.asyncio
    async def test_moderate_severity_sets_next_day_urgency(self, mock_userdata):
        """Severity 5-7 sets next_day urgency."""
        await record_symptoms(
            userdata=mock_userdata,
            chief_complaint="Moderate pain",
            severity_1_to_10=6,
        )

        mock_userdata.set_data.assert_any_call("urgency_level", "next_day")

    @pytest.mark.asyncio
    async def test_low_severity_sets_routine_urgency(self, mock_userdata):
        """Severity < 5 sets routine urgency."""
        await record_symptoms(
            userdata=mock_userdata,
            chief_complaint="Mild discomfort",
            severity_1_to_10=3,
        )

        mock_userdata.set_data.assert_any_call("urgency_level", "routine")


class TestFlagEmergency:
    """Tests for emergency flagging tool."""

    @pytest.mark.asyncio
    async def test_flag_call_911(self, mock_userdata):
        """Flags emergency with 911 recommendation."""
        result = await flag_emergency(
            userdata=mock_userdata,
            reason="Chest pain with shortness of breath",
            recommended_action="call_911",
        )

        assert result["flagged"] is True
        assert result["action"] == "call_911"
        assert "911" in result["message"]
        mock_userdata.set_data.assert_any_call("emergency_flagged", True)
        mock_userdata.set_data.assert_any_call("urgency_level", "emergency")

    @pytest.mark.asyncio
    async def test_flag_go_to_er(self, mock_userdata):
        """Flags emergency with ER recommendation."""
        result = await flag_emergency(
            userdata=mock_userdata,
            reason="Severe abdominal pain",
            recommended_action="go_to_er",
        )

        assert result["flagged"] is True
        assert result["action"] == "go_to_er"
        assert "emergency room" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_flag_urgent_care(self, mock_userdata):
        """Flags emergency with urgent care recommendation."""
        result = await flag_emergency(
            userdata=mock_userdata,
            reason="High fever for 3 days",
            recommended_action="urgent_care",
        )

        assert result["flagged"] is True
        assert result["action"] == "urgent_care"
        assert "urgent care" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_action_defaults_to_911(self, mock_userdata):
        """Invalid action defaults to call_911."""
        result = await flag_emergency(
            userdata=mock_userdata,
            reason="Unknown emergency",
            recommended_action="invalid_action",
        )

        assert result["action"] == "call_911"


class TestCheckAvailability:
    """Tests for appointment availability tool."""

    @pytest.mark.asyncio
    async def test_check_same_day_availability(self, mock_userdata):
        """Returns same-day slots."""
        result = await check_availability(
            userdata=mock_userdata,
            department="General Medicine",
            urgency="same_day",
        )

        assert result["available"] is True
        assert result["department"] == "General Medicine"
        assert result["urgency"] == "same_day"
        assert len(result["slots"]) > 0

    @pytest.mark.asyncio
    async def test_check_routine_availability(self, mock_userdata):
        """Returns routine appointment slots."""
        result = await check_availability(
            userdata=mock_userdata,
            department="Primary Care",
            urgency="routine",
        )

        assert result["available"] is True
        assert len(result["slots"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_urgency_defaults_to_routine(self, mock_userdata):
        """Invalid urgency level defaults to routine."""
        result = await check_availability(
            userdata=mock_userdata,
            department="General Medicine",
            urgency="invalid_urgency",
        )

        assert result["urgency"] == "routine"

    @pytest.mark.asyncio
    async def test_stores_request_in_session(self, mock_userdata):
        """Stores availability request in session."""
        await check_availability(
            userdata=mock_userdata,
            department="Cardiology",
            urgency="next_day",
        )

        mock_userdata.set_data.assert_any_call("requested_department", "Cardiology")
        mock_userdata.set_data.assert_any_call("requested_urgency", "next_day")


class TestBookAppointment:
    """Tests for appointment booking tool."""

    @pytest.mark.asyncio
    async def test_successful_booking(self, mock_userdata):
        """Books appointment successfully with confirmation."""
        result = await book_appointment(
            userdata=mock_userdata,
            department="General Medicine",
            date_time="Tomorrow at 10:00 AM",
            appointment_type="Follow-up",
        )

        assert result["booked"] is True
        assert result["confirmation"].startswith("APT-")
        assert len(result["confirmation"]) == 9  # APT-XXXXX
        assert "confirmation" in result["message"].lower()
        mock_userdata.set_data.assert_any_call("appointment_booked", True)

    @pytest.mark.asyncio
    async def test_booking_with_provider_preference(self, mock_userdata):
        """Books with provider preference."""
        result = await book_appointment(
            userdata=mock_userdata,
            department="Primary Care",
            date_time="Monday at 9:00 AM",
            appointment_type="Annual Physical",
            provider_preference="Dr. Johnson",
        )

        assert result["booked"] is True
        assert result["details"]["provider"] == "Dr. Johnson"

    @pytest.mark.asyncio
    async def test_booking_stores_details(self, mock_userdata):
        """Stores all appointment details in session."""
        await book_appointment(
            userdata=mock_userdata,
            department="General Medicine",
            date_time="Tomorrow at 2:00 PM",
            appointment_type="Urgent",
        )

        # Check that appointment was stored
        calls = [call for call in mock_userdata.set_data.call_args_list]
        stored_keys = [call[0][0] for call in calls]
        assert "appointment" in stored_keys
        assert "appointment_booked" in stored_keys
        assert "confirmation_number" in stored_keys


class TestCheckRefillStatus:
    """Tests for prescription refill status tool."""

    @pytest.mark.asyncio
    async def test_check_regular_medication(self, mock_userdata):
        """Checks status of regular medication."""
        result = await check_refill_status(
            userdata=mock_userdata,
            medication_name="Lisinopril",
        )

        assert result["found"] is True
        assert result["medication"] == "Lisinopril"
        assert "refills_remaining" in result
        assert "status" in result

    @pytest.mark.asyncio
    async def test_controlled_substance_requires_prescription(self, mock_userdata):
        """Controlled substances require new prescription."""
        result = await check_refill_status(
            userdata=mock_userdata,
            medication_name="Adderall",
        )

        assert result["found"] is True
        assert result["status"] == "requires_new_prescription"
        assert result["refills_remaining"] == 0

    @pytest.mark.asyncio
    async def test_stores_medication_in_session(self, mock_userdata):
        """Stores medication info in session."""
        await check_refill_status(
            userdata=mock_userdata,
            medication_name="Metformin",
        )

        mock_userdata.set_data.assert_any_call("medication_name", "Metformin")


class TestRequestRefill:
    """Tests for prescription refill request tool."""

    @pytest.mark.asyncio
    async def test_successful_refill_request(self, mock_userdata):
        """Submits refill request successfully."""
        result = await request_refill(
            userdata=mock_userdata,
            medication_name="Lisinopril",
        )

        assert result["submitted"] is True
        assert result["request_id"].startswith("RX-")
        assert "estimated_ready" in result
        mock_userdata.set_data.assert_any_call("refill_submitted", True)

    @pytest.mark.asyncio
    async def test_refill_with_pharmacy_location(self, mock_userdata):
        """Submits refill with pharmacy preference."""
        result = await request_refill(
            userdata=mock_userdata,
            medication_name="Metformin",
            pharmacy_location="Downtown branch",
        )

        assert result["submitted"] is True

    @pytest.mark.asyncio
    async def test_estimated_ready_time_provided(self, mock_userdata):
        """Provides estimated ready time."""
        result = await request_refill(
            userdata=mock_userdata,
            medication_name="Atorvastatin",
        )

        assert result["estimated_ready"] is not None
        assert len(result["estimated_ready"]) > 0
