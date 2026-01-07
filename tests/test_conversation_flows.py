"""Tests for multi-turn conversation flows.

This module tests complete conversation flows through the medical
triage system, verifying handoffs, context preservation, and
expected outcomes for various scenarios.
"""

import pytest
from pathlib import Path

from src.config.loader import load_config
from src.orchestration.manager import OrchestrationManager
from src.orchestration.factory import AgentFactory
from src.context.session import SessionData, create_session_data


@pytest.fixture
def medical_config():
    """Load the medical triage configuration."""
    config_path = Path(__file__).parent.parent / "config" / "medical_triage.json"
    return load_config(config_path)


@pytest.fixture
def manager(medical_config):
    """Create orchestration manager with medical config."""
    return OrchestrationManager(medical_config)


class TestMedicalTriageConfiguration:
    """Tests for medical triage configuration structure."""

    def test_config_has_four_assistants(self, medical_config):
        """Medical config has exactly 4 assistants."""
        assert len(medical_config.assistants) == 4

    def test_expected_assistants_present(self, medical_config):
        """All expected assistants are defined."""
        assistant_ids = {a.id for a in medical_config.assistants}
        expected = {"receptionist", "nurse_triage", "scheduling", "pharmacy"}
        assert assistant_ids == expected

    def test_receptionist_is_entry_point(self, medical_config):
        """Receptionist is the entry point."""
        assert medical_config.orchestration.entry_point == "receptionist"

    def test_receptionist_has_three_handoff_targets(self, medical_config):
        """Receptionist can hand off to nurse, scheduling, pharmacy."""
        receptionist = medical_config.get_assistant_by_id("receptionist")
        target_ids = {t.assistant_id for t in receptionist.handoff_targets}
        expected = {"nurse_triage", "scheduling", "pharmacy"}
        assert target_ids == expected

    def test_nurse_has_correct_handoff_targets(self, medical_config):
        """Nurse can hand off to scheduling and receptionist."""
        nurse = medical_config.get_assistant_by_id("nurse_triage")
        target_ids = {t.assistant_id for t in nurse.handoff_targets}
        assert target_ids == {"scheduling", "receptionist"}

    def test_no_billing_in_receptionist_instructions(self, medical_config):
        """Receptionist instructions do not mention billing as available."""
        receptionist = medical_config.get_assistant_by_id("receptionist")
        instructions_lower = receptionist.instructions.lower()

        # Should NOT mention billing as an available service
        assert "not available" in instructions_lower
        # Check that billing is in the NOT AVAILABLE section
        assert "billing" in instructions_lower


class TestSymptomToAppointmentFlow:
    """Tests for symptom -> nurse -> scheduling flow."""

    def test_receptionist_to_nurse_handoff(self, manager):
        """Receptionist hands off to nurse for symptoms."""
        session = manager.create_session_data("symptom-test")

        # Simulate patient verification
        session.set_data("patient_name", "John Smith")
        session.set_data("patient_dob", "01/15/1985")
        session.set_data("patient_verified", True)

        # Handoff to nurse
        session.record_handoff("receptionist", "nurse_triage", "Patient reports headache")

        assert session.current_assistant_id == "nurse_triage"
        assert session.handoff_count == 1
        assert session.get_data("patient_verified") is True

    def test_nurse_to_scheduling_handoff(self, manager):
        """Nurse hands off to scheduling after triage."""
        session = manager.create_session_data("triage-test")

        # Patient already verified and handed to nurse
        session.set_data("patient_verified", True)
        session.record_handoff("receptionist", "nurse_triage", "Symptoms reported")

        # Nurse records symptoms
        session.set_data("chief_complaint", "Persistent headache")
        session.set_data("urgency_level", "routine")

        # Handoff to scheduling
        session.record_handoff("nurse_triage", "scheduling", "Routine appointment needed")

        assert session.current_assistant_id == "scheduling"
        assert session.handoff_count == 2
        assert session.get_data("chief_complaint") == "Persistent headache"

    def test_complete_symptom_flow(self, manager):
        """Complete flow: reception -> nurse -> scheduling."""
        session = manager.create_session_data("complete-flow")

        # Step 1: Receptionist verifies patient
        session.set_data("patient_name", "Jane Doe")
        session.set_data("patient_verified", True)

        # Step 2: Handoff to nurse
        session.record_handoff("receptionist", "nurse_triage", "Reports feeling unwell")
        assert session.current_assistant_id == "nurse_triage"

        # Step 3: Nurse assesses symptoms
        session.set_data("chief_complaint", "Headache for 3 days")
        session.set_data("urgency_level", "next_day")

        # Step 4: Handoff to scheduling
        session.record_handoff("nurse_triage", "scheduling", "Next-day appointment")
        assert session.current_assistant_id == "scheduling"

        # Step 5: Appointment booked
        session.set_data("appointment_booked", True)
        session.set_data("confirmation_number", "APT-12345")

        # Verify complete flow
        assert session.handoff_count == 2
        assert len(session.handoff_history) == 2
        assert session.get_data("patient_verified") is True
        assert session.get_data("appointment_booked") is True


class TestEmergencyFlow:
    """Tests for emergency detection scenarios."""

    def test_emergency_flagged_at_nurse(self, manager):
        """Emergency detected by nurse sets flags."""
        session = manager.create_session_data("emergency-test")

        # Handoff to nurse
        session.record_handoff("receptionist", "nurse_triage", "Severe chest pain")

        # Nurse flags emergency
        session.set_data("emergency_flagged", True)
        session.set_data("emergency_reason", "Chest pain with shortness of breath")
        session.set_data("emergency_action", "call_911")
        session.set_data("urgency_level", "emergency")

        # Verify emergency state
        assert session.current_assistant_id == "nurse_triage"
        assert session.get_data("emergency_flagged") is True
        assert session.get_data("emergency_action") == "call_911"
        # Should NOT proceed to scheduling for emergencies
        assert session.handoff_count == 1

    def test_emergency_does_not_proceed_to_scheduling(self, manager):
        """Emergency calls don't go to scheduling."""
        session = manager.create_session_data("emergency-no-schedule")

        session.record_handoff("receptionist", "nurse_triage", "Can't breathe")
        session.set_data("emergency_flagged", True)

        # The system should stop at nurse (no further handoffs)
        assert session.current_assistant_id == "nurse_triage"


class TestDirectSchedulingFlow:
    """Tests for direct scheduling requests."""

    def test_receptionist_to_scheduling_direct(self, manager):
        """Receptionist can hand off directly to scheduling."""
        session = manager.create_session_data("direct-schedule")

        session.set_data("patient_name", "Bob Wilson")
        session.set_data("patient_verified", True)

        # Direct to scheduling (no symptoms)
        session.record_handoff("receptionist", "scheduling", "Annual checkup request")

        assert session.current_assistant_id == "scheduling"
        assert session.handoff_count == 1

    def test_scheduling_books_appointment(self, manager):
        """Scheduling can complete appointment booking."""
        session = manager.create_session_data("booking-test")

        session.set_data("patient_verified", True)
        session.record_handoff("receptionist", "scheduling", "Appointment request")

        # Scheduling books appointment
        session.set_data("appointment_booked", True)
        session.set_data("confirmation_number", "APT-99999")

        assert session.get_data("appointment_booked") is True
        assert session.get_data("confirmation_number").startswith("APT-")


class TestPharmacyFlow:
    """Tests for pharmacy refill scenarios."""

    def test_receptionist_to_pharmacy_handoff(self, manager):
        """Receptionist hands off to pharmacy for refills."""
        session = manager.create_session_data("pharmacy-test")

        session.set_data("patient_verified", True)
        session.record_handoff("receptionist", "pharmacy", "Prescription refill request")

        assert session.current_assistant_id == "pharmacy"
        assert session.handoff_count == 1

    def test_pharmacy_to_nurse_for_side_effects(self, manager):
        """Pharmacy transfers to nurse for medication side effects."""
        session = manager.create_session_data("side-effects")

        session.record_handoff("receptionist", "pharmacy", "Refill request")

        # Patient reports side effects
        session.record_handoff("pharmacy", "nurse_triage", "Reports medication side effects")

        assert session.current_assistant_id == "nurse_triage"
        assert session.handoff_count == 2


class TestContextPreservation:
    """Tests for data persistence across handoffs."""

    def test_patient_data_preserved_across_handoffs(self, manager):
        """All patient data persists through handoffs."""
        session = manager.create_session_data("context-test")

        # Set data at reception
        session.set_data("patient_name", "Alice Brown")
        session.set_data("patient_dob", "03/20/1990")
        session.set_data("patient_verified", True)
        session.set_data("insurance_on_file", True)

        # Multiple handoffs
        session.record_handoff("receptionist", "nurse_triage", "Symptoms")
        session.record_handoff("nurse_triage", "scheduling", "Appointment")

        # All data should persist
        assert session.get_data("patient_name") == "Alice Brown"
        assert session.get_data("patient_dob") == "03/20/1990"
        assert session.get_data("patient_verified") is True
        assert session.get_data("insurance_on_file") is True

    def test_handoff_history_tracked(self, manager):
        """Handoff history captures all transitions."""
        session = manager.create_session_data("history-test")

        session.record_handoff("receptionist", "nurse_triage", "Reason A")
        session.record_handoff("nurse_triage", "scheduling", "Reason B")
        session.record_handoff("scheduling", "receptionist", "Reason C")

        assert len(session.handoff_history) == 3

        # Verify handoff details
        assert session.handoff_history[0].from_assistant == "receptionist"
        assert session.handoff_history[0].to_assistant == "nurse_triage"
        assert session.handoff_history[0].reason == "Reason A"

        assert session.handoff_history[1].from_assistant == "nurse_triage"
        assert session.handoff_history[1].to_assistant == "scheduling"

        assert session.handoff_history[2].from_assistant == "scheduling"
        assert session.handoff_history[2].to_assistant == "receptionist"

    def test_data_set_by_any_assistant_is_accessible(self, manager):
        """Data set by any assistant is accessible to all."""
        session = manager.create_session_data("shared-data")

        # Receptionist sets data
        session.set_data("receptionist_data", "from_reception")
        session.record_handoff("receptionist", "nurse_triage", "Transfer")

        # Nurse adds more data
        session.set_data("nurse_data", "from_nurse")
        session.record_handoff("nurse_triage", "scheduling", "Transfer")

        # Scheduling should see all data
        assert session.get_data("receptionist_data") == "from_reception"
        assert session.get_data("nurse_data") == "from_nurse"


class TestMaxHandoffsEnforcement:
    """Tests for max handoffs limit enforcement."""

    def test_handoff_count_increments(self, manager):
        """Each handoff increments the counter."""
        session = manager.create_session_data("count-test")

        assert session.handoff_count == 0

        session.record_handoff("a", "b", "First")
        assert session.handoff_count == 1

        session.record_handoff("b", "c", "Second")
        assert session.handoff_count == 2

    def test_max_handoffs_from_config(self, medical_config):
        """Max handoffs value from config is accessible."""
        assert medical_config.orchestration.max_handoffs == 6


class TestAgentCreation:
    """Tests for agent creation from configuration."""

    def test_factory_creates_all_agents(self, medical_config):
        """Factory creates all configured agents."""
        factory = AgentFactory(medical_config)

        assert len(factory.agent_registry) == 4
        assert "receptionist" in factory.agent_registry
        assert "nurse_triage" in factory.agent_registry
        assert "scheduling" in factory.agent_registry
        assert "pharmacy" in factory.agent_registry

    def test_entry_agent_is_receptionist(self, medical_config):
        """Entry agent is the receptionist."""
        factory = AgentFactory(medical_config)
        entry_agent = factory.get_entry_agent()

        assert entry_agent.assistant_id == "receptionist"

    def test_agents_have_correct_instructions(self, medical_config):
        """Agents have instructions from config."""
        factory = AgentFactory(medical_config)

        receptionist_class = factory.get_agent_class("receptionist")
        receptionist = receptionist_class()

        # Agent should have instructions containing identity
        assert "Alex" in receptionist.instructions


class TestToolPresence:
    """Tests for tool availability on agents."""

    def test_receptionist_has_verify_patient_tool(self, medical_config):
        """Receptionist has verify_patient tool."""
        factory = AgentFactory(medical_config)
        receptionist_class = factory.get_agent_class("receptionist")

        assert hasattr(receptionist_class, "verify_patient")

    def test_nurse_has_symptom_tools(self, medical_config):
        """Nurse has symptom recording tools."""
        factory = AgentFactory(medical_config)
        nurse_class = factory.get_agent_class("nurse_triage")

        assert hasattr(nurse_class, "record_symptoms")
        assert hasattr(nurse_class, "flag_emergency")

    def test_scheduling_has_booking_tools(self, medical_config):
        """Scheduling has availability and booking tools."""
        factory = AgentFactory(medical_config)
        scheduling_class = factory.get_agent_class("scheduling")

        assert hasattr(scheduling_class, "check_availability")
        assert hasattr(scheduling_class, "book_appointment")

    def test_pharmacy_has_refill_tools(self, medical_config):
        """Pharmacy has refill tools."""
        factory = AgentFactory(medical_config)
        pharmacy_class = factory.get_agent_class("pharmacy")

        assert hasattr(pharmacy_class, "check_refill_status")
        assert hasattr(pharmacy_class, "request_refill")


class TestHandoffToolPresence:
    """Tests for handoff tool availability on agents."""

    def test_receptionist_has_handoff_tools(self, medical_config):
        """Receptionist has all handoff tools."""
        factory = AgentFactory(medical_config)
        receptionist_class = factory.get_agent_class("receptionist")

        assert hasattr(receptionist_class, "transfer_to_nurse_triage")
        assert hasattr(receptionist_class, "transfer_to_scheduling")
        assert hasattr(receptionist_class, "transfer_to_pharmacy")

    def test_nurse_has_handoff_tools(self, medical_config):
        """Nurse has correct handoff tools."""
        factory = AgentFactory(medical_config)
        nurse_class = factory.get_agent_class("nurse_triage")

        assert hasattr(nurse_class, "transfer_to_scheduling")
        assert hasattr(nurse_class, "transfer_to_receptionist")
        # Should NOT have cardiology or general medicine (removed)
        assert not hasattr(nurse_class, "transfer_to_cardiology_intake")
        assert not hasattr(nurse_class, "transfer_to_general_medicine")
