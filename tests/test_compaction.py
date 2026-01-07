"""Tests for smart context compaction module."""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context.compaction import (
    build_summarization_prompt,
    get_persisted_data_hints,
    smart_compact_context,
    summarize_with_llm,
    token_aware_truncate,
)
from src.context.session import SessionData


class TestGetPersistedDataHints:
    """Tests for extracting Session Memory hints."""

    def test_empty_session_returns_no_data_message(self) -> None:
        """Test that empty session returns appropriate message."""
        session_data = SessionData()
        hints = get_persisted_data_hints(session_data)
        assert hints == "(No data persisted yet)"

    def test_extracts_patient_name(self) -> None:
        """Test patient name extraction."""
        session_data = SessionData()
        session_data.set_data("patient_name", "John Smith")

        hints = get_persisted_data_hints(session_data)

        assert "Patient name: John Smith" in hints

    def test_extracts_verified_patient(self) -> None:
        """Test patient verified status is included."""
        session_data = SessionData()
        session_data.set_data("patient_name", "Jane Doe")
        session_data.set_data("patient_verified", True)

        hints = get_persisted_data_hints(session_data)

        assert "Patient name: Jane Doe (verified)" in hints

    def test_extracts_patient_dob(self) -> None:
        """Test date of birth extraction."""
        session_data = SessionData()
        session_data.set_data("patient_dob", "01/15/1980")

        hints = get_persisted_data_hints(session_data)

        assert "Date of birth: 01/15/1980" in hints

    def test_extracts_chief_complaint(self) -> None:
        """Test chief complaint extraction."""
        session_data = SessionData()
        session_data.set_data("chief_complaint", "persistent headaches")

        hints = get_persisted_data_hints(session_data)

        assert "Chief complaint: persistent headaches" in hints

    def test_extracts_symptoms_severity(self) -> None:
        """Test symptom severity extraction."""
        session_data = SessionData()
        session_data.set_data("symptoms", {"severity": 7, "onset": "2 days ago"})

        hints = get_persisted_data_hints(session_data)

        assert "Severity: 7/10" in hints
        assert "Onset: 2 days ago" in hints

    def test_extracts_urgency_level(self) -> None:
        """Test urgency level extraction."""
        session_data = SessionData()
        session_data.set_data("urgency_level", "same_day")

        hints = get_persisted_data_hints(session_data)

        assert "Urgency level: same_day" in hints

    def test_extracts_appointment_info(self) -> None:
        """Test appointment information extraction."""
        session_data = SessionData()
        session_data.set_data("confirmation_number", "APT-12345")
        session_data.set_data("appointment", {"date_time": "2024-01-15 10:00 AM"})

        hints = get_persisted_data_hints(session_data)

        assert "Appointment confirmation: APT-12345" in hints
        assert "Appointment time: 2024-01-15 10:00 AM" in hints

    def test_extracts_medication_info(self) -> None:
        """Test medication information extraction."""
        session_data = SessionData()
        session_data.set_data("medication_name", "Lisinopril")
        session_data.set_data("refill_request", {"request_id": "RX-67890"})

        hints = get_persisted_data_hints(session_data)

        assert "Medication: Lisinopril" in hints
        assert "Refill request: RX-67890" in hints

    def test_multiple_fields_combined(self) -> None:
        """Test multiple fields are combined correctly."""
        session_data = SessionData()
        session_data.set_data("patient_name", "Test Patient")
        session_data.set_data("chief_complaint", "fever")
        session_data.set_data("urgency_level", "urgent")

        hints = get_persisted_data_hints(session_data)

        assert "Patient name: Test Patient" in hints
        assert "Chief complaint: fever" in hints
        assert "Urgency level: urgent" in hints


class TestBuildSummarizationPrompt:
    """Tests for prompt building."""

    def test_builds_prompt_with_conversation(self) -> None:
        """Test prompt includes conversation content."""
        items = [
            MagicMock(role="user", content="Hello, I need help"),
            MagicMock(role="assistant", content="How can I assist you?"),
        ]
        persisted_hints = "(No data persisted yet)"

        prompt = build_summarization_prompt(items, persisted_hints)

        assert "User: Hello, I need help" in prompt
        assert "Assistant: How can I assist you?" in prompt
        assert "CONVERSATION SUMMARY" not in prompt  # That's output format
        assert "SUMMARY:" in prompt  # Asks for summary

    def test_excludes_system_messages(self) -> None:
        """Test system messages are excluded from summary."""
        items = [
            MagicMock(role="system", content="You are a helpful assistant"),
            MagicMock(role="user", content="Hello"),
        ]
        persisted_hints = "(No data persisted yet)"

        prompt = build_summarization_prompt(items, persisted_hints)

        assert "System:" not in prompt
        assert "User: Hello" in prompt

    def test_includes_persisted_hints(self) -> None:
        """Test persisted data hints are included in prompt."""
        items = [MagicMock(role="user", content="Test")]
        persisted_hints = "- Patient name: John Smith\n- Chief complaint: fever"

        prompt = build_summarization_prompt(items, persisted_hints)

        assert "ALREADY SAVED" in prompt
        assert "Patient name: John Smith" in prompt
        assert "Chief complaint: fever" in prompt

    def test_handles_list_content(self) -> None:
        """Test handling of list-format content."""
        items = [
            MagicMock(role="user", content=["Hello", "World"]),
        ]
        persisted_hints = "(No data persisted yet)"

        prompt = build_summarization_prompt(items, persisted_hints)

        assert "Hello" in prompt
        assert "World" in prompt

    def test_skips_items_without_role(self) -> None:
        """Test items without role attribute are skipped."""
        items = [
            {"content": "No role here"},
            MagicMock(role="user", content="Has role"),
        ]
        persisted_hints = "(No data persisted yet)"

        prompt = build_summarization_prompt(items, persisted_hints)

        assert "User: Has role" in prompt
        assert "No role here" not in prompt


class TestSummarizeWithLLM:
    """Tests for LLM summarization."""

    @pytest.mark.asyncio
    async def test_successful_summarization(self) -> None:
        """Test successful LLM summarization."""
        mock_llm = MagicMock()

        async def mock_chat(*args: Any, **kwargs: Any) -> Any:
            chunk = MagicMock()
            chunk.content = "This is a test summary."
            yield chunk

        mock_llm.chat = mock_chat

        result = await summarize_with_llm("Test prompt", mock_llm)

        assert result == "This is a test summary."

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self) -> None:
        """Test returns None when LLM returns empty response."""
        mock_llm = MagicMock()

        async def mock_chat(*args: Any, **kwargs: Any) -> Any:
            return
            yield  # Make it a generator

        mock_llm.chat = mock_chat

        result = await summarize_with_llm("Test prompt", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self) -> None:
        """Test returns None when LLM raises exception."""
        mock_llm = MagicMock()

        async def mock_chat(*args: Any, **kwargs: Any) -> Any:
            raise Exception("LLM error")
            yield  # Make it a generator

        mock_llm.chat = mock_chat

        result = await summarize_with_llm("Test prompt", mock_llm)

        assert result is None


class TestTokenAwareTruncate:
    """Tests for token-aware truncation fallback."""

    def test_truncates_to_target_tokens(self) -> None:
        """Test truncation based on target tokens."""
        mock_ctx = MagicMock()
        mock_ctx.items = [MagicMock() for _ in range(20)]
        mock_ctx.copy.return_value = mock_ctx

        mock_tracker = MagicMock()
        mock_tracker.last_prompt_tokens = 2000

        result = token_aware_truncate(mock_ctx, target_tokens=500, token_tracker=mock_tracker)

        mock_ctx.truncate.assert_called_once()

    def test_handles_empty_context(self) -> None:
        """Test handling of empty context."""
        mock_ctx = MagicMock()
        mock_ctx.items = []

        mock_tracker = MagicMock()
        mock_tracker.last_prompt_tokens = 0

        result = token_aware_truncate(mock_ctx, target_tokens=500, token_tracker=mock_tracker)

        assert result == mock_ctx
        mock_ctx.copy.assert_not_called()

    def test_keeps_minimum_items(self) -> None:
        """Test minimum 3 items are always kept."""
        mock_ctx = MagicMock()
        mock_ctx.items = [MagicMock() for _ in range(10)]
        mock_ctx.copy.return_value = mock_ctx

        mock_tracker = MagicMock()
        mock_tracker.last_prompt_tokens = 10000  # Very high to force low keep count

        result = token_aware_truncate(mock_ctx, target_tokens=10, token_tracker=mock_tracker)

        # Should keep at least 3 items
        call_args = mock_ctx.truncate.call_args
        assert call_args[1]["max_items"] >= 3


class TestSmartCompactContext:
    """Tests for the main smart compaction function."""

    @pytest.mark.asyncio
    async def test_no_compaction_when_not_needed(self) -> None:
        """Test returns original context when compaction not needed."""
        mock_ctx = MagicMock()
        session_data = SessionData()

        with patch("src.context.compaction.get_token_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.should_compact = False
            mock_tracker.get_stats.return_value = {
                "last_prompt_tokens": 1000,
                "usage_percent": 10.0,
                "context_limit": 10000,
            }
            mock_get_tracker.return_value = mock_tracker

            result = await smart_compact_context(mock_ctx, None, session_data)

            assert result == mock_ctx

    @pytest.mark.asyncio
    async def test_returns_original_when_no_conversation(self) -> None:
        """Test returns original when no conversation items."""
        mock_ctx = MagicMock()
        mock_ctx.items = []
        session_data = SessionData()

        with patch("src.context.compaction.get_token_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.should_compact = True
            mock_tracker.last_prompt_tokens = 5000
            mock_tracker.usage_percent = 50.0
            mock_tracker.threshold = 0.4
            mock_tracker.target_tokens = 2000
            mock_tracker.compact_target = 0.2
            mock_get_tracker.return_value = mock_tracker

            result = await smart_compact_context(mock_ctx, None, session_data)

            assert result == mock_ctx

    @pytest.mark.asyncio
    async def test_uses_llm_summary_when_available(self) -> None:
        """Test uses LLM summary when LLM succeeds."""
        # Create mock context with items
        mock_ctx = MagicMock()
        system_msg = MagicMock(role="system", content="System prompt")
        user_msg = MagicMock(role="user", content="Hello")
        assistant_msg = MagicMock(role="assistant", content="Hi there")
        mock_ctx.items = [system_msg, user_msg, assistant_msg]

        # Create mock LLM
        mock_llm = MagicMock()

        async def mock_chat(*args: Any, **kwargs: Any) -> Any:
            chunk = MagicMock()
            chunk.content = "Test summary of conversation."
            yield chunk

        mock_llm.chat = mock_chat

        session_data = SessionData()

        with patch("src.context.compaction.get_token_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.should_compact = True
            mock_tracker.last_prompt_tokens = 5000
            mock_tracker.usage_percent = 50.0
            mock_tracker.threshold = 0.4
            mock_tracker.target_tokens = 2000
            mock_tracker.compact_target = 0.2
            mock_get_tracker.return_value = mock_tracker

            result = await smart_compact_context(mock_ctx, mock_llm, session_data)

            # Check that summary was included
            assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_falls_back_to_truncation_when_llm_fails(self) -> None:
        """Test falls back to truncation when LLM returns None."""
        # Create mock context with items
        mock_ctx = MagicMock()
        system_msg = MagicMock(role="system", content="System prompt")
        user_msg = MagicMock(role="user", content="Hello")
        assistant_msg = MagicMock(role="assistant", content="Hi there")
        mock_ctx.items = [system_msg, user_msg, assistant_msg]
        mock_ctx.copy.return_value = mock_ctx

        # Create mock LLM that fails
        mock_llm = MagicMock()

        async def mock_chat(*args: Any, **kwargs: Any) -> Any:
            raise Exception("LLM failure")
            yield

        mock_llm.chat = mock_chat

        session_data = SessionData()

        with patch("src.context.compaction.get_token_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.should_compact = True
            mock_tracker.last_prompt_tokens = 5000
            mock_tracker.usage_percent = 50.0
            mock_tracker.threshold = 0.4
            mock_tracker.target_tokens = 2000
            mock_tracker.compact_target = 0.2
            mock_get_tracker.return_value = mock_tracker

            result = await smart_compact_context(mock_ctx, mock_llm, session_data)

            # Should have called truncate as fallback
            mock_ctx.truncate.assert_called()

    @pytest.mark.asyncio
    async def test_uses_truncation_when_no_llm(self) -> None:
        """Test uses truncation when LLM is None."""
        mock_ctx = MagicMock()
        user_msg = MagicMock(role="user", content="Hello")
        mock_ctx.items = [user_msg]
        mock_ctx.copy.return_value = mock_ctx

        session_data = SessionData()

        with patch("src.context.compaction.get_token_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.should_compact = True
            mock_tracker.last_prompt_tokens = 5000
            mock_tracker.usage_percent = 50.0
            mock_tracker.threshold = 0.4
            mock_tracker.target_tokens = 2000
            mock_tracker.compact_target = 0.2
            mock_get_tracker.return_value = mock_tracker

            result = await smart_compact_context(mock_ctx, None, session_data)

            # Should have called truncate
            mock_ctx.truncate.assert_called()


class TestTokenTrackerIntegration:
    """Integration tests for token tracker with compaction."""

    def test_compact_target_env_var(self) -> None:
        """Test CONTEXT_COMPACT_TARGET environment variable is respected."""
        from src.context.token_tracker import get_compact_target

        # Default value
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env var
            os.environ.pop("CONTEXT_COMPACT_TARGET", None)
            target = get_compact_target()
            assert target == 0.20

        # Custom value
        with patch.dict(os.environ, {"CONTEXT_COMPACT_TARGET": "0.30"}):
            target = get_compact_target()
            assert target == 0.30

        # Clamped value (too low)
        with patch.dict(os.environ, {"CONTEXT_COMPACT_TARGET": "0.01"}):
            target = get_compact_target()
            assert target == 0.05

        # Clamped value (too high)
        with patch.dict(os.environ, {"CONTEXT_COMPACT_TARGET": "0.90"}):
            target = get_compact_target()
            assert target == 0.50
