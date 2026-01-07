"""Smart context compaction using LLM summarization.

This module implements intelligent context compaction that:
1. Uses token-based thresholds (not item counts)
2. Integrates with Session Memory (knows what's already persisted)
3. Uses LLM to summarize contextual info, excluding already-saved facts
4. Falls back to token-aware truncation if LLM fails

The compaction is domain-agnostic - it works with any type of conversation.
"""

from typing import Any

from livekit.agents.llm import ChatContext, ChatMessage

from src.context.session import SessionData
from src.context.token_tracker import get_token_tracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_persisted_data_hints(session_data: SessionData) -> str:
    """Extract what data is already stored in Session Memory.

    This tells the LLM summarizer what facts are already persisted
    and should NOT be included in the summary (to avoid redundancy).

    Args:
        session_data: The session data to extract hints from.

    Returns:
        Formatted string of persisted data hints.
    """
    hints = []

    # Patient identity
    patient_name = session_data.get_data("patient_name")
    if patient_name:
        verified = session_data.get_data("patient_verified", False)
        status = "(verified)" if verified else ""
        hints.append(f"- Patient name: {patient_name} {status}".strip())

    patient_dob = session_data.get_data("patient_dob")
    if patient_dob:
        hints.append(f"- Date of birth: {patient_dob}")

    # Medical assessment
    chief_complaint = session_data.get_data("chief_complaint")
    if chief_complaint:
        hints.append(f"- Chief complaint: {chief_complaint}")

    symptoms = session_data.get_data("symptoms")
    if symptoms and isinstance(symptoms, dict):
        severity = symptoms.get("severity")
        if severity is not None:
            hints.append(f"- Severity: {severity}/10")
        onset = symptoms.get("onset")
        if onset:
            hints.append(f"- Onset: {onset}")

    urgency = session_data.get_data("urgency_level")
    if urgency:
        hints.append(f"- Urgency level: {urgency}")

    # Appointment info
    appointment = session_data.get_data("appointment")
    confirmation = session_data.get_data("confirmation_number")
    if appointment or confirmation:
        if confirmation:
            hints.append(f"- Appointment confirmation: {confirmation}")
        if appointment and isinstance(appointment, dict):
            date_time = appointment.get("date_time")
            if date_time:
                hints.append(f"- Appointment time: {date_time}")

    # Pharmacy info
    medication_name = session_data.get_data("medication_name")
    if medication_name:
        hints.append(f"- Medication: {medication_name}")

    refill_request = session_data.get_data("refill_request")
    if refill_request and isinstance(refill_request, dict):
        request_id = refill_request.get("request_id")
        if request_id:
            hints.append(f"- Refill request: {request_id}")

    if not hints:
        return "(No data persisted yet)"

    return "\n".join(hints)


def build_summarization_prompt(
    items: list[Any],
    persisted_hints: str,
) -> str:
    """Build the domain-agnostic summarization prompt.

    The prompt tells the LLM what's already saved and asks it to
    focus on contextual information not captured in structured data.

    Args:
        items: Chat context items to summarize.
        persisted_hints: String listing already-persisted data.

    Returns:
        Formatted prompt for LLM.
    """
    # Format conversation
    conversation_lines = []
    for item in items:
        if not hasattr(item, "role") or not hasattr(item, "content"):
            continue

        role = item.role
        content = ""
        if isinstance(item.content, str):
            content = item.content
        elif isinstance(item.content, list):
            # Extract text from content list
            for part in item.content:
                if isinstance(part, str):
                    content += part + " "

        content = content.strip()
        if content and role != "system":
            prefix = "User" if role == "user" else "Assistant"
            conversation_lines.append(f"{prefix}: {content}")

    conversation_text = "\n".join(conversation_lines)

    prompt = f"""You are a conversation summarizer. Create a concise summary of this conversation.

IMPORTANT - THESE FACTS ARE ALREADY SAVED (do NOT include them in summary):
{persisted_hints}

YOUR FOCUS - Summarize ONLY:
- Conversation flow and context (how we got here)
- Questions asked and answers given
- Decisions made during the conversation
- Any information NOT listed above

CONDENSE HEAVILY:
- Greetings, acknowledgments, pleasantries
- Procedural statements ("one moment", "let me check")
- Repeated mentions of saved facts
- Small talk and filler

Output a brief narrative (under 150 words) capturing conversation context.

CONVERSATION:
{conversation_text}

SUMMARY:"""

    return prompt


async def summarize_with_llm(
    prompt: str,
    llm: Any,
) -> str | None:
    """Call LLM to generate summary.

    Args:
        prompt: The summarization prompt.
        llm: The LLM instance to use.

    Returns:
        Summary text, or None if failed.
    """
    try:
        # Create a simple completion request
        summary_ctx = ChatContext()
        summary_ctx.items.append(
            ChatMessage(role="user", content=[prompt])
        )

        # Generate summary using streaming
        response_text = ""
        async for chunk in llm.chat(chat_ctx=summary_ctx):
            if hasattr(chunk, "content") and chunk.content:
                response_text += chunk.content

        if response_text:
            logger.config(f"LLM summarization successful: {len(response_text)} chars")
            return response_text.strip()

        return None

    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        return None


def token_aware_truncate(
    chat_ctx: ChatContext,
    target_tokens: int,
    token_tracker: Any,
) -> ChatContext:
    """Fallback: Token-aware truncation keeping recent messages.

    Estimates tokens per message and keeps as many recent messages
    as fit within target_tokens.

    Args:
        chat_ctx: Original context.
        target_tokens: Target token count.
        token_tracker: Token tracker for estimates.

    Returns:
        Truncated ChatContext.
    """
    if len(chat_ctx.items) == 0:
        return chat_ctx

    # Estimate average tokens per item based on current usage
    current_tokens = token_tracker.last_prompt_tokens
    if current_tokens == 0:
        current_tokens = len(chat_ctx.items) * 100  # Rough estimate

    avg_tokens_per_item = current_tokens / len(chat_ctx.items)

    # Calculate how many items to keep
    items_to_keep = max(3, int(target_tokens / avg_tokens_per_item))

    new_ctx = chat_ctx.copy()
    new_ctx.truncate(max_items=items_to_keep)

    logger.config(
        f"Token-aware truncate: {len(chat_ctx.items)} -> {len(new_ctx.items)} items "
        f"(target {target_tokens:,} tokens, ~{avg_tokens_per_item:.0f} per item)"
    )

    return new_ctx


async def smart_compact_context(
    chat_ctx: ChatContext,
    llm: Any,
    session_data: SessionData,
) -> ChatContext:
    """Smart context compaction using LLM summarization.

    This is the main entry point for smart compaction. It:
    1. Checks if compaction is needed based on token usage
    2. Queries Session Memory for already-persisted data
    3. Builds a prompt that excludes persisted data from summary
    4. Calls LLM to summarize
    5. Falls back to token-aware truncation if LLM fails

    Args:
        chat_ctx: Original chat context.
        llm: LLM instance for summarization.
        session_data: Session data to check for persisted info.

    Returns:
        Compacted ChatContext.
    """
    token_tracker = get_token_tracker()

    # Check if compaction is needed
    if not token_tracker.should_compact:
        stats = token_tracker.get_stats()
        logger.debug(
            f"No compaction needed: {stats['last_prompt_tokens']:,} tokens "
            f"({stats['usage_percent']:.1f}% of {stats['context_limit']:,} limit)"
        )
        return chat_ctx

    logger.config(
        f"Smart compaction triggered: {token_tracker.last_prompt_tokens:,} tokens "
        f"({token_tracker.usage_percent:.1f}%) exceeds {token_tracker.threshold:.0%} threshold. "
        f"Target: {token_tracker.target_tokens:,} tokens ({token_tracker.compact_target:.0%})"
    )

    # Separate system messages (preserve) from conversation (summarize)
    system_messages = []
    conversation_items = []

    for item in chat_ctx.items:
        if hasattr(item, "role"):
            if item.role == "system":
                system_messages.append(item)
            else:
                conversation_items.append(item)

    # If no conversation to summarize, just return original
    if not conversation_items:
        return chat_ctx

    # Get hints about what's already persisted in Session Memory
    persisted_hints = get_persisted_data_hints(session_data)

    # Attempt LLM summarization
    summary = None
    if llm is not None:
        prompt = build_summarization_prompt(conversation_items, persisted_hints)
        summary = await summarize_with_llm(prompt, llm)

    # Build new context
    new_ctx = ChatContext()

    if summary:
        # Success: Use LLM summary
        # Add system messages first
        for msg in system_messages:
            new_ctx.items.append(msg)

        # Add summary as a system message with clear framing
        summary_msg = ChatMessage(
            role="system",
            content=[
                f"=== CONVERSATION SUMMARY (Prior Exchange) ===\n"
                f"{summary}\n"
                f"=== END SUMMARY ===\n"
                f"Continue the conversation naturally based on this context."
            ]
        )
        new_ctx.items.append(summary_msg)

        # Keep last 2-3 exchanges for continuity
        recent_count = min(4, len(conversation_items))
        for item in conversation_items[-recent_count:]:
            new_ctx.items.append(item)

        logger.config(
            f"Smart compaction complete: {len(chat_ctx.items)} -> {len(new_ctx.items)} items "
            f"(LLM summary: {len(summary)} chars)"
        )

    else:
        # Fallback: Token-aware truncation
        logger.warning("LLM summarization failed, using token-aware truncation")
        new_ctx = token_aware_truncate(
            chat_ctx,
            token_tracker.target_tokens,
            token_tracker,
        )

    return new_ctx
