"""Main entry point for multi-assistant LiveKit worker.

This module sets up the LiveKit agent worker and handles session management.
The worker reads configuration from a JSON file and dynamically generates
all orchestration logic at runtime.

Usage:
    # Development mode (connects to LiveKit Cloud)
    python -m src.main dev

    # Console mode (local testing without LiveKit)
    python -m src.main console

    # Download required model files
    python -m src.main download-files
"""

import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import cartesia, deepgram, openai, silero

from src.artifacts.metrics import MetricsCollector
from src.artifacts.report import generate_session_report, save_session_artifacts
from src.artifacts.transcript import TranscriptCollector
from src.config.loader import ConfigurationError
from src.config.validator import SemanticValidationError
from src.context.session import SessionData
from src.context.token_tracker import get_token_tracker, reset_token_tracker
from src.orchestration.manager import OrchestrationManager
from src.persistence.database import init_database
from src.persistence.repository import SessionRepository
from src.utils.logger import get_logger

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Get colored logger
logger = get_logger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = "config/medical_triage.json"


def get_config_path() -> str:
    """Get the configuration file path from environment or default.

    Returns:
        Path to the configuration file.
    """
    return os.environ.get("CONFIG_FILE_PATH", DEFAULT_CONFIG_PATH)


def build_openrouter_config(manager: OrchestrationManager) -> dict[str, Any]:
    """Build OpenRouter configuration from environment variables.

    Reads all OpenRouter-related environment variables and builds
    a configuration dict for openai.LLM.with_openrouter().

    Environment Variables:
        OPENROUTER_MODEL: Model ID (falls back to config file)
        OPENROUTER_FALLBACK_MODELS: Comma-separated fallback models
        OPENROUTER_SITE_URL: Analytics site URL
        OPENROUTER_APP_NAME: Analytics app name
        OPENROUTER_PROVIDER_ORDER: Comma-separated preferred providers
        OPENROUTER_PROVIDER_IGNORE: Comma-separated providers to exclude
        OPENROUTER_PROVIDER_ONLY: Comma-separated exclusive providers
        OPENROUTER_PROVIDER_SORT: Sort by 'price', 'throughput', or 'latency'
        OPENROUTER_ALLOW_FALLBACKS: 'true' or 'false'
        OPENROUTER_DATA_COLLECTION: 'allow' or 'deny'
        OPENROUTER_REQUIRE_PARAMETERS: 'true' to only use providers supporting all params
        OPENROUTER_MAX_PRICE: Maximum price per token (float)
        OPENROUTER_QUANTIZATIONS: Comma-separated quantization levels

    Args:
        manager: The orchestration manager for config fallbacks.

    Returns:
        Configuration dict for with_openrouter().
    """
    config: dict[str, Any] = {}

    # Model (required) - falls back to config file
    config["model"] = os.environ.get(
        "OPENROUTER_MODEL",
        manager.config.global_settings.default_models.llm,
    )

    # Fallback models (comma-separated)
    fallback_models = os.environ.get("OPENROUTER_FALLBACK_MODELS")
    if fallback_models:
        config["fallback_models"] = [m.strip() for m in fallback_models.split(",")]

    # Analytics
    site_url = os.environ.get("OPENROUTER_SITE_URL")
    if site_url:
        config["site_url"] = site_url

    app_name = os.environ.get("OPENROUTER_APP_NAME")
    if app_name:
        config["app_name"] = app_name

    # Provider configuration
    provider_config: dict[str, Any] = {}

    # Provider order (comma-separated)
    provider_order = os.environ.get("OPENROUTER_PROVIDER_ORDER")
    if provider_order:
        provider_config["order"] = [p.strip() for p in provider_order.split(",")]

    # Provider ignore (comma-separated)
    provider_ignore = os.environ.get("OPENROUTER_PROVIDER_IGNORE")
    if provider_ignore:
        provider_config["ignore"] = [p.strip() for p in provider_ignore.split(",")]

    # Provider only (comma-separated)
    provider_only = os.environ.get("OPENROUTER_PROVIDER_ONLY")
    if provider_only:
        provider_config["only"] = [p.strip() for p in provider_only.split(",")]

    # Provider sort
    provider_sort = os.environ.get("OPENROUTER_PROVIDER_SORT")
    if provider_sort:
        provider_config["sort"] = provider_sort

    # Allow fallbacks
    allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS")
    if allow_fallbacks:
        provider_config["allow_fallbacks"] = allow_fallbacks.lower() == "true"

    # Data collection preference
    data_collection = os.environ.get("OPENROUTER_DATA_COLLECTION")
    if data_collection:
        provider_config["data_collection"] = data_collection

    # Require parameters support (filters to providers supporting all params)
    require_parameters = os.environ.get("OPENROUTER_REQUIRE_PARAMETERS")
    if require_parameters:
        provider_config["require_parameters"] = require_parameters.lower() == "true"

    # Max price per token
    max_price = os.environ.get("OPENROUTER_MAX_PRICE")
    if max_price:
        provider_config["max_price"] = float(max_price)

    # Quantizations (comma-separated)
    quantizations = os.environ.get("OPENROUTER_QUANTIZATIONS")
    if quantizations:
        provider_config["quantizations"] = [q.strip() for q in quantizations.split(",")]

    # Only add provider config if we have any settings
    if provider_config:
        config["provider"] = provider_config

    return config


def create_orchestration_manager() -> OrchestrationManager:
    """Create and initialize the orchestration manager.

    Loads and validates the configuration, then creates the
    orchestration manager with all agents.

    Returns:
        Initialized OrchestrationManager.

    Raises:
        SystemExit: If configuration loading or validation fails.
    """
    config_path = get_config_path()
    logger.config(f"Loading configuration from: {config_path}")

    try:
        manager = OrchestrationManager.from_config_file(config_path)
        logger.config(f"Orchestration manager created with assistants: {manager.list_assistants()}")
        return manager
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit(1) from e
    except SemanticValidationError as e:
        logger.error(f"Validation error: {e}")
        raise SystemExit(1) from e


# Global orchestration manager (created once on worker startup)
_orchestration_manager: OrchestrationManager | None = None


def get_orchestration_manager() -> OrchestrationManager:
    """Get or create the global orchestration manager.

    Returns:
        The global OrchestrationManager instance.
    """
    global _orchestration_manager
    if _orchestration_manager is None:
        _orchestration_manager = create_orchestration_manager()
    return _orchestration_manager


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for handling incoming calls.

    This function is called for each new participant that connects.
    It initializes the session with the entry point agent and
    handles the conversation lifecycle with full observability.

    Args:
        ctx: The job context from LiveKit.
    """
    logger.session(f"New job: {ctx.job.id}")

    # Wait for participant to connect
    await ctx.connect()

    # Get the orchestration manager
    manager = get_orchestration_manager()

    # Create session data
    session_id = str(uuid.uuid4())
    session_data = manager.create_session_data(
        session_id=session_id,
        initial_data={"job_id": ctx.job.id},
    )

    # Create session record in database
    session_repo = SessionRepository()
    session_repo.create(session_id)

    # Initialize artifact collectors for observability
    transcript = TranscriptCollector(session_id)
    metrics = MetricsCollector(session_id)

    logger.session(
        f"Session {session_id} started with entry agent: "
        f"{manager.config.orchestration.entry_point}"
    )

    # Get the entry point agent
    entry_agent = manager.get_entry_agent()

    # Build OpenRouter config from environment variables
    openrouter_config = build_openrouter_config(manager)
    logger.config(f"Using LLM model: {openrouter_config['model']}")
    if "provider" in openrouter_config:
        logger.config(f"Provider config: {openrouter_config['provider']}")
    if "fallback_models" in openrouter_config:
        logger.config(f"Fallback models: {openrouter_config['fallback_models']}")

    # Create the agent session with providers:
    # - LLM: OpenRouter (fully configurable via env vars)
    # - STT: Deepgram - $200 free credits
    # - TTS: Cartesia - 10,000 free credits
    session = AgentSession[SessionData](
        vad=silero.VAD.load(),
        llm=openai.LLM.with_openrouter(**openrouter_config),
        stt=deepgram.STT(model="nova-2"),
        tts=cartesia.TTS(),
        userdata=session_data,
    )

    # Initialize token tracker for this session (for context management)
    reset_token_tracker()  # Fresh tracker for new session
    token_tracker = get_token_tracker(openrouter_config.get("model"))

    # Register LLM metrics listener for real token tracking
    # This updates the token tracker with actual token counts after each LLM response
    if session.llm is not None and hasattr(session.llm, "on"):

        @session.llm.on("metrics_collected")  # type: ignore[untyped-decorator]
        def on_llm_metrics(metrics: Any) -> None:
            """Track actual token usage from LLM responses."""
            token_tracker.update_from_metrics(metrics)

            # Log if approaching threshold (80% of compaction threshold)
            warning_threshold = token_tracker.threshold * 0.8 * 100
            if token_tracker.usage_percent > warning_threshold:
                logger.warning(
                    f"Context usage high: {token_tracker.last_prompt_tokens:,} tokens "
                    f"({token_tracker.usage_percent:.1f}% of {token_tracker.context_limit:,})"
                )

    # Track previous agent for handoff detection
    previous_agent_id: str | None = None

    # Register session event handlers
    @session.on("agent_started")  # type: ignore[arg-type]
    def on_agent_started(agent: Agent) -> None:
        """Called when an agent starts handling the conversation."""
        nonlocal previous_agent_id
        assistant_id = getattr(agent, "assistant_id", "unknown")
        logger.agent(f"Agent started: {assistant_id}")
        session_data.current_assistant_id = assistant_id

        # Track handoffs in metrics and transcript
        if previous_agent_id is not None and previous_agent_id != assistant_id:
            metrics.record_handoff()
            transcript.add_handoff_event(
                from_assistant=previous_agent_id,
                to_assistant=assistant_id,
                reason="Agent handoff",
            )

        previous_agent_id = assistant_id

    @session.on("agent_stopped")  # type: ignore[arg-type]
    def on_agent_stopped(agent: Agent) -> None:
        """Called when an agent stops handling the conversation."""
        assistant_id = getattr(agent, "assistant_id", "unknown")
        logger.agent(f"Agent stopped: {assistant_id}")

    @session.on("user_input_transcribed")
    def on_user_input(event: Any) -> None:
        """Called when user speech is transcribed."""
        # Only capture final transcriptions (not interim results)
        if getattr(event, "is_final", False) and getattr(event, "transcript", ""):
            # Real-time logging for operator visibility
            logger.transcript(f"User: {event.transcript}")
            # Artifact collection
            transcript.add_user_message(event.transcript)
            # Record turn metrics
            metrics.record_turn(
                assistant_id=session_data.current_assistant_id,
                user_speech_duration_ms=None,  # Duration not available in this event
            )

    @session.on("conversation_item_added")
    def on_conversation_item(event: Any) -> None:
        """Called when a conversation item (user or assistant message) is added."""
        item = getattr(event, "item", None)
        if item is None:
            return

        # Only capture assistant messages (user messages handled by user_input_transcribed)
        if getattr(item, "role", None) != "assistant":
            return

        # Extract text content using ChatMessage.text_content property
        # ChatMessage.content is list[str | ImageContent | AudioContent]
        # The text_content property joins all string parts
        content = getattr(item, "text_content", None) or ""

        if content:
            # Real-time logging for operator visibility
            logger.transcript(f"{session_data.current_assistant_id}: {content}")
            # Artifact collection
            transcript.add_assistant_message(
                content=content,
                assistant_id=session_data.current_assistant_id,
            )

    end_reason = "completed"
    artifacts_dir = Path("artifacts") / session_id

    # Define shutdown callback - runs when session ACTUALLY ends (not when start() returns)
    async def save_artifacts_on_shutdown() -> None:
        """Save all artifacts when the session ends."""
        nonlocal end_reason

        # Finalize session in database
        session_repo.finalize(session_id, end_reason)

        # Finalize metrics
        final_metrics = metrics.finalize()

        # Generate session report
        report = generate_session_report(
            session_data=session_data,
            configuration_name=manager.config.metadata.name,
            configuration_version=manager.config.metadata.version,
            entry_assistant=manager.config.orchestration.entry_point,
            end_reason=end_reason,
            transcript=transcript,
            metrics=final_metrics,
        )

        # Save artifacts to disk
        try:
            save_session_artifacts(report, artifacts_dir)
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")

        # Log session summary
        logger.session(
            f"Session {session_id} summary - "
            f"Duration: {session_data.get_session_duration_seconds():.1f}s, "
            f"Handoffs: {session_data.handoff_count}, "
            f"Turns: {final_metrics.turn_count}"
        )

    # Register shutdown callback BEFORE starting session
    # This ensures artifacts are saved when session actually ends, not when start() returns
    ctx.add_shutdown_callback(save_artifacts_on_shutdown)

    # Start the session with the entry agent
    # NOTE: start() returns immediately - session continues in background
    # The shutdown callback will be called when the session actually ends
    try:
        await session.start(
            room=ctx.room,
            agent=entry_agent,
        )
        logger.session(f"Session {session_id} started successfully")
    except Exception as e:
        logger.error(f"Session error: {e}")
        end_reason = "error"
        raise


def main() -> None:
    """Main function to start the worker."""
    # Initialize database (creates tables if not exist)
    init_database()

    # Validate configuration on startup
    try:
        manager = create_orchestration_manager()
        logger.config(
            f"Configuration validated: {manager.config.metadata.name} "
            f"(v{manager.config.metadata.version})"
        )

        # Store globally for reuse
        global _orchestration_manager
        _orchestration_manager = manager

    except SystemExit:
        return

    # Run the worker
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )


if __name__ == "__main__":
    main()
