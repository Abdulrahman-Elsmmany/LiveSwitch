"""Pydantic schemas for multi-assistant configuration validation.

This module defines the complete schema for JSON configuration files that
drive the multi-assistant orchestration system. All configuration is
validated at startup to catch errors early.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HandoffType(str, Enum):
    """Type of handoff mechanism between assistants."""

    TOOL_BASED = "tool_based"
    RULE_BASED = "rule_based"
    STEP_BASED = "step_based"


class ContextTransferMode(str, Enum):
    """How context is transferred during handoffs.

    Currently only FULL mode is implemented, which includes:
    - Narrative reframing (previous assistant messages become third-person)
    - Smart compaction (LLM summarizes when context exceeds threshold)
    - Session memory injection (collected data persists as system message)
    """

    FULL = "full"  # Complete chat history with reframing, compaction, and memory


class ModelConfig(BaseModel):
    """Model configuration with provider/model syntax.

    Example: "openai/gpt-4o", "deepgram/nova-3"
    """

    stt: str = Field(
        default="openai/whisper-1",
        description="STT model identifier (provider/model)",
    )
    llm: str = Field(
        default="openai/gpt-4o",
        description="LLM model identifier (provider/model)",
    )
    tts: str = Field(
        default="openai/tts-1",
        description="TTS model identifier (provider/model)",
    )
    voice: str | None = Field(
        default="alloy",
        description="Voice identifier for TTS",
    )


class ToolParameter(BaseModel):
    """JSON Schema parameter definition for a tool."""

    type: str = Field(..., description="JSON Schema type")
    description: str | None = Field(default=None)
    enum: list[str] | None = Field(default=None)
    items: dict[str, Any] | None = Field(default=None)


class ToolDefinition(BaseModel):
    """Custom tool definition for an assistant.

    Tools allow assistants to perform actions like collecting data,
    making API calls, or triggering business logic.
    """

    name: str = Field(..., description="Tool function name")
    description: str = Field(..., description="Tool description for LLM")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for parameters",
    )
    handler: str | None = Field(
        default=None,
        description="Custom handler module path (e.g., 'src.tools.verify_patient')",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration (read by handler at runtime)",
    )


class HandoffTarget(BaseModel):
    """Definition of a handoff target from one assistant to another.

    Each handoff target specifies which assistant can be transferred to,
    when to trigger the transfer, and how context should be passed.
    """

    assistant_id: str = Field(..., description="Target assistant ID")
    description: str = Field(
        ...,
        description="When to trigger this handoff - used as tool description for LLM",
    )
    context_mode: ContextTransferMode = Field(
        default=ContextTransferMode.FULL,
        description="How to transfer context during handoff",
    )


class AssistantConfig(BaseModel):
    """Configuration for a single assistant.

    An assistant is an autonomous conversational entity with its own
    personality, capabilities, and handoff targets.
    """

    id: str = Field(..., description="Unique assistant identifier")
    name: str = Field(..., description="Human-readable name")
    instructions: str = Field(
        ...,
        description="System prompt defining personality and behavior",
    )
    model_overrides: ModelConfig | None = Field(
        default=None,
        description="Optional model overrides for this assistant",
    )
    tools: list[ToolDefinition] = Field(
        default_factory=list,
        description="Custom tools available to this assistant",
    )
    handoff_targets: list[HandoffTarget] = Field(
        default_factory=list,
        description="Assistants this one can transfer to",
    )
    on_enter_instructions: str | None = Field(
        default=None,
        description="Instructions for initial greeting when becoming active",
    )
    allow_interruptions: bool = Field(
        default=True,
        description="Whether user can interrupt assistant speech",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate assistant ID format.

        IDs must be alphanumeric with underscores/hyphens only.
        """
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Assistant ID '{v}' must be alphanumeric with underscores/hyphens only"
            )
        return v


class HandoffRule(BaseModel):
    """Rule-based handoff condition.

    Used for explicit condition-based routing (not recommended for most cases).
    """

    condition: str = Field(..., description="Python expression to evaluate")
    source: str = Field(..., description="Source assistant ID")
    target: str = Field(..., description="Target assistant ID")
    priority: int = Field(default=0, description="Rule priority (higher = first)")


class OrchestrationConfig(BaseModel):
    """Orchestration configuration defining how assistants interact.

    The entry_point specifies which assistant starts the conversation.
    handoff_type determines how transfers are triggered.
    """

    entry_point: str = Field(..., description="Initial assistant ID")
    handoff_type: HandoffType = Field(
        default=HandoffType.TOOL_BASED,
        description="Handoff mechanism type",
    )
    max_handoffs: int = Field(
        default=10,
        description="Maximum handoffs per session (prevents infinite loops)",
    )
    handoff_rules: list[HandoffRule] = Field(
        default_factory=list,
        description="Rule-based handoff conditions (for rule_based type)",
    )
    fallback_assistant: str | None = Field(
        default=None,
        description="Assistant ID for error recovery",
    )


class SharedContextField(BaseModel):
    """Field definition for shared context across assistants.

    Shared context fields are persisted across handoffs and can be
    accessed by all assistants via session.userdata.
    """

    name: str = Field(..., description="Field name")
    type: str = Field(
        ...,
        description="Python type name (str, int, float, bool, list, dict)",
    )
    required: bool = Field(default=False, description="Whether field is required")
    default: str | None = Field(
        default=None,
        description="Default value as string (will be parsed based on type)",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate field type is supported."""
        allowed_types = {"str", "int", "float", "bool", "list", "dict"}
        if v not in allowed_types:
            raise ValueError(f"Type must be one of: {allowed_types}")
        return v


class SharedContextConfig(BaseModel):
    """Shared context schema definition.

    Defines the structure of data that persists across all assistants
    during a session.
    """

    fields: list[SharedContextField] = Field(
        default_factory=list,
        description="Fields in the shared context",
    )


class GlobalSettings(BaseModel):
    """Global worker settings.

    These settings apply to the entire worker and can be overridden
    at the assistant level.
    """

    default_models: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Default model configuration",
    )
    session_timeout: int = Field(
        default=1800,
        description="Session timeout in seconds (30 minutes)",
    )
    min_endpointing_delay: float = Field(
        default=0.5,
        description="Minimum delay before considering speech complete",
    )
    max_endpointing_delay: float = Field(
        default=6.0,
        description="Maximum delay before forcing speech complete",
    )
    enable_transcription: bool = Field(
        default=True,
        description="Enable transcript logging",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection",
    )


class ConfigMetadata(BaseModel):
    """Configuration metadata.

    Describes the configuration file for documentation and versioning.
    """

    name: str = Field(..., description="Configuration name")
    description: str | None = Field(
        default=None,
        description="Configuration description",
    )
    version: str = Field(default="1.0.0", description="Configuration version")
    author: str | None = Field(default=None, description="Configuration author")


class MultiAssistantConfig(BaseModel):
    """Root configuration model for multi-assistant system.

    This is the top-level schema that validates the entire JSON configuration.
    All orchestration logic is derived from this configuration at runtime.
    """

    metadata: ConfigMetadata = Field(..., description="Configuration metadata")
    global_settings: GlobalSettings = Field(
        default_factory=GlobalSettings,
        description="Global worker settings",
    )
    assistants: list[AssistantConfig] = Field(
        ...,
        min_length=1,
        description="List of assistant configurations",
    )
    orchestration: OrchestrationConfig = Field(
        ...,
        description="Orchestration configuration",
    )
    shared_context: SharedContextConfig = Field(
        default_factory=SharedContextConfig,
        description="Shared context schema",
    )

    @field_validator("assistants")
    @classmethod
    def validate_unique_ids(cls, v: list[AssistantConfig]) -> list[AssistantConfig]:
        """Validate that all assistant IDs are unique."""
        ids = [a.id for a in v]
        if len(ids) != len(set(ids)):
            duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
            raise ValueError(f"Duplicate assistant IDs found: {set(duplicates)}")
        return v

    def get_assistant_by_id(self, assistant_id: str) -> AssistantConfig | None:
        """Get assistant configuration by ID.

        Args:
            assistant_id: The assistant ID to look up.

        Returns:
            AssistantConfig if found, None otherwise.
        """
        for assistant in self.assistants:
            if assistant.id == assistant_id:
                return assistant
        return None

    def get_all_assistant_ids(self) -> set[str]:
        """Get set of all assistant IDs.

        Returns:
            Set of assistant ID strings.
        """
        return {a.id for a in self.assistants}
