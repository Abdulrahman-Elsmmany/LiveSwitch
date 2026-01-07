"""Agent factory for dynamic agent class generation.

This module is the heart of the configuration-driven system. It dynamically
generates Agent subclasses at runtime based on JSON configuration.

NO HARDCODED ORCHESTRATION LOGIC - everything comes from configuration.
"""

from typing import Any

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import ChatContext, ChatMessage

from src.agents.tools import TOOL_REGISTRY
from src.config.schemas import (
    AssistantConfig,
    ContextTransferMode,
    HandoffTarget,
    MultiAssistantConfig,
    ToolDefinition,
)
from src.context.compaction import smart_compact_context
from src.context.memory import format_session_memory
from src.context.session import SessionData
from src.context.token_tracker import get_token_tracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_text_content(item: Any) -> str:
    """Extract text content from a chat message item.

    Args:
        item: A chat message item with content attribute.

    Returns:
        Extracted text content as a string.
    """
    if not hasattr(item, "content"):
        return ""
    content = item.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return " ".join(str(c) for c in content if isinstance(c, str))
    return str(content) if content else ""


def _transform_context_for_handoff(
    chat_ctx: ChatContext,
    source_assistant_name: str,
    target_assistant_name: str,
) -> ChatContext:
    """Transform chat context using narrative reframing for handoffs.

    Implements the Google ADK pattern: previous assistant messages are re-cast
    as narrative context rather than appearing as the new agent's outputs.
    This prevents identity confusion where the new agent thinks it said things
    from the previous conversation.

    Key insight from Google ADK:
    > "Prior assistant messages are re-cast as narrative context rather than
    >  appearing as the new agent's outputs"

    Args:
        chat_ctx: Original chat context.
        source_assistant_name: Name of the agent being transferred FROM.
        target_assistant_name: Name of the agent being transferred TO.

    Returns:
        New ChatContext with narrative-reframed messages.
    """
    new_ctx = ChatContext()

    # Collect conversation segments for narrative summary
    conversation_segments: list[str] = []
    user_messages_to_keep: list[Any] = []

    for item in chat_ctx.items:
        if not hasattr(item, "role"):
            continue

        if item.role == "user":
            # Keep user messages to preserve conversation flow
            user_messages_to_keep.append(item)
            content = _extract_text_content(item)
            if content:
                conversation_segments.append(f"Patient: {content}")

        elif item.role == "assistant":
            # Convert assistant messages to narrative (NOT kept as assistant role)
            # This is the key fix: previous agent's words become context, not "my words"
            content = _extract_text_content(item)
            if content:
                conversation_segments.append(f"{source_assistant_name}: {content}")

        # Skip system messages - they will be replaced with new agent's instructions

    # Build narrative context as a SINGLE system message at the start
    # Uses STRONG "DO NOT SPEAK" framing to prevent the LLM from echoing/summarizing
    if conversation_segments:
        narrative = (
            f"=== INTERNAL CONTEXT (DO NOT SPEAK ANY OF THIS) ===\n"
            f"You are {target_assistant_name}. The patient was just transferred to you.\n\n"
            f"CRITICAL RULES:\n"
            f"1. DO NOT summarize, repeat, or acknowledge what is written below\n"
            f"2. DO NOT say 'I will transfer you to...' - YOU are the agent they transferred TO\n"
            f"3. DO NOT echo or paraphrase the previous agent's words\n"
            f"4. Start fresh with your own greeting as {target_assistant_name}\n"
            f"5. The previous conversation below is REFERENCE ONLY - never speak it\n\n"
            f"Previous conversation (SILENT REFERENCE):\n"
            + "\n".join(conversation_segments)
            + f"\n\n=== END INTERNAL CONTEXT ===\n\n"
            f"Now respond as {target_assistant_name}. Use your on_enter greeting."
        )

        # Add narrative as system message
        narrative_msg = ChatMessage(role="system", content=[narrative])
        new_ctx.items.append(narrative_msg)

    # Add user messages back (preserves the actual conversation flow for the LLM)
    # These are kept as role="user" so the LLM sees the patient's actual words
    for user_msg in user_messages_to_keep:
        new_ctx.insert(user_msg)

    return new_ctx


class AgentFactory:
    """Factory for creating dynamic Agent classes from configuration.

    This factory reads the configuration and generates Agent subclasses
    at runtime. Each assistant defined in config becomes a unique Agent class
    with its own instructions, tools, and handoff capabilities.

    Attributes:
        config: The multi-assistant configuration.
        agent_registry: Dictionary mapping assistant IDs to Agent classes.
    """

    def __init__(self, config: MultiAssistantConfig) -> None:
        """Initialize the agent factory.

        Args:
            config: Validated multi-assistant configuration.
        """
        self.config = config
        self.agent_registry: dict[str, type[Agent]] = {}
        self._build_agents()

    def _build_agents(self) -> None:
        """Build all agent classes from configuration.

        This method iterates through all assistants in the configuration
        and creates a dynamic Agent class for each one.
        """
        logger.config(f"Building {len(self.config.assistants)} agent classes")

        for assistant in self.config.assistants:
            agent_class = self._create_agent_class(assistant)
            self.agent_registry[assistant.id] = agent_class
            logger.debug(f"Created agent class for '{assistant.id}'")

        # Second pass: add tools now that all agents are registered
        for assistant in self.config.assistants:
            agent_class = self.agent_registry[assistant.id]

            # Add custom tools (verify_patient, record_symptoms, etc.)
            if assistant.tools:
                for tool_config in assistant.tools:
                    self._add_custom_tool(agent_class, tool_config)

            # Add handoff tools
            for target in assistant.handoff_targets:
                self._add_handoff_tool(agent_class, target, assistant.id)

        logger.config(f"Agent registry built with {len(self.agent_registry)} agents")

    def _create_agent_class(self, assistant_config: AssistantConfig) -> type[Agent]:
        """Create a dynamic Agent class from configuration.

        Args:
            assistant_config: Configuration for this assistant.

        Returns:
            A new Agent subclass with configured behavior.
        """
        config = assistant_config  # Capture for closure

        class DynamicAgent(Agent):
            """Dynamically generated agent from configuration."""

            def __init__(
                self,
                chat_ctx: ChatContext | None = None,
            ) -> None:
                """Initialize the dynamic agent.

                Args:
                    chat_ctx: Optional chat context from previous assistant.
                """
                super().__init__(
                    instructions=config.instructions,
                    chat_ctx=chat_ctx,
                )

                # Store assistant metadata
                self.assistant_id = config.id
                self.assistant_name = config.name
                self.on_enter_instructions = config.on_enter_instructions

            async def on_enter(self) -> None:
                """Called when this agent becomes active.

                If on_enter_instructions is configured, generates an
                initial greeting using those instructions.
                """
                if self.on_enter_instructions and hasattr(self, "session"):
                    logger.agent(f"Agent '{self.assistant_id}' entering conversation")
                    await self.session.generate_reply(
                        instructions=self.on_enter_instructions
                    )

        # Set a meaningful class name for debugging
        DynamicAgent.__name__ = f"{config.id.title().replace('_', '')}Agent"
        DynamicAgent.__qualname__ = DynamicAgent.__name__

        return DynamicAgent

    def _add_handoff_tool(
        self,
        agent_class: type[Agent],
        target: HandoffTarget,
        source_id: str,
    ) -> None:
        """Add a handoff tool to an agent class.

        This creates a function_tool that, when called by the LLM,
        transfers control to another assistant.

        Args:
            agent_class: The agent class to add the tool to.
            target: The handoff target configuration.
            source_id: ID of the source assistant.
        """
        factory = self
        target_id = target.assistant_id
        tool_name = f"transfer_to_{target_id}"

        # Create the handoff tool function
        # Note: raw_arguments is required when using raw_schema with function_tool
        async def handoff_tool_fn(
            self: Agent,
            context: RunContext[SessionData],
            raw_arguments: dict[str, Any],
        ) -> tuple[Agent, str]:
            """Transfer the conversation to another assistant."""
            logger.handoff(f"Handoff: {source_id} -> {target_id}")

            # Get the target agent class
            target_class = factory.agent_registry.get(target_id)
            if target_class is None:
                logger.error(f"Target agent '{target_id}' not found in registry")
                raise ValueError(f"Unknown handoff target: {target_id}")

            # Check handoff limit
            max_handoffs = factory.config.orchestration.max_handoffs
            if context.userdata.handoff_count >= max_handoffs:
                logger.warning(f"Max handoffs ({max_handoffs}) reached")
                return self, f"Cannot transfer - maximum handoffs ({max_handoffs}) reached"

            # Record the handoff
            context.userdata.record_handoff(
                from_assistant=source_id,
                to_assistant=target_id,
                reason=target.description,
            )

            # Create new agent instance with appropriate context
            chat_ctx = None
            if target.context_mode == ContextTransferMode.FULL:
                if hasattr(context.session, "_chat_ctx"):
                    original_ctx = context.session._chat_ctx

                    # Get source and target agent names for narrative reframing
                    source_config = factory.config.get_assistant_by_id(source_id)
                    source_name = source_config.name if source_config else source_id
                    target_config = factory.config.get_assistant_by_id(target_id)
                    target_name = target_config.name if target_config else target_id

                    # Layer 1: Transform context using narrative reframing (Google ADK pattern)
                    # Previous assistant messages become narrative context, not "my words"
                    chat_ctx = _transform_context_for_handoff(
                        original_ctx, source_name, target_name
                    )

                    # Layer 2: Smart compact if conversation is getting long (token-based)
                    # Uses LLM summarization (Session Memory aware) with fallback to truncation
                    llm = getattr(context.session, "llm", None)
                    chat_ctx = await smart_compact_context(
                        chat_ctx, llm, context.userdata
                    )

                    # Layer 3: INJECT SESSION MEMORY (survives any compaction!)
                    # This is the persistent data from SessionData that tools have collected
                    memory_context = format_session_memory(context.userdata)
                    if memory_context:
                        # Prepend memory as first system message
                        # ChatMessage.content expects a list[ChatContent]
                        memory_msg = ChatMessage(role="system", content=[memory_context])
                        chat_ctx.items.insert(0, memory_msg)
                        logger.memory(
                            f"Injected session memory for {target_id} "
                            f"(patient: {context.userdata.get_data('patient_name', 'unknown')})"
                        )

                    # Log handoff context stats
                    token_tracker = get_token_tracker()
                    logger.handoff(
                        f"Context prepared for {target_id}: "
                        f"{len(chat_ctx.items)} items, "
                        f"last_tokens={token_tracker.last_prompt_tokens:,}, "
                        f"memory_injected={bool(memory_context)}"
                    )

            new_agent = target_class(chat_ctx=chat_ctx)  # type: ignore[call-arg]

            return new_agent, f"Transferring to {target_id}..."

        # Apply the function_tool decorator with explicit schema
        # (tools with no LLM-visible parameters need raw_schema to avoid invalid JSON schema)
        decorated_tool = function_tool(
            raw_schema={
                "name": tool_name,
                "description": f"Transfer the conversation to {target_id}. {target.description}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        )(handoff_tool_fn)

        # Set the docstring for the LLM
        decorated_tool.__doc__ = (
            f"Transfer the conversation to {target_id}. {target.description}"
        )

        # Add the tool to the agent class
        setattr(agent_class, tool_name, decorated_tool)

    def _add_custom_tool(
        self,
        agent_class: type[Agent],
        tool_config: ToolDefinition,
    ) -> None:
        """Add a custom tool to an agent class.

        This looks up the tool handler from TOOL_REGISTRY and creates
        a function_tool wrapper that integrates with LiveKit agents.

        Args:
            agent_class: The agent class to add the tool to.
            tool_config: The tool configuration from JSON.
        """
        handler = TOOL_REGISTRY.get(tool_config.name)
        if handler is None:
            logger.warning(
                f"No handler found for tool '{tool_config.name}' - "
                "tool will not be available"
            )
            return

        tool_name = tool_config.name
        # Capture tool-specific config for the closure
        tool_runtime_config = tool_config.config

        # Create a wrapper that extracts userdata and calls the handler
        async def tool_wrapper(
            self: Agent,
            context: RunContext[SessionData],
            raw_arguments: dict[str, Any],
        ) -> str:
            """Wrapper that invokes the actual tool handler."""
            logger.tool(f"Executing tool: {tool_name}")

            try:
                # Call the handler with userdata, tool config, and arguments
                result = await handler(
                    userdata=context.userdata,
                    tool_config=tool_runtime_config,
                    **raw_arguments,
                )

                # Return the message from the result for LLM context
                if isinstance(result, dict):
                    return result.get("message", str(result))  # type: ignore[no-any-return]
                return str(result)

            except Exception as e:
                logger.error(f"Tool '{tool_name}' error: {e}")
                return f"Tool error: {e}"

        # Build the schema for function_tool
        schema = {
            "name": tool_name,
            "description": tool_config.description,
            "parameters": tool_config.parameters,
        }

        # Apply the function_tool decorator
        decorated_tool = function_tool(raw_schema=schema)(tool_wrapper)
        decorated_tool.__doc__ = tool_config.description

        # Add to agent class
        setattr(agent_class, tool_name, decorated_tool)
        logger.debug(f"Added tool '{tool_name}' to {agent_class.__name__}")

    def get_agent_class(self, assistant_id: str) -> type[Agent] | None:
        """Get an agent class by assistant ID.

        Args:
            assistant_id: The assistant ID to look up.

        Returns:
            The Agent class if found, None otherwise.
        """
        return self.agent_registry.get(assistant_id)

    def get_entry_agent(self, chat_ctx: ChatContext | None = None) -> Agent:
        """Get an instance of the entry point agent.

        Args:
            chat_ctx: Optional initial chat context.

        Returns:
            An instance of the entry point agent.

        Raises:
            ValueError: If entry point agent not found.
        """
        entry_id = self.config.orchestration.entry_point
        agent_class = self.agent_registry.get(entry_id)

        if agent_class is None:
            raise ValueError(f"Entry point agent '{entry_id}' not found in registry")

        return agent_class(chat_ctx=chat_ctx)  # type: ignore[call-arg]

    def get_fallback_agent(self, chat_ctx: ChatContext | None = None) -> Agent | None:
        """Get an instance of the fallback agent if configured.

        Args:
            chat_ctx: Optional chat context.

        Returns:
            An instance of the fallback agent, or None if not configured.
        """
        fallback_id = self.config.orchestration.fallback_assistant
        if fallback_id is None:
            return None

        agent_class = self.agent_registry.get(fallback_id)
        if agent_class is None:
            logger.warning(f"Fallback agent '{fallback_id}' not found in registry")
            return None

        return agent_class(chat_ctx=chat_ctx)  # type: ignore[call-arg]

    def list_assistants(self) -> list[str]:
        """List all registered assistant IDs.

        Returns:
            List of assistant ID strings.
        """
        return list(self.agent_registry.keys())
