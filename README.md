# LiveSwitch

> **Dynamic Multi-Agent Voice Orchestration Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LiveKit Agents](https://img.shields.io/badge/LiveKit-Agents%201.3-green.svg)](https://docs.livekit.io/agents/)

LiveSwitch is a **configuration-driven multi-assistant conversation platform** built on the LiveKit Agents Framework. It enables seamless, real-time switching between specialized AI agents within a single voice call - all orchestrated from a simple JSON configuration file with zero hardcoded logic.

## Quick Start

```bash
# Create virtual environment
uv venv --python 3.13
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv sync

# Copy environment variables
cp .env.example .env.local
# Edit .env.local with your LiveKit credentials and free API keys:
#   - OpenRouter (LLM): https://openrouter.ai/settings/keys (google/gemini-3-flash-preview)
#   - Deepgram (STT): https://console.deepgram.com ($200 free credits)
#   - Cartesia (TTS): https://play.cartesia.ai (10,000 free credits)

# Download model files (Silero VAD)
uv run python -m src.main download-files

# Run in development mode
uv run python -m src.main dev

# Connect via LiveKit Playground: https://agents-playground.livekit.io
```

## Why LiveSwitch?

| Challenge | LiveSwitch Solution |
|-----------|---------------------|
| Monolithic assistants become unpredictable | Specialized agents with focused responsibilities |
| Context degrades in long conversations | Smart compaction with LLM summarization |
| Hardcoded routing is inflexible | 100% configuration-driven orchestration |
| Lost context during handoffs | Three-layer preservation: reframing + compaction + memory |
| No cross-call memory | SQLite persistence recognizes returning patients |

## Key Features

- **Zero Hardcoding** - All orchestration logic comes from JSON configuration
- **Dynamic Agent Generation** - Agents created at runtime, no code changes needed
- **Intelligent Handoffs** - LLM-decided transfers using LiveKit's recommended pattern
- **Context Preservation** - Narrative reframing + smart compaction + session memory
- **Cross-Call Memory** - SQLite V2 schema with UPSERT support for patient recognition
- **Production Observability** - Real-time transcripts, metrics, and session reports

---

## Analysis

This section provides the required analysis of multi-assistant conversation systems, including definitions, runtime behavior, and implementation patterns.

### What is a Multi-Assistant Configuration?

A **Multi-Assistant Configuration** is a declarative JSON document that defines the complete structure and behavior of a conversational system with multiple specialized agents. It contains:

1. **Assistant Registry**: All available assistants with their unique identifiers, names, system prompts, and capabilities
2. **Orchestration Rules**: How and when control transfers between assistants (entry point, max handoffs, fallback)
3. **Handoff Definitions**: Which assistants can transfer to which other assistants, with descriptions of trigger conditions
4. **Shared Context Schema**: Data structures that persist across handoffs (patient info, collected data, flags)
5. **Global Settings**: Default models (STT/LLM/TTS), timeouts, and feature flags

```json
{
  "metadata": { "name": "Medical Triage", "version": "1.0.0" },
  "global_settings": {
    "default_models": { "llm": "google/gemini-3-flash-preview", "stt": "deepgram/nova-2", "tts": "cartesia/sonic-2" }
  },
  "assistants": [ /* Assistant definitions */ ],
  "orchestration": { "entry_point": "receptionist", "handoff_type": "tool_based" },
  "shared_context": { "fields": [ /* Typed field definitions */ ] }
}
```

The configuration is validated in two layers:
- **Structural Validation** (Pydantic): Type checking, required fields, enum values
- **Semantic Validation** (Custom): Entry point exists, handoff targets valid, no orphaned assistants

### What is an Assistant?

An **Assistant** (in LiveKit terms: an `Agent`) is an autonomous conversational entity with focused responsibilities. Each assistant is defined by:

| Property                | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `id`                    | Unique identifier used in handoff references (e.g., `"nurse_triage"`)        |
| `name`                  | Human-readable name (e.g., `"Triage Nurse"`)                                 |
| `instructions`          | System prompt defining personality, role, and behavior boundaries            |
| `tools`                 | Callable functions for data collection and actions (e.g., `record_symptoms`) |
| `handoff_targets`       | List of assistants this one can transfer to, with trigger descriptions       |
| `model_overrides`       | Optional STT/LLM/TTS/voice customizations per assistant                      |
| `on_enter_instructions` | Greeting or introduction when this assistant becomes active                  |

In this implementation, assistants are **dynamically generated at runtime** as Python `Agent` subclasses. The `AgentFactory` reads the configuration and creates classes with:
- Injected system prompts
- Dynamically generated `@function_tool` decorated handoff methods
- Proper model configurations

### What is a Handoff?

A **Handoff** is the controlled transfer of conversational control from one assistant to another within the same call session. Key characteristics:

**Trigger Mechanism** (Tool-Based):
The LLM decides when to hand off by calling a generated tool function. Each handoff target becomes a tool like `transfer_to_nurse_triage()`. The LLM uses the tool's description to determine when the handoff is appropriate.

```python
@function_tool()
async def transfer_to_nurse_triage(self, context: RunContext[SessionData]):
    """Transfer to nurse triage when patient reports symptoms, feels unwell,
    has a medical concern, or needs clinical advice."""
    target_class = self.agent_registry["nurse_triage"]
    return target_class(chat_ctx=context.session._chat_ctx), "Transferring to nurse..."
```

**Context Transfer** (automatic on handoff):
1. **Narrative Reframing**: Previous assistant's messages become third-person ("The nurse said...") to prevent identity confusion
2. **Smart Compaction**: When context exceeds threshold (default 40%), LLM summarizes older messages while preserving recent exchanges
3. **Session Memory Injection**: Collected data (patient info, symptoms, appointments) injected as system message - survives any compaction

**Handoff Metadata** (automatically tracked):
- `from_assistant`: Previous assistant ID
- `to_assistant`: New assistant ID
- `reason`: Description of why transfer occurred
- `timestamp`: When the handoff happened
- `context_mode`: How context was transferred

**Handoff Limits**:
The `max_handoffs` setting prevents infinite loops. Once reached, the system stops accepting handoff requests.

### What is Full Conversation Context?

**Full Conversation Context** encompasses all data that must persist across handoffs to maintain conversational continuity:

| Component          | Description                                        | Implementation                                |
| ------------------ | -------------------------------------------------- | --------------------------------------------- |
| **Chat History**   | Complete transcript of user and assistant messages | `ChatContext` passed via `chat_ctx` parameter |
| **Session State**  | Typed structured data (patient info, flags)        | `AgentSession[T].userdata` dataclass          |
| **Collected Data** | Form fields, preferences, extracted info           | Stored in `SessionData.collected_data` dict   |
| **Handoff Trail**  | History of agent transitions                       | `SessionData.handoff_history` list            |
| **Call Metadata**  | Duration, participant info, timestamps             | `SessionData` properties                      |

The context preservation mechanism:
1. When a handoff tool is called, it receives the current `RunContext`
2. The context contains `session._chat_ctx` with full conversation history
3. The new agent is instantiated with `chat_ctx=context.session._chat_ctx`
4. The userdata dataclass automatically persists across agents

### Runtime Behavior

#### 1. Configuration Loading and Validation

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│ Load JSON   │───►│   Pydantic  │───►│  Semantic   │───►│ Generate │
│ Config File │    │ Validation  │    │ Validation  │    │ Agents   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
• File exists?     • Schema valid?    • Entry point      • Create Agent
• Valid JSON?      • Types correct?     exists?            subclasses
• UTF-8 encoded?   • Required fields? • Handoff targets  • Inject tools
                                        valid?           • Build registry
```

**Loading Steps**:
1. `load_config(path)` reads the JSON file
2. Pydantic parses and validates against `MultiAssistantConfig` schema
3. `validate_config()` performs semantic checks (entry point exists, targets valid)
4. `AgentFactory(config)` generates all agent classes

**Error Handling**:
- `ConfigFileNotFoundError`: File doesn't exist
- `ConfigParseError`: Invalid JSON syntax
- `ConfigValidationError`: Pydantic schema violations
- `SemanticValidationError`: Business logic violations (missing entry point, invalid target)

#### 2. Active Assistant Determination

The system maintains a **single active assistant** at any time:

```python
class OrchestrationState:
    current_assistant_id: str      # Currently active assistant
    previous_assistant_ids: list   # Trail of previous assistants
    handoff_count: int             # Number of handoffs so far
```

**Determination Flow**:
1. **Session Start**: Use `orchestration.entry_point` from config (e.g., `"receptionist"`)
2. **On Handoff**: New assistant from tool return becomes active
3. **On Error**: Route to `fallback_assistant` if configured

The `HandoffCoordinator` tracks state and validates transitions:
```python
coordinator.validate_handoff_target("receptionist", "nurse_triage")  # Returns (True, None)
coordinator.validate_handoff_target("receptionist", "cardiology")     # Returns (False, "Cannot hand off...")
```

#### 3. Dynamic Orchestration Generation

All orchestration logic is generated at runtime by the `AgentFactory`:

```python
class AgentFactory:
    def __init__(self, config: MultiAssistantConfig):
        self.config = config
        self.agent_registry: dict[str, type[Agent]] = {}
        self._build_agents()  # Generate all agent classes

    def _create_agent_class(self, config: AssistantConfig) -> type[Agent]:
        # Create a new Agent subclass dynamically
        class DynamicAgent(Agent):
            def __init__(self, chat_ctx=None):
                super().__init__(
                    instructions=config.instructions,
                    llm=model_config.llm,
                    chat_ctx=chat_ctx,
                )
                self.assistant_id = config.id

        # Inject handoff tools for each target
        for target in config.handoff_targets:
            self._add_handoff_tool(DynamicAgent, target)

        return DynamicAgent
```

**Key Principle**: No `if assistant_id == "billing"` anywhere in the code. All routing decisions come from configuration.

#### 4. Handoff Mechanism

When the LLM decides to hand off:

1. LLM calls the handoff tool (e.g., `transfer_to_nurse_triage`)
2. Tool function creates new agent instance with conversation context
3. Returns `(new_agent, transition_message)` tuple
4. LiveKit framework switches active agent
5. New agent's `on_enter` triggers greeting

```python
@function_tool()
async def transfer_to_nurse_triage(self, context: RunContext[SessionData]):
    """Transfer to nurse triage when patient reports symptoms..."""
    # Get the target agent class from registry
    target_class = self.agent_registry["nurse_triage"]

    # Preserve full conversation context
    new_agent = target_class(chat_ctx=context.session._chat_ctx)

    # Return agent and transition message
    return new_agent, "Let me transfer you to our triage nurse..."
```

#### 5. Data Persistence Across Handoffs

| Data Type       | Storage                       | How It Persists                    |
| --------------- | ----------------------------- | ---------------------------------- |
| Chat messages   | `ChatContext`                 | Passed via `chat_ctx` to new agent |
| Structured data | `SessionData.collected_data`  | UserData dataclass on session      |
| Handoff history | `SessionData.handoff_history` | Appended on each transition        |
| Flags/state     | `SessionData` fields          | UserData dataclass persists        |

**Session Data Structure**:
```python
@dataclass
class SessionData:
    session_id: str
    current_assistant_id: str
    handoff_count: int
    handoff_history: list[HandoffRecord]
    collected_data: dict[str, Any]  # patient_name, chief_complaint, etc.
    start_time: datetime
```

#### 6. Error Handling

| Error Type             | Handling                                    |
| ---------------------- | ------------------------------------------- |
| Config file not found  | Exit with clear error message and path      |
| Invalid JSON           | Exit with parsing error and line number     |
| Schema violation       | Exit with Pydantic validation details       |
| Missing entry point    | Exit with semantic validation error         |
| Invalid handoff target | Block handoff, log error, stay with current |
| Max handoffs reached   | Block further handoffs, log warning         |
| LLM API failure        | Retry with exponential backoff              |
| Session disconnect     | Generate session report, cleanup resources  |

**Fallback Mechanism**:
```python
# If configured, route to fallback on errors
if config.orchestration.fallback_assistant:
    fallback_agent = factory.get_agent_class(fallback_id)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LIVEKIT AGENT WORKER                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      CONFIGURATION LAYER                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │   config/    │  │   Pydantic   │  │  Semantic    │                 │  │
│  │  │   *.json     │  │   Schemas    │  │  Validator   │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      ORCHESTRATION LAYER                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │    Agent     │  │   Handoff    │  │   Session    │                 │  │
│  │  │   Factory    │  │ Coordinator  │  │    Data      │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        LIVEKIT LAYER                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │ AgentSession │  │   Dynamic    │  │   Plugins    │                 │  │
│  │  │   Manager    │  │   Agents     │  │ (STT/LLM/TTS)│                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      ARTIFACTS LAYER                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │  Transcript  │  │   Metrics    │  │   Session    │                 │  │
│  │  │   Logger     │  │   Collector  │  │   Report     │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Medical Triage Flow (Sample Configuration)

The included `config/medical_triage.json` implements a 4-assistant medical office:

```
                                    ┌─────────────────┐
                                    │   INCOMING      │
                                    │     CALL        │
                                    └────────┬────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │        RECEPTIONIST          │
                              │  • Greeting & Verification   │
                              │  • Reason for Call           │
                              └──────────────┬───────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
         ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
         │  NURSE TRIAGE    │    │    SCHEDULING    │    │    PHARMACY      │
         │ • Symptom Assess │    │ • Book Appts     │    │ • Refills        │
         │ • Urgency Level  │    │ • Reschedule     │    │ • Medication Qs  │
         │ • Red Flag Check │    │ • Cancel         │    │                  │
         └──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Assistants**:
1. **Receptionist** (entry point): Greeting, patient verification, initial routing
2. **Nurse Triage**: Symptom assessment, urgency determination, red flag screening
3. **Scheduling**: Appointment booking, availability checking
4. **Pharmacy**: Prescription refills, medication inquiries

---

## Project Structure

```
hams-assignment/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point, CLI, worker setup
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic models for configuration
│   │   ├── loader.py              # JSON loading and validation
│   │   └── validator.py           # Semantic validation rules
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── factory.py             # Dynamic agent class generation
│   │   ├── manager.py             # Orchestration state management
│   │   └── handoff.py             # Handoff coordination and validation
│   ├── agents/
│   │   ├── __init__.py
│   │   └── tools.py               # Tool definitions and TOOL_REGISTRY
│   ├── context/
│   │   ├── __init__.py
│   │   ├── session.py             # Session data management
│   │   ├── memory.py              # Session memory formatting
│   │   ├── history.py             # Chat history utilities
│   │   ├── compaction.py          # Smart context compaction
│   │   └── token_tracker.py       # Token usage tracking
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── database.py            # SQLite schema and connections (V2)
│   │   └── repository.py          # CRUD operations with UPSERT support
│   ├── artifacts/
│   │   ├── __init__.py
│   │   ├── transcript.py          # Transcript generation
│   │   ├── metrics.py             # Metrics collection
│   │   └── report.py              # Session report generation
│   └── utils/
│       └── logger.py              # Colored logging system
├── config/
│   └── medical_triage.json        # Sample 4-assistant configuration
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Test fixtures
│   ├── test_config.py             # Configuration validation tests
│   ├── test_factory.py            # Agent factory tests
│   ├── test_tools.py              # Tool handler tests
│   ├── test_integration.py        # Integration tests
│   ├── test_conversation_flows.py # Conversation flow tests
│   ├── test_error_scenarios.py    # Error handling tests
│   ├── test_compaction.py         # Context compaction tests
│   └── test_persistence.py        # Database persistence tests
├── artifacts/                     # Session artifacts (transcript, metrics, report)
├── data/                          # SQLite database (sessions.db)
├── .env.example                   # Example environment variables
├── pyproject.toml                 # Project configuration
├── CLAUDE.md                      # Development guidelines
├── PRD.md                         # Product Requirements Document
├── TASKS.md                       # Task tracking
└── README.md                      # This file
```

---

## Technology Stack

| Component                | Technology                             | Cost                    |
| ------------------------ | -------------------------------------- | ----------------------- |
| Runtime                  | Python 3.13+                           | Free                    |
| Agent Framework          | LiveKit Agents 1.3.x                   | Free                    |
| LLM                      | OpenRouter (google/gemini-3-flash-preview) | Varies (free tier avail)|
| Speech-to-Text           | Deepgram Nova-2                        | $200 free credits       |
| Text-to-Speech           | Cartesia Sonic                         | 10,000 free credits     |
| Voice Activity Detection | Silero VAD                             | Free (local)            |
| Persistence              | SQLite (V2 schema)                     | Free                    |
| Configuration Validation | Pydantic 2.x                           | Free                    |
| Package Management       | uv                                     | Free                    |
| Testing                  | pytest, pytest-asyncio                 | Free                    |
| Linting                  | ruff                                   | Free                    |
| Type Checking            | mypy                                   | Free                    |

---

## Development Commands

```bash
# Install dependencies
uv sync

# Run in development mode (connects to LiveKit Cloud)
uv run python -m src.main dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Format code
uv run ruff format src/
```

---

## Environment Variables

Create a `.env.local` file with the following variables. See `.env.example` for a complete template with all options documented.

```bash
# =============================================================================
# Required: LiveKit Configuration
# =============================================================================
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
LIVEKIT_URL=wss://your-project.livekit.cloud

# =============================================================================
# Required: AI Providers
# =============================================================================
OPENROUTER_API_KEY=your_openrouter_key  # https://openrouter.ai/settings/keys
DEEPGRAM_API_KEY=your_deepgram_key      # https://console.deepgram.com ($200 free)
CARTESIA_API_KEY=your_cartesia_key      # https://play.cartesia.ai (10K free)

# =============================================================================
# OpenRouter LLM Configuration
# =============================================================================
OPENROUTER_MODEL=google/gemini-3-flash-preview
OPENROUTER_PROVIDER_IGNORE=Venice,ModelRun     # Problematic providers to exclude
OPENROUTER_PROVIDER_SORT=latency               # Sort by: price, throughput, latency
OPENROUTER_ALLOW_FALLBACKS=true
OPENROUTER_REQUIRE_PARAMETERS=true             # Only providers supporting tools

# =============================================================================
# Context Management
# =============================================================================
CONTEXT_COMPACTION_THRESHOLD=0.40  # Compact when context exceeds 40%
CONTEXT_COMPACT_TARGET=0.20        # Target 20% after compaction

# =============================================================================
# Database Configuration
# =============================================================================
DATABASE_PATH=data/sessions.db     # SQLite database for cross-call memory

# =============================================================================
# Optional
# =============================================================================
CONFIG_FILE_PATH=config/medical_triage.json
# GROQ_API_KEY=your_groq_key       # For future use
```

---

## Artifacts (Call Output)

When a call ends, the system saves artifacts to the `artifacts/` directory:

```
artifacts/{session_id}/
├── {session_id}_report.json      # Full session report
├── {session_id}_transcript.json  # Conversation transcript
└── {session_id}_metrics.json     # Performance metrics
```

### Artifact Contents

**Session Report** (`_report.json`):
```json
{
  "session_id": "uuid-string",
  "start_time": "2025-01-06T10:30:00Z",
  "end_time": "2025-01-06T10:35:00Z",
  "duration_seconds": 300.5,
  "end_reason": "completed",
  "configuration": { "name": "Medical Triage", "version": "2.0.0" },
  "orchestration": {
    "entry_assistant": "receptionist",
    "final_assistant": "scheduling",
    "handoff_count": 2,
    "handoff_history": [...]
  },
  "collected_data": { "patient_name": "John Doe", "chief_complaint": "headache" },
  "transcript": [...],
  "metrics": {...}
}
```

**Transcript** (`_transcript.json`):
```json
{
  "session_id": "uuid-string",
  "entries": [
    { "timestamp": "...", "role": "assistant", "content": "Hello! How can I help?", "assistant_id": "receptionist" },
    { "timestamp": "...", "role": "user", "content": "I have a headache" },
    { "timestamp": "...", "role": "system", "content": "Handoff to nurse_triage", "metadata": { "event": "handoff" } }
  ]
}
```

**Metrics** (`_metrics.json`):
```json
{
  "session_id": "uuid-string",
  "duration_seconds": 300.5,
  "turn_count": 15,
  "handoff_count": 2,
  "total_llm_tokens": 2500,
  "assistants_used": ["receptionist", "nurse_triage", "scheduling"]
}
```

---

## SQLite Database (Cross-Call Memory)

The system uses SQLite (V2 schema) for **cross-call patient recognition**. When a patient calls again (even days later), the system recognizes them by name+DOB and injects their medical history into the conversation.

**Database Location**: `data/sessions.db`

**V2 Schema Features**:
| Feature | Description |
|---------|-------------|
| **UPSERT Support** | Prevents duplicate records (one assessment/appointment per session) |
| **Audit Trail** | `created_at`, `updated_at` on all tables |
| **Direct Patient Queries** | `patient_id` in all child tables (no JOINs needed) |
| **Consistent Design** | Uniform schema across all tables |

**Tables**:
- `patients` - Master patient data (name, DOB, unique constraint)
- `sessions` - Call records linked to patients
- `medical_assessments` - Symptoms recorded by nurse (UPSERT by session)
- `appointments` - Booked appointments (UPSERT by session)
- `pharmacy_requests` - Refill requests (UPSERT by session+medication)

**How It Works**:
1. When `verify_patient()` is called, system looks up patient by name+DOB
2. If found, loads their previous calls, complaints, and appointments
3. History is injected into SESSION MEMORY for the agent to see
4. Tools use UPSERT operations to prevent duplicate records

**Returning Patient Experience**:
```
## RETURNING PATIENT HISTORY
- Previous calls: 3
- Previous complaint: headache (2025-01-05)
- Had appointment: General Medicine on 2025-01-06 10:00
```

---

## Testing with LiveKit Playground

1. Start the agent worker: `uv run python -m src.main dev`
2. Open [LiveKit Agents Playground](https://agents-playground.livekit.io)
3. Connect to your LiveKit project
4. Test conversation flows and handoffs

**Test Scenarios**:
| Scenario      | Input                      | Expected Flow                     |
| ------------- | -------------------------- | --------------------------------- |
| Symptoms      | "I have a headache"        | Receptionist → Nurse Triage       |
| Appointment   | "I need a checkup"         | Receptionist → Scheduling         |
| Prescription  | "I need a refill"          | Receptionist → Pharmacy           |
| Back Transfer | In Pharmacy: "I feel sick" | Pharmacy → Nurse Triage           |

---

## License

MIT
