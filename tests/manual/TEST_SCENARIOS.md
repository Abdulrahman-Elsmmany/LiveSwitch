# Manual Test Scenarios

This document provides comprehensive manual test scenarios for validating the Multi-Assistant Medical Triage System through LiveKit Playground.

## Quick Start Checklist

Before testing, ensure all systems are working:

```bash
# 1. Run automated tests (should pass 148 tests)
uv run pytest

# 2. Type checking (should have no errors)
uv run mypy src/

# 3. Linting (should pass)
uv run ruff check src/

# 4. Start the worker
uv run python -m src.main dev
```

---

## Prerequisites

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   uv venv --python 3.13
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows

   # Install dependencies
   uv sync

   # Copy environment template
   cp .env.example .env.local
   ```

2. **Configure API Keys** in `.env.local`:
   ```
   LIVEKIT_API_KEY=your_api_key
   LIVEKIT_API_SECRET=your_api_secret
   LIVEKIT_URL=wss://your-project.livekit.cloud
   OPENROUTER_API_KEY=your_openrouter_key
   DEEPGRAM_API_KEY=your_deepgram_key
   CARTESIA_API_KEY=your_cartesia_key
   ```

3. **Start the Worker**
   ```bash
   uv run python -m src.main dev
   ```

4. **Connect via LiveKit Playground**
   - Go to: https://agents-playground.livekit.io/
   - Or go to your LiveKit Cloud dashboard → Playground
   - Connect to a room where the agent is listening

---

## What to Expect in Console Logs

When running, you should see colored log output like this:

```
[CONFIG] Loading configuration from: config/medical_triage.json
[CONFIG] Orchestration manager created with assistants: ['receptionist', 'nurse_triage', 'scheduling', 'pharmacy']
[SESSION] New job: AJ_xxxxx
[SESSION] Session abc123 started with entry agent: receptionist
[AGENT] Agent started: receptionist
[TRANSCRIPT] User: Hello, I need to schedule an appointment
[TRANSCRIPT] receptionist: Hello! Welcome to our medical center. I'm Alex...
[HANDOFF] Handoff: receptionist -> scheduling
[AGENT] Agent stopped: receptionist
[AGENT] Agent started: scheduling
[TRANSCRIPT] User: I need to see Dr. Smith next week
[TRANSCRIPT] scheduling: I'd be happy to help you schedule with Dr. Smith...
```

**Log Types:**
- `[CONFIG]` - Blue - Configuration loading
- `[SESSION]` - Green - Session lifecycle
- `[AGENT]` - Cyan - Agent start/stop
- `[TRANSCRIPT]` - Green - Real-time conversation (user and assistant messages)
- `[HANDOFF]` - Magenta - Agent transfers
- `[WARNING]` - Yellow - Warnings (e.g., high token usage)
- `[ERROR]` - Red - Errors

---

## Test Scenario 1: Happy Path - Symptom to Appointment

**Goal:** Verify complete flow from reception through nurse to scheduled appointment.

### Steps

1. **Greeting Phase**
   - Speak: "Hello, I'd like to speak with someone about not feeling well"

   **Expected:**
   - Alex (receptionist) greets you warmly
   - Asks for your name and date of birth

2. **Identity Verification**
   - Speak: "My name is John Smith and my date of birth is January 15, 1985"

   **Expected:**
   - Alex confirms your identity
   - Asks how they can help or what the reason for your call is

3. **Symptom Report**
   - Speak: "I've been having headaches for the past three days"

   **Expected:**
   - Alex acknowledges and transfers to nurse
   - Handoff message indicates transfer to Sarah (nurse)

4. **Nurse Assessment**
   - Sarah introduces herself
   - Speak: "The headache is mostly in my forehead, it started Monday, it's about a 5 out of 10"

   **Expected:**
   - Sarah asks follow-up questions (OLDCARTS)
   - Records symptoms
   - Determines urgency level (likely routine or next-day)

5. **Scheduling Handoff**
   - Sarah transfers to scheduling

   **Expected:**
   - Jordan (scheduling) introduces themselves
   - Offers appointment options based on urgency

6. **Appointment Booking**
   - Select a time slot when offered
   - Speak: "Yes, the morning appointment works for me"

   **Expected:**
   - Jordan confirms the appointment
   - Provides confirmation number (APT-XXXXX)
   - Reminds to arrive 15 minutes early

### Success Criteria
- [ ] Patient identity verified
- [ ] Handoff from receptionist to nurse occurred
- [ ] Symptoms recorded by nurse
- [ ] Handoff from nurse to scheduling occurred
- [ ] Appointment booked with confirmation number
- [ ] All context preserved (patient name mentioned throughout)

---

## Test Scenario 2: Emergency Detection

**Goal:** Verify emergency symptoms trigger appropriate response and do NOT proceed to scheduling.

### Steps

1. **Initial Contact**
   - Speak: "Hello, I need help"
   - Provide name and DOB when asked

2. **Emergency Symptoms**
   - After nurse transfer, speak: "I'm having severe chest pain and I can't catch my breath. It's been getting worse for the past hour."

   **Expected:**
   - Sarah (nurse) immediately recognizes red flags
   - Flags call as emergency
   - Advises to call 911 or go to ER immediately

3. **Verify No Scheduling Transfer**

   **Expected:**
   - Conversation stays with nurse
   - Does NOT transfer to scheduling for appointment
   - Clear emergency instructions provided

### Success Criteria
- [ ] Emergency recognized immediately
- [ ] 911/ER advice given
- [ ] Call flagged as emergency in system
- [ ] NO transfer to scheduling occurred
- [ ] Clear, calm emergency instructions

---

## Test Scenario 3: Direct Appointment Scheduling

**Goal:** Verify patient can go directly to scheduling without nurse triage.

### Steps

1. **Initial Contact**
   - Speak: "Hi, I just need to schedule my annual checkup"

2. **Identity Verification**
   - Provide name and DOB when asked

3. **Direct Transfer**

   **Expected:**
   - Alex transfers directly to scheduling (bypasses nurse)
   - Jordan offers appointment slots

4. **Book Appointment**
   - Select a time slot

   **Expected:**
   - Appointment confirmed with number
   - Pre-appointment instructions provided

### Success Criteria
- [ ] Direct transfer to scheduling (no nurse)
- [ ] Appointment booked successfully
- [ ] Only 1 handoff occurred (receptionist -> scheduling)

---

## Test Scenario 4: Pharmacy Refill Request

**Goal:** Verify pharmacy flow for prescription refills.

### Steps

1. **Initial Contact**
   - Speak: "I need to refill my blood pressure medication"

2. **Identity Verification**
   - Provide name and DOB when asked

3. **Pharmacy Transfer**

   **Expected:**
   - Alex transfers to pharmacy

4. **Refill Request**
   - Speak: "I need to refill my Lisinopril"

   **Expected:**
   - Pharmacy checks refill status
   - If refills available: submits request, provides pickup estimate
   - If no refills: explains need for new prescription

### Success Criteria
- [ ] Transfer to pharmacy occurred
- [ ] Medication name confirmed
- [ ] Refill status checked
- [ ] Request submitted with reference number (RX-XXXXXX)
- [ ] Estimated pickup time provided

---

## Test Scenario 5: Pharmacy to Nurse Escalation

**Goal:** Verify side effects during pharmacy call transfer to nurse.

### Steps

1. **Start at Pharmacy**
   - Get transferred to pharmacy for refill request

2. **Report Side Effects**
   - Speak: "Actually, I've been feeling dizzy and nauseous since I started taking this medication"

   **Expected:**
   - Pharmacy recognizes clinical concern
   - Transfers to nurse triage

3. **Nurse Assessment**

   **Expected:**
   - Sarah assesses the side effects
   - May recommend stopping medication and seeing provider
   - May offer appointment scheduling

### Success Criteria
- [ ] Side effects recognized as clinical concern
- [ ] Transfer to nurse occurred
- [ ] Appropriate clinical guidance provided

---

## Test Scenario 6: Unknown/Unavailable Service Request

**Goal:** Verify system handles requests for unavailable services gracefully.

### Steps

1. **Request Unavailable Service**
   - Speak: "I have a question about my bill" or "I need to talk to someone about insurance"

   **Expected:**
   - Alex explains billing/insurance is not available through this line
   - Does NOT hallucinate a transfer to non-existent department
   - May offer to help with available services

### Success Criteria
- [ ] No hallucinated department mentioned
- [ ] Clear explanation that service is unavailable
- [ ] Alternative assistance offered
- [ ] No transfer to non-existent agent

---

## Test Scenario 7: Identity Not Recognized Pattern

**Goal:** Verify DOB format validation works.

### Steps

1. **Provide Wrong Format**
   - When asked for DOB, speak: "January fifteenth, nineteen eighty-five"

   **Expected:**
   - System asks for MM/DD/YYYY format
   - Reprompts for correct format

2. **Provide Correct Format**
   - Speak: "Zero one, fifteen, nineteen eighty-five" (01/15/1985)

   **Expected:**
   - Verification succeeds

### Success Criteria
- [ ] Invalid format handled gracefully
- [ ] Clear format guidance provided
- [ ] Second attempt with correct format succeeds

---

## Test Scenario 8: Multi-Handoff Round Trip

**Goal:** Verify context preserved through multiple handoffs.

### Steps

1. **Complete Initial Flow**
   - Go through receptionist -> nurse -> scheduling

2. **Request Additional Help**
   - After appointment booked, speak: "Actually, I also have a question about my prescriptions"

   **Expected:**
   - Jordan transfers back to receptionist or pharmacy
   - Your name and appointment info still known

3. **Verify Context**
   - At any point, ask: "Do you have my information on file?"

   **Expected:**
   - Current agent confirms patient name
   - Previous context mentioned

### Success Criteria
- [ ] Multiple handoffs completed
- [ ] Patient name preserved throughout
- [ ] Collected data (symptoms, appointment) accessible
- [ ] No identity confusion between agents

---

## Test Scenario 9: Agent Identity Verification

**Goal:** Verify agents maintain correct identity and don't confuse themselves with previous agents.

### Steps

1. **Note Each Agent Name**
   - Receptionist should be "Alex"
   - Nurse should be "Sarah"
   - Scheduling should be "Jordan"

2. **Ask About Identity**
   - After a handoff, ask: "What's your name again?"

   **Expected:**
   - Current agent states their own name
   - Does NOT claim to be the previous agent

3. **Reference Previous Conversation**
   - Ask: "What did the receptionist say my name was?"

   **Expected:**
   - Agent correctly attributes previous conversation to correct staff member

### Success Criteria
- [ ] Alex identifies as Alex (receptionist)
- [ ] Sarah identifies as Sarah (nurse)
- [ ] Jordan identifies as Jordan (scheduling)
- [ ] No agent claims previous agent's identity
- [ ] Previous agent's words correctly attributed

---

## Test Scenario 10: Max Handoffs Limit

**Goal:** Verify system handles max handoffs gracefully.

### Steps

1. **Trigger Multiple Handoffs**
   - Request transfers back and forth multiple times
   - Config allows 6 max handoffs

2. **Reach Limit**

   **Expected:**
   - After 6 handoffs, transfer requests fail gracefully
   - System explains cannot transfer further
   - Encourages completing with current agent

### Success Criteria
- [ ] Handoffs tracked correctly
- [ ] Limit enforced (no more than 6)
- [ ] Graceful handling when limit reached

---

## Post-Test Verification

### 1. Verify Real-Time Logging Worked

During the call, you should have seen `[TRANSCRIPT]` lines in the console showing the conversation as it happened. If you didn't see these, check that the event handlers are working.

### 2. Verify Artifacts Were Generated

After the session ends (disconnect from playground or press Ctrl+C), check that artifacts were saved:

```bash
# Windows
dir artifacts\

# Linux/Mac
ls -la artifacts/

# Each session creates a folder with 3 files:
# artifacts/{session_id}/
#   ├── {session_id}_report.json      # Full session report
#   ├── {session_id}_transcript.json  # Conversation transcript
#   └── {session_id}_metrics.json     # Performance metrics
```

### 3. Verify Transcript Content

Open the transcript file and verify it contains actual conversation:

```bash
# Windows (PowerShell)
Get-Content artifacts\{session_id}\{session_id}_transcript.json | ConvertFrom-Json | Select-Object -ExpandProperty entries

# Linux/Mac
cat artifacts/{session_id}/{session_id}_transcript.json | jq '.entries'
```

**Expected Transcript Structure:**
```json
{
  "session_id": "abc123-...",
  "start_time": "2025-01-06T...",
  "end_time": "2025-01-06T...",
  "entry_count": 12,
  "entries": [
    {
      "timestamp": "2025-01-06T...",
      "role": "user",
      "content": "Hello, I need to schedule an appointment",
      "assistant_id": null,
      "metadata": {}
    },
    {
      "timestamp": "2025-01-06T...",
      "role": "assistant",
      "content": "Hello! Welcome to our medical center...",
      "assistant_id": "receptionist",
      "metadata": {}
    },
    {
      "timestamp": "2025-01-06T...",
      "role": "system",
      "content": "Handoff: receptionist -> scheduling",
      "assistant_id": null,
      "metadata": {
        "event_type": "handoff",
        "from_assistant": "receptionist",
        "to_assistant": "scheduling",
        "reason": "Agent handoff"
      }
    }
  ]
}
```

### 4. Verify Metrics Content

```bash
# View metrics
cat artifacts/{session_id}/{session_id}_metrics.json | jq '.'
```

**Expected Metrics Structure:**
```json
{
  "session_id": "abc123-...",
  "start_time": "2025-01-06T...",
  "end_time": "2025-01-06T...",
  "duration_seconds": 45.2,
  "turn_count": 8,
  "handoff_count": 2,
  "total_llm_tokens_input": 1234,
  "total_llm_tokens_output": 567,
  "total_llm_tokens": 1801,
  "assistants_used": ["receptionist", "nurse_triage", "scheduling"],
  "turn_metrics": [...]
}
```

### 5. Verify Report Content

```bash
# View full report
cat artifacts/{session_id}/{session_id}_report.json | jq '.'
```

**Expected Report Structure:**
```json
{
  "session_id": "abc123-...",
  "configuration": {
    "name": "Medical Triage System",
    "version": "1.0.0"
  },
  "entry_assistant": "receptionist",
  "end_reason": "completed",
  "session_data": {
    "patient_name": "John Smith",
    "patient_dob": "01/15/1985",
    "patient_verified": true,
    "chief_complaint": "headaches",
    ...
  },
  "transcript": {...},
  "metrics": {...}
}
```

### 6. Verify SQLite Database (Cross-Call Memory)

If you verified a patient during the call, their data is stored in SQLite:

```bash
# Windows
sqlite3 data\sessions.db "SELECT * FROM patients;"

# Linux/Mac
sqlite3 data/sessions.db "SELECT * FROM patients;"
```

**Expected Output:**
```
patient_id|full_name|date_of_birth|created_at|last_call_at
abc123...|John Smith|01/15/1985|2025-01-06T...|2025-01-06T...
```

**Test Cross-Call Recognition:**
1. Complete a call and verify patient (e.g., "John Smith", "01/15/1985")
2. Start a NEW call
3. Verify with the same name and DOB
4. The system should recognize them as a returning patient

```bash
# Check if patient was recognized (last_call_at should be updated)
sqlite3 data/sessions.db "SELECT full_name, last_call_at FROM patients ORDER BY last_call_at DESC LIMIT 5;"
```

### 7. Verify Session Was Finalized

```bash
# Check sessions table
sqlite3 data/sessions.db "SELECT session_id, status, start_time, end_time FROM sessions ORDER BY start_time DESC LIMIT 5;"
```

**Expected:** Status should be "completed" (or "error" if something went wrong)

---

## Artifact Verification Checklist

After each test, verify:

- [ ] `[TRANSCRIPT]` logs appeared in console during the call
- [ ] Artifacts folder created: `artifacts/{session_id}/`
- [ ] Report file exists and contains session data
- [ ] Transcript file exists and contains conversation entries (NOT empty)
- [ ] Metrics file exists and shows turn_count > 0
- [ ] SQLite database has session record with status = "completed"
- [ ] If patient verified: patient record exists in patients table

---

## Troubleshooting

### Agent Not Responding
- Check worker is running: `uv run python -m src.main dev`
- Verify API keys in `.env.local`
- Check LiveKit Cloud connection
- Verify you're connected to the correct room in Playground

### Wrong Agent Behavior
- Review configuration: `config/medical_triage.json`
- Check logs for errors
- Verify prompts contain correct instructions

### Handoffs Not Working
- Check handoff_targets in configuration
- Verify agent IDs match
- Review factory logs for tool injection

### No `[TRANSCRIPT]` Logs in Console
- This is a bug - the event handlers may not be firing
- Check that `user_input_transcribed` and `conversation_item_added` events are registered
- Verify STT (Deepgram) is working - you should see transcription in the logs

### No Artifacts Generated
- Check `artifacts/` directory exists after session ends
- Review session end logs for errors
- Verify write permissions on the artifacts directory
- Make sure session ended properly (disconnect or Ctrl+C)

### Empty Transcript in Artifacts
- Check that `[TRANSCRIPT]` logs appeared during the call
- If no logs: STT may not be transcribing (check Deepgram API key)
- If logs appeared but artifacts empty: event handlers not collecting properly

### SQLite Database Issues
- Check `data/` directory exists
- Verify database was initialized: `sqlite3 data/sessions.db ".tables"`
- Should show: `appointments  medical_assessments  patients  pharmacy_requests  sessions`

### Ctrl+C Doesn't Save Artifacts
- The `finally` block should always run
- If artifacts not saved, check for exceptions in the finally block
- Look for `Failed to save artifacts` error in logs

---

## Complete System Verification

Run this checklist to verify the entire system is working:

```bash
# 1. Verify tests pass
uv run pytest
# Expected: 148 passed

# 2. Verify type checking
uv run mypy src/
# Expected: No errors

# 3. Verify linting
uv run ruff check src/
# Expected: All checks passed!

# 4. Verify database initialized
sqlite3 data/sessions.db ".tables"
# Expected: appointments  medical_assessments  patients  pharmacy_requests  sessions

# 5. Start worker and test
uv run python -m src.main dev
# Connect via LiveKit Playground
# Have a conversation
# Disconnect or Ctrl+C

# 6. Verify artifacts created
dir artifacts\
# Expected: Folder with session_id containing 3 JSON files

# 7. Verify transcript has content
type artifacts\{session_id}\{session_id}_transcript.json
# Expected: JSON with entries array containing conversation
```

---

## Assignment Requirements Verification

This system meets all assignment requirements:

| Requirement | Status | How to Verify |
|-------------|--------|---------------|
| JSON config defines Multi-Assistant setup | ✅ | `config/medical_triage.json` exists |
| No UI - purely agent worker | ✅ | No frontend code in `src/` |
| Generate orchestration from JSON at runtime | ✅ | Check logs: `[CONFIG] Orchestration manager created...` |
| No hardcoded orchestration logic | ✅ | All assistants from JSON config |
| Custom tools definable in JSON | ✅ | Tools defined per-assistant in config |
| Multiple assistants in same call | ✅ | Test handoffs between agents |
| Controlled handoff between assistants | ✅ | Handoffs work as configured |
| Full conversation context across handoffs | ✅ | Patient name remembered after handoff |
| Track call state and transcript | ✅ | `[TRANSCRIPT]` logs + artifacts |
| Emit final call artifacts | ✅ | Check `artifacts/` after session |
