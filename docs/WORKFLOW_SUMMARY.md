# White Agent Workflow - Quick Summary

## What Changed

### 1. Audio File Prioritization Fix
**File**: `app/agents/tools/audio_tools.py`

**Problem**: The `find_wav_files_prioritized` function was correctly prioritizing vocal files, but immediately after, `random.shuffle()` was destroying that prioritization.

**Fix**: Removed the `random.shuffle(wav_files)` line in `get_audio_segments_as_chain_artifacts()` so vocal files are processed first.

**Result**: Now EVP transcriptions will use vocal audio segments instead of instrumental ones, improving the quality of "EVP-style" transcriptions.

**Added Logging**: Shows which files are vocal vs instrumental during processing.

### 2. White Agent Workflow Methods
**File**: `app/agents/white_agent.py`

**Added**:
- `start_workflow()` - Start a new workflow from scratch
- `resume_workflow()` - Resume after completing ritual tasks
- `resume_after_black_agent_ritual()` - Internal method for Black Agent resumption

**Changes**:
- Converted `resume_after_black_agent_ritual` from static method to instance method
- Properly handles paused states and task verification
- Continues workflow through Red Agent after resuming

## How to Use

### Starting a New Workflow

```bash
# Command line (easiest)
python run_white_agent.py start

# Or in Python
from app.agents.white_agent import WhiteAgent
white = WhiteAgent()
state = white.start_workflow()
```

### Resuming After a Sigil Ritual

When the workflow pauses for sigil charging:

1. Complete the ritual tasks in Todoist
2. Resume the workflow:

```bash
# Command line
python run_white_agent.py resume

# Or in Python
import pickle
with open("paused_state.pkl", "rb") as f:
    paused_state = pickle.load(f)

white = WhiteAgent()
final_state = white.resume_workflow(paused_state)
```

### Testing Without Task Verification

```bash
python run_white_agent.py resume --no-verify
```

## Files Created

1. **`run_white_agent.py`** - Command-line runner with start/resume commands
2. **`examples/white_agent_usage.py`** - Detailed Python usage examples
3. **`docs/WHITE_AGENT_USAGE.md`** - Complete documentation

## Workflow Sequence

1. **White Agent** → Creates initial song proposal
2. **Black Agent** → Generates counter-proposal with EVP + sigils
3. **⏸️ PAUSE** → Workflow pauses for human ritual action
4. **Human** → Completes sigil ritual, marks tasks done in Todoist
5. **▶️ RESUME** → White Agent resumes, processes Black's work
6. **Red Agent** → Creates final song proposal
7. **✅ Complete** → Saves all proposals and artifacts

## State Persistence

- Paused state automatically saved to `paused_state.pkl`
- Contains all context needed to resume
- Thread ID preserved for artifact tracking
- LangGraph checkpointer maintains sub-workflow state

## Quick Reference

| Action | Command |
|--------|---------|
| Start new workflow | `python run_white_agent.py start` |
| Resume workflow | `python run_white_agent.py resume` |
| Resume without verification | `python run_white_agent.py resume --no-verify` |
| Resume and cleanup | `python run_white_agent.py resume --cleanup` |

## Next Steps

1. Test the new vocal file prioritization by running a workflow
2. Check the logs to verify vocal files are being processed
3. Review the EVP transcriptions to see if they're more coherent
4. Complete a full workflow cycle: start → pause → ritual → resume

## Notes

- The vocal file fix will improve EVP quality immediately
- State files are Python pickles (not portable across Python versions)
- Task verification requires `TODOIST_API_TOKEN` in `.env`
- Mock mode available via `MOCK_MODE=true` for testing without API calls

