# White Agent Workflow Guide

This guide explains how to start and resume the White Agent workflow.

## Overview

The White Agent orchestrates the entire rainbow workflow, coordinating between agents. The workflow may pause when human action is required (e.g., for ritual sigil charging).

## Quick Start

### Option 1: Command Line (Recommended)

#### Start a new workflow:
```bash
python run_white_agent.py start
```

The workflow will run until it pauses for human action (e.g., sigil charging ritual). When it pauses, you'll see instructions and a saved state file (`paused_state.pkl`).

#### Resume after completing ritual tasks:
```bash
python run_white_agent.py resume
```

By default, this will verify that all Todoist tasks are complete before resuming.

#### Resume without verification (testing only):
```bash
python run_white_agent.py resume --no-verify
```

#### Resume and clean up the state file:
```bash
python run_white_agent.py resume --cleanup
```

### Option 2: Python API

#### Basic Usage:

```python
from app.agents.white_agent import WhiteAgent

# Create agent
white = WhiteAgent()

# Start workflow
state = white.start_workflow()

# Check if paused
if state.workflow_paused:
    print(f"Paused: {state.pause_reason}")
    # Save state for later
    import pickle
    with open("paused_state.pkl", "wb") as f:
        pickle.dump(state, f)
```

#### Resuming:

```python
from app.agents.white_agent import WhiteAgent
import pickle

# Load paused state
with open("paused_state.pkl", "rb") as f:
    paused_state = pickle.load(f)

# Create new agent instance
white = WhiteAgent()

# Resume workflow
final_state = white.resume_workflow(
    paused_state,
    verify_tasks=True  # Set to False to skip task verification
)
```

## Workflow States

### When Workflow Pauses

When the workflow pauses (e.g., for sigil charging), the state object contains:

- `workflow_paused`: `True`
- `pause_reason`: Human-readable explanation
- `pending_human_action`: Dict with details:
  - `agent`: Which agent needs action (e.g., "black")
  - `action`: What action is needed (e.g., "sigil_charging")
  - `instructions`: Step-by-step instructions
  - `tasks`: List of Todoist tasks with URLs
  - `black_config`: Configuration for resuming the sub-workflow

### After Resuming

After successful resume:

- `workflow_paused`: `False`
- `ready_for_red`: `True` (if Black Agent completed successfully)
- `artifacts`: List of generated artifacts (EVP, sigils, etc.)
- `song_proposals`: Updated with Black Agent's counter-proposal

## Examples

See `examples/white_agent_usage.py` for detailed examples:

- `example_start_workflow()` - Start and handle pausing
- `example_resume_workflow()` - Resume after completing tasks
- `example_resume_workflow_no_verify()` - Resume without task verification
- `example_full_workflow()` - Complete workflow (if no pausing)
- `example_manual_state_management()` - Advanced state management

## Typical Workflow Sequence

1. **Start**: White Agent creates initial proposal
2. **Black Agent Invoked**: Generates counter-proposal with EVP and sigils
3. **Pause**: Workflow pauses for sigil charging ritual
4. **Human Action**: Complete sigil ritual, mark Todoist tasks as done
5. **Resume**: White Agent resumes, processes Black Agent's work
6. **Red Agent Invoked**: Creates song proposal based on synthesized document
7. **Finalize**: Save all proposals and artifacts

## Troubleshooting

### State file not found
```
❌ State file not found: paused_state.pkl
```
**Solution**: Make sure you're in the correct directory or specify `--state-file` with the full path.

### Workflow not paused
```
⚠️ Workflow is not paused - nothing to resume
```
**Solution**: The workflow already completed. Start a new workflow instead.

### Task verification failed
```
❌ Not all tasks are complete
```
**Solution**: Complete all ritual tasks in Todoist, or use `--no-verify` to skip verification (testing only).

## File Locations

- **Runner script**: `run_white_agent.py`
- **Examples**: `examples/white_agent_usage.py`
- **Paused state**: `paused_state.pkl` (auto-saved)
- **Artifacts**: `chain_artifacts/{thread_id}/`
- **Proposals**: `app/structures/manifests/song_proposals/`

## Environment Variables

Make sure your `.env` file contains:

```bash
ANTHROPIC_API_KEY=your_api_key_here
AGENT_WORK_PRODUCT_BASE_PATH=/path/to/chain_artifacts
MANIFEST_PATH=/path/to/staged_raw_material
TODOIST_API_TOKEN=your_todoist_token
```

## Testing Mode

To run with mocked data (no API calls):

```bash
export MOCK_MODE=true
python run_white_agent.py start
```

This uses pre-generated YAML mock data instead of making live API calls.

