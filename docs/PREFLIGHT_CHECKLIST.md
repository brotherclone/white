# White Agent Workflow - Pre-Flight Checklist

## Before Your First Run

### ✅ Environment Setup

- [ ] `.env` file exists in project root
- [ ] `ANTHROPIC_API_KEY` is set
- [ ] `TODOIST_API_TOKEN` is set
- [ ] `AGENT_WORK_PRODUCT_BASE_PATH` points to `chain_artifacts/`
- [ ] `MANIFEST_PATH` points to `staged_raw_material/`

### ✅ Directory Structure

- [ ] `chain_artifacts/` directory exists
- [ ] `staged_raw_material/` has audio files organized by prefix (01_XX, 02_XX, etc.)
- [ ] Vocal files are named with "vocal" or "vox" in filename

### ✅ Dependencies

```bash
# Verify all packages installed
python -c "import librosa, soundfile, anthropic, todoist_api_python"
```

- [ ] All imports work without error

### ✅ Vocal Files Check

```bash
# Verify vocal files exist
find staged_raw_material -name "*vocal*" -o -name "*vox*" | head -10
```

- [ ] Vocal files found in multiple directories
- [ ] Files follow naming pattern: `XX_XX_NN_*_vocal.wav`

## First Workflow Run

### Step 1: Start the Workflow

```bash
python run_white_agent.py start
```

**Expected Output**:
- "🎵 Starting White Agent workflow"
- White Agent creates initial proposal
- Black Agent begins processing
- "⏸️ WORKFLOW PAUSED - HUMAN ACTION REQUIRED"

### Step 2: Check Paused State

- [ ] `paused_state.pkl` file created
- [ ] Instructions displayed with:
  - Todoist task URLs
  - Sigil charging ritual steps
  - EVP analysis details

### Step 3: Complete Ritual Tasks

- [ ] Open Todoist tasks from URLs in output
- [ ] Complete sigil charging ritual
- [ ] Mark all tasks as complete in Todoist
- [ ] (Optional) Review generated EVP transcripts in chain_artifacts/

### Step 4: Resume the Workflow

```bash
python run_white_agent.py resume
```

**Expected Output**:
- "🔄 Resuming workflow"
- Task verification passes
- Black Agent workflow completes
- White Agent processes results
- Red Agent invoked
- "✅ WORKFLOW COMPLETED"

### Step 5: Verify Results

- [ ] Check `chain_artifacts/{thread_id}/` for:
  - EVP audio segments (`.wav` files)
  - Sigil images (`.png` files)
- [ ] Song proposals saved in manifests directory
- [ ] No errors in output

## Testing Mode (No API Calls)

If you want to test without using Anthropic API credits:

```bash
export MOCK_MODE=true
python run_white_agent.py start
```

- [ ] Workflow uses mock YAML data
- [ ] No API calls made
- [ ] Workflow completes without pausing (if mock data configured)

## Troubleshooting Quick Checks

### Issue: "No vocal files found"

```bash
# Check file naming
ls staged_raw_material/01_02/*vocal* 2>/dev/null || echo "No vocal files in 01_02"
```

**Fix**: Ensure files have "vocal" or "vox" in the filename

### Issue: "Task verification failed"

```bash
# Test Todoist API
python -c "from todoist_api_python.api import TodoistAPI; api = TodoistAPI('YOUR_TOKEN'); print(api.get_tasks()[:2])"
```

**Fix**: 
- Verify `TODOIST_API_TOKEN` is correct
- Or use `--no-verify` flag for testing

### Issue: "State file not found"

```bash
# Check for state file
ls -lah paused_state.pkl
```

**Fix**: Make sure you're in the project root directory

## Vocal File Priority Verification

To verify vocal files are being prioritized:

```bash
# Run workflow and watch logs
python run_white_agent.py start 2>&1 | grep -E "\[VOCAL\]|\[INSTRUMENT\]"
```

**Expected**: You should see `[VOCAL]` tags appearing first in the processing order

## Quick Test Script

```python
# test_workflow.py
from app.agents.white_agent import WhiteAgent

white = WhiteAgent()
print("✓ WhiteAgent initialized")

state = white.start_workflow()
print(f"✓ Workflow started (thread: {state.thread_id})")

if state.workflow_paused:
    print(f"✓ Paused as expected: {state.pause_reason}")
else:
    print("⚠️ Workflow did not pause (might be in mock mode)")
```

Run with:
```bash
python test_workflow.py
```

## Ready to Go!

Once all checkboxes are complete, you're ready to run your first full workflow:

```bash
python run_white_agent.py start
# ... complete rituals ...
python run_white_agent.py resume
```

## Support

- Full docs: `docs/WHITE_AGENT_USAGE.md`
- Examples: `examples/white_agent_usage.py`
- Summary: `WORKFLOW_SUMMARY.md`

