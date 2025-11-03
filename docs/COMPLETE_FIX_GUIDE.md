# White Agent - Complete Fix & Usage Guide

## 🔧 Issues Fixed

### Issue 1: Vocal Files Not Being Prioritized (FIXED ✅)
**File**: `app/agents/tools/audio_tools.py`  
**Problem**: EVP transcriptions used instrumental audio instead of vocals  
**Solution**: Removed `random.shuffle()` that destroyed vocal file prioritization  
**Result**: Vocal files now processed first for better EVP quality

### Issue 2: Dict State AttributeError (FIXED ✅)
**Files**: `app/agents/white_agent.py`, `run_white_agent.py`, `examples/white_agent_usage.py`  
**Problem**: `AttributeError: 'dict' object has no attribute 'workflow_paused'`  
**Solution**: Added conversion from dict to MainAgentState object  
**Result**: All state attributes now accessible without errors

## 🚀 Usage

### Command Line (Recommended)

#### Start a new workflow:
```bash
python run_white_agent.py start
```

#### Resume after ritual completion:
```bash
python run_white_agent.py resume
```

#### Test mode (skip task verification):
```bash
python run_white_agent.py resume --no-verify
```

### Python API

```python
from app.agents.white_agent import WhiteAgent
import pickle

# Start
white = WhiteAgent()
state = white.start_workflow()

# If paused, save it
if state.workflow_paused:
    with open("../paused_state.pkl", "wb") as f:
        pickle.dump(state, f)

# Later, resume
with open("../paused_state.pkl", "rb") as f:
    paused_state = pickle.load(f)

white = WhiteAgent()
final_state = white.resume_workflow(paused_state)
```

## 📁 Files Created/Modified

### Modified Files
1. **`app/agents/tools/audio_tools.py`**
   - Removed `random.shuffle()` in `get_audio_segments_as_chain_artifacts()`
   - Added logging for vocal vs instrumental files

2. **`app/agents/white_agent.py`**
   - Added `start_workflow()` method
   - Added `resume_workflow()` method
   - Modified `resume_after_black_agent_ritual()` to instance method
   - Added dict→MainAgentState conversion

3. **`run_white_agent.py`** (NEW)
   - Command-line interface
   - `ensure_state_object()` helper

4. **`examples/white_agent_usage.py`** (NEW)
   - Python usage examples
   - `ensure_state_object()` helper

### Documentation Files (NEW)
- `docs/WHITE_AGENT_USAGE.md` - Complete usage guide
- `docs/FIX_DICT_STATE_ISSUE.md` - Technical fix details
- `WORKFLOW_SUMMARY.md` - High-level summary
- `PREFLIGHT_CHECKLIST.md` - Pre-run checklist
- `QUICK_FIX_SUMMARY.md` - Quick reference

### Test Files (NEW)
- `test_state_conversion.py` - Verify dict conversion fix

## ✅ Verification Steps

### 1. Syntax Check
```bash
python -m py_compile run_white_agent.py
# Should complete with no output
```

### 2. State Conversion Test
```bash
python test_state_conversion.py
# Should show: ✅ All tests passed!
```

### 3. Run Workflow
```bash
python run_white_agent.py start
# Should start without errors
```

## 🎯 What to Expect

### Starting a Workflow
1. White Agent creates initial proposal
2. Black Agent generates counter-proposal
3. EVP transcription from vocal files
4. Sigil generation
5. **Workflow pauses** for ritual completion
6. State saved to `paused_state.pkl`

### After Completing Rituals
1. Mark Todoist tasks complete
2. Run `python run_white_agent.py resume`
3. Black Agent work integrated
4. White Agent processes results
5. Red Agent invoked
6. Final proposals saved

## 🐛 Troubleshooting

### "No vocal files found"
- Check files have "vocal" or "vox" in filename
- Verify files exist: `find staged_raw_material -name "*vocal*"`

### "State file not found"
- Run from project root directory
- Or specify: `--state-file /full/path/to/paused_state.pkl`

### "AttributeError: 'dict' object..."
- Update to latest code
- This should be fixed now with `ensure_state_object()` helper

### "Task verification failed"
- Complete all Todoist tasks
- Or use `--no-verify` for testing

## 📚 Documentation

- **Quick Start**: This file
- **Full Guide**: `docs/WHITE_AGENT_USAGE.md`
- **Examples**: `examples/white_agent_usage.py`
- **Checklist**: `PREFLIGHT_CHECKLIST.md`
- **Technical Details**: `docs/FIX_DICT_STATE_ISSUE.md`

## 🎨 Workflow Sequence

```
┌─────────────────┐
│  White Agent    │ Creates initial proposal
└────────┬────────┘
         ↓
┌─────────────────┐
│  Black Agent    │ Counter-proposal + EVP + Sigils
└────────┬────────┘
         ↓
┌─────────────────┐
│  ⏸️  PAUSE      │ Human performs ritual
└────────┬────────┘
         ↓
┌─────────────────┐
│  White Agent    │ Processes Black's work
└────────┬────────┘
         ↓
┌─────────────────┐
│  Red Agent      │ Final song proposal
└────────┬────────┘
         ↓
┌─────────────────┐
│  ✅ Complete    │ Proposals saved
└─────────────────┘
```

## 🔑 Key Points

1. **Vocal files now prioritized** - Better EVP transcriptions
2. **State conversion handled** - No more AttributeError
3. **Easy start/resume** - Simple command-line interface
4. **State persistence** - Workflow can pause and resume
5. **Task verification** - Optional Todoist integration

## 🚦 Ready to Go!

Everything is fixed and ready to use. Run:

```bash
python run_white_agent.py start
```

Then after completing rituals:

```bash
python run_white_agent.py resume
```

That's it! 🎵

