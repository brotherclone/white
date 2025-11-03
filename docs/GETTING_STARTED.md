# White Agent - Getting Started

## ✅ Status: Ready to Use

All fixes have been applied and verified. The White Agent workflow is ready to run.

## Quick Start

```bash
# Verify everything is working
python check_status.py

# Start a new workflow
python run_white_agent.py start

# After completing ritual tasks in Todoist
python run_white_agent.py resume
```

## What Was Fixed

### 1. Vocal File Prioritization ✅
EVP transcriptions now use vocal audio files instead of instrumental ones, resulting in much better quality transcriptions.

### 2. Dict State Conversion ✅
Fixed `AttributeError: 'dict' object has no attribute 'workflow_paused'` by adding proper conversion from dict to MainAgentState objects.

## Documentation

- **Quick Reference**: `QUICK_REFERENCE.txt` - One-page reference card
- **Complete Guide**: `COMPLETE_FIX_GUIDE.md` - Comprehensive usage guide
- **Status Report**: `FINAL_STATUS_REPORT.txt` - Detailed fix report
- **Full Docs**: `docs/WHITE_AGENT_USAGE.md` - Complete documentation
- **Checklist**: `PREFLIGHT_CHECKLIST.md` - Pre-flight checklist

## Workflow

1. **White Agent** → Creates initial song proposal
2. **Black Agent** → Generates counter-proposal with EVP + sigils
3. **⏸️ Pause** → Workflow pauses for ritual completion
4. **Human** → Completes sigil ritual in Todoist
5. **▶️ Resume** → White Agent resumes and processes
6. **Red Agent** → Creates final song proposal
7. **✅ Complete** → All proposals saved

## Examples

See `examples/white_agent_usage.py` for Python API examples.

## Troubleshooting

Run the status check:
```bash
python check_status.py
```

For more help, see `docs/WHITE_AGENT_USAGE.md`.

## Next Steps

1. Verify setup: `python check_status.py`
2. Start workflow: `python run_white_agent.py start`
3. Complete rituals (Todoist tasks)
4. Resume: `python run_white_agent.py resume`
5. Check results in `chain_artifacts/` and `song_proposals/`

---

**All systems operational. Ready to create music! 🎵**

