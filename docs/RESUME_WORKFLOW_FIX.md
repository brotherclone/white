# Resume Workflow Fix - Summary

## Problem
When trying to resume the White Agent workflow after completing Black Agent ritual tasks, the system was failing with the error:
```
langgraph.errors.EmptyInputError: Received no input for __start__
```

## Root Cause
The issue had multiple layers:

1. **In-Memory Checkpointer**: The BlackAgent workflow was using `MemorySaver()`, an in-memory checkpointer that doesn't persist across process restarts.

2. **Lost Checkpoint Data**: When the paused state was pickled and saved to disk, the BlackAgent instance (and its checkpointer with checkpoint data) wasn't serialized. On resume, a new BlackAgent instance was created with a new in-memory checkpointer that had no knowledge of the saved workflow state.

3. **Wrong Resume Function**: The code was calling `resume_black_agent_workflow()` which created a new BlackAgent, instead of using the existing instance from WhiteAgent.

## Solution

### 1. Switch to Persistent Checkpointer
Changed from `MemorySaver()` to `SqliteSaver` with a persistent SQLite database:

```python
# Before
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# After
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
conn = sqlite3.connect("checkpoints/black_agent.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
```

**Key Points:**
- Database file: `checkpoints/black_agent.db`
- `check_same_thread=False`: Required because LangGraph uses threads internally
- Checkpoint data persists across process restarts

### 2. Use Existing BlackAgent Instance
Changed `resume_after_black_agent_ritual()` to use `resume_black_agent_workflow_with_agent()`:

```python
# Before
final_black_state = resume_black_agent_workflow(black_config, verify_tasks=verify_tasks)

# After
black_agent = self.agents.get('black')
final_black_state = resume_black_agent_workflow_with_agent(
    black_agent, 
    black_config, 
    verify_tasks=verify_tasks
)
```

This ensures we use the same BlackAgent instance that's part of WhiteAgent.

### 3. Handle Missing Workflow
Updated `resume_black_agent_workflow_with_agent()` to recreate the compiled workflow if it doesn't exist (after unpickling):

```python
if not hasattr(black_agent, '_compiled_workflow') or black_agent._compiled_workflow is None:
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    import os
    os.makedirs("checkpoints", exist_ok=True)
    conn = sqlite3.connect("checkpoints/black_agent.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    black_agent._compiled_workflow = black_agent.create_graph().compile(
        checkpointer=checkpointer,
        interrupt_before=["await_human_action"]
    )
```

### 4. Fixed Input Parameter
Changed the invoke call from `None` to `{}` (empty dict) when resuming:

```python
# Resume workflow - it will continue from 'await_human_action' node
result = black_agent._compiled_workflow.invoke(
    {},  # Empty dict when resuming - state comes from checkpoint
    config=black_config
)
```

## Files Modified

1. `/Volumes/LucidNonsense/White/app/agents/black_agent.py`
   - Added sqlite3 import
   - Updated checkpointer creation to use SqliteSaver

2. `/Volumes/LucidNonsense/White/app/agents/white_agent.py`
   - Changed import from `resume_black_agent_workflow` to `resume_black_agent_workflow_with_agent`
   - Updated function call to pass black_agent instance

3. `/Volumes/LucidNonsense/White/app/agents/workflow/resume_black_workflow.py`
   - Updated both resume functions to use SqliteSaver
   - Changed `resume_black_agent_workflow_with_agent` to create workflow if missing
   - Fixed invoke parameter from `None` to `{}`

## Testing

Tested with:
```bash
python run_white_agent.py resume --no-verify
```

Result:
```
INFO:root:Resuming Black Agent workflow after human action...
INFO:root:✓ Workflow completed: The Frequency of Forgetting
INFO:root:✓ Black Agent workflow resumed and completed
```

## Notes

- The `checkpoints/` directory is created automatically if it doesn't exist
- The SQLite database persists workflow state across process restarts
- Multiple workflow threads can coexist in the same database (they're identified by thread_id)
- The `check_same_thread=False` parameter is safe here because LangGraph handles thread safety internally

## Future Considerations

- Consider adding database cleanup/archival for old threads
- Could add migration script if switching checkpoint storage mechanisms
- May want to configure SQLite with WAL mode for better concurrent access

