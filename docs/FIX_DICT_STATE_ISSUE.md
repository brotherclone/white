# Fix: AttributeError - 'dict' object has no attribute 'workflow_paused'

## Problem

When running `python run_white_agent.py start`, the workflow crashed with:

```
AttributeError: 'dict' object has no attribute 'workflow_paused'
```

This occurred because LangGraph's `workflow.invoke()` returns a dictionary, not a `MainAgentState` object, but the code was trying to access attributes like `state.workflow_paused`.

## Root Cause

In `white_agent.py`, the `start_workflow()` method was doing:

```python
final_state = workflow.invoke(initial_state, config)
return final_state  # This is a dict!
```

LangGraph workflows return state as dictionaries by default, even when the state schema is a Pydantic model.

## Solution

### 1. Added conversion in `start_workflow()` method

**File**: `app/agents/white_agent.py`

```python
result = workflow.invoke(initial_state, config)

# Convert dict result to MainAgentState if needed
if isinstance(result, dict):
    final_state = MainAgentState(**result)
else:
    final_state = result

return final_state
```

### 2. Added `ensure_state_object()` helper function

**Files**: `run_white_agent.py`, `examples/white_agent_usage.py`

```python
def ensure_state_object(state):
    """Convert dict state to MainAgentState if needed."""
    if isinstance(state, dict):
        return MainAgentState(**state)
    return state
```

### 3. Applied conversion at all state access points

**In `run_white_agent.py`**:
- `start_workflow()`: Convert after calling `white.start_workflow()`
- `resume_workflow()`: Convert after loading from pickle and after resuming

**In `examples/white_agent_usage.py`**:
- All example functions now call `ensure_state_object()` after getting state

## Files Modified

1. **`app/agents/white_agent.py`**
   - Modified `start_workflow()` to convert dict to MainAgentState

2. **`run_white_agent.py`**
   - Added `ensure_state_object()` helper
   - Convert state in `start_workflow()` function
   - Convert state in `resume_workflow()` function

3. **`examples/white_agent_usage.py`**
   - Added `ensure_state_object()` helper
   - Convert state in all example functions

## Testing

Run the test script to verify the fix:

```bash
python test_state_conversion.py
```

Expected output:
```
✓ WhiteAgent created
✓ start_workflow() returned: <class 'app.agents.states.white_agent_state.MainAgentState'>
✓ State is MainAgentState object
✓ Can access workflow_paused attribute
✓ Can access thread_id: <some-uuid>

✅ All tests passed! State conversion working correctly.
```

## Verification

Now you can successfully run:

```bash
python run_white_agent.py start
```

The workflow will start properly and you'll be able to access state attributes without errors.

## Why This Happened

LangGraph's design philosophy is to use plain dictionaries for state to maximize flexibility and avoid coupling to specific data structures. Even when you define a Pydantic model as the state schema, the runtime representation is a dict.

This is why we need to explicitly convert back to `MainAgentState` objects when we want to use Pydantic features like:
- Attribute access (`.workflow_paused` vs `['workflow_paused']`)
- Type validation
- Model methods
- IDE autocomplete

## Future-Proofing

The `ensure_state_object()` helper makes the code robust to both:
- Dict states (from LangGraph)
- MainAgentState objects (from our conversions or tests)

This means the code won't break if LangGraph changes its behavior or if we use different state sources.

