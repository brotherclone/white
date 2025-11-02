# Bug Fix Summary: Todoist API Error Handling

## Date: 2025-01-02

## Problem
The Black Agent workflow was encountering 403 Forbidden errors when trying to create Todoist tasks for sigil charging. This was causing:
1. Exception stack traces in the logs
2. Duplicate section creation attempts
3. Unclear error handling
4. The workflow was pausing correctly but with noisy error output

## Root Causes

### 1. Todoist API Permissions
The Todoist API token doesn't have write permissions for the project, resulting in 403 Forbidden errors when attempting to:
- Create new sections
- Create new tasks

### 2. Exception-Based Error Handling
The `create_sigil_charging_task` function was raising exceptions on failure, requiring try-catch blocks at call sites.

### 3. Duplicate generate_sigil Calls
The `evaluate_evp` method had a fallback that directly called `generate_sigil()`, bypassing the workflow routing and potentially causing the method to be called twice.

### 4. Incorrect Type Handling in EVP Evaluation
The `evaluate_evp` method wasn't properly handling the `YesOrNo` object returned by LangChain's `with_structured_output`.

## Solutions Implemented

### 1. Graceful Degradation in Todoist Integration
**File**: `/app/reference/mcp/todoist/main.py`

Changed `create_sigil_charging_task` to return a status dictionary instead of raising exceptions:

```python
# Success case:
{
    "success": True,
    "id": task.id,
    "content": task.content,
    "url": task.url,
    ...
}

# Failure case:
{
    "success": False,
    "error": "403 Forbidden: Cannot create task...",
    "status_code": 403
}
```

Benefits:
- No exceptions propagated to calling code
- Cleaner error logs (ERROR level, no stack traces)
- Allows calling code to handle failures gracefully

### 2. Enhanced Black Agent Error Handling
**File**: `/app/agents/black_agent.py`

Updated `generate_sigil` to check the `success` flag:

```python
task_result = create_sigil_charging_task(...)

if task_result.get("success", False):
    # Create Todoist task reference
    state.pending_human_tasks.append({...})
    state.human_instructions = "...with Todoist URL..."
else:
    # Provide manual instructions
    logging.warning(f"⚠️ Todoist task creation failed: {error_msg}")
    state.human_instructions = "...manual charging instructions..."
```

Benefits:
- Single code path for sigil generation
- Clear WARNING messages instead of ERROR stack traces
- Manual fallback instructions when Todoist unavailable

### 3. Fixed Workflow Routing
**File**: `/app/agents/black_agent.py`

Removed direct `generate_sigil()` call from `evaluate_evp`:

```python
# Before:
else:
    return self.generate_sigil(state)

# After:
else:
    logging.warning(f"EVP evaluation returned unexpected type: {type(result)}")
    state.should_update_proposal_with_evp = False
```

Benefits:
- Respects workflow graph routing
- No duplicate sigil generation
- Consistent workflow execution

### 4. Proper Type Handling for YesOrNo
**File**: `/app/agents/black_agent.py`

Added handling for both dict and YesOrNo object types:

```python
if isinstance(result, dict):
    state.should_update_proposal_with_evp = result.get("answer", False)
elif isinstance(result, YesOrNo):
    state.should_update_proposal_with_evp = result.answer
else:
    logging.warning(f"EVP evaluation returned unexpected type: {type(result)}")
    state.should_update_proposal_with_evp = False
```

Benefits:
- Handles both LangChain return types
- No warnings for expected behavior
- Graceful fallback for unexpected types

### 5. Added Debug Logging
**File**: `/app/reference/mcp/todoist/main.py`

Added logging to help diagnose section lookup issues:

```python
logging.debug(f"Checking section: {name} == {section_name}?")
logging.info(f"Found existing section: {section_name} (id={section.id})")
logging.info(f"Section '{section_name}' not found, attempting to create...")
```

## Testing Results

### Before Fix
```
ERROR:root:403 Forbidden: Cannot create task in project 6CrfWqXrxppjhqMJ...
ERROR:root:Failed creating sigil charging task; sections=[[Section(...)]]
Traceback (most recent call last):
  File ".../todoist/main.py", line 102, in create_sigil_charging_task
    section = api.add_section(...)
  ...
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: ...
ERROR:root:✗ Failed to create Todoist task: 403 Client Error: Forbidden...
[DUPLICATE ATTEMPTS - TWO SIMILAR ERRORS]
```

### After Fix
```
INFO:root:🜏 Entering generate_sigil method
INFO:root:Section 'Black Agent - Sigil Work' not found, attempting to create...
ERROR:root:403 Forbidden: Cannot create task in project 6CrfWqXrxppjhqMJ. Check API token permissions...
WARNING:root:⚠️ Todoist task creation failed: 403 Forbidden: Cannot create task...
INFO:root:⏸️  Black Agent workflow paused at: ('await_human_action',)
INFO:root:============================================================
INFO:root:⏸️  WORKFLOW PAUSED - HUMAN ACTION REQUIRED
INFO:root:============================================================
Instructions:
⚠️ SIGIL CHARGING REQUIRED (Todoist task creation failed)
Manually charge the sigil for '...':
**Wish:** ...
**Glyph:** ...
...charging instructions...
```

## Impact

✅ **No more exception stack traces** - Clean ERROR/WARNING logs only
✅ **Single sigil generation attempt** - No duplicates
✅ **Graceful degradation** - Workflow continues with manual instructions
✅ **Clear user feedback** - Helpful manual instructions displayed
✅ **Workflow pauses correctly** - Can be resumed after manual completion
✅ **Better maintainability** - Status-based error handling pattern

## Next Steps

### To Fix Todoist API Access (Optional)
1. Check API token permissions in Todoist settings
2. Ensure token has write access to project `6CrfWqXrxppjhqMJ`
3. Verify token can create sections and tasks

### To Clean Up Duplicate Sections (Optional)
Manually delete duplicate "Black Agent - Sigil Work" sections in Todoist project.

## Files Modified

1. `/app/reference/mcp/todoist/main.py` - Return status dict instead of raising exceptions
2. `/app/agents/black_agent.py` - Check success flag, fix workflow routing, handle YesOrNo type

