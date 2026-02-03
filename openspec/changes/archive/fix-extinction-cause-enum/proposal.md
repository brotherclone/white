# Change: Fix ExtinctionCause Enum Missing Values

## Why

Green agent fails when loading species data with extinction causes not in the enum:

```
ERROR:app.agents.green_agent:Failed to load species extinction artifact: 'ocean_warming' is not a valid ExtinctionCause
```

The enum has `ocean_acidification` but not `ocean_warming`. LLM-generated species data may use terms not in the predefined enum.

## What Changes

### Option A: Expand Enum (Recommended)
Add missing legitimate extinction causes to `ExtinctionCause`:
- `OCEAN_WARMING = "ocean_warming"`
- Audit for other missing common causes

### Option B: Add Fallback Mapping
Create alias mapping for close-enough terms:
```python
CAUSE_ALIASES = {
    "ocean_warming": ExtinctionCause.CLIMATE_CHANGE,
}
```

### Option C: Graceful Degradation
Fall back to `UNKNOWN` when cause isn't recognized instead of raising.

## Impact

- Affected code: `app/structures/enums/extinction_cause.py`, possibly `green_agent.py`
- Prevents runtime errors during species artifact loading
- Backward compatible
