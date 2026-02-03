# Tasks: Fix ExtinctionCause Enum

## Phase 1: Fix Data Loading (COMPLETED)

- [x] **1.1** Remove ExtinctionCause enum conversion in `extinction_tools.py` - corpus has free-form descriptions
- [x] **1.2** Change `size_category` from Literal to str - corpus has "colonial/modular" for coral
- [x] **1.3** Update test to reflect new flexible types

## Phase 2: Add Resilience (Optional - Not Needed)

- [~] **2.1** ~~Add fallback to `UNKNOWN` in green_agent species loader~~ - Not needed, string passthrough works
- [~] **2.2** ~~Log warning instead of error for unknown causes~~ - Not needed

## Summary

The issue wasn't missing enum values - it was that `extinction_tools.py` was forcing enum conversion when the artifact already accepts strings. The corpus intentionally has:
- Free-form extinction causes like "Ship strike mortality", "ocean_warming"
- Non-standard size categories like "colonial/modular"

Fixed by removing unnecessary enum conversion and relaxing the Literal type.

## Validation

```bash
# All 6 species now load
python -c "from app.agents.tools.extinction_tools import load_green_corpus, get_random_species; [print(get_random_species(load_green_corpus()).common_name) for _ in range(6)]"

# Tests pass
pytest tests/ -k "extinction or species or green" -v
```
