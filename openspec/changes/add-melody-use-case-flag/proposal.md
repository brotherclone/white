# Change: Add vocal/instrumental use-case flag to melody candidates

## Why
Melody candidates are generated for both vocal lines and instrumental hooks, but this
distinction is lost when candidates land in `review.yml`. Arrangers and ACE Studio
import workflows have no machine-readable signal to filter by intent.

## What Changes
- Propagate `use_case` field (`vocal` | `instrumental`) from `MelodyPattern` into each
  candidate entry written to `review.yml`
- Approved entries in the promoted loop list also carry `use_case`
- No scoring or generation logic changes — purely additive metadata

## Impact
- Affected specs: melody-generation
- Affected code: `app/generators/midi/pipelines/melody_pipeline.py`, `app/generators/midi/production/promote_part.py`
