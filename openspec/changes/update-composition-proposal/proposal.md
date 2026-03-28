# Change: Rework production plan as genuine Claude composition proposal

## Why
The current `production_plan.py` simply lists approved sections in label order with
`play_count=1`. It is a mechanical inventory, not a compositional act. The drift report
(which compares the plan to the final Logic arrangement) is therefore meaningless —
there is nothing to drift from. Making Claude author a real arrangement arc gives the
drift report teeth and makes the plan a genuine creative artefact.

## What Changes
- `generate_plan()` calls the Claude API with the full song proposal (concept, mood,
  color, singer, key) plus the approved loop inventory (sections, bar lengths, energy
  scores) and asks for an arrangement arc: intro/verse/chorus ordering, repeat strategy,
  dynamic arc, rationale
- Response is parsed into `ProductionPlan` with a new `rationale` field (string) and
  `proposed_by: claude` marker
- `PlanSection` gains a `reason` field (one-sentence Claude note per section placement)
- `refresh_plan()` preserves human edits to `play_count`, `reason`, and section order while
  re-syncing bar counts
- Old mechanical `generate_plan()` becomes `generate_plan_mechanical()` (kept for tests)

## Impact
- Affected specs: composition-proposal, production-plan
- Affected code: `app/generators/midi/production_plan.py`
- Requires Anthropic SDK available in environment (already present)
- Not breaking — `refresh_plan()` still works on existing plan files
