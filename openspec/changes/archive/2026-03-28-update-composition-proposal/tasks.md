## 1. Schema
- [x] 1.1 Add `rationale: str` field to `ProductionPlan` dataclass (empty string default)
- [x] 1.2 Add `proposed_by: str` field to `ProductionPlan` (default `"mechanical"`)
- [x] 1.3 Add `reason: str` field to `PlanSection` dataclass (empty string default)
- [x] 1.4 Update YAML save/load round-trip to include all three new fields

## 2. Claude Composition Step
- [x] 2.1 Write `_build_composition_prompt(proposal, loop_inventory) → str`
       — includes color/mood/concept/key/singer, loop list with bar counts and energy labels
- [x] 2.2 Write `_parse_claude_plan(response_text, existing_plan) → ProductionPlan`
       — extracts section order, repeat counts, per-section reason, top-level rationale
- [x] 2.3 Wrap in `generate_plan(production_dir, song_proposal_path, use_claude=True)`
       — falls back to mechanical if `use_claude=False` or API unavailable

## 3. Refresh Compatibility
- [x] 3.1 `refresh_plan()`: preserve `reason` and `proposed_by` fields on reload
- [x] 3.2 Warn (not error) if a section in the plan has no matching approved loop

## 4. Tests
- [x] 4.1 Update `test_production_plan.py`: mock Claude response, assert `rationale` and
       `proposed_by: claude` present
- [x] 4.2 Assert `refresh_plan()` preserves `reason` fields from prior plan
- [x] 4.3 `generate_plan_mechanical()` still passes all existing tests unchanged
