# Tasks: redesign-melody-templates

## Ordered Work Items

1. **Add `use_case` field to `MelodyPattern` dataclass**
   - Add `use_case: str = "vocal"` field with default "vocal"
   - Update `select_templates()` to filter `use_case == "vocal"` when generating singer parts
   - Update `make_fallback_pattern()` to include `use_case="vocal"`
   - Validate: existing tests still pass; new `use_case` field present on all patterns

2. **Reclassify existing templates as `"lead"`**
   - Audit all 31 current templates for note density (> 6 onsets/bar in 4/4 = lead candidate)
   - Mark dense/ornamental templates `use_case="lead"` (expect ~12-15)
   - Validate: after reclassification, at least 16 vocal templates remain

3. **Write new vocal templates — 4/4, target ≥ 30 total vocal**
   - Add templates per archetype (≥ 4 per archetype):
     - `declarative`: one long phrase, ~1 note/beat, held resolution
     - `call_and_rest`: phrase + silence + phrase (Nebraska phrasing)
     - `haiku`: three short motifs with rests between
     - `incantatory`: minimal interval movement, repeated short motif
     - `drone_and_step`: held root + one descent or ascent (Lankum)
     - `conversational`: irregular durations, speech-rhythm onsets
   - Each template: ≤ 6 onsets/bar, ≥ 1 rest of 0.5+ beats, ≥ 1 note of 1.5+ beats
   - Validate: `test_melody_patterns.py` — add density assertions for vocal templates

4. **Write new vocal templates — 3/4 and 6/8 (≥ 4 each)**
   - Waltz-feel and compound meter vocal patterns
   - Validate: pipeline generates output for 3/4 and 6/8 song proposals

5. **Update singability scoring**
   - Penalise > 6 onsets/bar in 4/4 (new density component)
   - Reward held notes (duration ≥ 1.5 beats)
   - Require ≥ 1 rest per bar (not per 4 bars as currently)
   - Rebalance: `singability = (interval + range + rest + density + hold) / 5`
   - Validate: unit tests for each scoring component; dense patterns score lower than sparse

6. **Update `review.yml` output to include `use_case` per candidate**
   - Add `use_case` field to candidate dict in melody pipeline output
   - Validate: review.yml contains `use_case: vocal` for all generated candidates

7. **Regression: run full melody pipeline on Supertanker Townie**
   - Run melody pipeline on indigo song with new templates
   - Confirm top candidates are sparse/singable (subjective check)
   - Promote new melody loops, compare to prior attempt

8. **Update tests**
   - `test_melody_patterns.py`: assert vocal templates have ≤ 6 onsets/bar and ≥ 1 rest
   - `test_melody_pipeline.py`: assert `use_case` field present in review output
   - All existing tests pass

## Dependencies

- Tasks 1–2 must complete before 3–5 (need the field to reclassify/author)
- Task 5 (scoring) can run in parallel with tasks 3–4 (template authoring)
- Task 6 depends on task 1 (field exists)
- Task 7 depends on all prior tasks
- Task 8 can be written in parallel with implementation

## Notes

- No changes to bass, drum, or chord pipelines
- No changes to promotion workflow or composite scoring weights
- Lead templates are preserved in the library for future instrument track work
- Supertanker Townie melody phase should be re-run once this change is applied
