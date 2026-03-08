# Proposal: auto-chord-kickoff

## Summary

After The Prism finalises a song concept, automatically invoke the chord generation
pipeline so the first musical artefact (chord candidates + `review.yml`) is ready
for human review without a manual CLI step.

This is the first step toward a fully automated production pipeline. It is
deliberately minimal: one opt-in flag, one safe wrapper call, no new agents, no
changes to the concept chain or the chord pipeline internals.

## Motivation

Currently the workflow is:
1. Run The Prism → song proposal written to `chain_artifacts/<thread>/yml/`
2. Manually shrinkwrap
3. Manually run `python -m app.generators.midi.pipelines.chord_pipeline --thread ...`

Step 3 is pure boilerplate. The Prism already knows everything the chord pipeline
needs at the moment `finalize_song_proposal` completes: the thread path, the final
proposal's color, and its `iteration_id` (which forms the filename).

## Scope

**In scope**
- Add `auto_chord_generation: bool = False` parameter to `WhiteAgent.start_workflow()`
- Add a private `_invoke_chord_pipeline_safe()` helper on `WhiteAgent` that calls
  `run_chord_pipeline()` in-process, redirecting `print()` to the logger and
  converting any `SystemExit` to a logged warning (non-fatal)
- Call the helper at the end of `finalize_song_proposal` when the flag is set
- Honour `MOCK_MODE` (skip in mock)
- Log the output path on success

**Out of scope**
- Per-color processor refactor (separate future change)
- Automated drum / bass / melody generation
- Any changes to `chord_pipeline.py` internals
- Any new orchestration agent

## Key Design Decisions

### Call against `chain_artifacts/`, not `shrink_wrapped/`

The chord pipeline just needs a directory with `yml/<proposal>.yml`. The
chain_artifacts thread dir satisfies this. Writing `production/` into
`chain_artifacts/<thread_id>/production/` is fine — shrinkwrap will include it
on the next run.

This avoids the chicken-and-egg of needing to shrinkwrap before generating chords.

### Proposal filename construction

The final proposal in chain_artifacts is saved as:
```
song_proposal_{iteration.rainbow_color}_{iteration.iteration_id}.yml
```
Both `rainbow_color` and `iteration_id` are available on
`state.song_proposals.iterations[-1]` at the end of `finalize_song_proposal`.

### Non-fatal by design

A chord generation failure must never lose the concept chain result. The wrapper
catches all exceptions (including `SystemExit`) and logs them as warnings.
`state.run_finished` is set to `True` before the chord pipeline is called.

### `AUTO_CHORD_GENERATION` env var

For convenience, the env var `AUTO_CHORD_GENERATION=true` enables the feature
without changing call sites (e.g. for the CLI runner). The `start_workflow` flag
takes precedence when explicitly passed.

## Files Affected

- `app/agents/white_agent.py` — `start_workflow`, `finalize_song_proposal`, new
  `_invoke_chord_pipeline_safe` method
- `tests/agents/test_white_agent_chord_kickoff.py` — new test file

## Open Questions

None — scope is well-bounded.
