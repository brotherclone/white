# prism-auto-chord-generation Specification

## Purpose
TBD - created by archiving change auto-chord-kickoff. Update Purpose after archive.
## Requirements
### Requirement: auto_chord_generation flag on start_workflow

`WhiteAgent.start_workflow()` SHALL accept an `auto_chord_generation: bool = False`
keyword argument. When `True`, or when env var `AUTO_CHORD_GENERATION=true` is set,
chord generation MUST run after finalization. The explicit flag takes precedence over
the env var.

#### Scenario: default off — chord pipeline not called

Given `start_workflow()` is called without `auto_chord_generation`
When the workflow completes
Then `run_chord_pipeline` is not called

#### Scenario: flag enabled — chord pipeline called

Given `start_workflow(auto_chord_generation=True)` is called
When `finalize_song_proposal` completes successfully
Then `_invoke_chord_pipeline_safe` is called once with the correct thread dir
and proposal filename

#### Scenario: env var enables auto chord generation

Given env var `AUTO_CHORD_GENERATION=true` is set
And `start_workflow()` is called without the explicit flag
When the workflow completes
Then `_invoke_chord_pipeline_safe` is called

---

### Requirement: _invoke_chord_pipeline_safe wrapper

`WhiteAgent` MUST expose a private method `_invoke_chord_pipeline_safe(thread_dir, song_filename)`
that calls `run_chord_pipeline()` in-process and SHALL never raise or call `sys.exit`.
All `Exception` and `SystemExit` instances MUST be caught, logged as WARNING, and
silently discarded.

#### Scenario: successful run — output path logged

Given chord pipeline runs without error
When `_invoke_chord_pipeline_safe` returns
Then a log message at INFO level records the `production/` output directory

#### Scenario: chord pipeline raises exception — non-fatal

Given `run_chord_pipeline` raises any exception
When `_invoke_chord_pipeline_safe` catches it
Then a WARNING is logged with the error detail
And no exception propagates to the caller

#### Scenario: chord pipeline calls sys.exit — non-fatal

Given `run_chord_pipeline` internally calls `sys.exit(1)` (e.g. no candidates)
When `_invoke_chord_pipeline_safe` catches the `SystemExit`
Then a WARNING is logged
And the process does not exit

---

### Requirement: finalize_song_proposal triggers chord generation

`finalize_song_proposal` SHALL call `_invoke_chord_pipeline_safe` after
`state.run_finished` is set to `True`, using the chain_artifacts thread directory
and the final proposal's filename. It MUST NOT call the chord pipeline in MOCK_MODE
or when no proposal iterations exist.

#### Scenario: correct paths derived from state

Given `state.thread_id` is a valid UUID
And `state.song_proposals.iterations[-1]` has `rainbow_color` and `iteration_id`
When `finalize_song_proposal` runs with `auto_chord_generation=True`
Then `_invoke_chord_pipeline_safe` receives:
- `thread_dir` = `<AGENT_WORK_PRODUCT_BASE_PATH>/<thread_id>`
- `song_filename` = `song_proposal_<rainbow_color>_<iteration_id>.yml`

#### Scenario: skipped when MOCK_MODE is true

Given env var `MOCK_MODE=true`
When `finalize_song_proposal` runs with `auto_chord_generation=True`
Then `_invoke_chord_pipeline_safe` is NOT called

#### Scenario: skipped when no iterations exist

Given `state.song_proposals.iterations` is empty
When `finalize_song_proposal` runs with `auto_chord_generation=True`
Then `_invoke_chord_pipeline_safe` is NOT called
And a WARNING is logged

#### Scenario: run_finished remains True even if chord gen fails

Given `run_chord_pipeline` raises an exception
When `finalize_song_proposal` completes
Then `state.run_finished` is `True`

---

### Requirement: chord output written inside chain_artifacts thread directory

The chord pipeline MUST write `production/<slug>/chords/` inside the chain_artifacts
thread directory. Shrinkwrap SHALL pick this up automatically on the next run.

#### Scenario: production directory created under chain_artifacts thread

Given chord pipeline succeeds
When output is written
Then a directory exists at `<AGENT_WORK_PRODUCT_BASE_PATH>/<thread_id>/production/<slug>/chords/candidates/`
And `review.yml` exists at `<AGENT_WORK_PRODUCT_BASE_PATH>/<thread_id>/production/<slug>/chords/review.yml`

