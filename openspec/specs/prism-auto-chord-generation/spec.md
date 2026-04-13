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
`finalize_song_proposal` SHALL call `_invoke_chord_pipeline_safe` after `state.run_finished` is set to `True` for **every** `is_final=True` proposal iteration in `state.song_proposals.iterations`. White's proposal SHALL be processed last. It MUST NOT call the chord pipeline in MOCK_MODE or when no `is_final` iterations exist.

#### Scenario: all final proposals get chord generation
- **WHEN** `finalize_song_proposal` runs with `auto_chord_generation=True`
- **AND** `state.song_proposals.iterations` contains multiple `is_final=True` proposals
- **THEN** `_invoke_chord_pipeline_safe` is called once per `is_final=True` iteration
- **AND** the White proposal is processed after all non-White proposals

#### Scenario: White is last
- **WHEN** both a non-White and White `is_final=True` proposal exist
- **THEN** the non-White proposal's chord pipeline runs before White's

#### Scenario: skipped when MOCK_MODE is true
- **WHEN** env var `MOCK_MODE=true`
- **THEN** `_invoke_chord_pipeline_safe` is NOT called for any proposal

#### Scenario: skipped when no is_final iterations exist
- **WHEN** `state.song_proposals.iterations` has no `is_final=True` entries
- **THEN** `_invoke_chord_pipeline_safe` is NOT called
- **AND** a WARNING is logged

### Requirement: chord output written inside chain_artifacts thread directory

The chord pipeline MUST write `production/<slug>/chords/` inside the chain_artifacts
thread directory. Shrinkwrap SHALL pick this up automatically on the next run.

#### Scenario: production directory created under chain_artifacts thread

Given chord pipeline succeeds
When output is written
Then a directory exists at `<AGENT_WORK_PRODUCT_BASE_PATH>/<thread_id>/production/<slug>/chords/candidates/`
And `review.yml` exists at `<AGENT_WORK_PRODUCT_BASE_PATH>/<thread_id>/production/<slug>/chords/review.yml`

### Requirement: Browser auto-launch after chord generation
After all chord pipelines complete, Prism SHALL check whether the candidate review servers are running and launch them if not, then open the browser to the first non-White song's chord review. This behaviour SHALL be gated behind `AUTO_BROWSER_LAUNCH=true` and SHALL be suppressed when `MOCK_MODE=true`.

#### Scenario: servers already running — browser opens directly
- **WHEN** ports 8000 and 3000 are both already listening
- **THEN** no new processes are launched
- **AND** the browser opens to `http://localhost:3000?production-dir=<first-non-white-dir>&phase=chords`

#### Scenario: FastAPI server not running — launched automatically
- **WHEN** port 8000 is not listening
- **THEN** `candidate_server.py` is launched as a non-blocking subprocess with the first non-White production dir
- **AND** Prism waits up to 5 seconds for port 8000 to respond before opening the browser

#### Scenario: Next.js server not running — launched automatically
- **WHEN** port 3000 is not listening
- **THEN** `npm run dev` is launched in `web/` as a non-blocking subprocess

#### Scenario: auto-launch suppressed with --no-browser
- **WHEN** `run_white_agent start --no-browser` is invoked
- **THEN** `AUTO_BROWSER_LAUNCH` is not set and no servers are launched or browser opened

#### Scenario: auto-launch suppressed in MOCK_MODE
- **WHEN** `MOCK_MODE=true`
- **THEN** browser auto-launch is skipped regardless of `AUTO_BROWSER_LAUNCH`

