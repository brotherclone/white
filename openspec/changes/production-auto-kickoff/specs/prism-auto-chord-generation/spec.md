## MODIFIED Requirements

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

## ADDED Requirements

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
