## ADDED Requirements
### Requirement: Pipeline Run Orchestrator
The system SHALL provide a state-aware CLI (`pipeline_runner.py`) that reads
`song_context.yml` phase statuses and drives the pipeline forward without
requiring the operator to remember phase order, flags, or paths.

The orchestrator SHALL support the following subcommands:
- `status` — print per-phase status and next action
- `next` — print the next runnable command without executing it
- `run` — execute the next `pending` phase, update status to `generated`, then stop
- `promote` — wrap `promote_part.py` with a summary confirmation step and update status to `promoted`
- `batch` — run a named phase for all `pending` songs in a thread directory

The orchestrator SHALL never auto-promote; it SHALL always pause for human review
after each `run` and print the exact `promote` command to run next.

Phase dependency order SHALL be: `init_production → chords → drums → bass → melody → lyrics`.

#### Scenario: Status shows per-phase state
- **WHEN** `pipeline status --production-dir <path>` is run
- **THEN** each phase is shown with its current status (pending/generated/promoted)

#### Scenario: Next phase identified
- **WHEN** chords=promoted, drums=pending
- **THEN** `pipeline next` prints the drum pipeline command

#### Scenario: Run advances to generated
- **WHEN** `pipeline run` is invoked with drums pending and chords promoted
- **THEN** drum pipeline is executed and `song_context.yml` drums status is updated to `generated`

#### Scenario: Run stops before auto-promote
- **WHEN** a phase completes via `pipeline run`
- **THEN** the orchestrator stops and prints the promote command — it does NOT auto-promote

#### Scenario: Batch runs pending phases across thread
- **WHEN** `pipeline batch --thread <dir> --phase drums` is run
- **THEN** drum pipeline is run for each production dir where drums=pending

## ADDED Requirements
### Requirement: Phase Status Sync in promote_part
`promote_part.py` SHALL write `promoted` status back to `song_context.yml` after
a successful promotion, so status stays in sync even when promote is called
directly (without using the orchestrator).

#### Scenario: Promote writes status to song_context
- **WHEN** `promote_part` successfully promotes a candidate
- **THEN** `song_context.yml` phases dict for that phase is set to `promoted`
