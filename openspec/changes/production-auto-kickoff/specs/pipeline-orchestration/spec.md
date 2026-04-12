## MODIFIED Requirements

### Requirement: Canonical Song Context
Every production directory SHALL contain a `song_context.yml` written by `init_production.py` before any MIDI phase runs, serving as the single source of truth for static song metadata across all phases. `init_production.py` SHALL NOT write a separate `initial_proposal.yml` — all proposal fields SHALL be present directly in `song_context.yml`.

#### Scenario: song_context.yml written on init — no initial_proposal.yml
- **WHEN** `init_production` is run on a production directory
- **THEN** `song_context.yml` is written containing `title`, `color`, `concept`, `key`, `bpm`, `time_sig`, `singer`, `sounds_like`, `genres`, `mood`, `song_proposal`, `thread`, `proposed_by`, `generated`, and `phases` (all initially `pending`)
- **AND** `initial_proposal.yml` is NOT written

#### Scenario: existing dirs with initial_proposal.yml unaffected
- **WHEN** a pipeline phase runs on a directory that already has `initial_proposal.yml`
- **THEN** the phase reads from `song_context.yml` and ignores `initial_proposal.yml`
- **AND** no error is raised

#### Scenario: all pipeline phases read concept from song_context.yml
- **WHEN** drum, bass, or melody pipeline is run on a directory containing `song_context.yml`
- **THEN** the concept embedding is computed from `song_context["concept"]` rather than any fallback string
