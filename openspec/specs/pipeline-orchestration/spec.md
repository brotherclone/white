# pipeline-orchestration Specification

## Purpose
Defines how song metadata flows through the music production pipeline — from the initial
song proposal through chord, drum, bass, melody, and lyric generation phases.
## Requirements
### Requirement: Canonical Song Context
Every production directory SHALL contain a `song_context.yml` written by `init_production.py` before any MIDI phase runs, serving as the single source of truth for static song metadata across all phases. `init_production.py` SHALL NOT write a separate `initial_proposal.yml` — all proposal fields SHALL be present directly in `song_context.yml`.

#### Scenario: song_context.yml written on init — no initial_proposal.yml
- **WHEN** `init_production` is run on a production directory
- **THEN** `song_context.yml` is written containing `title`, `color`, `concept`, `key`, `bpm`, `time_sig`, `singer`, `sounds_like`, `genres`, `mood`, `song_proposal`, `thread`, `proposed_by`, `generated`, and `phases` (all initially `pending`)
- **AND** `initial_proposal.yml` is NOT written

#### Scenario: dirs with both files prefer song_context.yml
- **WHEN** a pipeline phase runs on a directory that has both `song_context.yml` and `initial_proposal.yml`
- **THEN** the phase reads canonical song metadata from `song_context.yml`
- **AND** `initial_proposal.yml` does not override `song_context.yml`
- **AND** no error is raised

#### Scenario: legacy dirs with only initial_proposal.yml still work
- **WHEN** a pipeline phase runs on a legacy directory that has `initial_proposal.yml` but no `song_context.yml`
- **THEN** the phase falls back to `initial_proposal.yml` as the available song metadata source
- **AND** no error is raised

#### Scenario: all pipeline phases read concept from song_context.yml
- **WHEN** drum, bass, or melody pipeline is run on a directory containing `song_context.yml`
- **THEN** the concept embedding is computed from `song_context["concept"]` rather than any fallback string

### Requirement: Unified Song Proposal Loader
A single `load_song_proposal_unified()` function in `production_plan.py` SHALL replace
the four divergent `load_song_proposal` implementations.

#### Scenario: unified loader returns canonical field set
- **WHEN** `load_song_proposal_unified(proposal_path)` is called
- **THEN** the returned dict contains `title`, `bpm`, `time_sig` (always a string),
  `key`, `color`, `concept`, `genres`, `mood`, `singer`, `sounds_like`, `key_root`,
  `mode`

#### Scenario: time_sig is always a string
- **WHEN** any load_song_proposal variant is called
- **THEN** `time_sig` in the returned dict is always a string (e.g. `"4/4"`), never a
  tuple — callers that need integer components parse it themselves

---

### Requirement: Production Dir Migration
A migration script SHALL allow existing production directories to gain `song_context.yml`
without re-running any MIDI generation phase.

#### Scenario: migration is non-destructive
- **WHEN** `migrate_production_dir.py` is run on an existing production directory
- **THEN** no existing files (review.yml, candidates, MIDI files) are modified; only
  `song_context.yml` is created

#### Scenario: migration succeeds from chord_review alone
- **WHEN** a production directory has `chords/review.yml` and a reachable song proposal
  YAML
- **THEN** migration produces a valid `song_context.yml` with all required fields

### Requirement: White Song Proposal — Sub-Proposals Field
The song proposal schema SHALL support an optional `sub_proposals` field: an ordered list
of production directory paths whose approved chord and lyric material is used as source
material for White synthesis.

`load_song_proposal_unified()` SHALL read `sub_proposals` from the proposal YAML and
include it in the returned dict as a list of strings (empty list if absent).
`song_context.yml` SHALL record `sub_proposals` when written by `init_production.py`.

#### Scenario: sub_proposals read from proposal YAML

- **WHEN** a song proposal YAML contains a `sub_proposals` list of path strings
- **THEN** `load_song_proposal_unified()` returns `sub_proposals` as a list of strings
- **AND** `song_context.yml` written by `init_production.py` includes the `sub_proposals` list

#### Scenario: sub_proposals absent defaults to empty list

- **WHEN** a song proposal YAML has no `sub_proposals` field
- **THEN** `load_song_proposal_unified()` returns `sub_proposals: []`
- **AND** non-White pipelines behave identically to before

#### Scenario: White pipeline resolves sub_proposals at runtime

- **WHEN** the chord or lyric pipeline is invoked on a White production directory
- **THEN** `sub_proposals` paths are resolved relative to the project root
- **AND** a clear error is raised for any path that does not exist or has no
  `chords/review.yml`

### Requirement: Phase Status Sync in promote_part
`promote_part.py` SHALL write `promoted` status back to `song_context.yml` after
a successful promotion, so status stays in sync even when promote is called
directly (without using the orchestrator).

#### Scenario: Promote writes status to song_context
- **WHEN** `promote_part` successfully promotes a candidate
- **THEN** `song_context.yml` phases dict for that phase is set to `promoted`

