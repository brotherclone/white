# pipeline-orchestration Specification

## Purpose
Defines how song metadata flows through the music production pipeline — from the initial
song proposal through chord, drum, bass, melody, and lyric generation phases.

## Requirements

### Requirement: Canonical Song Context
Every production directory SHALL contain a `song_context.yml` written by
`init_production.py` before any MIDI phase runs, serving as the single source of truth
for static song metadata across all phases.

#### Scenario: song_context.yml written on init
- **WHEN** `init_production` is run on a production directory
- **THEN** `song_context.yml` is written containing `title`, `color`, `concept`, `key`,
  `bpm`, `time_sig`, `singer`, `sounds_like`, `genres`, `mood`, `song_proposal`,
  `thread`, and `phases` (all initially `pending`)

#### Scenario: all pipeline phases read concept from song_context.yml
- **WHEN** drum, bass, or melody pipeline is run on a directory containing
  `song_context.yml`
- **THEN** the concept embedding is computed from `song_context["concept"]` rather than
  the fallback `f"{color_name} chromatic concept"` string

#### Scenario: graceful fallback for pre-migration dirs
- **WHEN** a pipeline phase runs on a directory that has no `song_context.yml`
- **THEN** the phase falls back to the previous behavior (chord_review + proposal YAML
  navigation) without error

---

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

