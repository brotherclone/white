## ADDED Requirements

### Requirement: Production Plan Generation

The system SHALL provide a command that generates a `production_plan.yml` file in a song's production directory, bootstrapped from the approved chord sections and song proposal.

#### Scenario: Generate from approved chords

- **WHEN** `python -m app.generators.midi.production_plan --production-dir <path>` is run
- **AND** the song has an approved chord `review.yml` with at least one approved candidate
- **THEN** a `production_plan.yml` is written to the production directory root
- **AND** it contains one section entry per unique approved chord label
- **AND** bar counts are derived from approved harmonic rhythm MIDI if present, otherwise from approved chord MIDI, otherwise from chord count
- **AND** all sections default to `repeat: 1` and `vocals: false`
- **AND** sections appear in the order they were labeled in the chord review

#### Scenario: Refresh existing plan

- **WHEN** `--refresh` flag is passed and a `production_plan.yml` already exists
- **THEN** bar counts are recalculated from current approved loops
- **AND** all human-edited fields (`repeat`, `vocals`, `notes`, `sounds_like`, section order) are preserved
- **AND** sections present in the plan but no longer in approved chords are flagged with a warning but retained

#### Scenario: No approved chords

- **WHEN** no approved chord candidates exist
- **THEN** the command exits with an error message and does not write a plan

### Requirement: Production Plan Schema

The `production_plan.yml` SHALL conform to a defined schema capturing song structure and production metadata.

#### Scenario: Required top-level fields

- **WHEN** a production plan is generated
- **THEN** it SHALL contain: `song_slug`, `generated` (ISO timestamp), `bpm`, `time_sig`, `key`, `color`, `vocals_planned`, `sounds_like`, and `sections`

#### Scenario: Section entry fields

- **WHEN** a section is written to the plan
- **THEN** each section entry SHALL contain: `name`, `bars` (integer), `repeat` (integer ≥ 1), `vocals` (bool), and `notes` (string, may be empty)

#### Scenario: Human-editable

- **WHEN** the human edits `production_plan.yml` directly
- **THEN** the file remains valid and is read correctly by downstream phases
- **AND** the human MAY reorder sections, change `repeat`, `vocals`, `notes`, and `sounds_like` without breaking any pipeline

### Requirement: Drum Pipeline Section Context

When a `production_plan.yml` exists, the drum pipeline SHALL annotate each candidate in `drums/review.yml` with the name of the section that follows it in the song arrangement.

#### Scenario: next_section annotation present

- **WHEN** `drums/review.yml` is generated and `production_plan.yml` exists
- **THEN** each candidate entry SHALL include a `next_section` field containing the name of the following section (or `null` if it is the last section)

#### Scenario: No production plan

- **WHEN** `production_plan.yml` does not exist
- **THEN** the drum pipeline proceeds without `next_section` annotations
- **AND** no error is raised

### Requirement: Manifest Bootstrap

The system SHALL provide a command that reads a completed `production_plan.yml` and emits a partial `Manifest`-compatible YAML with all derivable fields pre-filled.

#### Scenario: Bootstrap manifest fields

- **WHEN** `python -m app.generators.midi.production_plan --production-dir <path> --bootstrap-manifest` is run
- **AND** a `production_plan.yml` exists
- **THEN** a `manifest_bootstrap.yml` is written to the production directory
- **AND** it SHALL contain: `bpm`, `tempo`, `key`, `rainbow_color`, `title`, `vocals`, `sounds_like`, and `structure` (sections with bar counts converted to approximate durations in seconds at the song BPM)
- **AND** fields that require a final render (`main_audio_file`, `TRT`, `release_date`, `album_sequence`, `audio_tracks`, `lrc_file`) SHALL be present with `null` values and a comment indicating they are filled at render time
