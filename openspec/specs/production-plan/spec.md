# production-plan Specification

## Purpose
DEPRECATED — `production_plan.yml` is no longer part of the standard workflow.
The arrangement (`arrangement.txt` exported from Logic) is the source of truth
for song structure. Downstream pipelines read from `arrangement.txt` + the song
proposal YAML directly. The code (`production_plan.py`) is retained but not
invoked. See change `update-lyric-pipeline-remove-plan`.
## Requirements
### Requirement: Production Plan Generation

The system SHALL provide a command that generates a `production_plan.yml` file in a song's production directory, bootstrapped from the approved chord sections and song proposal.

#### Scenario: Generate from approved chords

- **WHEN** `python -m app.generators.midi.production_plan --production-dir <path>` is run
- **AND** the song has an approved chord `review.yml` with at least one approved candidate
- **THEN** a `production_plan.yml` is written to the production directory root
- **AND** it contains one section entry per unique approved chord label
- **AND** bar counts are derived from the `hr_distribution` field in the chord `review.yml` if present, otherwise from approved chord MIDI length, otherwise from chord count in the candidate
- **AND** all sections default to `play_count: 1` and `vocals: false`
- **AND** sections appear in the order they were labeled in the chord review

#### Scenario: Refresh existing plan

- **WHEN** `--refresh` flag is passed and a `production_plan.yml` already exists
- **THEN** bar counts are recalculated from current approved loops
- **AND** all human-edited fields (`play_count`, `vocals`, `notes`, `loops`, section order) are preserved
- **AND** sections present in the plan but no longer in approved chords are flagged with a warning but retained

#### Scenario: No approved chords

- **WHEN** no approved chord candidates exist
- **THEN** the command exits with an error message and does not write a plan

### Requirement: Production Plan Schema
The `ProductionPlan` dataclass SHALL include the following fields:
- `sections: list[PlanSection]` — ordered section entries
- `rationale: str` — top-level compositional reasoning (empty string for mechanical plans)
- `proposed_by: str` — `"claude"` when AI-authored, empty string for mechanical plans

The `PlanSection` dataclass SHALL include:
- `name: str`, `bars: int`, `play_count: int`, `vocals: bool`, `notes: str`, `loops: dict`
- `reason: str` — one-sentence note on placement (empty string for mechanical plans)

All fields SHALL survive a YAML save/load round-trip with no data loss.

#### Scenario: Round-trip preserves rationale and reasons
- **WHEN** a Claude-authored plan is saved to YAML and reloaded
- **THEN** `rationale`, `proposed_by`, and all per-section `reason` fields are identical
  to the original

#### Scenario: Refresh preserves human edits
- **WHEN** `refresh_plan()` is called on a plan where the user has manually edited
  `play_count`, `reason`, or section order
- **THEN** those edits are preserved and only `bars` is updated from the loop inventory

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

