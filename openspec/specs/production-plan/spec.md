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
The `PlanSection` dataclass SHALL include an `arc` field: a float 0.0–1.0
representing intended emotional intensity for that section (0.0 = near-silence,
1.0 = peak). The field SHALL default to 0.0 and be serialised as a plain float
in YAML. It SHALL be populated by `_infer_arc_from_label` when `generate_plan`
creates sections, and preserved by `refresh_plan` (human overrides survive refresh).

`_infer_arc_from_label(label: str) → float` SHALL return:
- `intro`, `outro` → 0.15
- `verse` → 0.35
- `pre_chorus` → 0.55
- `chorus`, `refrain`, `hook` → 0.75
- `bridge` → 0.20
- `climax`, `peak` → 0.90
- anything else → 0.40

#### Scenario: Arc field round-trips through YAML
- **WHEN** a plan is saved and reloaded
- **THEN** `arc` values are preserved as floats

#### Scenario: Arc auto-seeded from label
- **WHEN** `generate_plan` creates sections
- **THEN** chorus sections have arc > 0.6 and bridge sections have arc < 0.3

#### Scenario: Human override survives refresh
- **WHEN** a human sets arc=0.9 on a verse and `refresh_plan` is called
- **THEN** the verse arc remains 0.9 after refresh

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

