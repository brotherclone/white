## MODIFIED Requirements

### Requirement: Production Plan Generation
The system SHALL provide a command that generates a `production_plan.yml` file in a song's production directory, bootstrapped from the approved chord sections and song proposal. `vocals_planned` SHALL default to `True`. Each `PlanSection` SHALL default `vocals` to `True` unless its label matches a conventionally instrumental type (`intro`, `outro`, `instrumental`, `solo`, `interlude`, `break`), in which case `vocals` SHALL default to `False`.

#### Scenario: Generate from approved chords
- **WHEN** `python -m app.generators.midi.production_plan --production-dir <path>` is run
- **AND** the song has an approved chord `review.yml` with at least one approved candidate
- **THEN** a `production_plan.yml` is written to the production directory root
- **AND** it contains one section entry per unique approved chord label
- **AND** bar counts are derived from the `hr_distribution` field in the chord `review.yml` if present, otherwise from approved chord MIDI length, otherwise from chord count in the candidate
- **AND** `vocals_planned` is `true`
- **AND** sections whose label matches an instrumental type have `vocals: false`; all others have `vocals: true`
- **AND** sections appear in the order they were labeled in the chord review

#### Scenario: Refresh existing plan
- **WHEN** `--refresh` flag is passed and a `production_plan.yml` already exists
- **THEN** bar counts are recalculated from current approved loops
- **AND** all human-edited fields (`play_count`, `vocals`, `vocals_planned`, `notes`, `loops`, section order) are preserved
- **AND** sections present in the plan but no longer in approved chords are flagged with a warning but retained

#### Scenario: No approved chords
- **WHEN** no approved chord candidates exist
- **THEN** the command exits with an error message and does not write a plan
