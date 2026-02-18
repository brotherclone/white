# production-review Specification

## Purpose
TBD - created by archiving change add-music-production-pipeline. Update Purpose after archive.
## Requirements
### Requirement: Review File Generation

The chord pipeline SHALL generate a YAML review file alongside the MIDI candidates, listing each candidate with its scores and placeholders for human annotation.

#### Scenario: Review file creation

- **WHEN** the chord pipeline completes scoring and MIDI output
- **THEN** it SHALL write a `review.yml` file in the song's chords directory
- **AND** the file SHALL contain metadata (song proposal path, thread, generation timestamp, seed, scoring weights)
- **AND** each candidate entry SHALL include: id, midi file path, rank, composite score, theory score breakdown, chromatic score breakdown, and a human-readable progression summary

#### Scenario: Annotation placeholders

- **WHEN** the review file is generated
- **THEN** each candidate SHALL have `label: null`, `status: pending`, and `notes: ""` fields for the human to fill in

### Requirement: Human Labeling

The human SHALL label candidates in the review YAML file by editing the `label` and `status` fields.

#### Scenario: Structural labeling

- **WHEN** the human reviews a chord candidate
- **THEN** they MAY set `label` to one of: `verse-candidate`, `chorus-candidate`, `bridge-candidate`, `intro-candidate`, `outro-candidate`, or a custom label

#### Scenario: Approval status

- **WHEN** the human reviews a chord candidate
- **THEN** they SHALL set `status` to `approved`, `rejected`, or leave as `pending`

#### Scenario: Freeform notes

- **WHEN** the human reviews a chord candidate
- **THEN** they MAY add freeform text to the `notes` field for context or revision requests

### Requirement: Approved Candidate Promotion

Approved candidates SHALL be promoted to the song's approved chords directory for use by subsequent pipeline phases.

#### Scenario: Copy on approval

- **WHEN** a promotion command is run against a review file
- **THEN** all candidates with `status: approved` SHALL have their MIDI files copied to `<song>/chords/approved/`
- **AND** the file SHALL be renamed to match the label (e.g., `verse.mid`, `chorus.mid`)

#### Scenario: Multiple candidates per label

- **WHEN** multiple candidates share the same label and are approved
- **THEN** they SHALL be numbered (e.g., `verse_1.mid`, `verse_2.mid`)

#### Scenario: Promotion summary

- **WHEN** promotion completes
- **THEN** the command SHALL print a summary of promoted files and their labels

