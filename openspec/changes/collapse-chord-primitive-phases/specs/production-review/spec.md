## MODIFIED Requirements

### Requirement: Review File Generation

The chord pipeline SHALL generate a YAML review file alongside the MIDI candidates, listing each
candidate with its scores and placeholders for human annotation. Scratch beat companions SHALL be
listed but flagged as non-promotable.

#### Scenario: Review file creation

- **WHEN** the chord pipeline completes scoring and MIDI output
- **THEN** it SHALL write a `review.yml` file in the song's chords directory
- **AND** the file SHALL contain metadata (song proposal path, thread, generation timestamp, seed, scoring weights)
- **AND** each candidate entry SHALL include: id, midi file path, scratch_midi file path, rank,
  composite score, theory score breakdown, chromatic score breakdown, HR distribution, strum
  pattern name, and a human-readable progression summary

#### Scenario: Annotation placeholders

- **WHEN** the review file is generated
- **THEN** each candidate SHALL have `label: null`, `status: pending`, and `notes: ""` fields for
  the human to fill in
- **AND** each candidate SHALL have a `scratch_midi` field pointing to its companion scratch beat

### Requirement: Approved Candidate Promotion

Approved candidates SHALL be promoted to the song's approved chords directory. Promotion enforces
one file per section label; scratch beats are saved but never promoted.

#### Scenario: Copy on approval

- **WHEN** a promotion command is run against a review file
- **THEN** all candidates with `status: approved` SHALL have their MIDI files copied to
  `<song>/chords/approved/`
- **AND** the file SHALL be renamed to match the label (e.g., `verse.mid`, `chorus.mid`)

#### Scenario: One approved per label — strict enforcement

- **WHEN** a promotion command is run and two or more approved candidates share the same label
- **THEN** promotion SHALL fail with an error listing the conflicting candidates
- **AND** the user SHALL resolve the conflict by rejecting all but one before re-running promotion

#### Scenario: Scratch beats excluded from promotion

- **WHEN** promotion runs
- **THEN** scratch beat MIDI files SHALL NOT be copied to `approved/`
- **AND** scratch files SHALL be retained in `candidates/` for reference

#### Scenario: Promotion summary

- **WHEN** promotion completes
- **THEN** the command SHALL print a summary of promoted files and their labels
