## ADDED Requirements

### Requirement: Album Phase-Completion Matrix
The song dashboard SHALL display a table showing all songs in the album with their
completion status for each pipeline phase (chords, drums, bass, melody, quartet).

Status values SHALL be: `approved` (≥1 approved candidate), `pending` (candidates
present but none approved), `not_started` (no `review.yml` exists for that phase).

#### Scenario: Full album scan
- **WHEN** `song_dashboard` is run with a valid `--album-dir`
- **THEN** every song directory under `shrink_wrapped/` appears as a row in the table

#### Scenario: Phase filter shows incomplete songs only
- **WHEN** `--phase bass` is passed
- **THEN** only songs without an approved bass candidate are shown

#### Scenario: Color filter
- **WHEN** `--color violet` is passed
- **THEN** only violet songs appear in the table

### Requirement: Song Metadata Columns
The dashboard SHALL display singer, key, color, total approved bar count, and whether
a `production_plan.yml` is present for each song.

#### Scenario: Bar count from approved loops
- **WHEN** a song has approved chord, bass, and melody candidates
- **THEN** the `bars` column shows the sum of bar counts across all approved sections

#### Scenario: Plan column reflects file presence
- **WHEN** `production_plan.yml` exists in the song production directory
- **THEN** the plan column shows ✓; otherwise it shows ✗

### Requirement: Read-Only Operation
The song dashboard SHALL make no modifications to any file.

#### Scenario: No writes on scan
- **WHEN** the dashboard is run
- **THEN** no `review.yml`, `production_plan.yml`, or any other file is modified
