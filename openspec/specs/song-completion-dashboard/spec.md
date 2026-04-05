# song-completion-dashboard Specification

## Purpose
Define a read-only dashboard that summarizes each song's pipeline completion status
and key production metadata across an album so users can quickly identify progress
and remaining work by song and phase.
## Requirements
### Requirement: Album Phase-Completion Matrix
The song dashboard SHALL display a table showing all songs in the album with their
completion status for each pipeline phase (chords, drums, bass, melody, quartet).

Status values SHALL be: `approved` (≥1 approved or accepted candidate), `pending`
(candidates present but none approved), `no_candidates` (phase directory exists but
`review.yml` is missing, empty, or unreadable), `not_started` (phase directory does
not exist).

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

#### Scenario: Bar count from production plan
- **WHEN** a song has a `production_plan.yml` with defined sections
- **THEN** the `bars` column shows the sum of `bars * play_count` across all plan sections

#### Scenario: Plan column reflects file presence
- **WHEN** `production_plan.yml` exists in the song production directory
- **THEN** the plan column shows ✓; otherwise it shows —

### Requirement: Read-Only Operation
The song dashboard SHALL make no modifications to any file.

#### Scenario: No writes on scan
- **WHEN** the dashboard is run
- **THEN** no `review.yml`, `production_plan.yml`, or any other file is modified

