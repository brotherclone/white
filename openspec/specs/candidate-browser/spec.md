# candidate-browser Specification

## Purpose
TBD - created by archiving change add-candidate-browser. Update Purpose after archive.
## Requirements
### Requirement: Candidate Listing
The candidate browser SHALL load and display all pipeline candidates (chords, drums, bass,
melody, quartet) from a given `production_dir`, grouped by phase and section.
Non-generated entries (those with `generated: false` in review.yml) SHALL be displayed
with a `[H]` marker in place of score columns. Null values for `scores` and `rank` in
review.yml SHALL not cause errors.

#### Scenario: All phases shown
- **WHEN** the browser is launched with a `--production-dir` pointing to a song with
  candidates across multiple phases
- **THEN** all phases with at least one candidate appear as groups in the display

#### Scenario: Phase filter
- **WHEN** `--phase melody` is passed
- **THEN** only melody candidates are displayed

#### Scenario: Non-generated entry displayed with marker
- **WHEN** a review.yml entry has `generated: false`
- **THEN** the browser displays `[H]` in the score column for that entry instead of numeric scores

#### Scenario: Null scores and rank tolerated
- **WHEN** a review.yml candidate entry has `scores: null` or `rank: null`
- **THEN** the browser loads without error, substituting 0.0 for composite score and 99 for rank

### Requirement: In-place Approval and Rejection
The candidate browser SHALL allow the user to approve or reject any `candidate`-labelled
entry by keystroke, updating `review.yml` without leaving the terminal.

#### Scenario: Approve a candidate
- **WHEN** a candidate row is selected and the user presses `a`
- **THEN** the corresponding `review.yml` entry is updated to `status: approved` and
  the row's status column updates immediately

#### Scenario: Reject a candidate
- **WHEN** a candidate row is selected and the user presses `r`
- **THEN** the corresponding `review.yml` entry is updated to `status: rejected`

#### Scenario: Already-approved entries not editable
- **WHEN** an `approved` entry is selected and the user presses `a` or `r`
- **THEN** no change is made and a status message explains why

### Requirement: MIDI Playback
The candidate browser SHALL open a selected candidate's MIDI file in the system default
MIDI player when the user presses `p`.

#### Scenario: Play candidate
- **WHEN** a candidate row is selected and the user presses `p`
- **THEN** the associated `.mid` file is opened via the OS default application (macOS `open`)

### Requirement: Score Breakdown Panel
The candidate browser SHALL display a score breakdown (theory score, chromatic score,
diversity factor, composite score) for the currently selected candidate.

#### Scenario: Score detail visible
- **WHEN** a candidate row is selected
- **THEN** a panel shows the four score components for that candidate

