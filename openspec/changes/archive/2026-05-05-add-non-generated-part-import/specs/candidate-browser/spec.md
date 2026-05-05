## MODIFIED Requirements

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
