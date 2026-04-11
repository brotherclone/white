## MODIFIED Requirements

### Requirement: Composite Scoring
The drum pipeline composite scoring SHALL incorporate style reference profile
adjustments when `style_reference_profile` is present in `song_context.yml`.

- Low `note_density` (< 2.0 notes/bar) → boost sparse/ambient drum patterns
- High `velocity_variance` (> 20) → boost patterns with ghost notes
- Low `note_density` (< 1.5 notes/bar) → penalise dense/busy patterns

These SHALL be applied as score adjustments after arc and aesthetic hint adjustments.

#### Scenario: Low density reference boosts sparse drum patterns
- **WHEN** `style_reference_profile.note_density` is 1.8
- **AND** a sparse and a dense pattern are candidates
- **THEN** the sparse pattern receives a higher score adjustment than the dense pattern

#### Scenario: Missing profile — no adjustment
- **WHEN** no `style_reference_profile` is present in song_context
- **THEN** drum scoring proceeds unchanged (existing behaviour)
