## MODIFIED Requirements

### Requirement: Composite Scoring
The melody pipeline composite scoring SHALL incorporate style reference profile
adjustments when `style_reference_profile` is present in `song_context.yml`.

- High `rest_ratio` (> 0.5) → boost sparse/stepwise melody templates
- Low `note_density` (< 2.0) → boost sparse templates; penalise dense
- High `mean_duration_beats` (> 1.5) → boost descent/stepwise templates

#### Scenario: Sparse reference boosts sparse melody
- **WHEN** `style_reference_profile.rest_ratio` is 0.61
- **AND** a sparse and a dense melody template are candidates
- **THEN** the sparse template receives a higher score adjustment

#### Scenario: Missing profile — no adjustment
- **WHEN** no `style_reference_profile` is present
- **THEN** melody scoring proceeds unchanged
