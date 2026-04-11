## MODIFIED Requirements

### Requirement: Composite Scoring
The bass pipeline composite scoring SHALL incorporate style reference profile
adjustments when `style_reference_profile` is present in `song_context.yml`.

- High `mean_duration_beats` (> 1.5 beats) → boost pedal/drone bass templates
- High `rest_ratio` (> 0.5) → boost minimal/drone templates; penalise walking
- Low `harmonic_rhythm` (< 0.5 changes/bar) → boost pedal/drone templates

#### Scenario: Long note reference boosts pedal bass
- **WHEN** `style_reference_profile.mean_duration_beats` is 2.3
- **AND** a pedal and a walking bass are candidates
- **THEN** the pedal pattern receives a higher score adjustment

#### Scenario: Missing profile — no adjustment
- **WHEN** no `style_reference_profile` is present
- **THEN** bass scoring proceeds unchanged
