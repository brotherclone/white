## MODIFIED Requirements
### Requirement: Composite Scoring
The drum pipeline composite scoring SHALL incorporate arc-weighted energy scoring.
When `arc` is available for a section (from `production_plan.yml`), the target
energy SHALL be blended between the label heuristic and the arc value. The arc
SHALL map to a target energy band: arc < 0.3 → low, 0.3–0.65 → medium,
arc > 0.65 → high. The arc target SHALL take precedence over the label heuristic
when both are present.

#### Scenario: High arc section prefers high-energy patterns
- **WHEN** section arc=0.85 and a dense pattern and a sparse pattern are both candidates
- **THEN** the dense pattern scores higher than the sparse pattern

#### Scenario: Low arc section prefers low-energy patterns
- **WHEN** section arc=0.10 and a sparse pattern and a dense pattern are both candidates
- **THEN** the sparse pattern scores higher than the dense pattern

#### Scenario: Missing arc falls back to label heuristic
- **WHEN** no production_plan.yml is present
- **THEN** energy target is derived from section label as before
