## MODIFIED Requirements
### Requirement: Composite Scoring
The bass pipeline SHALL read the section `arc` value from `production_plan.yml`
when available, and use it to weight tag-based selection: arc ≤ 0.25 prefers
pedal/drone templates; arc 0.25–0.65 is balanced; arc ≥ 0.65 penalises root_drone.
The arc influence SHALL be expressed as a score adjustment applied before ranking.

#### Scenario: Low arc boosts drone/pedal templates
- **WHEN** section arc=0.10 and drone-tagged patterns compete with walking patterns
- **THEN** drone-tagged patterns receive a positive score adjustment

#### Scenario: High arc penalises root_drone
- **WHEN** section arc=0.80 and root_drone is a candidate
- **THEN** root_drone receives a negative score adjustment
