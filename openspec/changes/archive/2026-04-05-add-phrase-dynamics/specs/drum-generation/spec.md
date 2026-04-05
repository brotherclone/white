## ADDED Requirements

### Requirement: Phrase-Level Velocity Shaping (Drums)
The drum pipeline SHALL apply the dynamic curve to all drum note velocities within a
section (scaling all voices uniformly), with the drum velocity clamp (45–127) enforced.

#### Scenario: Crash accent preserved during crescendo
- **WHEN** a LINEAR_CRESC curve is applied to a section containing a crash accent note
- **THEN** the crash velocity is scaled but clamped to 127

#### Scenario: Ghost notes stay soft
- **WHEN** any dynamic curve is applied
- **THEN** ghost notes (originally at velocity 45) are scaled proportionally but
  never rise above 65 (one-third of the dynamic range)
