## ADDED Requirements

### Requirement: Phrase-Level Velocity Shaping (Bass)
The bass pipeline SHALL apply the same dynamic curve mechanism as the melody pipeline,
with the bass velocity clamp (50–110) enforced after curve application.

#### Scenario: Diminuendo on outro
- **WHEN** no dynamics map is present and the section is labelled `outro`
- **THEN** the LINEAR_DIM curve is applied and all velocities remain ≥ 50

#### Scenario: Accent notes respect clamp ceiling
- **WHEN** a dynamic curve would push an accent note above 110
- **THEN** the velocity is clamped to 110
