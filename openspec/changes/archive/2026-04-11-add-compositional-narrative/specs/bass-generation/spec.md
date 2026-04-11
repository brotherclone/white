## MODIFIED Requirements

### Requirement: Composite Scoring
The bass pipeline composite scoring SHALL incorporate narrative texture and lead_voice
constraints when `composition_narrative.yml` is present.

- `texture: absent` → prefer rest-heavy templates; penalise walking/arpeggiated
- `lead_voice: bass` → prefer templates with movement; boost walking/arpeggiated
- `lead_voice: melody` → prefer pedal/drone; penalise walking

The narrative constraint adjustment SHALL be a tag-weighted score delta applied
alongside the arc-based adjustment.

#### Scenario: Bass leads section — movement preferred
- **WHEN** section narrative declares `lead_voice: bass`
- **AND** a walking bass and a pedal pattern are candidates
- **THEN** the walking bass scores higher than the pedal pattern

#### Scenario: Melody leads — bass supports
- **WHEN** section narrative declares `lead_voice: melody`
- **AND** a drone and a walking bass are candidates
- **THEN** the drone scores higher than the walking bass
