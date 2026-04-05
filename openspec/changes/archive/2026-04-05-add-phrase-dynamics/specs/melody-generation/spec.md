## ADDED Requirements

### Requirement: Phrase-Level Velocity Shaping (Melody)
The melody pipeline SHALL apply a dynamic curve to the velocity of all notes within a
generated section before writing the MIDI candidate file.

The curve SHALL be determined by (in priority order):
1. The `dynamics` map in `song_proposal.yml` for the current section label
2. `infer_curve(section_energy)` heuristic
3. Default: FLAT (no change)

#### Scenario: Linear crescendo applied
- **WHEN** a section is configured with `linear_cresc`
- **THEN** note velocities increase monotonically from the first note to the last,
  bounded by the melody velocity clamp (60–127)

#### Scenario: Swell on intro by default
- **WHEN** no dynamics map is present and the section is labelled `intro`
- **THEN** the SWELL curve is applied (velocities rise then fall across the section)

#### Scenario: Flat preserves existing velocities
- **WHEN** the effective curve is FLAT
- **THEN** note velocities are identical to the pre-curve values (no-op)
