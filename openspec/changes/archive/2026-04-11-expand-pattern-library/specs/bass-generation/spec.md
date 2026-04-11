## MODIFIED Requirements
### Requirement: Bass Pattern Templates
Each `BassPattern` in the template library SHALL carry an optional `tags: list[str]`
field drawn from a controlled vocabulary: `drone`, `pedal`, `walking`, `arpeggiated`,
`sustained`, `minimal`. Existing patterns without tags behave identically.

The library SHALL include the following additional drone/pedal templates:
- `root_drone` — single root note, whole-note duration, no movement
- `slow_pedal` — root on beat 1, octave below on beat 3
- `descending_sigh` — root → major 7th → 5th over 4 bars, stepwise descent
- `sustained_fifth` — held 5th drone across the bar, slight velocity swell
- `minimal_walk` — root + one passing tone approaching the next chord

All new templates SHALL carry `drone`, `pedal`, or `minimal` tags as appropriate.

#### Scenario: Tag field present on all patterns
- **WHEN** the bass pattern library is loaded
- **THEN** every `BassPattern` has a `tags` attribute (empty list if none assigned)

#### Scenario: Drone/pedal templates available
- **WHEN** the library is filtered for patterns tagged `drone` or `pedal`
- **THEN** at least 4 patterns are returned
