# strum-generation Specification Delta

## MODIFIED Requirements

### Requirement: Strum Candidate Generation

The strum generator SHALL support variable chord durations from the harmonic rhythm phase, applying strum patterns that scale to each chord's assigned duration.

#### Scenario: Variable duration from harmonic rhythm

- **WHEN** an approved harmonic rhythm exists for the section (in `harmonic_rhythm/approved/`)
- **THEN** the strum pipeline SHALL read the approved duration map
- **AND** each chord SHALL receive its approved duration (in bars) instead of the default 1.0 bar
- **AND** strum patterns SHALL repeat to fill longer durations or truncate for shorter durations

#### Scenario: Backward compatibility without harmonic rhythm

- **WHEN** no approved harmonic rhythm exists for a section
- **THEN** the strum pipeline SHALL fall back to uniform 1.0 bar per chord
- **AND** behavior SHALL be identical to the current implementation
