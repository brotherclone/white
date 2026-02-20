## MODIFIED Requirements

### Requirement: Section-Aware Generation

The drum generator SHALL read approved chord labels to determine song sections and generate
section-appropriate drum candidates. Drum generation targets section labels only — it does not
distinguish between chord variants or HR-derived filenames.

#### Scenario: Read approved chord sections

- **WHEN** the drum pipeline is invoked with a song production directory
- **THEN** it SHALL read all `.mid` files in `chords/approved/` and derive section names from
  their filenames (e.g., `verse.mid` → section `verse`)
- **AND** reject if no approved chords exist
- **AND** ignore any `_scratch.mid` files present in `candidates/`

#### Scenario: Section energy mapping

- **WHEN** drum candidates are generated for a section
- **THEN** the generator SHALL apply a default energy mapping (intro=low, verse=medium,
  chorus=high, bridge=low, outro=medium)
- **AND** the user MAY override the energy for any section via CLI

#### Scenario: Energy-adjacent inclusion

- **WHEN** templates are selected for a section
- **THEN** the generator SHALL include templates matching the target energy level AND templates
  one energy level away
- **AND** exact-match templates SHALL rank higher in the default ordering
