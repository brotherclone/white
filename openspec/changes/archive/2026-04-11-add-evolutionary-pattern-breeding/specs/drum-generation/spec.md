## ADDED Requirements

### Requirement: Drum Pipeline Evolve Flag
The drum pipeline CLI SHALL accept `--evolve`, `--generations` (int, default 8), and
`--population` (int, default 30) flags. When `--evolve` is passed, evolved drum
candidates SHALL be merged into the standard candidate pool before scoring. Evolved
candidates SHALL be written to `candidates/` with an `evolved_` filename prefix and
their `id` field SHALL begin with `evolved_`.

#### Scenario: --evolve flag merges candidates
- **GIVEN** the drum pipeline is run with `--evolve`
- **WHEN** candidate generation completes
- **THEN** the candidate pool contains both hand-coded and evolved patterns

#### Scenario: Evolved candidates use evolved_ prefix
- **GIVEN** `--evolve` is passed
- **WHEN** MIDI files are written
- **THEN** at least one filename begins with `evolved_`
