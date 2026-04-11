## ADDED Requirements

### Requirement: Melody Pipeline Evolve Flag
The melody pipeline CLI SHALL accept `--evolve`, `--generations` (int, default 8), and
`--population` (int, default 30) flags. When `--evolve` is passed, evolved melody
candidates SHALL be merged into the standard candidate pool before scoring. Evolved
candidates SHALL have their `id` field begin with `evolved_`.

#### Scenario: --evolve flag merges melody candidates
- **GIVEN** the melody pipeline is run with `--evolve`
- **WHEN** candidate generation completes
- **THEN** the candidate pool contains both hand-coded and evolved patterns
