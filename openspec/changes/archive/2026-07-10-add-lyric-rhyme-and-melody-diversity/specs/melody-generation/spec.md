## MODIFIED Requirements

### Requirement: Melody Pipeline Evolve Flag
The melody pipeline CLI SHALL breed evolved melody candidates by default, accepting
`--no-evolve` to opt out, and `--generations` (int, default 8) and `--population` (int,
default 30) flags to control breeding. When evolution runs, evolved melody candidates
SHALL be merged into the standard candidate pool before scoring. Evolved candidates
SHALL have their `id` field begin with `evolved_`.

#### Scenario: Default invocation breeds candidates
- **GIVEN** the melody pipeline is run with no `--evolve`/`--no-evolve` flag
- **WHEN** candidate generation completes
- **THEN** the candidate pool contains both hand-coded and evolved patterns

#### Scenario: --no-evolve disables breeding
- **GIVEN** the melody pipeline is run with `--no-evolve`
- **WHEN** candidate generation completes
- **THEN** the candidate pool contains only hand-coded patterns

#### Scenario: --evolve is accepted as a no-op for backward compatibility
- **GIVEN** the melody pipeline is run with `--evolve` explicitly
- **WHEN** candidate generation completes
- **THEN** behavior is identical to the default (evolution runs)
