# pattern-evolution Specification

## Purpose
TBD - created by archiving change add-evolutionary-pattern-breeding. Update Purpose after archive.
## Requirements
### Requirement: Evolutionary Pattern Engine
`app/generators/midi/patterns/pattern_evolution.py` SHALL provide three public functions:
`breed_drum_patterns`, `breed_bass_patterns`, and `breed_melody_patterns`.

Each function SHALL accept a concept embedding, seed patterns, and optional
`generations` (default 8) and `population_size` (default 30) parameters, and return
`top_n` (default 5) evolved patterns of the same dataclass type as the inputs.

The engine SHALL implement: tournament selection (k=3), elitism (top 2 survive each
generation), crossover (voice-row swap for drums; bar-boundary splice for bass/melody),
and mutation with probability 0.15.

Evolved patterns SHALL carry an `evolved` tag plus any tags inherited from their
highest-fitness parent.

#### Scenario: breed_drum_patterns returns top_n results
- **GIVEN** a set of seed DrumPatterns and a concept embedding
- **WHEN** `breed_drum_patterns` is called with `top_n=3`
- **THEN** exactly 3 DrumPattern instances are returned

#### Scenario: Evolved drum patterns are valid
- **GIVEN** seed patterns with well-formed voice grids
- **WHEN** crossover is applied
- **THEN** all voices in the child pattern are lists of (float, str) tuples

#### Scenario: Evolved patterns carry evolved tag
- **GIVEN** any seed patterns
- **WHEN** breeding completes
- **THEN** every returned pattern has "evolved" in its tags

#### Scenario: breed_bass_patterns returns top_n results
- **GIVEN** seed BassPatterns, a chord progression, and a concept embedding
- **WHEN** `breed_bass_patterns` is called with `top_n=3`
- **THEN** exactly 3 BassPattern instances are returned

#### Scenario: breed_melody_patterns returns top_n results
- **GIVEN** seed MelodyPatterns, a chord progression, and a concept embedding
- **WHEN** `breed_melody_patterns` is called with `top_n=3`
- **THEN** exactly 3 MelodyPattern instances are returned

#### Scenario: Fitness ordering preserved
- **GIVEN** a population scored by the Refractor
- **WHEN** top_n is selected
- **THEN** the returned patterns are ordered highest fitness first

