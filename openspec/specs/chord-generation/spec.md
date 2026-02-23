# chord-generation Specification

## Purpose
TBD - created by archiving change add-music-production-pipeline. Update Purpose after archive.
## Requirements
### Requirement: Chord Pipeline Input

The chord generation pipeline SHALL accept a song proposal YAML file path and a shrinkwrapped thread directory as input. It SHALL extract key, mode, BPM, time signature, concept text, and rainbow color from the song proposal and thread manifest.

#### Scenario: Load song proposal from shrinkwrapped thread

- **WHEN** the pipeline is invoked with a thread directory and song proposal filename
- **THEN** it SHALL parse the song proposal YAML for key, BPM, time signature, and concept
- **AND** parse the thread manifest for the full concept text and rainbow color
- **AND** reject proposals with missing key or rainbow color fields

#### Scenario: Derive chromatic target from rainbow color

- **WHEN** the rainbow color is extracted from the song proposal
- **THEN** the pipeline SHALL map it to target mode distributions (temporal, spatial, ontological)
- **AND** use uniform distributions for White and Black proposals

### Requirement: Candidate Generation

The pipeline SHALL generate chord primitive candidates by combining a Markov chord progression
with a randomly-sampled harmonic rhythm (HR) distribution and strum articulation pattern. Each
candidate is a complete chord primitive — voicings, rhythm, and articulation — ready for
promotion without further post-processing.

#### Scenario: Graph-guided generation

- **WHEN** the pipeline generates candidates
- **THEN** it SHALL use the function transition graph for weighted Markov sampling
- **AND** generate at least 50 candidates per invocation (configurable)
- **AND** constrain all candidates to the target key and mode from the song proposal

#### Scenario: HR and strum baked into each candidate

- **WHEN** a chord progression is generated
- **THEN** the pipeline SHALL randomly sample a harmonic rhythm distribution (from the half-bar
  duration grid) and a strum articulation pattern (from the strum template library)
- **AND** apply both to the progression's voicings before writing the candidate MIDI
- **AND** the same seed SHALL produce identical HR + strum pairings for reproducibility

#### Scenario: Progression length from time signature

- **WHEN** the song proposal specifies a time signature
- **THEN** the pipeline SHALL use an appropriate default progression length (e.g., 4 bars for 4/4, 7 bars for 7/8)
- **AND** allow the user to override the length via CLI parameter

#### Scenario: Reproducible generation

- **WHEN** a random seed is provided
- **THEN** the same seed SHALL produce identical candidates for the same song proposal

### Requirement: Composite Scoring

The pipeline SHALL score each candidate using both music theory metrics (from the chord prototype) and chromatic fitness (from ChromaticScorer), producing a single composite ranking.

#### Scenario: Music theory scoring

- **WHEN** a chord progression candidate is scored
- **THEN** the pipeline SHALL compute melody score, voice leading score, variety score, and graph probability score using the existing scoring functions

#### Scenario: Chromatic scoring

- **WHEN** a chord progression candidate is scored
- **THEN** the pipeline SHALL convert the candidate to MIDI bytes, encode the concept text via `ChromaticScorer.prepare_concept()`, and score with `ChromaticScorer.score()`
- **AND** the concept embedding SHALL be computed once and reused across all candidates in the batch

#### Scenario: Composite ranking

- **WHEN** all candidates are scored
- **THEN** the pipeline SHALL compute a weighted composite score (default: 30% theory, 70% chromatic)
- **AND** rank candidates by composite score descending
- **AND** allow the user to configure scoring weights via CLI or config

### Requirement: MIDI Output

The pipeline SHALL export each top-ranked candidate as a standard MIDI file alongside a scratch
beat MIDI for auditioning.

#### Scenario: MIDI file generation

- **WHEN** the top N candidates are selected (default N=10)
- **THEN** the pipeline SHALL write each as a `.mid` file in the song's production directory
- **AND** the MIDI file SHALL use the song proposal's BPM for tempo
- **AND** chord notes SHALL reflect the baked-in HR distribution and strum articulation

#### Scenario: Output directory structure

- **WHEN** MIDI files are generated
- **THEN** they SHALL be placed in `<thread>/production/<song_slug>/chords/candidates/`
- **AND** the directory SHALL be created if it does not exist

#### Scenario: Scratch beat generation

- **WHEN** a candidate MIDI file is written
- **THEN** the pipeline SHALL also write a companion scratch beat MIDI named
  `<candidate>_scratch.mid` in the same candidates directory
- **AND** the scratch beat SHALL use the lowest-energy template from the genre family inferred
  from the song proposal, matching the candidate's bar length and BPM
- **AND** scratch files SHALL be listed in `review.yml` with `scratch: true` and SHALL NOT be
  eligible for promotion

### Requirement: CLI Interface

The chord pipeline SHALL be invocable from the command line.

#### Scenario: Basic invocation

- **WHEN** the user runs the pipeline CLI
- **THEN** it SHALL accept `--thread` (shrinkwrapped thread directory), `--song` (song proposal filename), and optional `--seed`, `--num-candidates`, `--top-k`, `--theory-weight`, `--chromatic-weight` parameters

#### Scenario: Progress output

- **WHEN** the pipeline is running
- **THEN** it SHALL print progress (loading, generating, scoring, writing) to stdout
- **AND** print the top candidates with their composite scores and score breakdowns

### Requirement: CLI Interface — HR and Strum Parameters

The chord pipeline CLI SHALL expose controls for HR and strum generation.

#### Scenario: HR and strum seed propagation

- **WHEN** the user provides `--seed`
- **THEN** the seed SHALL deterministically control Markov generation, HR distribution sampling,
  and strum pattern sampling together

#### Scenario: Strum pattern override

- **WHEN** the user provides `--strum-patterns` (comma-separated list)
- **THEN** only the specified strum patterns SHALL be used when pairing with chord progressions
- **AND** if the flag is omitted, all patterns applicable to the song's time signature are eligible

