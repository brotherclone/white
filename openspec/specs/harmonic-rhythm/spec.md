# harmonic-rhythm Specification

## Purpose
TBD - created by archiving change add-harmonic-rhythm-generation. Update Purpose after archive.
## Requirements
### Requirement: Half-Bar Duration Grid

The harmonic rhythm generator SHALL express all chord durations as multiples of half a bar, producing candidate distributions that redistribute chord timing within a section.

#### Scenario: Duration granularity

- **WHEN** chord durations are generated for a section
- **THEN** each chord SHALL have a duration that is a multiple of 0.5 bars
- **AND** the minimum duration for any chord SHALL be 0.5 bars

#### Scenario: Distribution enumeration

- **WHEN** a section has N approved chords
- **THEN** the generator SHALL enumerate distributions where each chord gets at least 0.5 bars
- **AND** total section length SHALL range from `N * 0.5` bars (minimum) to `N * 2.0` bars (maximum)
- **AND** the uniform distribution (1.0 bar per chord) SHALL always be included as a baseline candidate

#### Scenario: Candidate cap

- **WHEN** the number of valid distributions exceeds 200
- **THEN** the generator SHALL randomly sample 200 distributions (using the pipeline seed)
- **AND** always include the uniform baseline in the sample

### Requirement: Drum Accent Extraction

The harmonic rhythm generator SHALL parse approved drum MIDI files to identify strong beat positions for chord change alignment.

#### Scenario: Accent identification

- **WHEN** an approved drum MIDI file is parsed
- **THEN** the generator SHALL identify note-on events with velocity >= 100 as accented hits
- **AND** quantize accent positions to half-bar grid boundaries (within a tolerance of ± 1 eighth note)

#### Scenario: Accent mask generation

- **WHEN** accents are extracted from a drum pattern
- **THEN** the generator SHALL produce an accent mask listing which half-bar boundary positions within the bar are "strong"
- **AND** beat 0 (bar start) SHALL always be marked as strong

#### Scenario: No approved drums fallback

- **WHEN** no approved drum MIDI files exist for a section
- **THEN** the generator SHALL use a default accent mask where only bar starts (every 1.0 bar) are strong
- **AND** log a warning about the fallback

### Requirement: Drum Alignment Scoring

The harmonic rhythm generator SHALL score each candidate distribution based on how well chord onsets align with drum accent positions.

#### Scenario: Alignment calculation

- **WHEN** a candidate distribution is scored for drum alignment
- **THEN** the score SHALL equal the fraction of chord onsets that land on strong half-bar positions from the drum accent mask
- **AND** the first chord onset (always at beat 0) SHALL count as aligned

#### Scenario: Alignment with repeating drum bar

- **WHEN** the chord distribution spans multiple bars
- **THEN** the drum accent mask SHALL tile (repeat) across the full section length
- **AND** chord onsets SHALL be checked against the tiled mask

### Requirement: Chromatic Temporal Scoring

The harmonic rhythm generator SHALL score each candidate using ChromaticScorer, focusing on temporal mode alignment.

#### Scenario: MIDI generation for scoring

- **WHEN** a candidate distribution is scored
- **THEN** the generator SHALL produce MIDI bytes with each chord's voicing sustained for its assigned duration
- **AND** the MIDI SHALL use the song proposal's BPM

#### Scenario: Concept embedding reuse

- **WHEN** multiple candidates are scored for the same section
- **THEN** the concept embedding SHALL be computed once and reused across all candidates

#### Scenario: Temporal match extraction

- **WHEN** ChromaticScorer returns results for a candidate
- **THEN** the pipeline SHALL extract the temporal mode match against the color's chromatic target
- **AND** use this as the chromatic component of the composite score

### Requirement: Composite Scoring and Ranking

The harmonic rhythm pipeline SHALL compute a weighted composite score from drum alignment and chromatic temporal match.

#### Scenario: Composite calculation

- **WHEN** a candidate has both drum alignment and chromatic scores
- **THEN** the composite score SHALL be `0.3 * drum_alignment + 0.7 * chromatic_temporal_match`

#### Scenario: Per-section ranking

- **WHEN** all candidates for a section are scored
- **THEN** they SHALL be ranked by composite score descending
- **AND** the top-k candidates (default k=20) SHALL be included in the review file

### Requirement: Harmonic Rhythm MIDI Output

The harmonic rhythm pipeline SHALL write candidate MIDI files and a review YAML for human annotation.

#### Scenario: Output directory structure

- **WHEN** harmonic rhythm candidates are generated
- **THEN** they SHALL be placed in `<song>/harmonic_rhythm/candidates/`
- **AND** files SHALL be named `hr_<section>_<NNN>.mid`
- **AND** an `approved/` directory SHALL be created for promoted candidates
- **AND** old candidate files SHALL be removed before writing new ones

#### Scenario: Review file generation

- **WHEN** candidates are written
- **THEN** a `review.yml` SHALL be generated in `<song>/harmonic_rhythm/`
- **AND** each candidate SHALL include: id, midi file path, rank, section, duration distribution (as list of floats), total bars, drum alignment score, chromatic scores, composite score
- **AND** each candidate SHALL have `label: null`, `status: pending`, and `notes: ""` fields

### Requirement: Harmonic Rhythm CLI Interface

The harmonic rhythm pipeline SHALL be invocable from the command line.

#### Scenario: Basic invocation

- **WHEN** the user runs the harmonic rhythm pipeline CLI
- **THEN** it SHALL accept `--production-dir` (path to song production directory) and optional `--seed`, `--top-k` (per section), `--onnx-path` (ChromaticScorer model path)

#### Scenario: Progress output

- **WHEN** the harmonic rhythm pipeline runs
- **THEN** it SHALL print: sections found, approved drum patterns loaded, candidate count per section, scoring progress, and top candidates per section with score breakdowns

