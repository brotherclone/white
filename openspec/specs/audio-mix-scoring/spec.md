# audio-mix-scoring Specification

## Purpose
TBD - created by archiving change add-audio-mix-scoring. Update Purpose after archive.
## Requirements
### Requirement: Audio Bounce Encoding
The system SHALL encode an audio file (WAV, AIFF, or MP3) using the CLAP model to produce
a 512-dimensional audio embedding suitable for Refractor inference.

#### Scenario: Supported format encoded successfully
- **WHEN** a valid WAV, AIFF, or MP3 file is provided
- **THEN** a 512-dim numpy array is returned with no exception

#### Scenario: Unsupported or corrupt file
- **WHEN** an unreadable or unsupported file path is provided
- **THEN** a clear error is raised before any ONNX inference is attempted

---

### Requirement: Mix Chromatic Scoring
The system SHALL score a rendered audio bounce against the song's chromatic target using
Refractor in audio-only mode (no MIDI, no concept text), returning temporal, spatial, and
ontological probability distributions plus a scalar confidence value.

#### Scenario: Audio-only Refractor inference
- **WHEN** only an audio embedding is provided (no MIDI, no concept embedding)
- **THEN** Refractor returns a valid score dict with temporal/spatial/ontological dicts and
  a confidence in [0, 1]

#### Scenario: Chromatic match computed
- **WHEN** the Refractor result and the color's CHROMATIC_TARGETS entry are available
- **THEN** `compute_chromatic_match()` returns a scalar in [0, 1] representing alignment

---

### Requirement: Per-Dimension Drift Report
The system SHALL compute a drift report comparing the mix's predicted chromatic distribution
against the song's target distribution, reporting per-dimension signed deltas.

#### Scenario: Drift computed for all three dimensions
- **WHEN** a Refractor result and a CHROMATIC_TARGETS target are provided
- **THEN** the drift report contains temporal_delta, spatial_delta, ontological_delta, and
  an overall_drift scalar (mean absolute delta across dimensions)

#### Scenario: On-target mix
- **WHEN** predicted distributions closely match the target
- **THEN** overall_drift is near 0.0 and all dimension deltas are small

---

### Requirement: Mix Score File
The system SHALL write `melody/mix_score.yml` containing the Refractor score, chromatic
match, drift report, and metadata (audio file path, timestamp, Refractor ONNX path).

#### Scenario: File written successfully
- **WHEN** scoring completes without error
- **THEN** `mix_score.yml` exists in the song's `melody/` directory and is valid YAML

#### Scenario: Existing file overwritten
- **WHEN** `mix_score.yml` already exists from a previous run
- **THEN** the file is overwritten with the latest result (not appended)

---

### Requirement: Score Mix CLI
The system SHALL provide a CLI entry point `score_mix.py` with `--mix-file` and
`--production-dir` flags that scores a bounce and prints a human-readable summary.

#### Scenario: CLI produces console summary
- **WHEN** invoked with valid `--mix-file` and `--production-dir`
- **THEN** stdout shows temporal/spatial/ontological scores, confidence, chromatic match,
  and overall drift; `mix_score.yml` is written

#### Scenario: Missing production directory
- **WHEN** `--production-dir` does not exist
- **THEN** CLI exits with a clear error message before loading any models

