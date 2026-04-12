## MODIFIED Requirements

### Requirement: Audio Bounce Encoding
The system SHALL encode a full-length audio file by chunking it into overlapping windows
(default 30s, 5s stride) at 48kHz, encoding each chunk independently via CLAP to produce
a 512-dim embedding per chunk.

#### Scenario: Supported format encoded successfully
- **WHEN** a valid WAV, AIFF, or MP3 file is provided
- **THEN** a list of 512-dim numpy arrays (one per chunk) is returned with no exception

#### Scenario: Audio shorter than one chunk window
- **WHEN** the audio file is shorter than the configured window size
- **THEN** the full audio is treated as a single chunk and one embedding is returned

#### Scenario: Unsupported or corrupt file
- **WHEN** an unreadable or unsupported file path is provided
- **THEN** a clear error is raised before any ONNX inference is attempted

---

### Requirement: Mix Chromatic Scoring
The system SHALL score a rendered audio bounce against the song's chromatic target by
scoring each chunk individually with Refractor, then aggregating per-chunk
temporal/spatial/ontological distributions using confidence-weighted mean pooling.

#### Scenario: Multi-chunk audio-only Refractor inference
- **WHEN** a full-length mix is provided with no MIDI or concept text
- **THEN** Refractor scores each chunk independently and returns aggregated temporal,
  spatial, and ontological distributions plus a mean confidence value

#### Scenario: Chromatic match computed from aggregated scores
- **WHEN** the aggregated Refractor result and the color's CHROMATIC_TARGETS entry
  are available
- **THEN** `compute_chromatic_match()` returns a scalar in [0, 1] representing
  alignment of the aggregated prediction to the target

---

### Requirement: Mix Score File
The system SHALL write `melody/mix_score.yml` containing the aggregated Refractor
score, chromatic match, drift report, chunk metadata, and provenance fields (audio
file path, timestamp, Refractor ONNX path, chunk_count, chunk_stride_s).

#### Scenario: File written with chunk metadata
- **WHEN** scoring completes without error
- **THEN** `mix_score.yml` exists in the song's `melody/` directory, is valid YAML,
  and contains `chunk_count` and `chunk_stride_s` fields

#### Scenario: Existing file overwritten
- **WHEN** `mix_score.yml` already exists from a previous run
- **THEN** the file is overwritten with the latest result (not appended)

---

### Requirement: Score Mix CLI
The system SHALL provide a CLI entry point `score_mix.py` with `--mix-file`,
`--production-dir`, `--chunk-size` (default 30), and `--chunk-stride` (default 5)
flags that scores a bounce and prints a human-readable summary.

#### Scenario: CLI produces console summary with chunk info
- **WHEN** invoked with valid `--mix-file` and `--production-dir`
- **THEN** stdout shows chunk count, temporal/spatial/ontological scores, confidence,
  chromatic match, and overall drift; `mix_score.yml` is written

#### Scenario: Missing production directory
- **WHEN** `--production-dir` does not exist
- **THEN** CLI exits with a clear error message before loading any models

---

## ADDED Requirements

### Requirement: Full-Song Validation Suite
The system SHALL provide `training/validate_mix_scoring.py` that scores all labeled
`staged_raw_material/<id>/<id>_main.wav` files using the chunked scoring approach,
compares predicted top-1 color to the known color, and reports per-color and overall
top-1 accuracy.

#### Scenario: Validation run produces accuracy report
- **WHEN** `validate_mix_scoring.py` is run against the staged_raw_material directory
- **THEN** stdout shows a per-color accuracy table and overall top-1 accuracy percentage

#### Scenario: Results written to file
- **WHEN** validation completes
- **THEN** `training/data/mix_scoring_validation.yml` is written with per-song
  predicted color, ground-truth color, confidence, and correct/incorrect flag

#### Scenario: Low accuracy warning
- **WHEN** overall top-1 accuracy is below 70%
- **THEN** a prominent warning is emitted recommending a Phase 2 calibration head proposal
