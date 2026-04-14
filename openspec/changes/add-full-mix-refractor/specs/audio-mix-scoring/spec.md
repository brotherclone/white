## MODIFIED Requirements

### Requirement: Audio Bounce Encoding
The system SHALL encode an audio file (WAV, AIFF, or MP3) by chunking it into overlapping
30s windows (5s stride), encoding each chunk via CLAP to a 512-dim embedding, and returning
a list of per-chunk embeddings for downstream aggregation.

#### Scenario: Supported format encoded successfully
- **WHEN** a valid WAV, AIFF, or MP3 file is provided
- **THEN** a list of one or more 512-dim numpy arrays is returned, one per chunk

#### Scenario: Audio shorter than one chunk window
- **WHEN** the audio file is shorter than 30 seconds
- **THEN** a single zero-padded 512-dim embedding is returned

#### Scenario: Unsupported or corrupt file
- **WHEN** an unreadable or unsupported file path is provided
- **THEN** a clear error is raised before any ONNX inference is attempted

---

### Requirement: Mix Chromatic Scoring
The system SHALL score a rendered audio bounce against the song's chromatic target by
scoring each chunk individually via Refractor and aggregating results using
confidence-weighted mean pooling, returning temporal, spatial, and ontological probability
distributions plus a scalar confidence value.

#### Scenario: Chunked audio-only Refractor inference
- **WHEN** an audio file is provided with a known production directory
- **THEN** each 30s chunk is scored independently and results are aggregated;
  the final score dict contains temporal/spatial/ontological dicts and a confidence in [0,1]

#### Scenario: Refractor CDM used when available
- **WHEN** `refractor_cdm.onnx` exists at the configured path
- **THEN** `Refractor` routes audio-only scoring through the calibration head,
  producing confidence > 0.20 on typical full-mix audio

#### Scenario: Graceful fallback to base model
- **WHEN** `refractor_cdm.onnx` is absent or `--cdm-onnx-path ""` is set
- **THEN** scoring proceeds with the base Refractor ONNX model without error

#### Scenario: Chromatic match computed
- **WHEN** the aggregated Refractor result and the color's CHROMATIC_TARGETS entry are available
- **THEN** `compute_chromatic_match()` returns a scalar in [0, 1] representing alignment

---

### Requirement: Refractor CDM Training
The system SHALL provide a training script that trains a small calibration MLP on top of
frozen CLAP embeddings from the labeled `staged_raw_material/` catalog, exporting the
result to `training/data/refractor_cdm.onnx`.

#### Scenario: Training completes and exports ONNX
- **WHEN** `train_refractor_cdm.py` is run with access to `staged_raw_material/`
- **THEN** the script trains for the configured number of epochs, logs per-epoch val
  accuracy, and writes `refractor_cdm.onnx` for the best checkpoint

#### Scenario: Validation accuracy reported
- **WHEN** training completes
- **THEN** per-color top-1 accuracy and overall accuracy are printed;
  a warning is emitted if overall accuracy is below 70%

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
match, drift report, chunk metadata, and provenance (which model produced the score).

#### Scenario: File written successfully
- **WHEN** scoring completes without error
- **THEN** `mix_score.yml` exists in the song's `melody/` directory and is valid YAML

#### Scenario: Chunk metadata recorded
- **WHEN** scoring completes
- **THEN** `mix_score.yml` metadata contains `chunk_count`, `chunk_stride_s`, and
  `refractor_cdm` (path or null) indicating the model used

#### Scenario: Existing file overwritten
- **WHEN** `mix_score.yml` already exists from a previous run
- **THEN** the file is overwritten with the latest result (not appended)

---

### Requirement: Score Mix CLI
The system SHALL provide a CLI entry point `score_mix.py` with `--mix-file`,
`--production-dir`, `--chunk-size`, `--chunk-stride`, and `--cdm-onnx-path` flags.

#### Scenario: CLI produces console summary
- **WHEN** invoked with valid `--mix-file` and `--production-dir`
- **THEN** stdout shows temporal/spatial/ontological scores, confidence, chromatic match,
  chunk count, and overall drift; `mix_score.yml` is written

#### Scenario: Missing production directory
- **WHEN** `--production-dir` does not exist
- **THEN** CLI exits with a clear error message before loading any models
