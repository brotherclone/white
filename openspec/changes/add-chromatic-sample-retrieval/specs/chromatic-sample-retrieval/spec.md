## ADDED Requirements

### Requirement: CLAP Index Loading
The system SHALL load the precomputed CLAP embeddings and Refractor scores from the
training parquet (`training_data_clap_embeddings.parquet`), either from a local path or
via `hf_hub_download` from `earthlyframes/white-training-data`.

#### Scenario: Local parquet loaded
- **WHEN** a valid local parquet path is provided
- **THEN** a DataFrame is returned with segment_id, song_slug, color, and clap_embedding
  columns; Refractor score columns included if present

#### Scenario: HuggingFace fallback
- **WHEN** no local path is provided and network is available
- **THEN** the parquet is downloaded via `hf_hub_download` and loaded successfully

#### Scenario: Missing Refractor score columns
- **WHEN** the parquet does not contain temporal/spatial/ontological columns
- **THEN** the system logs a warning and falls back to computing chromatic match from
  CLAP distributions via Refractor at retrieval time

---

### Requirement: Color-Targeted Sample Retrieval
The system SHALL return the top-N audio segments ranked by chromatic match score for a
given color name, using precomputed Refractor scores from the parquet.

#### Scenario: Top-N by color
- **WHEN** a valid color name and top-n value are provided
- **THEN** results are sorted descending by chromatic_match and exactly top-n rows are
  returned (or all rows if fewer exist for that color)

#### Scenario: Unknown color name
- **WHEN** a color name not in CHROMATIC_TARGETS is provided
- **THEN** an error is raised listing valid color names before any parquet loading

---

### Requirement: CLAP Cosine Similarity Retrieval
The system SHALL support retrieval by cosine similarity between a query CLAP embedding
and all embeddings in the index, independent of color label.

#### Scenario: Cross-color similarity search
- **WHEN** a 512-dim query embedding is provided
- **THEN** results are sorted by descending cosine similarity and top-n rows are returned
  with their color labels and similarity scores

---

### Requirement: Sample Map Output
The system SHALL write a `sample_map.yml` file listing ranked results with rank,
segment_id, song_slug, color, chromatic_match, and audio_path (null if unavailable).

#### Scenario: YAML written with all fields
- **WHEN** retrieval completes successfully
- **THEN** `sample_map.yml` exists at the output path and contains all required fields
  for each result entry

#### Scenario: Audio copy requested but files missing
- **WHEN** `--copy-audio` is set but media files are not in the local cache
- **THEN** audio_path entries are set to null and a warning is printed; the YAML is still
  written successfully

---

### Requirement: Sample Retrieval CLI
The system SHALL provide a CLI `retrieve_samples.py` with `--color` and `--top-n` flags
that prints a ranked table and writes `sample_map.yml`.

#### Scenario: CLI prints ranked summary
- **WHEN** invoked with valid `--color` and `--top-n`
- **THEN** stdout shows a ranked table of segment_id, song_slug, and chromatic_match;
  `sample_map.yml` is written to `--output-dir`
