## ADDED Requirements

### Requirement: Sounds-Like Embedding Build Script

A local CPU script SHALL precompute per-segment `sounds_like` embeddings from the
artist catalog, ready for upload to the Modal training volume.

#### Scenario: Build embeddings — happy path

- **WHEN** `python training/build_sounds_like_embeddings.py` is run
- **AND** `app/data/artist_catalog.yml` contains at least one `reviewed` or `draft` entry
- **THEN** the script reads `training/data/training_data_with_embeddings.parquet`
- **AND** for each unique song slug, parses the `sounds_like` column into artist names
- **AND** looks up each artist name in the catalog (preferring `reviewed` over `draft`)
- **AND** embeds each found artist's `description` via DeBERTa (same model used for
  concept/lyric embeddings)
- **AND** mean-pools the per-artist embeddings into a single 768-dim vector
- **AND** writes `training/data/sounds_like_embeddings.parquet` with one row per
  `segment_id` containing: `segment_id`, `song_slug`, `sounds_like_raw`,
  `artists_found`, `artists_total`, `has_sounds_like`, `sounds_like_emb` (list[float32])

#### Scenario: Song with no catalog matches

- **WHEN** all artists in a song's `sounds_like` list are absent from the catalog
- **THEN** the row is written with `has_sounds_like: False` and
  `sounds_like_emb` as a zero vector (768 zeros)
- **AND** a warning is printed listing the uncatalogued artists
- **AND** the build does not fail

#### Scenario: Partial catalog match

- **WHEN** only some artists in a song's `sounds_like` list have catalog entries
- **THEN** mean-pooling is performed over the found subset only
- **AND** `artists_found` reflects the actual count matched, `artists_total` the full count
- **AND** `has_sounds_like` is `True`

#### Scenario: Reviewed entries preferred over draft

- **WHEN** a `sounds_like` artist has both a `reviewed` and a `draft` entry in the catalog
  (which should not occur but is guarded against)
- **THEN** the `reviewed` entry's description is used

#### Scenario: Rebuild is idempotent

- **WHEN** the script is run a second time without catalog changes
- **THEN** the output parquet is overwritten with identical content
- **AND** no error is raised

---

### Requirement: Sounds-Like Parquet Format

The output parquet SHALL be joinable with `training_data_with_embeddings.parquet`
on `segment_id` and SHALL contain sufficient metadata for debugging coverage.

#### Scenario: Parquet schema

- **WHEN** `sounds_like_embeddings.parquet` is written
- **THEN** it SHALL contain exactly the columns:
  `segment_id` (str), `song_slug` (str), `sounds_like_raw` (str),
  `artists_found` (int), `artists_total` (int), `has_sounds_like` (bool),
  `sounds_like_emb` (list[float32], length 768)
- **AND** the number of rows SHALL equal the number of rows in the source
  training parquet (one row per segment, not per song)

#### Scenario: Coverage summary printed

- **WHEN** the script completes
- **THEN** it prints: total segments processed, songs with full coverage,
  songs with partial coverage, songs with no coverage, overall `has_sounds_like` rate
