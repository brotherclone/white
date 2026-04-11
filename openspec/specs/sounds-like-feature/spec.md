# sounds-like-feature Specification

## Purpose
TBD - created by archiving change add-sounds-like-training-feature. Update Purpose after archive.
## Requirements
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

### Requirement: MIDI Style Reference Extraction
The system SHALL extract statistical features from locally available MIDI files
for each `sounds_like` artist and write an aggregate `StyleProfile` to
`song_context.yml` under `style_reference_profile`. If no local MIDI files exist
for any artist, this block SHALL be omitted and pipeline behaviour SHALL be unchanged.

The local directory structure SHALL be:
```
style_references/<artist_slug>/*.mid
```

Where artist_slug is the artist name lowercased with spaces replaced by underscores.

Extracted features SHALL include: note_density, mean_duration_beats,
velocity_mean, velocity_variance, interval_histogram, rest_ratio,
harmonic_rhythm, phrase_length_mean. All are averaged across all available MIDI
files for the artist, then averaged across all artists.

Profiles SHALL be cached as `style_references/<artist_slug>/profile.yml` and only
recomputed when source MIDI files change.

#### Scenario: Local MIDI files present — profile written
- **WHEN** `init_production` runs for a song with `sounds_like: [Grouper]`
- **AND** `style_references/grouper/` contains one or more MIDI files
- **THEN** `song_context.yml` contains a `style_reference_profile` block
- **AND** the profile reflects the statistical features of the MIDI files

#### Scenario: No local MIDI files — no profile written
- **WHEN** no MIDI files exist in `style_references/` for any `sounds_like` artist
- **THEN** `song_context.yml` does NOT contain a `style_reference_profile` block
- **AND** all pipelines run with existing behaviour

#### Scenario: Partial coverage — profile from available artists only
- **WHEN** `sounds_like` lists two artists but only one has local MIDI files
- **THEN** the profile is derived from only the artist with files
- **AND** a warning is logged for the missing artist

