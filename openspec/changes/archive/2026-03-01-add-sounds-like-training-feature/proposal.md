# Change: Add Sounds-Like Training Feature

## Why

The `sounds_like` field exists on every training row (83 songs, 166 unique artists, all
with Discogs IDs) but contributes nothing to the fusion model's chromatic predictions.
These are curated aesthetic reference pointers — already human-validated — that encode
information about each song's stylistic neighbourhood that no other modality carries.

The mechanism: use the `add-artist-style-catalog` catalog (Claude-generated aesthetic
descriptions, human-reviewed) to embed each artist's description via DeBERTa into a
768-dim vector, then mean-pool the per-song set of embeddings into one
`sounds_like_emb`. Add it as a 5th input modality to the fusion model using the same
null-embedding + modality-dropout pattern already in use for lyric and audio.

Because `sounds_like` is a song-level signal (not segment-level), one vector is
broadcast across all segments of the same song — analogous to `concept_emb`.

## What Changes

- **`training/data/artist_catalog.yml`** (symlink / copy of `app/data/artist_catalog.yml`) —
  the catalog must be accessible to Modal training jobs; handled by uploading to the Modal
  volume or embedding in the parquet pre-processing step
- **`training/build_sounds_like_embeddings.py`** — local CPU script that reads the
  DeBERTa parquet, parses `sounds_like` strings, looks up artist descriptions in the
  catalog, embeds each description, mean-pools per song, and writes
  `training/data/sounds_like_embeddings.parquet` (segment_id + 768-dim float32 vector +
  `has_sounds_like` bool)
- **`training/modal_midi_fusion.py`** — updated to load the sounds-like parquet alongside
  the DeBERTa and CLAP parquets; passes `sounds_like_emb` + `has_sounds_like` as a 5th
  modality to the model
- **`training/models/multimodal_fusion.py`** — adds `null_sounds_like` learned parameter
  (768-dim) and expands fusion input from 2560 to 3328; `forward()` gains `sounds_like_emb`
  + `has_sounds_like` arguments
- **`training/export_onnx.py`** — updated dummy inputs for ONNX export to include the new
  5th modality tensor
- **`training/refractor.py`** — extended to optionally accept
  `sounds_like_emb` (pre-computed externally) or `sounds_like_texts` (list of artist
  description strings to embed on-the-fly); null vector used when neither is provided
- **`tests/training/test_sounds_like_embeddings.py`** — unit tests for the
  embedding build pipeline

## What This Is NOT

- Not training on audio, lyrics, MIDI, or any content from the reference artists
- Not using Discogs API for lookups — Discogs IDs remain informational metadata
- Not adding a 5th dimension to any Refractor-exposed axis — the output shape
  (temporal/spatial/ontological/confidence) is unchanged
- Not blocking the re-train on catalog completeness — a song with no catalog matches
  gets `has_sounds_like: False` and the null embedding, same as a song with no audio

## Dependencies

- **`add-artist-style-catalog`** must be implemented and reviewed entries present in the
  catalog before training can benefit; however the code can be written and tested against
  an empty catalog (all `has_sounds_like: False`)
- The retrain runs on Modal A10G (same as Phase 3); ONNX re-export required after

## Impact

- Affected specs: `sounds-like-feature` (ADDED), `multimodal-fusion` (MODIFIED)
- Affected code:
  - `training/build_sounds_like_embeddings.py` — new local CPU script (~150 lines)
  - `training/models/multimodal_fusion.py` — 5th modality, 2560→3328 input dim
  - `training/modal_midi_fusion.py` — load + pass sounds_like modality
  - `training/export_onnx.py` — dummy input for new modality
  - `training/refractor.py` — optional sounds_like injection
  - `tests/training/test_sounds_like_embeddings.py` — new tests
