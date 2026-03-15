# Change: Add Chromatic Sample Retrieval

## Why
The training corpus contains 9,907 labeled audio segments with precomputed CLAP embeddings
and Refractor chromatic scores. This data is currently only used for training — it is not
accessible during production. A retrieval tool would let producers find real reference
segments that score highest for a given color target, providing chromatic-aligned
audio material for mood reference, sampling, or arrangement inspiration.

## What Changes
- New CLI tool `app/generators/midi/production/retrieve_samples.py` that loads CLAP
  embeddings + Refractor scores from the HuggingFace parquet, finds the top-N segments
  matching a color target, and writes a `sample_map.yml` with ranked results
- Optional: copies matched audio files to an output directory if a local media cache is
  available (gracefully skipped if media is not downloaded)
- No new models; uses precomputed scores from `training_data_clap_embeddings.parquet`
  and optionally the DeBERTa parquet for concept/lyric metadata

## Impact
- Affected specs: `chromatic-sample-retrieval` (new)
- Affected code:
  - New: `app/generators/midi/production/retrieve_samples.py`
  - New: `tests/generators/midi/test_retrieve_samples.py`
  - Read: `training_data_clap_embeddings.parquet` (HuggingFace or local cache)
  - Read: `training/refractor.py` (optional re-scoring if parquet scores unavailable)
