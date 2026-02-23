# Change: Prepare Multimodal Training Data

## Why

The 2026-02-06 re-extraction fixed MIDI binary embedding (0% → 43.3%) but the new parquet files lack DeBERTa text embeddings. The multimodal fusion model (Phase 3.1) requires all modalities to have computed features before training can begin.

Current state:
- `training_segments_metadata.parquet`: 10,544 segments, 75 columns, no binary blobs
- `training_segments_media.parquet`: 10,544 segments, adds `audio_waveform`, `audio_sample_rate`, `midi_binary`
- **Text embeddings**: 0% (old file had them; new extraction dropped them)
- **MIDI binary**: 43.3% (4,563 segments — up from 0% pre-bugfix)
- **Audio waveform**: 85.1% (8,972 segments)
- **UNLABELED segments**: 3,506 (33%) — no `rainbow_color` assigned
- **Missing albums**: Green, Indigo, Violet absent entirely

## What Changes

- Run DeBERTa-v3 embedding pass on 10,544 segments
- Audit UNLABELED segments for recoverable `rainbow_color` labels
- Verify data integrity: binary presence matches boolean flags
- Document coverage gaps and their impact on downstream training

## Impact

- Affected specs: multimodal-fusion (data readiness prerequisite)
- Affected code: `training/core/embedding_loader.py`, `app/extractors/segment_extractor/build_training_segments_db.py`
- **BREAKING**: Training scripts referencing old `training_data_with_embeddings.parquet` must update to new split parquet files
- Blocks: `add-multimodal-fusion` (cannot train without embeddings)
