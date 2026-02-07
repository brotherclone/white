# Implementation Tasks

## 1. Text Embedding Generation
- [ ] 1.1 Run DeBERTa-v3 embedding pass on `training_segments_metadata.parquet` (10,544 segments)
- [ ] 1.2 Store embeddings in new column or separate embeddings parquet
- [ ] 1.3 Verify embedding dimensions match Phase 1-4 expectations (`[768]` half-float)

## 2. Label Audit
- [ ] 2.1 Audit 3,506 UNLABELED segments — cross-reference with manifest data for recoverable `rainbow_color` labels
- [ ] 2.2 Document which songs/albums are missing color assignments
- [ ] 2.3 Report on Green, Indigo, Violet absence and impact on training balance

## 3. Coverage Verification
- [ ] 3.1 Verify `has_midi` flag matches actual `midi_binary` presence (currently aligned: both 4,563)
- [ ] 3.2 Verify `has_audio` flag matches actual `audio_waveform` presence (currently aligned: both 8,972)
- [ ] 3.3 Spot-check MIDI binary decodability via `app/util/midi_segment_utils.py`
- [ ] 3.4 Spot-check audio binary decodability with sample rate consistency

## 4. Documentation
- [ ] 4.1 Document coverage by album color (MIDI sparsity: Blue 12%, overall 43%)
- [ ] 4.2 Document migration path from legacy `training_data_with_embeddings.parquet`
