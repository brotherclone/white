# Implementation Tasks

## 1. Text Embedding Generation
- [x] 1.1 Run DeBERTa-v3 embedding pass on all training segments (11,605 segments) — 2026-02-12 via Modal
- [x] 1.2 Store embeddings in separate parquet (`training_data_with_embeddings.parquet`, 4.5 MB)
- [x] 1.3 Verify embedding dimensions: concept_embedding [768] + lyric_embedding [768] float32

## 2. Label Audit
- [x] 2.1 All 11,605 segments have rainbow_color labels — zero UNLABELED after 2026-02-10 rebuild
- [x] 2.2 All 8 colors present: Black, Red, Orange, Yellow, Green, Blue, Indigo, Violet
- [x] 2.3 Green (393), Yellow (656) are instrumental — spatial mode bottleneck documented

## 3. Coverage Verification
- [x] 3.1 Verify `has_midi` flag matches actual `midi_binary` presence — verified 2026-02-10
- [x] 3.2 Verify `has_audio` flag matches actual `audio_waveform` presence — verified 2026-02-10
- [x] 3.3 MIDI binary decodability spot-checked via verify_extraction.py — 10/10 passed
- [x] 3.4 Audio binary decodability spot-checked — 10/10 passed

## 4. CLAP Audio Embedding Precomputation
- [x] 4.1 Extract CLAP embeddings (laion/larger_clap_music, 512-dim) for all audio segments — 2026-02-12 via Modal
- [x] 4.2 Resample 44.1kHz → 48kHz via librosa before CLAP processor
- [x] 4.3 Store in `training_data_clap_embeddings.parquet` (20.5 MB, 9,981 with audio / 11,692 total)
- [x] 4.4 Verify: 512-dim, unit-normalized, ~85% coverage matches has_audio flag

## 5. Documentation
- [x] 5.1 Document coverage by album color (MIDI sparsity: Blue 12%, overall 44.3%)
- [ ] 5.2 Document migration path from legacy `training_data_with_embeddings.parquet`

## Notes

- GPU execution migrated from RunPod to Modal (2026-02-12) due to RunPod network storage region matching issues
- Modal scripts: `training/modal_embedding_extraction.py` (DeBERTa), `training/modal_clap_extraction.py` (CLAP)
- Media parquet cached in Modal Volume `white-training-data` for reuse
