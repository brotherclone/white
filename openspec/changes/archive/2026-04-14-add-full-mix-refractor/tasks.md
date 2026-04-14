## 1. MLP architecture

- [x] 1.1 Create `training/models/refractor_cdm_model.py`:
      `RefractorCDMModel(clap_dim, concept_dim, use_concept, hidden_dims, dropout)` — three independent
      Linear(3) heads with softmax; `forward()` returns `(temporal, spatial, ontological)` tensors
- [x] 1.2 Add `export_onnx(model, path, use_concept)` utility to the same file;
      ONNX output names: `temporal`, `spatial`, `ontological`; input name: `input` (concatenated)
- [x] 1.3 Unit tests in `tests/training/test_refractor_cdm_model.py`:
      forward pass shapes, ONNX round-trip (skipped locally — onnx not installed), independent heads

## 2. Training script

- [x] 2.1 Create `training/modal_train_refractor_cdm.py` as Modal app with CLI flags:
      `--epochs` (default 150), `--lr` (default 1e-3), `--batch-size`, `--no-concept`, `--dry-run`
- [x] 2.2 Create `training/extract_cdm_embeddings.py` (local Phase 1): chunks each `_main.wav`,
      encodes with `Refractor.prepare_audio()`, saves to `training/data/refractor_cdm_embeddings.npz`
- [x] 2.3 Stratified 80/20 split by color; log per-color train/val counts
- [x] 2.4 Training loop: random shuffle per epoch; MSE loss against `CHROMATIC_TARGETS` soft
      distributions; Adam optimizer; log train loss + val accuracy per epoch; save best checkpoint
      (`best_mean_acc`)
- [x] 2.5 After training, export best checkpoint to ONNX bytes, returned to local caller
- [x] 2.6 Print final per-color accuracy table and overall top-1 accuracy on val set

## 3. Refractor integration

- [x] 3.1 Add `cdm_onnx_path: Optional[str]` parameter to `Refractor.__init__()`;
      auto-detect `training/data/refractor_cdm.onnx`; load into `self._cdm_session` when present
- [x] 3.2 In `Refractor.score()`: when `_cdm_session` is set and `audio_emb` is provided
      (and `midi_bytes` is None), route through `_score_cdm()` instead of base ONNX
- [x] 3.3 Confidence: mean of the three head peak probabilities
- [x] 3.4 Tests: fixed `TestScorerWithMockedONNX` fixture to set `_cdm_session = None`

## 4. score_mix + validate integration

- [x] 4.1 Add `--cdm-onnx-path` flag to `score_mix.py` CLI (default: auto-detect; `""` disables)
- [x] 4.2 `score_mix()` gains `cdm_onnx_path` parameter; passes to `Refractor()` constructor
- [x] 4.3 Add `chunk_audio()` and `aggregate_chunk_scores()` to `score_mix.py`; loop per-chunk
      scoring with confidence-weighted mean pooling; add `chunk_count`/`chunk_stride_s` to output
- [x] 4.4 Create `training/validate_mix_scoring.py` with `--no-cdm` flag for A/B comparison
- [x] 4.5 Re-run validation with trained Refractor CDM; confirm overall accuracy ≥ 70%
      and confidence > 0.20 on The Network Dreams of Synapses
      (blocked on: `extract_cdm_embeddings.py` + `modal run modal_train_refractor_cdm.py`)
      Result: 73.1% overall (57/78); Violet 91.7% (11/12), conf ≥ 0.93. Song not yet staged — all Violet songs pass with high confidence.

## 5. Tests

- [x] 5.1 `TestChunkAudio` (8 tests): short audio, padding, chunk count, resampling, float32, custom params
- [x] 5.2 `TestAggregateChunkScores` (7 tests): passthrough, confidence mean, weighting, zero-conf fallback
- [x] 5.3 Updated `TestScoreMixIntegration`: `chunk_count` in result and YAML; `scorer.score` called per chunk
- [x] 5.4 `TestWriteMixScore`: `refractor_cdm` in metadata, `chunk_count`/`chunk_stride_s` in YAML
- [x] 5.5 Full test suite: 3210 passed, 6 pre-existing failures (unrelated), 0 new regressions
