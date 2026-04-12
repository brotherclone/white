## 1. score_mix.py — chunked encoding

- [ ] 1.1 Replace `encode_audio_file` single-pass with `chunk_audio` helper: load
      waveform, resample to 48kHz, yield overlapping 30s windows (default stride 5s),
      return list of numpy arrays
- [ ] 1.2 For each chunk, call `scorer._encode_audio` (or `prepare_audio`) to get a
      512-dim CLAP embedding — collect into a list
- [ ] 1.3 Replace the single `scorer.score(audio_emb=...)` call in `score_mix()` with
      a loop: score each chunk, collect per-chunk result dicts
- [ ] 1.4 Add `aggregate_chunk_scores(results: list[dict]) -> dict` function:
      confidence-weighted mean of temporal/spatial/ontological distributions; overall
      confidence = mean of per-chunk confidences
- [ ] 1.5 Add `chunk_count` and `chunk_stride_s` fields to `mix_score.yml` output
- [ ] 1.6 Update CLI `--chunk-size` and `--chunk-stride` optional flags (defaults:
      30s / 5s) so scoring parameters are visible in the output file

## 2. Tests

- [ ] 2.1 Unit test `chunk_audio`: correct number of chunks, correct lengths, handles
      audio shorter than one chunk window
- [ ] 2.2 Unit test `aggregate_chunk_scores`: weighted mean math, single-chunk
      passthrough, all-zero-confidence fallback (uniform mean)
- [ ] 2.3 Update existing `score_mix` integration tests: stub Refractor to return
      per-chunk results, assert `mix_score.yml` contains `chunk_count`

## 3. validate_mix_scoring.py

- [ ] 3.1 Create `training/validate_mix_scoring.py` with `--artifacts-dir` flag
      (default: `staged_raw_material/`)
- [ ] 3.2 For each song dir, load color from `<id>.yml`, find `<id>_main.wav`
- [ ] 3.3 Score each main.wav using chunked `score_mix` logic; record predicted
      top-1 color vs ground-truth color
- [ ] 3.4 Print per-color accuracy table and overall top-1 accuracy
- [ ] 3.5 Write `training/data/mix_scoring_validation.yml` with per-song results
- [ ] 3.6 If overall accuracy < 70%, emit a prominent warning recommending Phase 2
      calibration head proposal

## 4. Cleanup

- [ ] 4.1 Run full test suite; confirm no regressions in existing score_mix tests
- [ ] 4.2 Re-score The Network Dreams of Synapses with updated score_mix; verify
      confidence > 0.20 and prediction is plausible for Violet
