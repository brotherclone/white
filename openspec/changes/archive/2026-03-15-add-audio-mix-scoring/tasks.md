## 1. Core Scorer
- [x] 1.1 Add `encode_audio_file(path) -> np.ndarray` in `score_mix.py` — loads WAV/AIFF/MP3
      via librosa, runs through CLAP processor, returns 512-dim embedding
- [x] 1.2 Add `score_mix(audio_path, production_dir) -> dict` — encodes audio, calls
      `Refractor.score()` with audio embedding only (no MIDI, null concept), computes
      `compute_chromatic_match()` vs song target from proposal
- [x] 1.3 Add `chromatic_drift_report(score_result, target) -> dict` — per-dimension delta
      between predicted and target distributions (temporal/spatial/ontological)
- [x] 1.4 Add `write_mix_score(result, drift, melody_dir)` — writes `mix_score.yml`

## 2. Song Metadata
- [x] 2.1 Reuse `_find_and_load_proposal()` from `lyric_pipeline.py` to get color/concept
      (imported directly — same pattern as `lyric_feedback_export.py`; dedicated shared
      utility deferred)

## 3. CLI
- [x] 3.1 `--mix-file` (required): path to rendered audio bounce
- [x] 3.2 `--production-dir` (required): song production directory
- [x] 3.3 `--onnx-path` (optional): override Refractor ONNX path
- [x] 3.4 Print per-dimension scores + drift summary; print mix_score.yml path on completion

## 4. Tests
- [x] 4.1 Unit test: `chromatic_drift_report()` computes correct deltas for a known target
- [x] 4.2 Unit test: `write_mix_score()` round-trips through YAML correctly
- [x] 4.3 Integration test: stub Refractor + CLAP; verify full `score_mix()` flow writes
      expected mix_score.yml structure
