## 1. Core Scorer
- [ ] 1.1 Add `encode_audio_file(path) -> np.ndarray` in `score_mix.py` — loads WAV/AIFF/MP3
      via librosa, runs through CLAP processor, returns 512-dim embedding
- [ ] 1.2 Add `score_mix(audio_path, production_dir) -> dict` — encodes audio, calls
      `Refractor.score()` with audio embedding only (no MIDI, no concept), computes
      `compute_chromatic_match()` vs song target from proposal
- [ ] 1.3 Add `drift_report(score_result, target) -> dict` — per-dimension delta between
      predicted and target distributions (temporal/spatial/ontological)
- [ ] 1.4 Add `write_mix_score(result, drift, melody_dir)` — writes `mix_score.yml`

## 2. Song Metadata
- [ ] 2.1 Reuse `_find_and_load_proposal()` from `lyric_pipeline.py` to get color/concept;
      extract to shared utility in `production/` to avoid duplication

## 3. CLI
- [ ] 3.1 `--mix-file` (required): path to rendered audio bounce
- [ ] 3.2 `--production-dir` (required): song production directory
- [ ] 3.3 `--onnx-path` (optional): override Refractor ONNX path
- [ ] 3.4 Print per-dimension scores + drift summary; print mix_score.yml path on completion

## 4. Tests
- [ ] 4.1 Unit test: `drift_report()` computes correct deltas for a known target
- [ ] 4.2 Unit test: `write_mix_score()` round-trips through YAML correctly
- [ ] 4.3 Integration test: stub Refractor + CLAP; verify full `score_mix()` flow writes
      expected mix_score.yml structure
