## 1. Pattern Library
- [x] 1.1 Create `app/generators/midi/patterns/quartet_patterns.py`
- [x] 1.2 Define `VoicePattern` dataclass: name, voice_type (alto/tenor/bass_voice), interval_offsets, rhythm_offsets
- [x] 1.3 Implement voice range constants: alto (48–67), tenor (43–62), bass_voice (36–55)
- [x] 1.4 Implement counterpoint constraint checker: `check_parallels(soprano, voice) → list[str]`
- [x] 1.5 Write `tests/generators/midi/test_quartet_patterns.py` (range clamping, parallel detection)

## 2. Pipeline
- [x] 2.1 Create `app/generators/midi/pipelines/quartet_pipeline.py`
- [x] 2.2 Implement `generate_quartet(production_dir, section, singer)` — reads approved melody MIDI
- [x] 2.3 Generate violin II, viola, cello from soprano via string quartet voice generators
- [x] 2.4 Apply counterpoint constraints, re-roll violating beats
- [x] 2.5 Write multi-channel MIDI: ch0=violin I, ch1=violin II, ch2=viola, ch3=cello
- [x] 2.6 Write `review.yml` candidates with per-voice counterpoint score
- [x] 2.7 Write `tests/generators/midi/test_quartet_pipeline.py`

## 3. CLI
- [x] 3.1 Add `--phase quartet` to pipeline orchestration CLI (or standalone entry point)
