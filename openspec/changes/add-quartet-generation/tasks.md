## 1. Pattern Library
- [ ] 1.1 Create `app/generators/midi/patterns/quartet_patterns.py`
- [ ] 1.2 Define `VoicePattern` dataclass: name, voice_type (alto/tenor/bass_voice), interval_offsets, rhythm_offsets
- [ ] 1.3 Implement voice range constants: alto (48–67), tenor (43–62), bass_voice (36–55)
- [ ] 1.4 Implement counterpoint constraint checker: `check_parallels(soprano, voice) → list[str]`
- [ ] 1.5 Write `tests/generators/midi/test_quartet_patterns.py` (range clamping, parallel detection)

## 2. Pipeline
- [ ] 2.1 Create `app/generators/midi/pipelines/quartet_pipeline.py`
- [ ] 2.2 Implement `generate_quartet(production_dir, section, singer)` — reads approved melody MIDI
- [ ] 2.3 Generate alto, tenor, bass-voice from soprano via interval offset templates
- [ ] 2.4 Apply `check_parallels`, re-roll violating beats
- [ ] 2.5 Write multi-channel MIDI: ch0=soprano, ch1=alto, ch2=tenor, ch3=bass-voice
- [ ] 2.6 Write `review.yml` candidates with per-voice counterpoint score
- [ ] 2.7 Write `tests/generators/midi/test_quartet_pipeline.py`

## 3. CLI
- [ ] 3.1 Add `--phase quartet` to pipeline orchestration CLI (or standalone entry point)
