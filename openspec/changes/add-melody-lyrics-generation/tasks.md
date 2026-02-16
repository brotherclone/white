# Tasks — add-melody-lyrics-generation

## Group 1: Template library (`melody_patterns.py`)

- [ ] Define `MelodyPattern` dataclass and `SingerRange` dataclass
- [ ] Define `SINGERS` registry with vocal ranges for all 5 singers
- [ ] Implement melody resolution: `resolve_melody_notes(pattern, chord_voicing, singer_range, next_voicing=None)` → list of MIDI notes
- [ ] Implement `select_templates(templates, time_sig, energy)` with energy-adjacent inclusion (reuse pattern from bass)
- [ ] Implement `make_fallback_pattern(time_sig)` for unsupported time signatures
- [ ] Create 12+ 4/4 templates across contour types and energy levels
- [ ] Create 6+ 7/8 templates with asymmetric groupings
- [ ] Implement `clamp_to_singer_range(note, singer)` with interval mirroring
- [ ] Implement `strong_beat_chord_snap(notes, chord_tones, time_sig)`

**Verify**: Unit tests for all resolution functions, template validation, singer range clamping

## Group 2: Theory scoring (`melody_patterns.py`)

- [ ] Implement `singability_score(notes, singer_range)` — interval penalty, range usage, rest placement
- [ ] Implement `chord_tone_alignment(notes, chord_tones, time_sig)` — strong-beat chord-tone fraction
- [ ] Implement `contour_quality(notes)` — arch shape, variety, resolution
- [ ] Implement `melody_theory_score(singability, chord_tone, contour)` → 0.0–1.0

**Verify**: Unit tests for each scoring function with known-good/known-bad inputs

## Group 3: Pipeline (`melody_pipeline.py`)

- [ ] Read approved chords, harmonic rhythm, and section labels (reuse from bass pipeline)
- [ ] Read song proposal including singer assignment
- [ ] Infer singer from key center when not specified
- [ ] Generate melody candidates per section: resolve templates → write MIDI → score
- [ ] Composite scoring: 30% theory + 70% chromatic (consistent with all phases)
- [ ] Write `review.yml` with chromatic synthesis thematic excerpts per section
- [ ] Write candidate MIDI files to `<song>/melody/candidates/`
- [ ] CLI interface: `--production-dir`, `--singer`, `--seed`, `--top-k`, `--onnx-path`

**Verify**: Integration tests with mock ChromaticScorer, MIDI output validation

## Group 4: Vocal prep (`vocal_prep.py`)

- [ ] Parse promoted melody MIDI → (onset_sec, duration_sec, midi_note) list
- [ ] Parse `lyrics.txt` → syllable sequences (whitespace + hyphen splitting)
- [ ] Align syllables to melody notes (1:1, overflow wraps)
- [ ] Output SoulX-Singer metadata JSON per section
- [ ] CLI interface: `--production-dir`, `--section` (or all)

**Verify**: Unit tests for syllable splitting, alignment, and JSON output format
