# Tasks — add-melody-lyrics-generation

## Group 1: Template library (`melody_patterns.py`)

- [x] Define `MelodyPattern` dataclass and `SingerRange` dataclass
- [x] Define `SINGERS` registry with vocal ranges for all 5 singers
- [x] Implement melody resolution: `resolve_melody_notes(pattern, chord_voicing, singer_range, next_voicing=None)` → list of MIDI notes
- [x] Implement `select_templates(templates, time_sig, energy)` with energy-adjacent inclusion (reuse pattern from bass)
- [x] Implement `make_fallback_pattern(time_sig)` for unsupported time signatures
- [x] Create 12+ 4/4 templates across contour types and energy levels
- [x] Create 6+ 7/8 templates with asymmetric groupings
- [x] Implement `clamp_to_singer_range(note, singer)` with interval mirroring
- [x] Implement `strong_beat_chord_snap(notes, chord_tones, time_sig)`

**Verify**: Unit tests for all resolution functions, template validation, singer range clamping — 40 tests pass

## Group 2: Theory scoring (`melody_patterns.py`)

- [x] Implement `singability_score(notes, singer_range)` — interval penalty, range usage, rest placement
- [x] Implement `chord_tone_alignment(notes, chord_tones, time_sig)` — strong-beat chord-tone fraction
- [x] Implement `contour_quality(notes)` — arch shape, variety, resolution
- [x] Implement `melody_theory_score(singability, chord_tone, contour)` → 0.0–1.0

**Verify**: Unit tests for each scoring function with known-good/known-bad inputs — included in test_melody_patterns.py

## Group 3: Pipeline (`melody_pipeline.py`)

- [x] Read approved chords, harmonic rhythm, and section labels (reuse from bass pipeline)
- [x] Read song proposal including singer assignment
- [x] Infer singer from key center when not specified
- [x] Generate melody candidates per section: resolve templates → write MIDI → score
- [x] Composite scoring: 30% theory + 70% chromatic (consistent with all phases)
- [x] Write `review.yml` with chromatic synthesis thematic excerpts per section
- [x] Write candidate MIDI files to `<song>/melody/candidates/`
- [x] CLI interface: `--production-dir`, `--singer`, `--seed`, `--top-k`, `--onnx-path`

**Verify**: Integration tests with mock ChromaticScorer, MIDI output validation — 9 tests pass

## Group 4: Vocal synthesis (external — ACE Studio)

_No pipeline code required._ Vocal synthesis workflow:
1. Import approved melody MIDI into ACE Studio
2. ACE Studio handles syllable parsing and phoneme alignment natively
3. Human adds lyrics (Claude can draft from chromatic synthesis docs)
4. Render vocal audio in ACE Studio, assemble in Logic Pro

~~`vocal_prep.py` removed~~ — originally planned for SoulX-Singer but ACE Studio accepts standard MIDI directly.
