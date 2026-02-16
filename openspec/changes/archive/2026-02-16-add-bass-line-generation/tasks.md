# Tasks: Add Bass Line Generation

## Implementation Order

### 1. Bass Pattern Template Library
- [x] Create `app/generators/midi/bass_patterns.py`
- [x] Define `BassPattern` dataclass with tone-selection-based note specs
- [x] Implement tone resolution logic (root, 5th, 3rd, octave, approach, passing)
- [x] Define ~15 templates for 4/4 (root, walking, pedal, arpeggiated, octave, syncopated)
- [x] Define ~5 templates for 7/8 (group-aligned positions)
- [x] Implement fallback pattern for unsupported time signatures
- [x] Implement template selection by time signature and energy
- [x] Write unit tests for tone resolution (all tone types, edge cases, register clamping)
- [x] Write unit tests for template selection and fallback

**Validates**: Templates can be selected by time sig/energy, tone resolution produces correct MIDI notes

### 2. Theory Scoring
- [x] Add theory scoring functions to `bass_patterns.py`
- [x] Implement `root_adherence()` — fraction of strong-beat notes that are chord root
- [x] Implement `kick_alignment()` — fraction of bass onsets matching kick hits
- [x] Implement `voice_leading_score()` — smoothness of bass movement between chords
- [x] Implement `bass_theory_score()` — mean of available components
- [x] Write unit tests for each scoring component

**Validates**: Theory scoring functions return correct values for known inputs

### 3. Bass Pipeline
- [x] Create `app/generators/midi/bass_pipeline.py`
- [x] Implement chord root extraction from approved MIDI (reuse `parse_chord_voicings` from strum_pipeline)
- [x] Implement kick onset extraction from approved drum MIDI
- [x] Implement harmonic rhythm reading (reuse from strum_pipeline)
- [x] Implement bass MIDI generation (resolve templates to notes, write channel 0)
- [x] Integrate ChromaticScorer for chromatic scoring
- [x] Implement composite scoring (theory + chromatic)
- [x] Implement review YAML generation
- [x] Implement CLI with argparse
- [x] Write unit tests for chord root extraction
- [x] Write unit tests for MIDI generation (correct register, channel, note count)
- [x] Write unit tests for composite scoring
- [x] Write integration test (end-to-end with mock production directory)

**Validates**: Pipeline runs end-to-end, produces MIDI files + review.yml, candidates are scored and ranked

### 4. Verification
- [x] Run pipeline against an existing production directory with approved chords/drums/harmonic rhythm
- [x] Verify MIDI files play correctly (notes in bass register, correct channel)
- [x] Verify review.yml has correct structure and scores
- [x] Verify promote_chords.py works with bass review.yml

**Validates**: Full integration with existing pipeline stages — 183 tests passing, 0 failures

## Dependencies

- Task 2 depends on Task 1 (needs BassPattern for test data)
- Task 3 depends on Tasks 1 and 2
- Task 4 depends on Task 3
- Tasks 1 and 2 can be partially parallelized (tone resolution needed by both)

## Notes

- Reuse `parse_chord_voicings` from `strum_pipeline.py` for chord MIDI parsing
- Reuse `read_approved_harmonic_rhythm` from `strum_pipeline.py` for harmonic rhythm
- Reuse `load_song_proposal`, `compute_chromatic_match`, `get_chromatic_target`, `_to_python` from `chord_pipeline.py`
- Reuse `promote_chords.py` as-is for promotion workflow
