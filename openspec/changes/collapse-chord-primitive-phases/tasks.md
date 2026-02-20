## 1. Chord Pipeline — Bake HR + Strum into Candidates

- [x] 1.1 Import `strum_patterns.py` HR distribution sampler and strum template library into `chord_pipeline.py`
- [x] 1.2 After generating each Markov progression, randomly sample an HR distribution and strum pattern (seeded)
- [x] 1.3 Apply HR distribution to chord durations and strum pattern to voicings before writing candidate MIDI
- [x] 1.4 Record `hr_distribution` and `strum_pattern` in the candidate metadata for inclusion in `review.yml`
- [x] 1.5 Update `review.yml` generation to include `hr_distribution`, `strum_pattern`, and `scratch_midi` fields per candidate

## 2. Scratch Beat Generation

- [x] 2.1 Add `generate_scratch_beat(section, bpm, bars, genre_family, time_sig) → bytes` helper in `chord_pipeline.py`
- [x] 2.2 Select the lowest-energy drum template from the matched genre family
- [x] 2.3 Write `<candidate>_scratch.mid` alongside each chord candidate MIDI
- [x] 2.4 Add `scratch: true` flag in `review.yml` entries for scratch files (or list as a field on the candidate)

## 3. Promotion — One-Per-Label Enforcement

- [x] 3.1 Update `promote_chords.py` to detect duplicate labels among approved candidates (renamed to `promote_part.py`)
- [x] 3.2 Fail with a clear error listing conflicts when duplicates are found
- [x] 3.3 Ensure scratch MIDI files are excluded from promotion (filter `_scratch.mid` from candidate list)
- [x] 3.4 Remove the `_1`, `_2` numbering logic for multi-approved-per-label (now an error condition)

## 4. Drum Pipeline — Section Labels Only

- [x] 4.1 Update `drum_pipeline.py` to read section names from `chords/approved/*.mid` filenames
- [x] 4.2 Remove any logic that reads from `harmonic_rhythm/approved/` or handles HR-derived filenames
- [x] 4.3 Add filter to ignore `_scratch.mid` files when scanning candidates directory

## 5. Remove HR and Strum Pipeline Modules

- [x] 5.1 Delete `app/generators/midi/harmonic_rhythm_pipeline.py`
- [x] 5.2 Delete `app/generators/midi/strum_pipeline.py`
- [x] 5.3 Remove or archive associated tests: `tests/generators/midi/test_harmonic_rhythm_pipeline.py`,
       `tests/generators/midi/test_strum_pipeline.py`
- [x] 5.4 Retain `strum_patterns.py` and `harmonic_rhythm.py` as internal libraries (used by chord pipeline)

## 6. Update `add-production-plan` Spec Delta

- [x] 6.1 Edit `openspec/changes/add-production-plan/specs/production-plan/spec.md` to remove the
       `harmonic rhythm MIDI if present` fallback from the bar count scenario
- [x] 6.2 Validate `add-production-plan` after update

## 7. Tests

- [x] 7.1 Update `tests/generators/midi/test_chord_pipeline.py` to cover HR + strum baking and scratch beat output
- [x] 7.2 Update `tests/generators/midi/test_promote_chords.py` to cover one-per-label enforcement (renamed to `test_promote_part.py` via `TestPromotePart`)
- [x] 7.3 Update `tests/generators/midi/test_drum_pipeline.py` to confirm section-label-only reading
- [x] 7.4 Run full test suite and confirm no regressions

## 8. Validate

- [x] 8.1 All tests passing (280+ tests, no regressions)
