## 1. Dependency
- [ ] 1.1 Add `pyphen>=0.14.0` to `packages/generation/pyproject.toml`

## 2. Core Algorithm
- [ ] 2.1 Implement `syllabify(word) -> list[str]` using pyphen `en_US` dictionary
        — strip punctuation before lookup; fall back to whole-word if pyphen has no split
- [ ] 2.2 Implement `assign_syllables_to_notes(notes, syllables) -> list[tuple[Note, str]]`
        — greedy left-to-right; one syllable per note until syllables exhausted,
          then remaining notes get empty string (melisma continuation)
- [ ] 2.3 Implement `split_note(note, n, ticks_per_beat) -> list[Note]`
        — divides note duration equally into n sub-notes; last sub-note absorbs rounding
          remainder; preserves pitch, channel, velocity
- [ ] 2.4 Implement `auto_split_melody(midi_path, lyrics_path, min_split_ticks, output_path)`
        — orchestrates: parse MIDI phrases → parse lyric lines → assign → split long notes
          → write output MIDI
- [ ] 2.5 Output path convention: `<source_stem>_split.mid` alongside source file

## 3. Phrase-Lyric Alignment
- [ ] 3.1 Reuse `extract_phrases()` from `melody_pipeline.py` to identify phrase boundaries
- [ ] 3.2 Match lyric lines to phrases by index (same 1:1 mapping the lyric pipeline uses)
- [ ] 3.3 Parse lyric line into words, then syllabify each word into ordered syllable list

## 4. API Endpoint
- [ ] 4.1 Add `POST /api/v1/production/auto-split-melody` to `candidate_server.py`
        — accepts `production_dir`, `phase_label` (which approved melody to split),
          optional `min_split_beats` (float, default 1.0)
        — returns path to generated `*_split.mid` and a per-phrase alignment report

## 5. Tests
- [ ] 5.1 Unit test `syllabify()`: common English words, punctuation stripping, fallback
- [ ] 5.2 Unit test `assign_syllables_to_notes()`: equal count, more notes than syllables,
          more syllables than notes
- [ ] 5.3 Unit test `split_note()`: even split, odd-tick remainder absorbed in last note
- [ ] 5.4 Integration test `auto_split_melody()` with a real 4-bar MIDI + lyric fixture:
          assert output note count >= input note count, assert no pitch changes
