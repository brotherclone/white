## 1. Dependency
- [x] 1.1 Add `pyphen>=0.14.0` to `packages/generation/pyproject.toml`

## 2. Core Algorithm
- [x] 2.1 Implement `syllabify(word) -> list[str]` using pyphen `en_US` dictionary
        — strip punctuation before lookup; fall back to whole-word if pyphen has no split
- [x] 2.2 Implement `assign_syllables_to_notes(notes, syllables) -> list[tuple[Note, str]]`
        — greedy left-to-right; one syllable per note until syllables exhausted,
          then remaining notes get empty string (melisma continuation)
- [x] 2.3 Implement `split_note(note, n, ticks_per_beat) -> list[Note]`
        — divides note duration equally into n sub-notes; last sub-note absorbs rounding
          remainder; preserves pitch, channel, velocity
- [x] 2.4 Implement `auto_split_melody(midi_path, lyrics_path, min_split_ticks, output_path)`
        — orchestrates: parse MIDI phrases → parse lyric lines → assign → split long notes
          → write output MIDI
- [x] 2.5 Output path convention: `<source_stem>_split.mid` alongside source file

## 3. Phrase-Lyric Alignment
- [x] 3.1 Reuse `extract_phrases()` from `lyric_pipeline.py` to identify phrase boundaries
- [x] 3.2 Match lyric lines to phrases by index (same 1:1 mapping the lyric pipeline uses)
- [x] 3.3 Parse lyric line into words, then syllabify each word into ordered syllable list

## 4. API Endpoint
- [x] 4.1 Add `POST /production/auto-split-melody` to `candidate_server.py`
        — accepts `phase_label` (which approved melody to split),
          optional `min_split_beats` (float, default 1.0)
        — returns path to generated `*_split.mid` and a per-phrase alignment report

## 5. Tests
- [x] 5.1 Unit test `syllabify()`: common English words, punctuation stripping, fallback
- [x] 5.2 Unit test `assign_syllables_to_notes()`: equal count, more notes than syllables,
          more syllables than notes
- [x] 5.3 Unit test `split_note()`: even split, odd-tick remainder absorbed in last note
- [x] 5.4 Integration test `auto_split_melody()` with a real 4-bar MIDI + lyric fixture:
          assert output note count >= input note count, assert no pitch changes
