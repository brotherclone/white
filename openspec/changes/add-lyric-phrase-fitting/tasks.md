## 1. Fitting math
- [x] 1.1 Add `SHORT_PHRASE_NOTE_THRESHOLD` constant and a `_phrase_syllable_range(notes)` helper in `lyric_pipeline.py` that widens the low end of the target range (and disables the "spacious" penalty framing) for phrases at/below the threshold
- [x] 1.2 Wire the helper into the phrase-target loop shared by `_build_prompt` and `_build_white_cutup_prompt`
- [x] 1.3 Update the phrase instruction text to mention melisma as a legal option when a phrase is short

## 2. Verification
- [x] 2.1 Add/adjust unit tests in `packages/generation/tests/test_lyric_pipeline.py` covering: a 1-note phrase gets a widened range, a 2-note phrase gets a widened range, a 4+ note phrase is unaffected (matches current behavior)
- [x] 2.2 Run `pytest packages/generation/tests/test_lyric_pipeline.py` and confirm no regressions to existing fitting-verdict tests
- [x] 2.3 Generate a real candidate against a song with several short melody phrases and manually confirm the output no longer defaults to single monosyllables on those lines — verified against `filing_cipher_prism_v1/melody/lyrics.txt`: short phrases now land on real multisyllable words ("colander," "junipers," "forgotten," "untouched," "blackened") instead of defaulting to monosyllables
