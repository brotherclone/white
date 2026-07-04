# Change: Widen syllable targets for short MIDI phrases in lyric generation

## Why
Very short MIDI phrases (1–2 notes) currently get a syllable target of `floor(notes*0.8)`–`ceil(notes*1.15)` — often 1–2 syllables — and the prompt explicitly instructs Claude to hit that target on every line. Across candidates and across songs this repeatedly steers lyrics toward the same small pool of blunt monosyllables ("blue," "dead," "gone"), even though ACE Studio handles melisma (one syllable/word sustained across several notes) automatically. The tight per-note ratio was the right fix for the original "splits needed" (too many syllables) problem, but it overcorrected against short phrases, where a *wider* range — not a tighter one — is appropriate.

## What Changes
- Introduce a note-count threshold (`SHORT_PHRASE_NOTE_THRESHOLD`, default 3) below which the syllable target range is widened rather than computed from the strict 0.8x–1.15x multiplier, and a spacious (fewer syllables than notes) ratio is not treated as a fitting concern for these phrases.
- Update the per-phrase prompt instructions (`_build_prompt`, `_build_white_cutup_prompt`) to explicitly offer melisma as a legal option for short phrases, instead of only ever asking for one-syllable-per-note density.
- No change to phrase extraction, MIDI note counting, or the paste-ready/tight/splits-needed verdict bands for phrases above the threshold.

## Impact
- Affected specs: `lyric-generation` (MODIFIED: Lyric Fitting Score)
- Affected code: `packages/generation/src/white_generation/pipelines/lyric_pipeline.py` — phrase target computation and prompt text in `_build_prompt` (~lines 1065–1085) and `_build_white_cutup_prompt` (~lines 939–956)
