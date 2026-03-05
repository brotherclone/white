# Change: ACE Studio Import — Round-trip MIDI Parser + LRC Export

## Why
`ace_studio_export.py` is a one-way street. After the human finalises vocal synthesis
in ACE Studio, ACE exports a MIDI containing every note with its lyric meta message
attached. That export is the authoritative record of what was actually sung — but nothing
currently reads it back into the pipeline. Parsing it unlocks word-level lyric timing,
syllable-to-note alignment, and LRC subtitle generation without any extra manual work.

## What Changes
- Add `ace_studio_import.py` in `app/generators/midi/` — parses `VocalSynthvN_*.mid`
  exports from a production directory's `VocalSynthvN/` folder
- Reconstructs word tokens from syllable-split lyric meta messages (`iron#1 iron#2` → `iron`)
- Emits a structured note list: `[{word, syllable_index, start_beat, end_beat, pitch, velocity}]`
- Writes an LRC file (`.lrc`) to the production directory with word-level timestamps derived
  from the MIDI tempo and ticks-per-beat
- Exposes a clean Python API consumed by the drift report (see companion change
  `update-production-review-with-drift`)

## Impact
- Affected specs: `ace-studio-mcp`
- Affected code: new file `app/generators/midi/ace_studio_import.py`; new tests
  `tests/generators/midi/test_ace_studio_import.py`
- No breaking changes — purely additive
