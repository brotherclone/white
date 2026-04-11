# Change: Add monophonic four-part (quartet) generation

## Why
Approved melody lines are currently solo voices. Adding counterpoint voices (alto, tenor,
bass-voice) turns a solo sketch into a full choral or string/brass quartet arrangement
with minimal additional human effort. Each voice is a separate MIDI channel, importable
directly into Logic or ACE Studio.

## What Changes
- New pipeline phase: `quartet_pipeline.py` — reads approved melody MIDI + chord voicings,
  generates three additional voices following basic counterpoint rules
- New template library: `quartet_patterns.py` — counterpoint rules expressed as interval
  relationships to the soprano (melody) voice
- Voices: soprano (existing melody), alto, tenor, bass-voice (distinct from bass-line phase)
- Counterpoint constraints: no parallel 5ths/octaves, prefer contrary motion, voice range
  enforcement per voice type
- Output: single multi-channel MIDI file (`<section>_quartet.mid`) + `review.yml` entries
- `promote_part.py` reused as-is

## Impact
- Affected specs: quartet-generation (new capability)
- Affected code: new `app/generators/midi/pipelines/quartet_pipeline.py`,
  new `app/generators/midi/patterns/quartet_patterns.py`
- Not breaking — purely additive phase; existing songs unaffected
