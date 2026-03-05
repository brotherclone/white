# Change: Production Review — Drift Report + Actual Vocal Metrics

## Why
Two gaps surfaced while finishing the blue song:

1. **No drift record.** The pipeline generates melody loops and a human refines them
   in ACE Studio. Once ACE exports the final vocal MIDI, we have no tool to quantify
   how much changed — which pitches moved, whether the rhythm shifted, which lyrics
   were edited. That comparison is the creative audit trail for the song.

2. **Song evaluator uses estimates.** `song_evaluator.py` currently approximates vocal
   coverage and syllable density from bar counts and `lyrics.txt`. After an ACE export
   exists, the evaluator can compute these from real note durations and the actual
   lyric-note alignment, making the report authoritative rather than approximate.

## What Changes
- Add `drift_report.py` in `app/generators/midi/` — compares approved melody loop MIDIs
  (from `melody/approved/`) against the section-segmented notes from the ACE Studio
  import (`load_ace_export`) and writes `drift_report.yml` to the production directory
- Extend `song_evaluator.py` with an `--ace-import` flag: when set, vocal coverage,
  syllable density, and chromatic lyric alignment are derived from the ACE export MIDI
  rather than from estimates
- Adds `actual_vocal_coverage`, `actual_syllable_density`, and `ace_chromatic_alignment`
  fields to `song_evaluation.yml` when ACE data is available

## Impact
- Affected specs: `production-review`
- Affected code:
  - New: `app/generators/midi/drift_report.py`
  - Modified: `app/generators/midi/song_evaluator.py`
  - New tests: `tests/generators/midi/test_drift_report.py`
- Depends on: `add-ace-studio-import` (uses `load_ace_export`)
- No breaking changes — evaluator extension is opt-in via flag
