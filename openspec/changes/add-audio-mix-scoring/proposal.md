# Change: Add Chromatic Scoring for Rendered Audio Mixes

## Why
The production pipeline scores MIDI throughout generation (chords → drums → bass → melody)
but has no way to score the final Logic Pro bounce against its chromatic target. A rendered
mix may drift significantly from the MIDI-time scores — instrumental timbre, mix decisions,
and performance all affect perceived color. This adds a post-render scoring step that closes
the feedback loop.

## What Changes
- New CLI tool `app/generators/midi/production/score_mix.py` that encodes an audio file via
  CLAP and passes the embedding to Refractor in audio-only mode
- Writes `melody/mix_score.yml` with temporal/spatial/ontological scores + per-dimension
  drift delta vs. the song's chromatic target
- Refractor already supports audio-only inference (CLAP pathway); no model changes needed

## Impact
- Affected specs: `audio-mix-scoring` (new), `lyric-generation` (minor — mix_score.yml is
  written alongside lyrics_review.yml in melody/)
- Affected code:
  - New: `app/generators/midi/production/score_mix.py`
  - New: `tests/generators/midi/test_score_mix.py`
  - Read-only: `training/refractor.py`, `training/data/refractor.onnx`
