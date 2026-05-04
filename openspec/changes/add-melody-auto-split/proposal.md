# Change: Melody Auto-Split

## Why
When a generated (or hand-written) melody has more notes than a lyric line has syllables,
the composer must manually subdivide notes in ACE Studio before syllables can be placed.
This is slow and error-prone. The system already identifies when splitting is needed
(`ratio > 1.30` verdict in the lyric pipeline) but takes no action. A first-pass
auto-split reduces manual work from "rebuild the note grid by hand" to "nudge the split
points."

## What Changes
- New `auto_split_melody(midi_path, lyrics_path, production_dir, ...)` function in a
  new module `white_generation/pipelines/melody_auto_split.py`
- Uses `pyphen` (rule-based hyphenation, no ML) to break words into syllables
- Greedily assigns syllables to notes left-to-right within each phrase
- Any note that carries a multi-syllable word AND whose duration ≥ `min_split_ticks`
  (configurable, default = 1 beat) is subdivided into N equal-duration sub-notes at the
  same pitch and velocity
- Outputs a `*_split.mid` alongside the source MIDI — source is never modified
- New API endpoint `POST /api/v1/production/auto-split-melody` triggers the above

## Non-Goals
- Stress-weighted or linguistically informed split timing (equal subdivision only, for now)
- Automatic melisma detection or contraction (N notes → 1 syllable handled by ACE Studio)
- Non-English lyrics (pyphen supports other locales but only `en_US` is in scope here)

## Impact
- Affected specs: `melody-auto-split` (new)
- Affected code:
  - New: `packages/generation/src/white_generation/pipelines/melody_auto_split.py`
  - Modified: `packages/api/src/white_api/candidate_server.py` — new endpoint
  - New dependency: `pyphen>=0.14.0` in `packages/generation/pyproject.toml`
