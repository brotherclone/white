# Change: Add White Synthesis Pipeline

## Why

The White album is the chromatic synthesis of all seven color songs. Rather than generating
new chord progressions from scratch, White should materially inherit from its sub-proposals —
transposing approved chord progressions into its key, reassembling their bars using a
cut-up technique (Burroughs/Gysin), and regenerating lyrics from the same cut-up principle
applied to accepted sub-lyrics. This makes White a true synthesis artifact, not just another
generated song.

## What Changes

- **White song proposal schema**: new optional `sub_proposals` field — a list of production
  directory paths to draw donor chord and lyric material from.
- **White chord generation mode**: when `rainbow_color` is `White`, `chord_pipeline.py`
  switches to "donor mode" — it reads approved MIDI files from each sub-proposal, transposes
  them to the White key, adjusts BPM, extracts individual bars, and generates candidates by
  randomly splicing bars into new progressions (cut-up).
- **MIDI rebracketing utilities**: new module
  `app/generators/midi/pipelines/white_rebracketing.py` that handles MIDI transposition,
  BPM rescaling, and bar extraction/reassembly.
- **White lyric cut-up mode**: when the lyric pipeline detects White color, it reads approved
  lyric files from each sub-proposal and feeds them to Claude as cut-up source material
  instead of generating lyrics from scratch from the concept alone.
- **Drum, bass, and melody phases**: unchanged — they run on the White chords exactly as they
  do for any color song. No modifications needed.

## Impact

- Affected specs: `chord-generation`, `lyric-generation`, `pipeline-orchestration`
- Affected code:
  - `app/generators/midi/pipelines/chord_pipeline.py` — White branch in main pipeline
  - `app/generators/midi/pipelines/lyric_pipeline.py` — cut-up prompt branch
  - New: `app/generators/midi/pipelines/white_rebracketing.py`
  - `openspec/specs/pipeline-orchestration/spec.md` — sub_proposals field
- **No breaking changes** to existing color song pipelines
