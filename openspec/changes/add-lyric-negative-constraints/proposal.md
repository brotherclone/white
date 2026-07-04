# Change: Add lyric-scoped negative constraints to avoid word/imagery convergence

## Why
Lyric candidates converge on the same small set of concrete words and images across
songs (and across candidates within a song) — e.g. "blue thing" in one song, "dead" in
the next — because nothing in the lyric pipeline tracks what language has already been
used. `packages/extraction/src/white_extraction/util/generate_negative_constraints.py`
solves a structurally similar convergence problem, but it operates on song-proposal
metadata (key, BPM, title words, concept phrases, dialogue openers) for the White
ideation agent's `negative_constraints.yml`, and has no visibility into lyric text —
it is not wired into `lyric_pipeline.py` at all. This is a genuinely different concern
(word/imagery frequency in generated lyric text) consumed by a different pipeline, and
needs its own mechanism rather than extending the agent-level one.

## What Changes
- New module `lyric_negative_constraints.py` (same analysis pattern as
  `generate_negative_constraints.py`: `Counter`-based frequency counting + threshold +
  severity) scoped entirely to lyric text.
- Walks an album/thread's promoted `melody/lyrics.txt` files (optionally including
  pending `melody/candidates/lyrics_*.txt`) and computes: overused short/monosyllabic
  words, overused concrete nouns/images, and per-word frequency across the album.
- Writes `lyrics_negative_constraints.yml` at the shrink-wrapped album root (same
  location pattern as the existing `negative_constraints.yml` and `index.yml`).
- `lyric_pipeline.py` loads this file when present (no error if absent) and injects a
  formatted avoidance block into the generation prompt, the same way `artist_context`
  is injected today.
- Explicitly does **not** modify `generate_negative_constraints.py` or
  `negative_constraints.yml` — the two mechanisms remain independent, with different
  inputs (proposal metadata vs. lyric text) and different consumers (White agent vs.
  lyric pipeline).

## Impact
- Affected specs: `lyric-negative-constraints` (new capability)
- Affected code: new `packages/generation/src/white_generation/lyric_negative_constraints.py`;
  `packages/generation/src/white_generation/pipelines/lyric_pipeline.py` (prompt building,
  new CLI flag)
