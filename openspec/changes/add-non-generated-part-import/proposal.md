# Change: Non-Generated Part Import

## Why
When the pipeline fails to produce a usable candidate for a given section (e.g. chorus
melody), the composer writes their own MIDI file. Currently there is no way to bring that
file into the pipeline — downstream phases cannot consume it, strum/bass/lyric generation
cannot align to it, and the candidate browser cannot display it.

## What Changes
- New `register_part()` function (lives beside `promote_part.py`) that accepts an external
  MIDI file and writes it into a phase's `approved/` directory with a synthetic review.yml
  entry (`generated: false`, `status: approved`)
- New API endpoint `POST /api/v1/production/register-part` wrapping the above
- `candidate_browser` updated to render non-generated entries with a visual marker and skip
  score columns (no scores exist for hand-written parts)
- `candidate_browser` updated to tolerate `scores: null` and `rank: null` in review.yml
  (these were already causing crashes — see bugs fixed in this session)

## Impact
- Affected specs: `non-generated-part-import` (new), `candidate-browser` (modified)
- Affected code:
  - `packages/generation/src/white_generation/pipelines/promote_part.py` — add `register_part()`
  - `packages/api/src/white_api/candidate_server.py` — add `/register-part` endpoint
  - `packages/api/src/white_api/candidate_browser.py` — already partially fixed; add display marker
