# Change: Add song completion dashboard

## Why
With 10+ songs in progress across the album, it is not obvious which songs have all
pipeline phases complete and which are blocked on a specific phase. This information
currently exists only in the filesystem and in the producer's head.

## What Changes
- New CLI tool: `app/tools/song_dashboard.py` — reads all songs under `shrink_wrapped/`
  and prints a phase-completion matrix using `rich.table`
- For each song: shows chord / drum / bass / melody / quartet phase status
  (✓ approved, ⚠ pending candidates, ✗ no candidates, — not started)
- Also shows: singer, key, color, total approved bar count, production plan status
- Reads directly from `review.yml` files and `production_plan.yml` — no new state
- Optionally filters to a single color with `--color red`

## Impact
- Affected specs: song-completion-dashboard (new capability)
- Affected code: new `app/tools/song_dashboard.py`
- Read-only — makes no writes to any file
