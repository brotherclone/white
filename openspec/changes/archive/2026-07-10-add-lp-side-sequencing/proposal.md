# Change: Add LP-side sequencing (drag-and-drop mix assembly against a 20-minute limit)

## Why
White is the only double album in the catalog, and there is currently no way to see how
finished mixes add up against real LP-side runtime limits — `song_context.yml` doesn't
even track mix duration. The user wants a lightweight, duration-aware sequencing surface:
assign completed mixes into 4 sides (A–D) and see cumulative time against a ~20-minute
soft ceiling per side, used as a compositional/sequencing constraint even though the
record won't be physically pressed.

## What Changes
- Extract and cache mix duration (via `soundfile`, already a project dependency) for any
  song with a mix file, exposed through the existing `GET /production/mix/info` endpoint.
- New `sides.yml` data file at the album root (`$SHRINKWRAP_OUTPUT_DIR`, alongside
  `index.yml`) storing 4 sides (A–D), each an ordered list of song IDs with cached
  durations, and a configurable `side_limit_seconds` (default 1200 = 20 minutes).
- New `candidate_server.py` endpoints: list sides with computed totals, assign a song to
  a side at a position, reorder within a side, move between sides, remove from a side.
- New client page (`packages/client/app/sides/`) with 4 drag-and-drop columns, one per
  side, each showing assigned songs (title, duration) and a running total with a visual
  warning when the side exceeds the soft limit.
- Only songs with `has_mix: true` are assignable; songs without a mix are shown as
  unavailable/non-draggable in the source list.

## Impact
- Affected specs: `lp-side-sequencing` (new capability)
- Affected code: `packages/api/src/white_api/candidate_server.py` (new endpoints), new
  `packages/composition/src/white_composition/lp_sides.py` (data model + duration
  extraction + read/write helpers), new `packages/client/app/sides/page.tsx`,
  `packages/client/lib/api.ts` / `lib/types.ts` additions, new client dependency for
  drag-and-drop (see `design.md`)
