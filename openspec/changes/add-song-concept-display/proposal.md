# Change: Display song concept at the top of the candidate browser

## Why
When reviewing MIDI candidates the user has no reminder of the song's concept — which
can be elaborate and "out there" even when the musical parameters look simple. Having
the concept visible avoids the need to switch windows to the proposal YAML.

## What Changes
- `scan_songs` in `candidate_server.py` reads `concept` from `song_context.yml` and
  includes it in each song entry (null if `song_context.yml` is absent or has no concept)
- `SongEntry` TypeScript interface gains `concept: string | null`
- `/candidates` page renders the concept in a collapsible block between the breadcrumb
  and the toolbar — collapsed by default to a 3-line clamp, expandable on click.
  Hidden when `activeSong.concept` is null (single-song mode with no context file).

## Notes
- `song_context.yml` is the canonical source; `manifest_bootstrap.yml` does **not**
  carry concept and should not be changed.
- The active change `refactor-song-index-navigation` also touches `scan_songs` and
  `SongEntry` (adds `stage`). This change is additive and does not conflict, but
  implementers should be aware of the overlap when merging.

## Impact
- Affected specs: `candidate-browser-web`
- Affected code:
  - `packages/api/src/white_api/candidate_server.py` — `scan_songs` adds `concept`
  - `packages/api/tests/test_candidate_server.py` — shape test updated
  - `packages/client/lib/types.ts` — `SongEntry` gains `concept`
  - `packages/client/app/candidates/page.tsx` — concept block rendered
