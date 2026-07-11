# Change: Add a per-song listen + diary notes panel to the sides screen

## Why
There are now enough songs that critique/notes are hard to hold in memory across the
catalog. Both building blocks already exist — a diary system (`white_diary` +
`/diary/{song_slug}` API, already wired into the client and used on the composition
board) and mix audio playback (an `<audio>` element on the board page) — but neither is
reachable from the sides screen, which is the one place that already lists every mixed
song in one view. There's no way today to play a song's current mix and jot a note
about it without leaving the sides screen to find the right song elsewhere.

## What Changes
- Add a new per-song mix endpoint (`GET /songs/{song_id}/mix` +
  `GET /songs/{song_id}/mix/info`) so an arbitrary song's mix can be streamed by id,
  independent of the server's single global "active song" state that the existing
  `/production/mix` endpoints depend on.
- Add a small "notes" subbutton to each song row on the sides screen (both the
  available-songs pool and songs already assigned to a side) that opens a modal
  combining: the song's mix player (when it has one) and its diary entries — existing
  ones listed, plus the existing diary create-entry form reused as-is.
- **Not** a drag-and-drop target: song rows are already full drag handles for side
  reassignment (dnd-kit, `SideSongRow`/`AvailableSongRow`); a subbutton with
  `stopPropagation` (matching the existing "×" remove button's escape-hatch pattern)
  avoids the row's click/drag activation racing against a new drop target.

## Impact
- Affected specs: `lp-side-sequencing` (adds the per-song mix endpoint and the notes
  panel UI; both are new requirements, no existing requirement changes)
- Affected code:
  - `packages/api/src/white_api/candidate_server.py` — two new routes, reusing the
    existing `_resolve_song_for_sides`, `_song_mix_duration`, and `load_song_context`
    helpers already used by the sides feature
  - `packages/client/app/sides/page.tsx` — new subbutton on `AvailableSongRow` and
    `SideSongRow`; new `SongNotesModal` component
  - `packages/client/lib/api.ts` — new `fetchSongMixInfo`/`songMixStreamUrl` helpers
    (mirroring the existing `fetchMixInfo`/`mixStreamUrl` pair); no changes to the
    diary API, which is reused as-is
- No changes to `white_diary` or its API — this change is purely a new consumer of an
  already-complete capability (see `song-diary` spec, unchanged)
