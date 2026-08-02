## 1. Backend — per-song mix endpoint
- [x] 1.1 Add `GET /songs/{song_id}/mix/info` to `candidate_server.py`, reusing
      `_resolve_song_for_sides`, `_song_mix_duration`, and `load_song_context`
- [x] 1.2 Add `GET /songs/{song_id}/mix`, mirroring `stream_mix()`'s content-type
      resolution but resolving the production dir via `_resolve_song_for_sides`
- [x] 1.3 Tests: has_mix true/false, unknown song_id → 404, correct content-type per
      extension, missing file on disk → 404

## 2. Client — API helpers
- [x] 2.1 Add `fetchSongMixInfo(songId)` and `songMixStreamUrl(songId)` to
      `packages/client/lib/api.ts`, mirroring `fetchMixInfo`/`mixStreamUrl`

## 3. Client — SongNotesModal
- [x] 3.1 Build `SongNotesModal` component (visual pattern matching the existing
      `DiaryModal`/`LyricModal`): header with song title, close button
- [x] 3.2 Conditionally render `<audio controls>` (has_mix true) or a "no mix file yet"
      message (has_mix false)
- [x] 3.3 Fetch and list existing diary entries via `fetchDiaryEntries(production_slug)`,
      reversed for most-recent-first display
- [x] 3.4 Reuse the existing diary create-entry form fields/behavior, submitting via
      `createDiaryEntry(production_slug, entry)`; append the new entry to the list on
      success without a full reload

## 4. Client — wire the notes button into the rows
- [x] 4.1 Add a small notes button to `SideSongRow`, next to the existing "×" button,
      with `stopPropagation` in its `onClick`
- [x] 4.2 Add the same button to `AvailableSongRow` (independent of `has_mix` /
      `disabled` — the button must work even when the row itself isn't draggable)
- [x] 4.3 Track which song's modal is open in `SidesPage` state; render one
      `SongNotesModal` instance conditionally

## 5. Validation
- [x] 5.1 `openspec validate add-song-notes-panel --strict`
- [x] 5.2 `pytest packages/api/tests/` for the new endpoint tests
- [x] 5.3 Manual smoke test in the browser: open notes panel from both the available
      pool and an assigned side row, confirm drag-to-reorder still works, play a mix,
      add a diary entry and see it appear
