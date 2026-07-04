## 1. Duration extraction
- [x] 1.1 Add a `mix_duration_seconds(path)` helper (using `soundfile.info`) to
      `packages/composition/src/white_composition/lp_sides.py`
- [x] 1.2 Extend `GET /production/mix/info` in `candidate_server.py` to include
      `duration_seconds` (null if no mix or unreadable file)
- [x] 1.3 Unit tests for duration extraction against a short fixture WAV, and for the
      missing-file / no-mix cases

## 2. Sides data model
- [x] 2.1 Define `sides.yml` read/write helpers in `lp_sides.py`: `load_sides(album_dir)`,
      `save_sides(album_dir, sides)`, initializing empty A–D sides if the file is absent
- [x] 2.2 Implement `assign_song(sides, song_id, side, position, duration_seconds)`,
      `move_song(sides, song_id, to_side, to_position)`, `remove_song(sides, song_id)`
- [x] 2.3 Implement `side_totals(sides)` returning per-side summed duration and an
      `over_limit` flag against `side_limit_seconds`
- [x] 2.4 Unit tests for assign/move/remove/reorder and total/over-limit computation

## 3. API endpoints
- [x] 3.1 `GET /sides` — return all 4 sides with songs, cached durations, and totals
- [x] 3.2 `POST /sides/{side}/assign` — assign a song at a position (body: song_id, position)
- [x] 3.3 `POST /sides/{side}/move` — move a song within/between sides
- [x] 3.4 `DELETE /sides/{side}/songs/{song_id}` — remove a song from a side
- [x] 3.5 Integration tests in `packages/api/tests` covering the above against a temp
      album dir with fixture manifests

## 4. Client UI
- [x] 4.1 Add `@dnd-kit/core` to `packages/client` — `@dnd-kit/sortable` turned out to be
      unnecessary: the page refetches `/sides` after every drop rather than maintaining
      optimistic client-side reordering, so plain `useDraggable`/`useDroppable` covers it
- [x] 4.2 New page `packages/client/app/sides/page.tsx` with 4 drop-column layout
- [x] 4.3 Side column: song list (title, duration), running total, over-limit visual warning
- [x] 4.4 Available-songs source list: only `has_mix: true` songs draggable
- [x] 4.5 `lib/api.ts` / `lib/types.ts` additions for the new endpoints
- [~] 4.6 Manual verification: drag a mixed song onto side A, confirm total updates and
      warning appears once the side exceeds 20 minutes — **partially verified**: `tsc`
      and `next build` pass, and I ran the exact API sequence the drag handlers issue
      (assign/move/remove) live against a fixture album, confirming durations, ordering,
      and the no-mix 400 rejection all work end-to-end. I could not drive an actual
      pointer-drag gesture in a browser (Chrome extension wasn't connected in this
      session) — **please try the real drag interaction yourself** before treating this
      as fully verified.
