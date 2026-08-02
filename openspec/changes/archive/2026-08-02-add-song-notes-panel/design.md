## Context
The sides screen (`packages/client/app/sides/page.tsx`) is the one view that already
lists every mixed song across every thread in one place. Two things needed for
per-song critique already exist elsewhere in the app — diary entries (board page) and
mix playback (board page, but scoped to a single global "active" production) — this
change wires both into the sides screen without duplicating either system.

## Goals / Non-Goals
- Goals: play a song's current mix and read/add diary notes without leaving the sides
  screen; do it without disturbing the existing drag-and-drop reordering interaction.
- Non-Goals: editing or deleting existing diary entries from this panel (the existing
  `DiaryModal` on the board page doesn't support that either — creation-only is the
  established pattern); changing the diary or mix-file-set APIs; a rubric/score field
  on diary entries (explicitly deferred — see conversation history, revisit once a
  batch of freeform notes exists to mine for structure).

## Decisions

### Subbutton, not a new drop target
Both `AvailableSongRow` and `SideSongRow` are already full dnd-kit drag handles
(`{...attributes}{...listeners}` spread across the whole row) for side reassignment.
The only existing escape hatch is `SideSongRow`'s "×" remove button, which calls
`e.stopPropagation()` in its `onClick` before the drag listeners see the pointer event.
A new "open notes panel" subbutton follows the exact same pattern. A drag-to-open
alternative was considered and rejected: it would require `onDragEnd` to disambiguate
"dropped on a side column" (reassign) from "dropped on the notes panel" (open panel),
which is real new ambiguity for no benefit over a plain button click.

### Per-song mix endpoint keyed by the sides screen's composite song id, not diary's song_slug
The existing mix endpoints (`GET /production/mix(/info)`) resolve against the server's
single global `_production_dir`/`_active_song` state — fine for the board page (one
song open at a time) but unusable from the sides screen, which shows many songs from
many threads simultaneously. The new endpoints take `song_id` in the URL and resolve it
via the existing `_resolve_song_for_sides()` helper (already used by the sides
assign/move/remove routes), which matches on `SongEntry.id` —
`f"{thread_slug}__{production_slug}"`, guaranteed unique across the whole album.

This deliberately does *not* reuse the diary API's keying scheme: `fetchDiaryEntries`/
`createDiaryEntry` key by plain `production_slug` (see `board/page.tsx`, which passes
`composition.production_slug`), which is fine for the board page (one thread's songs at
a time) but could theoretically collide across threads on the sides screen (e.g. two
different threads both producing a `_v1`-suffixed slug). That's pre-existing diary
behavior, out of scope to change here — noted so a future session doesn't assume the
two systems share a key format. The notes panel keeps using `production_slug` for
diary calls (matching the board page's existing usage) and the composite `id` only for
the new mix endpoint.

### Reuse `_song_mix_duration` and `load_song_context`, don't reload the song list
The sides screen already fetches the full `SongEntry` list (including `has_mix`) on
load; the panel doesn't need a new "does this song have a mix" check — it already
knows from the row's own `song.has_mix` field. The new `/mix/info` endpoint's
`duration_seconds` is fetched lazily only when the panel opens (not prefetched for
every row), since it requires a `soundfile.info()` call per song.

## Risks / Trade-offs
- Two mix-serving endpoints now exist (`/production/mix` for the board page's "active
  song" flow, `/songs/{id}/mix` for arbitrary songs) with near-identical bodies. Not
  consolidated in this change — the active-song endpoints have a different resolution
  path (global state, not `_resolve_song_for_sides`) and unifying them is a separate,
  larger refactor not needed to ship this feature.
