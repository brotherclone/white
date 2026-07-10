# Change: Add Song Lifecycle Statuses (Merged, Abandoned, Scrapped)

## Why

Songs don't always finish as independent tracks. Some get spliced into a multi-part suite;
others stall permanently; others get cannibalized for parts in later productions. The system
has no way to record or surface any of these outcomes, so dead or absorbed songs clutter the
song index with no way to distinguish them from active work. Three new lifecycle statuses —
**Merged**, **Abandoned**, and **Scrapped** — give each end-state a first-class representation
with dedicated UI actions and filtered views.

## What Changes

- `manifest_bootstrap.yml` gains an optional `lifecycle_status` field
  (`null` / `"merged"` / `"abandoned"` / `"scrapped"`) and an optional
  `merged_with` list of song IDs (populated only for the `merged` status).
  An optional `uses_parts_from` list of scrapped song IDs records which
  scrapped songs contributed material to the active song.
- `_compute_stage()` returns the lifecycle status string directly when it is set,
  bypassing mix-stage logic.
- `SongEntry.stage` type extended with `"merged" | "abandoned" | "scrapped"`.
- Song index filter: `all` pill **excludes** merged / abandoned / scrapped songs;
  three new explicit pills (`merged`, `abandoned`, `scrapped`) let you opt in.
- `/board` page gains a **Song Lifecycle** panel with three actions:
  - **Merge** — opens a song-picker modal; both the active song and the chosen
    song are set to `merged`; each records the other's ID in `merged_with`.
  - **Abandon** — opens a confirmation dialog with a whimsical plea for the
    song's life before writing `abandoned`.
  - **Scrap** — confirmation dialog (distinct flavour from Abandon) writes `scrapped`.
- `/board` page Song Lifecycle panel gains a **"Uses parts from"** disclosure widget
  showing a list of all scrapped songs; selections are persisted to `uses_parts_from`
  in the active song's `manifest_bootstrap.yml`. This is a bookkeeping link only —
  it lets the producer track which scrapped songs donated material to this production.
- New API endpoints:
  - `POST /songs/{id}/lifecycle` — set lifecycle status (body: `{status, merged_with?}`)
  - `GET /songs/scrapped` — return all songs with `lifecycle_status: scrapped`
  - `PATCH /songs/{id}/uses-parts-from` — update `uses_parts_from` list

## Impact

- Affected specs: `chain-artifacts`, `candidate-browser-web`
- Affected code:
  - `packages/composition/src/white_composition/shrinkwrap_chain_artifacts.py` — manifest schema
  - `packages/api/src/white_api/candidate_server.py` — `_compute_stage`, `scan_songs`, new endpoints
  - `packages/client/lib/types.ts` — `SongEntry.stage` union
  - `packages/client/lib/api.ts` — new API helpers
  - `packages/client/app/page.tsx` — filter pills, card rendering, exclude logic
  - `packages/client/app/board/page.tsx` — Song Lifecycle panel
  - `packages/client/app/board/page.tsx` — Song Lifecycle panel + "Uses parts from" widget

## Out of Scope

- Reversing a lifecycle status (un-abandon, un-merge) — deliberate omission; these
  are intentional terminal states. Can be added later if needed.
- Bulk lifecycle operations from the song index.
