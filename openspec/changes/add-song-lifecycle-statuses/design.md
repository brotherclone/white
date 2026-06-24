# Design: Song Lifecycle Statuses

## Context

Lifecycle statuses are terminal song states distinct from the production pipeline stages
(ideation → generation → … → complete). They're not stages in the mix pipeline; they
describe *why a song stopped progressing*.  The system already stores per-song metadata in
`manifest_bootstrap.yml` and computes a display stage in `_compute_stage()`, making the
manifest the natural home for lifecycle state.

## Goals / Non-Goals

- **Goals**: surface Merged / Abandoned / Scrapped in the song index and board with
  minimal new persistence surface; no new files or directories.
- **Non-Goals**: reversibility; bulk operations; workflow automation triggered by
  lifecycle changes.

## Decisions

### Storage: `manifest_bootstrap.yml`, not a separate file

`manifest_bootstrap.yml` is already the canonical per-song metadata file read by
`scan_songs()` and surfaced in `GET /songs`. Adding two optional fields
(`lifecycle_status`, `merged_with`, `uses_parts_from`) keeps reads in one place and
requires no migration for existing songs (absent field → active/null).

### `_compute_stage()` checks lifecycle first

Insert a guard at the top of `_compute_stage()`:

```python
lc = mb.get("lifecycle_status")
if lc in ("merged", "abandoned", "scrapped"):
    return lc
```

This means the existing mix-stage logic is only reached for active songs, and the
type union `SongEntry["stage"]` simply gains three new string members.

### "All" filter excludes lifecycle-terminal songs

The existing `all` pill shows `songs.length` — with lifecycle statuses this would
include dead songs and inflate the count misleadingly. Redefine `all` as
"songs with no lifecycle status set" (i.e. `stage` not in
`["merged", "abandoned", "scrapped"]`). Each lifecycle status gets its own pill so
they're reachable without typing.

### Merge is bilateral

When Song A merges into Song B, both records must be updated atomically (within one
request). The `POST /songs/{id}/lifecycle` endpoint accepts `merged_with: [id]` and
the server writes both manifests. The client sends `{ status: "merged",
merged_with: [partner_id] }` from the active song's board; the server resolves the
partner's production path and writes its manifest too.

`merged_with` is a **list** (not a scalar) to accommodate N-part suites without a
schema change later. For a two-part merge, each song's list contains the other's ID.

### "Uses parts from" lives on the consuming song

The association between an active song and a scrapped donor belongs to the consumer's
`manifest_bootstrap.yml` as `uses_parts_from: [song_id, ...]`. The scrapped donor's
manifest is not modified. This keeps the scrapped song's record pristine and makes the
relationship directional and explicit.

### Lifecycle actions and "Uses parts from" both on `/board`

The board page is the natural home for song-level decisions (advance/regress stage,
handoff). The lifecycle panel fits alongside the existing RegressionModal. "Uses parts
from" is a producer bookkeeping link — it records which scrapped songs donated material
to this production — so it belongs in the same lifecycle panel on `/board`, not in the
Chromatic Samples table on `/candidates`.

## Risks / Trade-offs

- **No undo**: terminal states are permanent in the current design. Risk is low because
  the manifest file can be edited manually to clear `lifecycle_status` if needed.
- **Partner manifest write**: merging requires writing two manifests in one API call.
  If the write to the second manifest fails after the first succeeds, state is
  inconsistent. Mitigation: write both to temp buffers, then rename atomically (or
  accept the rare inconsistency given the low stakes of this operation).

## Open Questions

- Should clicking a merged/abandoned/scrapped song card still activate and navigate to
  `/board`? Current proposal: yes, for inspection purposes. The lifecycle panel on the
  board will show the terminal status prominently.
