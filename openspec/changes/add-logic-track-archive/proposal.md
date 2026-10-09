# Change: Archive unplaced Logic tracks from the production drive

## Why

The album track order is now mostly settled in `sides.yml`. The Mac HD has 14 GB free,
and `LOGIC_OUTPUT_DIR` (`~/Documents/Music Production/Earthly Frames/White/Tracks`)
holds 74 GB across 155 Logic song folders. Only around 15 of those are placed on a side.
The rest can move to LucidNonsense.

Moving them by hand breaks things, because the handoff derives and records absolute Logic paths:

- `_song_dir()` always resolves to `$LOGIC_OUTPUT_DIR/<thread>/<title> (<slug>)`. Once a
  folder has moved, `/composition`, the board, regression checks and sample export all
  look in the wrong place. A re-handoff would also quietly scaffold an **empty duplicate**
  on the Mac HD.
- `composition.yml` stores an absolute `logic_project_path`. The board's "copy path"
  button uses it.
- Some `song_context.yml` files store an absolute `suite_logic_path`.

## What Changes

- **New `logic_archive` module + CLI** (`white_composition`). It moves Logic song folders
  from `LOGIC_OUTPUT_DIR` to a new `LOGIC_ARCHIVE_DIR`
  (`/Volumes/LucidNonsense/Earthly Frames Archive/The Rainbow Table/09- White - TBD/Tracks/Extras`).
  It keeps the same `<thread>/<song folder>` layout.
- **Selection**: a song is *placed* if its `song_id` appears in `sides.yml`. An optional
  `archive_keep.yml` at the album root lists extra song IDs or whole threads to keep, such
  as songs used as sources for interstitials. Everything else is a candidate.
- **Safety**: dry run by default. With `--execute` it copies, verifies every file by size
  and SHA-256, rewrites the stored paths in the destination copy, records a manifest
  entry, and only then deletes the source. Any failure leaves the source untouched.
- **Restore**: `--restore <song_id>` moves a folder back the same safe way, for an
  interstitial source that's needed again.
- **Path rewriting**: updates `composition.yml` `logic_project_path` and any
  `song_context.yml` path fields (`suite_logic_path`) that point into a moved folder.
- **Archive-aware resolution (MODIFIED logic-handoff)**: `resolve_song_dir()` checks the
  primary location first, then the archive. Handoff refuses to scaffold a fresh folder on
  the primary drive for a song that's already archived. If the archive volume isn't
  mounted, resolution fails with a clear "archive offline" error instead of guessing.
- **Manifest**: `archive_manifest.yml` at the album root (Pydantic model in `white_core`)
  records for each song: source path, destination path, file count, byte total,
  timestamp, and status.

Out of scope: mix bounces under `.../White/Listening` (none of the `mix_file` paths point
into `Tracks/`, so playlist sync and side durations are unaffected), and anything that
edits the inside of `.logicx` bundles.

## Impact

- Affected specs: `logic-handoff` (MODIFIED: project scaffold), `logic-archive` (ADDED)
- Affected code: `packages/composition/src/white_composition/logic_handoff.py`,
  new `packages/composition/src/white_composition/logic_archive.py`,
  new `white_core` manifest model + enum, `packages/api/src/white_api/candidate_server.py`
  (error mapping for archive-offline), `.env` (`LOGIC_ARCHIVE_DIR`)
- Data: moves about 60 GB of Logic folders off the Mac HD. No data is deleted until it
  has been verified on LucidNonsense.
