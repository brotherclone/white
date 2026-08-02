# Change: Add listening-playlist sync (Rejects / Review / White Album WiP)

## Why
The pipeline now generates a large volume of candidate mixes, and there's no way to
listen to them away from the desktop — on a phone or in the car — without manually
copying files around. The user wants a one-button sync that buckets every song with a
finished mix into 3 folders by its current triage status, so those folders can be
dragged into Apple Music (or synced via Finder/iCloud) for offline listening.

## What Changes
- New `playlist_config.yml` at the album root (`$SHRINKWRAP_OUTPUT_DIR/playlist_config.yml`,
  alongside `sides.yml`/`index.yml`), storing a single configurable `output_dir` (default
  `/Users/gabrielwalsh/Documents/Music Production/Earthly Frames/White/Listening`).
- New `packages/composition/src/white_composition/playlist_sync.py` module (mirroring the
  `lp_sides.py` pattern): reads/writes `playlist_config.yml`, classifies songs into
  Rejects / Review / White Album WiP per the rules below, and performs a full
  deterministic rebuild of each of the 3 destination subfolders (copy real mix files in,
  remove anything no longer matching).
  - **Rejects**: `lifecycle_status in {scrapped, abandoned}` AND `has_mix`.
  - **Review**: `lp_consideration != placed` AND `lifecycle_status not in {scrapped,
    abandoned}` AND `has_mix`.
  - **White Album WiP**: `lp_consideration == placed`. Ordered by `sides.yml`'s Side
    A→D order and in-side position; filenames get a zero-padded sequence prefix
    (`01_A_songtitle.mp3`) so Music/Finder sort matches the intended play order.
- New `candidate_server.py` endpoints: `GET /playlists/config`, `POST /playlists/config`
  (update `output_dir`), `POST /playlists/sync` (runs the rebuild, returns per-folder
  counts).
- New "Sync to Playlists" button plus an inline output-directory field on the existing
  Sides page (`packages/client/app/sides/page.tsx`), showing a result summary
  (`Rejects: 5, Review: 12, White Album WiP: 9`) after sync.

## Impact
- Affected specs: `listening-playlist-sync` (new capability)
- Affected code: `packages/api/src/white_api/candidate_server.py` (new endpoints), new
  `packages/composition/src/white_composition/playlist_sync.py`, new
  `packages/client/lib/api.ts` additions, `packages/client/app/sides/page.tsx`
- No changes to `lp-side-sequencing` behavior — this change only *reads* `sides.yml`,
  it doesn't write to it.
- Branch: stays on the current feature branch (`feature/dupes` — see note below) per
  user instruction; treated as part of this in-flight feature branch rather than a new
  one.
