## 1. Playlist config
- [x] 1.1 Define `playlist_config.yml` read/write helpers in
      `packages/composition/src/white_composition/playlist_sync.py`:
      `load_playlist_config(album_dir)`, `save_playlist_config(album_dir, config)`,
      materializing the default `output_dir` if the file is absent
- [x] 1.2 Unit tests: default config on first read, round-trip save/load, updating
      `output_dir`

## 2. Classification
- [x] 2.1 Implement `classify_songs(songs, sides_doc) -> dict[str, list[SongEntry]]`
      returning `{"rejects": [...], "review": [...], "wip": [...]}` per the rules in
      `proposal.md`, given `scan_songs()` output and a loaded `SidesDocument`
- [x] 2.2 WiP ordering: songs present in `sides.yml` ordered by Side A→D + in-side
      position; songs with `lp_consideration == placed` but absent from `sides.yml`
      appended at the end (unsequenced)
- [x] 2.3 Unit tests: each of the 3 buckets' membership rules (including the "has no
      mix" exclusion and the mutual-exclusivity of the 3 buckets), WiP ordering with a
      mix of sequenced/unsequenced placed songs

## 3. Filesystem rebuild
- [x] 3.1 Implement `sync_playlists(songs, sides_doc, output_dir) -> dict[str, int]`:
      for each of the 3 subfolders, compute the target filename set (sanitized title +
      collision-suffix rule for Rejects/Review, numeric-prefix rule for WiP per
      `design.md`), copy any file whose target doesn't yet exist or whose source has a
      newer mtime/different size, delete any existing file in the subfolder not in the
      target set, and return per-folder synced counts
- [x] 3.2 Filename sanitization helper (strip characters invalid in filenames, preserve
      source extension)
- [x] 3.3 Guard against a misconfigured `output_dir` (empty, `/`, or home directory
      root) — refuse to sync rather than deleting broadly
- [x] 3.4 Unit tests against a temp directory: full rebuild adds new files, removes
      stale ones, leaves unrelated files/folders outside the 3 subfolders untouched,
      collision suffixing, WiP numeric prefixing order

## 4. API endpoints
- [x] 4.1 `GET /playlists/config` — return current `output_dir`
- [x] 4.2 `POST /playlists/config` — update `output_dir` (body: `{output_dir}`)
- [x] 4.3 `POST /playlists/sync` — run classification + rebuild, return
      `{"rejects": n, "review": n, "wip": n}`
- [x] 4.4 Integration tests in `packages/api/tests` covering the above against a temp
      album dir with fixture manifests/mixes across all 3 buckets

## 5. Client UI
- [x] 5.1 `lib/api.ts` additions: `fetchPlaylistConfig`, `setPlaylistConfig`,
      `syncPlaylists`
- [x] 5.2 Sides page (`packages/client/app/sides/page.tsx`): inline editable
      output-directory field (loads current value, saves via `setPlaylistConfig`) and
      a "Sync to Playlists" button
- [x] 5.3 Post-sync result display: per-folder counts (`Rejects: 5, Review: 12, White
      Album WiP: 9`) and an error state if sync fails (e.g. misconfigured directory)
- [x] 5.4 Manual verification: verified against a fixture album (not real data) via
      direct API calls through the running server/client stack — confirmed correct
      bucketing and Side A→D WiP filename ordering on disk (`01_A_...`, `02_B_...`).
      The Chrome browser extension wasn't connected this session, so the actual button
      click wasn't exercised visually in a real browser tab — worth a quick manual
      click-through against real data before considering this fully done end-to-end.
