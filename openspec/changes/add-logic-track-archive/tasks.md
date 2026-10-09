## 1. Models and config
- [x] 1.1 Add `ArchiveClassification` (`placed`/`kept`/`candidate`/`unmatched`) and `ArchiveStatus` (`archived`/`restored`) `str, Enum`s in `white_core/enums/`
- [x] 1.2 Add Pydantic `ArchiveManifest` / `ArchiveManifestEntry` and `ArchiveKeepList` models in `white_core`
- [x] 1.3 Add `LOGIC_ARCHIVE_DIR="/Volumes/LucidNonsense/Earthly Frames Archive/The Rainbow Table/09- White - TBD/Tracks/Extras"` to `.env` (and `.env.example` if present)

## 2. Archive-aware resolution (logic-handoff)
- [x] 2.1 Add `LogicArchiveOfflineError` and primary → archive → offline → primary resolution in `_song_dir()` / `resolve_song_dir()`
- [x] 2.2 Make sure `handoff()` never scaffolds under `LOGIC_OUTPUT_DIR` for an archived song
- [x] 2.3 Map `LogicArchiveOfflineError` to HTTP 503 at the `resolve_song_dir` call sites in `candidate_server.py`
- [x] 2.4 Tests: archive fallback, re-handoff into the archive, offline error, unchanged behaviour when `LOGIC_ARCHIVE_DIR` is unset

## 3. Archive module + CLI
- [x] 3.1 `logic_archive.classify()`: walk `LOGIC_OUTPUT_DIR`, match `(<slug>)` suffixes, apply `sides.yml` and `archive_keep.yml`
- [x] 3.2 Dry-run report (classification, sizes, destination)
- [x] 3.3 `move_song()`: copy to `.partial`, verify size and SHA-256, rename, rewrite paths, write manifest, delete source; clean up on failure; skip on conflict
- [x] 3.4 Thread-folder preservation: create the destination thread dir, remove the source thread dir when it holds only `.DS_Store` or nothing, recreate it on restore
- [x] 3.5 Path rewrite for `composition.yml` and `song_context.yml` (`width=float("inf")`)
- [x] 3.6 `--restore`, `--only`, `--limit`, `--include-unmatched`, `--execute` flags
- [x] 3.7 Tests using tmp dirs: classification, thread hierarchy kept, empty-thread cleanup and retention, thread recreated on restore, dry run makes no changes, checksum-mismatch rollback, conflict skip, missing archive root, path rewrite, restore round trip

## 4. Rollout (manual, with the user)
- [x] 4.1 Dry run against the real `LOGIC_OUTPUT_DIR` (21 placed / 131 candidates, 45.3 GB / 3 unmatched); `archive_keep.yml` left for the user to fill in with interstitial sources
- [x] 4.2 `--execute --only <one song>`, then open that project in Logic from LucidNonsense to confirm audio and MIDI references resolve
- [x] 4.3 Bulk `--execute` (130 archived; 2 accented-name failures fixed via NFC normalisation and retried), all 131 manifest entries resolve to the archive with correct `logic_project_path`

## 5. Finish
- [x] 5.1 Bump `packages/composition` and `packages/core` versions (minor); bump `packages/api` if the 503 mapping changes it
- [x] 5.2 `graphify update .`
