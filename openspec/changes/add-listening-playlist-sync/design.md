## Context
`scan_songs()` in `candidate_server.py` already returns, per song, everything needed to
classify it: `has_mix`, `lifecycle_status` (`None`/`merged`/`abandoned`/`scrapped`),
`lp_consideration` (`not_considered`/`candidate`/`placed`), `title`, `production_path`.
`lp_sides.py`'s `load_sides()` gives the ordered per-side (A–D) song_id list needed for
WiP sequencing. `song_context.yml`'s `mix_file` field is the absolute path to the actual
audio file to copy — set via the existing `/songs/{song_id}/mix/set` endpoint, arbitrary
format (`.mp3`/`.wav`/`.aiff`/`.m4a`).

There is no existing "export to external directory" pattern in the codebase — everything
so far reads/writes inside `$SHRINKWRAP_OUTPUT_DIR`. This is the first feature that
writes real files *outside* the repo/album tree, onto the user's Music-import folder.

## Goals / Non-Goals
- Goals: classify every song with a mix into exactly one of Rejects/Review/WiP (or none,
  if it has no mix); on each sync, make each destination folder's contents exactly match
  the current classification (add what's newly matching, remove what no longer is);
  preserve Side A→D + position order for WiP via filename prefixing so Music/Finder sort
  order matches intended playback order.
- Non-Goals: no audio transcoding/format conversion (files are copied byte-for-byte, kept
  in their original format); no actual Apple Music / iTunes Library API integration —
  the user drags/imports the synced folders themselves; no incremental/diffed copy
  (every sync recomputes and rewrites full folder contents, per user's explicit choice);
  no dedup across the three folders (a song can only ever match one bucket by
  construction, since the three classifications are mutually exclusive and exhaustive
  over "has a mix").

## Decisions

- **Decision: classification is computed fresh from `scan_songs()` on every sync, not
  cached.** There's no new persisted "which bucket is this song in" state — bucket
  membership is fully derived from existing `lifecycle_status`/`lp_consideration`/
  `has_mix` fields already on disk. This keeps `playlist_sync.py` a pure
  classify-then-rebuild function with no state of its own beyond the configurable
  `output_dir`.

- **Decision: full deterministic rebuild per folder, not incremental.** Each of the 3
  subfolders (`Rejects/`, `Review/`, `White Album WiP/`) is fully recomputed: existing
  files not in the new computed set are deleted, files in the new set not yet present
  are copied. This means a song moving from Review → Rejects (e.g. getting marked
  Abandoned) cleanly disappears from Review and appears in Rejects on the next sync,
  with no manual cleanup. Confirmed with the user as the desired behavior over an
  additive-only approach.
  - Risk: if `output_dir` is misconfigured to point at a folder containing unrelated
    files, those files get deleted on sync. Mitigated by scoping deletion to only the 3
    named subfolders under `output_dir` (never deleting `output_dir` itself or sibling
    content), and by only ever deleting files this tool itself would have created
    (matched by extension against known audio formats, to avoid nuking e.g. a stray
    `.DS_Store` — though harmless, or a user's own unrelated file if they pointed the
    config at an existing folder).

- **Decision: copy real files, not symlinks.** Confirmed with the user — reliability for
  phone/car listening (works even if the source production directory is later moved or
  cleaned up, and definitely resolves correctly when synced via Finder/iCloud/USB to a
  phone) outweighs the disk-space cost.

- **Decision: WiP filenames get a zero-padded global sequence prefix** —
  `{seq:02d}_{side}_{sanitized_title}{ext}`, where `seq` runs 1..N across the full
  Side A→D + in-side-position order (not reset per side), e.g. `01_A_song_one.mp3`,
  `02_A_song_two.mp3`, `09_B_song_nine.mp3`. This guarantees Finder/Music's default
  alphabetical file sort exactly matches the intended play order without needing an
  audio-tagging dependency. Confirmed with the user over writing ID3 track-number tags.
  - A WiP song not yet assigned to any side (i.e. `lp_consideration == placed` but
    absent from `sides.yml`) is classified as WiP but placed after all sequenced songs,
    prefixed `99_unsequenced_{title}{ext}`, so it's still synced (nothing with
    `lp_consideration == placed` is silently dropped) without corrupting the numbering
    of songs that *are* sequenced.

- **Decision: `playlist_config.yml` at the album root**, sibling to `sides.yml`, storing
  only `{output_dir: <path>}`. Not `.env` — confirmed with the user, since `.env` holds
  secrets and this is a UI-configurable, per-checkout path setting. Created with the
  default path on first read if absent (same "materialize on first use" pattern as
  `sides.yml`).

- **Decision: filename collisions within a folder are disambiguated by appending a short
  suffix from the song's `production_slug`.** Song titles aren't guaranteed unique
  (two versions/attempts of a song can share a title); sanitized-title-only filenames
  would silently overwrite each other during copy. Suffix is only appended when a
  collision is actually detected within that sync's file set, so the common case stays
  a clean `songtitle.mp3`.

## Risks / Trade-offs
- Full-rebuild-with-deletion is a destructive operation on a directory outside the repo.
  Mitigated by scoping to the 3 named subfolders and by validating `output_dir` is set
  (non-empty, not `/` or the user's home directory root) before any deletion runs.
- No incremental copy means large libraries re-copy unchanged files' bytes are *not*
  re-copied — sync only touches files that are new/changed (compares mtime+size before
  copying) or need deletion; only the *decision* of what belongs in each folder is fully
  recomputed each time, not the file I/O itself.

## Migration Plan
Net-new feature; no existing data migrates. `playlist_config.yml` is created with the
default `output_dir` on first sync if absent, mirroring `sides.yml`'s creation pattern.
The three destination subfolders are created under `output_dir` on first sync if absent.

## Open Questions
None outstanding — sync behavior, file placement, WiP ordering mechanism, and config
location were all confirmed with the user before scaffolding.
