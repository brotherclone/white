# Design: Logic track archive

## Context

There are three places a Logic song folder's location matters:

1. **Derived at runtime**: `logic_handoff._song_dir()` builds
   `$LOGIC_OUTPUT_DIR/<thread>/<safe_title> (<production_slug>)`. Every API path goes
   through `resolve_song_dir()`.
2. **Stored inside the folder**: `composition.yml` → `logic_project_path`. This moves with
   the folder, but its value goes stale.
3. **Stored in the production dir**: `song_context.yml` → `suite_logic_path` (3 files
   today). It's data only; no code reads it.

Mix bounces live in `.../White/Listening`, not `Tracks/`, so `mix_file`, side durations
and playlist sync are unaffected.

## Goals / Non-Goals

- Goals: free Mac HD space safely, keep every tool working for archived songs (or failing
  clearly), make archiving reversible.
- Non-Goals: editing `.logicx` internals, archiving `Listening/` bounces, automatic or
  scheduled archiving.

## Decisions

### Resolution: primary first, archive as fallback

`resolve_song_dir()` returns the primary path if it exists. Otherwise it returns the
archive path if `LOGIC_ARCHIVE_DIR` is set and that path exists. If the song has a
manifest entry with status `archived` but the archive root isn't reachable (the volume
isn't mounted), it raises `LogicArchiveOfflineError`. If neither path exists, it returns
the primary path, so a song that has never been handed off behaves exactly as it does today.

- Alternative: a per-song `logic_dir` override in `song_context.yml`. Rejected because
  every consumer would have to read it, and it duplicates the manifest.
- Alternative: symlinks left on the Mac HD. Rejected because they fail silently when the
  volume is unmounted, and Logic doesn't resolve them consistently.

### Handoff guard

`handoff()` scaffolds into whatever `resolve_song_dir()` returns. An archived song gets
re-handed-off into its archive folder, and no empty duplicate is created on the Mac HD.

### Copy → verify → rewrite → delete

1. Copy the tree to `<archive>/<thread>/<folder>.partial` with `shutil.copytree`, keeping
   metadata.
2. Verify that the file count matches, and that size and SHA-256 match for every regular
   file. Symlinks inside the tree are copied as links and compared by target.
3. Atomically rename `.partial` to the final name.
4. Rewrite `logic_project_path` in the **destination** `composition.yml`, and any
   `song_context.yml` path field under the old prefix.
5. Write the manifest entry (`archived`).
6. `shutil.rmtree` the source.

If anything fails before step 6, the `.partial` folder is removed and the source is left
untouched. A destination that already exists is a conflict: the song is skipped and
reported, never merged or overwritten. Restore runs the same steps in the other direction.

### Selection

- Placed: every `song_id` in `sides.yml` (format `<thread>__<production_slug>`).
- Kept: `archive_keep.yml` → `{songs: [song_id...], threads: [thread_slug...]}`.
- Candidates: every `<thread>/<folder>` under `LOGIC_OUTPUT_DIR` that isn't placed or kept.
  Folders are matched to song IDs by the `(<production_slug>)` suffix plus the thread dir.
- A folder that doesn't map to a known production dir is reported as `unmatched`, and is
  only archived when it's passed explicitly with `--include-unmatched`.

### Logic project integrity

Song folders contain the `.logicx` bundle plus sibling `MIDI/`, `Samples/` and
`Recordings/` folders. Logic records both absolute and project-relative file references,
and finds relative ones when the whole folder moves together. Because the folder is
moved as one unit, references should still resolve. The task list includes a manual
smoke test: open one archived project from LucidNonsense before running the bulk archive.
Any audio a project references from *outside* its song folder (for example a sampler
instrument pointing at another drive) is not touched, and Logic may ask for it to be located.

## Risks / Trade-offs

- **Disk space during copy**: copying only writes to LucidNonsense (4.7 TB free), so the
  Mac HD doesn't need extra space. Songs are processed one at a time, and each one frees
  space as soon as it's verified.
- **Hashing time**: 60 GB on an external drive takes a while. Progress is printed for each
  song, and `--limit N` allows batches.
- **Archive unmounted**: archived songs return a clear 503 "archive offline" from the API
  instead of 404s or quiet scaffolding.

## Migration Plan

1. Set `LOGIC_ARCHIVE_DIR` in `.env`.
2. Dry run and review the list. Add interstitial sources to `archive_keep.yml`.
3. Archive one song with `--execute --only <song_id>`, then open it in Logic from
   LucidNonsense.
4. Run the bulk archive with `--execute`.
5. Rollback: `--restore <song_id>` for any song.

## Open Questions

- The recommended defaults (sides + keep-list, archive fallback, auto-delete after
  verification) were chosen without confirmation. Revisit them before apply if they're wrong.
- Should archived songs get a badge on the `/board` page? That would be a small follow-up
  and isn't included here.
