## MODIFIED Requirements

### Requirement: Logic Project Scaffold
`white_composition.logic_handoff` SHALL create a Logic Pro project folder on the fast
drive when `handoff(production_dir)` is called.

The folder SHALL be created at:
`$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/`

The seed Logic project at `packages/composition/logic/seed.logicx` SHALL be
copied (full directory copy) into that folder and renamed to `<song_title>.logicx`.

If the destination folder already exists, the function SHALL skip the copy and log
a warning rather than raising an error.

`LOGIC_OUTPUT_DIR` SHALL be read from the environment. If unset, the function SHALL
raise `EnvironmentError` with a descriptive message.

`resolve_song_dir(production_dir)` and `handoff()` SHALL resolve the song folder in this order:
1. If `$LOGIC_OUTPUT_DIR/<thread_slug>/<folder>/` exists, return it.
2. Otherwise, if `LOGIC_ARCHIVE_DIR` is set and
   `$LOGIC_ARCHIVE_DIR/<thread_slug>/<folder>/` exists, return that archive path.
3. Otherwise, if `archive_manifest.yml` records the song as `archived` but the archive
   path can't be reached, raise `LogicArchiveOfflineError`. Callers in
   `candidate_server.py` SHALL map this to HTTP 503 with a detail message naming the
   archive volume.
4. Otherwise, return the primary path (today's behaviour for songs that haven't been
   handed off).

`handoff()` SHALL never scaffold a new folder under `LOGIC_OUTPUT_DIR` for a song that
resolves to, or is recorded as being in, the archive.

#### Scenario: Successful scaffold
- **WHEN** `handoff(production_dir)` is called with `LOGIC_OUTPUT_DIR` set and a
  valid production dir containing `song_context.yml`
- **THEN** `$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/<song_title>.logicx` exists
  as a copy of the seed bundle
- **AND** `composition.yml` is created in the same folder (see Composition File requirement)

#### Scenario: Destination already exists
- **WHEN** the Logic song folder already exists at the target path
- **THEN** the copy is skipped, a warning is printed, and the function continues to
  update `composition.yml`

#### Scenario: LOGIC_OUTPUT_DIR not set
- **WHEN** `LOGIC_OUTPUT_DIR` is not set in the environment
- **THEN** `EnvironmentError` is raised with the message
  `"LOGIC_OUTPUT_DIR is not set — add it to .env"`

#### Scenario: Archived song resolves to archive
- **WHEN** a song's folder exists only under `LOGIC_ARCHIVE_DIR`
- **THEN** `resolve_song_dir()` returns the archive path
- **AND** `GET /composition` reads that archive folder's `composition.yml`

#### Scenario: Re-handoff of archived song does not duplicate
- **WHEN** `handoff()` is called for a song whose folder exists only under `LOGIC_ARCHIVE_DIR`
- **THEN** MIDI, text and samples are synced into the archive folder
- **AND** no folder is created under `LOGIC_OUTPUT_DIR`

#### Scenario: Archive offline
- **WHEN** the manifest records a song as `archived` and the archive volume is not mounted
- **THEN** `resolve_song_dir()` raises `LogicArchiveOfflineError`
- **AND** `handoff()` creates nothing under `LOGIC_OUTPUT_DIR`
