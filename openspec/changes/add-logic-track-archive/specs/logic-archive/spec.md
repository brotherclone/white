## ADDED Requirements

### Requirement: Archive Candidate Selection
`white_composition.logic_archive` SHALL classify every Logic song folder under
`$LOGIC_OUTPUT_DIR/<thread>/<folder>/` as one of `placed`, `kept`, `candidate`, or
`unmatched`. The status SHALL be represented by a `str, Enum` in `white_core/enums/`.

- `placed`: the folder's song ID (`<thread>__<production_slug>`, where the production slug
  is taken from the folder's trailing `(<production_slug>)` suffix) appears in any side of
  `$SHRINKWRAP_OUTPUT_DIR/sides.yml`.
- `kept`: the song ID is listed under `songs`, or its thread under `threads`, in the
  optional `$SHRINKWRAP_OUTPUT_DIR/archive_keep.yml`.
- `unmatched`: the folder name has no `(<production_slug>)` suffix, or no matching
  production dir exists.
- `candidate`: anything else.

#### Scenario: Placed song is never a candidate
- **WHEN** `sides.yml` side B contains `the-breathing-machine-learns-to-sing__the_archivists_rebellion`
- **THEN** the folder `the-breathing-machine-learns-to-sing/the_archivists_rebellion (the_archivists_rebellion)` is classified `placed`

#### Scenario: Keep-list thread protects all its songs
- **WHEN** `archive_keep.yml` lists `violet-fallback-defensive-violet-response` under `threads`
- **THEN** every folder under that thread directory is classified `kept`

#### Scenario: Missing keep-list
- **WHEN** `archive_keep.yml` does not exist
- **THEN** classification proceeds using `sides.yml` alone, without error

### Requirement: Dry-Run By Default
The archive CLI (`python -m white_composition.logic_archive`) SHALL default to a dry run.
The dry run prints each folder's classification, the total size of candidates, and the
destination path, and SHALL NOT create, modify or delete any file. `--execute` SHALL be
required to move anything. `--only <song_id>` SHALL restrict the run to one song, and
`--limit N` SHALL cap the number of songs moved.

#### Scenario: Dry run makes no changes
- **WHEN** the CLI is run without `--execute`
- **THEN** a report of placed, kept, candidate and unmatched folders, with sizes, is printed
- **AND** no files under `LOGIC_OUTPUT_DIR` or `LOGIC_ARCHIVE_DIR` are created, modified or deleted

#### Scenario: Unmatched folders need explicit opt-in
- **WHEN** `--execute` is passed without `--include-unmatched`
- **THEN** `unmatched` folders are reported but not moved

### Requirement: Verified Move
With `--execute`, each candidate folder SHALL be moved to
`$LOGIC_ARCHIVE_DIR/<thread>/<folder>/` by copying it to a `.partial` sibling, verifying
it, renaming it into place, rewriting stored paths, recording a manifest entry, and only
then deleting the source.

Verification SHALL confirm that the file counts are equal, and that every regular file's
size and SHA-256 match. If any step before source deletion fails, the `.partial` copy
SHALL be removed, the source SHALL be left untouched, and the song SHALL be reported as
failed while the run continues to the next song.

If the destination folder already exists, the song SHALL be skipped as a conflict and
never merged or overwritten.

`LOGIC_ARCHIVE_DIR` SHALL be read from the environment. If it is unset, or its root does
not exist, `--execute` SHALL raise `EnvironmentError` before any copying starts.

#### Scenario: Successful archive
- **WHEN** a candidate folder is archived with `--execute`
- **THEN** it exists at `$LOGIC_ARCHIVE_DIR/<thread>/<folder>/` with identical file contents
- **AND** the source folder under `LOGIC_OUTPUT_DIR` no longer exists

#### Scenario: Checksum mismatch aborts that song only
- **WHEN** a copied file's SHA-256 differs from its source
- **THEN** the `.partial` destination is removed, the source is unchanged
- **AND** the song is reported as failed and the remaining candidates are still processed

#### Scenario: Destination conflict
- **WHEN** `$LOGIC_ARCHIVE_DIR/<thread>/<folder>/` already exists
- **THEN** the song is skipped and reported as a conflict, and neither copy is modified

#### Scenario: Archive volume not mounted
- **WHEN** `--execute` is passed and the `LOGIC_ARCHIVE_DIR` root does not exist
- **THEN** `EnvironmentError` is raised and nothing is copied or deleted

### Requirement: Thread Folder Preservation
Moves SHALL preserve the `<thread>/<folder>` hierarchy and SHALL never flatten song
folders into the archive root. The destination thread directory SHALL be created when it
doesn't exist.

After a song's source folder is deleted, its source thread directory SHALL be removed if
the only thing left in it is `.DS_Store`. If it contains anything else, it SHALL be left
in place. Restore SHALL recreate the primary thread directory when needed, and SHALL
apply the same empty-thread cleanup to the archive side.

#### Scenario: Thread hierarchy kept in archive
- **WHEN** `violet-fallback-defensive-violet-response/The Cataloguer's Lament (flesh_circuit_taxonomy_v2)` is archived
- **THEN** it exists at `$LOGIC_ARCHIVE_DIR/violet-fallback-defensive-violet-response/The Cataloguer's Lament (flesh_circuit_taxonomy_v2)/`

#### Scenario: Last song in a thread archived
- **WHEN** the final remaining song folder in a thread directory under `LOGIC_OUTPUT_DIR` is archived
- **AND** the thread directory then contains only `.DS_Store` or nothing
- **THEN** the thread directory is removed from `LOGIC_OUTPUT_DIR`

#### Scenario: Thread directory with other content retained
- **WHEN** a song is archived and its source thread directory still holds other files or folders
- **THEN** the thread directory is left in place

#### Scenario: Restore recreates thread directory
- **WHEN** a song is restored and `$LOGIC_OUTPUT_DIR/<thread>/` does not exist
- **THEN** the thread directory is created and the song folder is restored inside it

### Requirement: Stored Path Rewrite
When a folder is moved (archived or restored), the system SHALL rewrite every stored
absolute path that starts with the old folder path so that it starts with the new folder
path instead. This covers:

- `logic_project_path` in the moved folder's `composition.yml`
- any string value in the matching production dir's `song_context.yml` (for example
  `suite_logic_path`)

YAML SHALL be written with `width=float("inf")` so paths are not wrapped.

#### Scenario: composition.yml updated
- **WHEN** a song is archived
- **THEN** its archived `composition.yml` `logic_project_path` points at the
  `LOGIC_ARCHIVE_DIR` location

#### Scenario: suite_logic_path updated
- **WHEN** a production dir's `song_context.yml` has a `suite_logic_path` inside the moved folder
- **THEN** that value is rewritten to the new location, and other keys are left unchanged

### Requirement: Archive Manifest
The system SHALL keep `$SHRINKWRAP_OUTPUT_DIR/archive_manifest.yml`, validated by a
Pydantic model in `white_core`, with one entry per song ID. Each entry records the
source path, destination path, file count, total bytes, last-moved timestamp, and status
(`archived` or `restored`).

#### Scenario: Entry written after archive
- **WHEN** a song is archived successfully
- **THEN** the manifest has an entry for that song ID with status `archived` and the
  verified file count and byte total

### Requirement: Restore
`--restore <song_id> --execute` SHALL move an archived folder back to
`LOGIC_OUTPUT_DIR` using the same copy, verify, rewrite and delete sequence, and SHALL
set the manifest entry's status to `restored`.

#### Scenario: Restore an interstitial source
- **WHEN** `--restore <song_id> --execute` is run for an archived song
- **THEN** the folder is back at `$LOGIC_OUTPUT_DIR/<thread>/<folder>/` with verified contents
- **AND** its `composition.yml` `logic_project_path` points at the primary location
- **AND** the manifest entry status is `restored`
