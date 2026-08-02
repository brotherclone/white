# listening-playlist-sync Specification

## Purpose
Bucket every mixed song into one of three Apple-Music-importable listening folders
(Rejects, Review, White Album WiP) so the local filesystem always mirrors each song's
current `lifecycle_status`/`lp_consideration`, without hand-managing playlists as songs
move through review.

## Requirements
### Requirement: Playlist Output Configuration
The system SHALL persist a configurable playlist output directory in
`playlist_config.yml` at the album root (`$SHRINKWRAP_OUTPUT_DIR/playlist_config.yml`),
defaulting to `<home directory>/Documents/Music Production/Earthly Frames/White/Listening`
when unset.

#### Scenario: Default config on first read
- **WHEN** no `playlist_config.yml` exists and the config is read
- **THEN** the default `output_dir` is returned and the file is materialized on disk
  with that default

#### Scenario: Config update persists
- **WHEN** `POST /playlists/config` is called with a new `output_dir`
- **THEN** subsequent reads (and subsequent syncs) use the new directory

### Requirement: Song Classification Into Three Buckets
The system SHALL classify every song with `has_mix: true` into exactly one of three
mutually-exclusive buckets — Rejects, Review, or White Album WiP — based on
`lifecycle_status` and `lp_consideration`. Songs without a mix are excluded from all
three buckets.

#### Scenario: Rejects bucket
- **WHEN** a song has `has_mix: true` and `lifecycle_status` is `scrapped` or
  `abandoned`
- **THEN** it is classified into Rejects

#### Scenario: Review bucket
- **WHEN** a song has `has_mix: true`, `lp_consideration` is not `placed`, and
  `lifecycle_status` is neither `scrapped` nor `abandoned`
- **THEN** it is classified into Review

#### Scenario: White Album WiP bucket
- **WHEN** a song has `has_mix: true` and `lp_consideration` is `placed`
- **THEN** it is classified into White Album WiP, regardless of `lifecycle_status`

#### Scenario: No mix excludes from all buckets
- **WHEN** a song has `has_mix: false`
- **THEN** it does not appear in Rejects, Review, or White Album WiP regardless of its
  other status fields

### Requirement: White Album WiP Sequencing
White Album WiP files SHALL be named so that a default alphabetical file sort (as used
by Finder and Apple Music folder import) matches the Side A→D, in-side-position order
recorded in `sides.yml`.

#### Scenario: Sequenced song numeric prefix
- **WHEN** a WiP song is present in `sides.yml` at side `S` and position `P`
- **THEN** its synced filename is prefixed `{seq:02d}_{S}_`, where `seq` is a global
  1-based counter across all sides in A→D, in-side-position order

#### Scenario: Unsequenced placed song
- **WHEN** a song has `lp_consideration == placed` but is absent from every side in
  `sides.yml`
- **THEN** it is still synced into White Album WiP, with a filename prefix that sorts
  after every sequenced song (e.g. `99_unsequenced_`)

### Requirement: Full Deterministic Folder Rebuild
Each sync SHALL make each of the 3 destination subfolders under `output_dir` exactly
match the current classification: copying in newly-matching songs' mix files and
deleting any existing file in that subfolder no longer classified into it.

#### Scenario: New match is copied in
- **WHEN** a song newly matches a bucket's criteria that it didn't match on the
  previous sync
- **THEN** its mix file is copied into that bucket's subfolder on the next sync

#### Scenario: Stale file is removed
- **WHEN** a song no longer matches the criteria for the bucket its file currently sits
  in (e.g. a Review song is later marked Abandoned)
- **THEN** its file is removed from that bucket's subfolder on the next sync (it may
  reappear in a different bucket's subfolder if it now matches that bucket)

#### Scenario: Unrelated content outside the 3 subfolders is untouched
- **WHEN** a sync runs
- **THEN** only files inside the `Rejects/`, `Review/`, and `White Album WiP/`
  subfolders of `output_dir` are added or removed; nothing else under `output_dir`, and
  nothing outside it, is modified

#### Scenario: Unchanged files are not re-copied
- **WHEN** a song's target file already exists in its bucket's subfolder with a
  matching size and an mtime no older than the source mix file
- **THEN** the file is left in place rather than being re-copied

### Requirement: Sync Trigger and Result Reporting
The system SHALL expose a sync action via `POST /playlists/sync` returning per-bucket
counts, and the client SHALL expose this as a button on the Sides page.

#### Scenario: Sync returns counts
- **WHEN** `POST /playlists/sync` completes
- **THEN** the response includes `{"rejects": <int>, "review": <int>, "wip": <int>}`
  reflecting the number of files present in each subfolder after the rebuild

#### Scenario: Button triggers sync from the Sides page
- **WHEN** the user clicks "Sync to Playlists" on the Sides page
- **THEN** `POST /playlists/sync` is called and the returned counts are displayed to
  the user

#### Scenario: Misconfigured output directory refuses to sync
- **WHEN** `output_dir` is empty, `/`, or a user's home directory root
- **THEN** the sync is refused with an error rather than performing any file deletion

