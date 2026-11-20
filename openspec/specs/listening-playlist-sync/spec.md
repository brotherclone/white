# listening-playlist-sync Specification

## Purpose
Bucket every mixed song into one of three Apple-Music-importable listening folders
(Rejects, Review, White Album WiP) so the local filesystem always mirrors each song's
current `lifecycle_status`/`lp_consideration`, without hand-managing playlists as songs
move through review.

## Requirements

### Requirement: Playlist Output Configuration
The system SHALL persist a configurable playlist output directory and a WiP album
title in `playlist_config.yml` at the album root
(`$SHRINKWRAP_OUTPUT_DIR/playlist_config.yml`). `output_dir` SHALL default to
`<home directory>/Documents/Music Production/Earthly Frames/White/Listening` and
`wip_album_title` SHALL default to `White Album WiP` when unset.

#### Scenario: Default config on first read
- **WHEN** no `playlist_config.yml` exists and the config is read
- **THEN** the default `output_dir` and `wip_album_title` are returned and the file is
  materialized on disk with those defaults

#### Scenario: Existing config without album title
- **WHEN** `playlist_config.yml` exists with only `output_dir`
- **THEN** the config loads with the stored `output_dir` and the default
  `wip_album_title`

#### Scenario: Config update persists
- **WHEN** `POST /playlists/config` is called with a new `output_dir`
- **THEN** subsequent reads (and subsequent syncs) use the new directory

#### Scenario: Partial config update preserves other fields
- **WHEN** `POST /playlists/config` is called with only `wip_album_title`
- **THEN** the stored `output_dir` is unchanged and the new title is persisted

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

### Requirement: White Album WiP Metadata Tagging
After copying a song's mix into `White Album WiP/`, the system SHALL write title,
album (`wip_album_title`) and track-number ID3 tags on the copy when it is an mp3. The
title SHALL come from `song_context.yml` (`song_slug`, else `title`, else the scanned
title). Only the copy is tagged; source mixes and `Rejects/`/`Review/` files SHALL NOT
be modified.

#### Scenario: Sequenced WiP mp3 is tagged
- **WHEN** a sync copies an mp3 for a song at global sequence position `seq` out of
  `total` sequenced songs
- **THEN** the copy's album tag equals `wip_album_title` and its track number is
  `seq/total`

#### Scenario: Title tag prefers song_slug
- **WHEN** a WiP song has a non-blank `song_slug`
- **THEN** the copy's title tag equals `song_slug`, while its filename is still derived
  from the scanned title (`manifest_bootstrap.yml`)

#### Scenario: Blank song_slug falls back to the context title
- **WHEN** a WiP song's `song_slug` is empty or blank and `song_context.yml` has a
  non-blank `title`
- **THEN** the copy's title tag equals that `title`, replacing any title the source mix
  carried, while the filename is still derived from the scanned title

#### Scenario: Unsequenced WiP mp3 is tagged without a track number
- **WHEN** a sync copies an mp3 for a placed song absent from `sides.yml`
- **THEN** the copy's album tag equals `wip_album_title` and it has no track number

#### Scenario: Source mix is untouched
- **WHEN** a sync tags a WiP copy
- **THEN** the source mix file referenced by `song_context.yml` is byte-for-byte
  unchanged

#### Scenario: Other buckets are not tagged
- **WHEN** a song is synced into `Rejects/` or `Review/`
- **THEN** its copy keeps the source file's original tags

#### Scenario: Non-mp3 WiP mix is not tagged
- **WHEN** a WiP song's mix is not an mp3 (e.g. `.wav`)
- **THEN** it is copied without tagging and the sync completes normally

#### Scenario: Tagging failure does not abort the sync
- **WHEN** writing tags to one WiP copy fails
- **THEN** a warning is logged, the untagged copy remains, and the remaining songs are
  still synced and counted
