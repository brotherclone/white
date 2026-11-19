## MODIFIED Requirements
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

## ADDED Requirements
### Requirement: White Album WiP Metadata Tagging
After copying a song's mix into `White Album WiP/`, the system SHALL write ID3 tags on
the copied file when it is an mp3: the album tag set to the configured
`wip_album_title`, and the track number set from the song's position in the WiP
sequence. Tags SHALL be written to the copy only; source mix files and files in
`Rejects/` and `Review/` SHALL NOT be tagged or modified.

#### Scenario: Sequenced WiP mp3 is tagged
- **WHEN** a sync copies an mp3 for a song at global sequence position `seq` out of
  `total` sequenced songs
- **THEN** the copy's album tag equals `wip_album_title` and its track number is
  `seq/total`

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
