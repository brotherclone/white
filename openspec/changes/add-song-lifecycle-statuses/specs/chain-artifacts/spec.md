## MODIFIED Requirements

### Requirement: Manifest Bootstrap Schema
The `manifest_bootstrap.yml` file SHALL contain the following fields:
`schema_version`, `stub`, `title`, `rainbow_color`, `bpm`, `key`, `singer`.
Optional fields SHALL include `suite`, `suite_part`, `suite_logic_path`, `sounds_like`,
`time_sig`, and the new lifecycle fields described below.

The `manifest_bootstrap.yml` file MAY contain a `lifecycle_status` field whose value is
one of `"merged"`, `"abandoned"`, or `"scrapped"`. When absent or `null`, the song is
considered active.

When `lifecycle_status` is `"merged"`, the file MAY also contain a `merged_with` field
holding a list of song IDs (strings in `{thread_slug}__{production_slug}` format) that
this song was merged into or from.

The `manifest_bootstrap.yml` file MAY contain a `uses_parts_from` field holding a list
of song IDs (same format) referring to scrapped songs whose material was reused. This
field is independent of `lifecycle_status` and may appear on any active song.

Writing or updating `manifest_bootstrap.yml` via `_synthesize_bootstrap_stub()` SHALL NOT
write `lifecycle_status`, `merged_with`, or `uses_parts_from` (they default absent and
are set only by lifecycle API actions).

#### Scenario: Active song manifest has no lifecycle field
- **WHEN** `manifest_bootstrap.yml` is written by `_synthesize_bootstrap_stub()`
- **THEN** no `lifecycle_status`, `merged_with`, or `uses_parts_from` keys are present

#### Scenario: Merged song manifest
- **WHEN** a merge action completes for songs A and B
- **THEN** Song A's `manifest_bootstrap.yml` contains `lifecycle_status: merged` and
  `merged_with: [<song_B_id>]`
- **AND** Song B's `manifest_bootstrap.yml` contains `lifecycle_status: merged` and
  `merged_with: [<song_A_id>]`

#### Scenario: Abandoned song manifest
- **WHEN** an abandon action completes for song A
- **THEN** Song A's `manifest_bootstrap.yml` contains `lifecycle_status: abandoned`

#### Scenario: Scrapped song manifest
- **WHEN** a scrap action completes for song A
- **THEN** Song A's `manifest_bootstrap.yml` contains `lifecycle_status: scrapped`

#### Scenario: Uses parts from
- **WHEN** a song's "Uses parts from" list is saved with scrapped song IDs
- **THEN** the active song's `manifest_bootstrap.yml` contains
  `uses_parts_from: [<scrapped_song_id>, ...]`
- **AND** the scrapped song's `manifest_bootstrap.yml` is NOT modified
