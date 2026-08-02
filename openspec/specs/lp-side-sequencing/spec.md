# lp-side-sequencing Specification

## Purpose
TBD - created by archiving change add-lp-side-sequencing. Update Purpose after archive.
## Requirements
### Requirement: Mix Duration Extraction
The system SHALL compute mix audio duration on demand using `soundfile.info()` and
surface it through the existing mix-info endpoint.

#### Scenario: Duration available
- **WHEN** `GET /production/mix/info` is called for a song with a valid `mix_file`
- **THEN** the response includes `duration_seconds` as a float

#### Scenario: No mix file
- **WHEN** `GET /production/mix/info` is called for a song with no `mix_file` set
- **THEN** `duration_seconds` is `null` and `has_mix` is `false`

#### Scenario: Unreadable mix file
- **WHEN** the mix file path exists in `song_context.yml` but the file is missing or
  unreadable by `soundfile`
- **THEN** `duration_seconds` is `null` and no exception is raised to the caller

### Requirement: Sides Data File
The system SHALL persist LP-side assignments in `sides.yml` at the album root
(`$SHRINKWRAP_OUTPUT_DIR/sides.yml`), containing exactly 4 sides (A–D) and a
`side_limit_seconds` soft limit (default 1200).

#### Scenario: File created on first write
- **WHEN** no `sides.yml` exists and a song is first assigned to a side
- **THEN** `sides.yml` is created with all 4 sides (A–D) and the assigned song placed
  in the target side

#### Scenario: Cached duration stored per assignment
- **WHEN** a song is assigned to a side
- **THEN** its `duration_seconds` (read via mix duration extraction at assignment time)
  is cached in `sides.yml` alongside the song reference

### Requirement: Side Assignment API
`candidate_server.py` SHALL expose endpoints to list, assign, move, and remove songs
from sides, plus computed per-side totals.

#### Scenario: List sides with totals
- **WHEN** `GET /sides` is called
- **THEN** all 4 sides are returned, each with its ordered song list, cached durations,
  a summed `total_seconds`, and an `over_limit` boolean (`total_seconds > side_limit_seconds`)

#### Scenario: Assign a song
- **WHEN** `POST /sides/A/assign` is called with `{song_id, position}` for a song with
  `has_mix: true`
- **THEN** the song is inserted into side A at `position`, its duration is cached, and
  the updated side is returned

#### Scenario: Reject assignment of a song without a mix
- **WHEN** `POST /sides/{side}/assign` is called for a song with `has_mix: false`
- **THEN** a 400 response is returned and `sides.yml` is not modified

#### Scenario: Move a song between sides
- **WHEN** `POST /sides/{side}/move` is called with `{song_id, to_side, to_position}`
- **THEN** the song is removed from its current side and inserted into `to_side` at
  `to_position`

#### Scenario: Remove a song from a side
- **WHEN** `DELETE /sides/{side}/songs/{song_id}` is called
- **THEN** the song is removed from that side's list and `sides.yml` is updated

### Requirement: Drag-and-Drop Sides UI
The client SHALL provide a page with 4 drop-target columns (one per side) showing
assigned songs with duration and a running total against the soft limit.

#### Scenario: Running total display
- **WHEN** the sides page loads
- **THEN** each side column shows its songs (title + duration) and a running total
  formatted as `MM:SS` (or `H:MM:SS` if over an hour)

#### Scenario: Over-limit warning
- **WHEN** a side's total exceeds `side_limit_seconds`
- **THEN** the column displays a visual warning (e.g. total shown in a warning color)
  without blocking further assignment

#### Scenario: Non-mixed songs are not draggable
- **WHEN** the available-songs list includes a song with `has_mix: false`
- **THEN** that song is rendered in a disabled state and cannot be dragged onto a side

#### Scenario: Drag updates assignment
- **WHEN** a song is dragged from the available list (or another side) onto a side column
- **THEN** the corresponding assign/move API call is made and the UI reflects the new
  side totals without a full page reload

### Requirement: Per-Song Mix Streaming Endpoint
The system SHALL expose `GET /songs/{song_id}/mix/info` and `GET /songs/{song_id}/mix`
that resolve an arbitrary song by its composite id (`{thread_slug}__{production_slug}`,
the same id format used throughout the sides API) independent of any single "active"
production. `song_id` SHALL be resolved the same way the existing sides
assign/move/remove endpoints resolve it (matching against `scan_songs()` results),
returning 404 for an unknown id.

`GET /songs/{song_id}/mix/info` SHALL return `{has_mix, mix_file, duration_seconds}`
with the same shape and semantics as the existing `GET /production/mix/info` (duration
computed via `soundfile.info()`, `null` when there is no mix file or it can't be read).

`GET /songs/{song_id}/mix` SHALL stream the file at that song's `mix_file` (from its
own `song_context.yml`) with the same content-type resolution as the existing
`GET /production/mix` (`.mp3`→`audio/mpeg`, `.wav`→`audio/wav`, `.aiff`/`.aif`→
`audio/aiff`), returning 404 if no `mix_file` is set or the file doesn't exist on disk.

#### Scenario: Info for a song with a mix
- **WHEN** `GET /songs/{song_id}/mix/info` is called for a song whose
  `song_context.yml` has a valid `mix_file`
- **THEN** the response has `has_mix: true`, the `mix_file` path, and a numeric
  `duration_seconds`

#### Scenario: Info for a song without a mix
- **WHEN** `GET /songs/{song_id}/mix/info` is called for a song with no `mix_file` set
- **THEN** the response has `has_mix: false`, `mix_file: null`, `duration_seconds: null`

#### Scenario: Unknown song id
- **WHEN** either endpoint is called with a `song_id` that doesn't match any song
  returned by `scan_songs()`
- **THEN** a 404 response is returned

#### Scenario: Stream returns the file with correct content type
- **WHEN** `GET /songs/{song_id}/mix` is called for a song with a `.wav` mix file
- **THEN** the response body is that file's bytes with `Content-Type: audio/wav`

#### Scenario: Stream 404s when no mix file exists
- **WHEN** `GET /songs/{song_id}/mix` is called for a song with no `mix_file` set, or
  a `mix_file` path that no longer exists on disk
- **THEN** a 404 response is returned

### Requirement: Song Notes Panel
The sides screen SHALL provide a way to open a per-song panel showing that song's mix
player (when it has one) and its diary entries, without disturbing the existing
drag-and-drop reassignment interaction on the same row.

Both `AvailableSongRow` and `SideSongRow` SHALL render a small button that opens the
panel for that row's song, using `stopPropagation` on its click handler so it does not
trigger the row's drag-handle listeners — the same escape-hatch pattern already used by
`SideSongRow`'s existing "×" remove button.

The panel SHALL:
- Render an `<audio controls>` element sourced from `GET /songs/{song_id}/mix` when the
  song's `has_mix` is true, and an explanatory message ("no mix file yet") when false
- List the song's existing diary entries (via the existing `fetchDiaryEntries`,
  keyed by the song's `production_slug` — matching how the composition board already
  calls it) ordered most-recent-first
- Reuse the existing diary create-entry form fields and submit behavior (author, phase,
  title, body → `createDiaryEntry`) so a new note can be added without leaving the panel

Opening the panel SHALL NOT require the song to have a mix file — diary notes remain
addable for any song.

#### Scenario: Notes button does not start a drag
- **WHEN** the notes button on a song row is clicked
- **THEN** the panel opens for that song
- **AND** the row's drag-and-drop reordering is not triggered

#### Scenario: Panel shows player for a mixed song
- **WHEN** the panel opens for a song with `has_mix: true`
- **THEN** an audio player sourced from `GET /songs/{song_id}/mix` is rendered

#### Scenario: Panel omits player for an unmixed song
- **WHEN** the panel opens for a song with `has_mix: false`
- **THEN** no audio player is rendered, and an explanatory message is shown instead

#### Scenario: Panel lists existing diary entries, most recent first
- **WHEN** the panel opens for a song with existing diary entries
- **THEN** `fetchDiaryEntries` is called with that song's `production_slug`
- **AND** the entries are displayed with the most recently created entry first

#### Scenario: New entry can be added from the panel
- **WHEN** the create-entry form is filled in and submitted from the panel
- **THEN** `createDiaryEntry` is called for that song's `production_slug`
- **AND** the newly created entry appears in the panel's entry list without a full
  page reload

#### Scenario: Available (unassigned) songs also get the panel
- **WHEN** a song is in the available-songs pool (not yet assigned to any side)
- **THEN** its row still renders the notes button, regardless of `has_mix`

