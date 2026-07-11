## ADDED Requirements

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
