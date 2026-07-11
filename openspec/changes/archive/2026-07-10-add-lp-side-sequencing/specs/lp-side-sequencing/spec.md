## ADDED Requirements

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
