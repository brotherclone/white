## ADDED Requirements

### Requirement: Lifecycle Stage Computation
`_compute_stage()` SHALL check `lifecycle_status` in `manifest_bootstrap.yml` before any
mix-stage logic. If `lifecycle_status` is one of `"merged"`, `"abandoned"`, or
`"scrapped"`, that value SHALL be returned directly as the song's stage.

#### Scenario: Merged song stage
- **WHEN** `manifest_bootstrap.yml` contains `lifecycle_status: merged`
- **THEN** `_compute_stage()` returns `"merged"`

#### Scenario: Abandoned song stage
- **WHEN** `manifest_bootstrap.yml` contains `lifecycle_status: abandoned`
- **THEN** `_compute_stage()` returns `"abandoned"`

#### Scenario: Scrapped song stage
- **WHEN** `manifest_bootstrap.yml` contains `lifecycle_status: scrapped`
- **THEN** `_compute_stage()` returns `"scrapped"`

#### Scenario: Active song falls through to existing logic
- **WHEN** `manifest_bootstrap.yml` has no `lifecycle_status` or `lifecycle_status: null`
- **THEN** `_compute_stage()` proceeds with existing mix-stage computation

---

### Requirement: Lifecycle API Endpoints
The API SHALL expose three new endpoints for managing song lifecycle status.

`POST /songs/{id}/lifecycle` SHALL accept a JSON body `{status, merged_with?}` where
`status` is one of `"merged"`, `"abandoned"`, `"scrapped"`. When `status` is `"merged"`,
`merged_with` SHALL be a list containing at least one other song ID. The endpoint SHALL
write `lifecycle_status` (and `merged_with` if applicable) to the target song's
`manifest_bootstrap.yml`. When `status` is `"merged"`, the endpoint SHALL also write
the reciprocal `lifecycle_status` and `merged_with` entry to each partner song's
`manifest_bootstrap.yml`. Return `{"ok": true, "status": "<status>"}`.

`GET /songs/scrapped` SHALL return a JSON array of song entries (same shape as
`GET /songs`) filtered to songs whose `lifecycle_status` is `"scrapped"`.

`PATCH /songs/{id}/uses-parts-from` SHALL accept `{uses_parts_from: [song_id, ...]}` and
write that list to the target song's `manifest_bootstrap.yml`, returning `{"ok": true}`.

#### Scenario: Set abandoned
- **WHEN** `POST /songs/{id}/lifecycle` is called with `{status: "abandoned"}`
- **THEN** the song's `manifest_bootstrap.yml` is updated with `lifecycle_status: abandoned`
- **AND** `{"ok": true, "status": "abandoned"}` is returned

#### Scenario: Set scrapped
- **WHEN** `POST /songs/{id}/lifecycle` is called with `{status: "scrapped"}`
- **THEN** the song's `manifest_bootstrap.yml` is updated with `lifecycle_status: scrapped`

#### Scenario: Merge two songs
- **WHEN** `POST /songs/{id}/lifecycle` is called with `{status: "merged", merged_with: [partner_id]}`
- **THEN** the active song's manifest gets `lifecycle_status: merged`, `merged_with: [partner_id]`
- **AND** the partner song's manifest gets `lifecycle_status: merged`, `merged_with: [active_id]`
- **AND** `{"ok": true, "status": "merged"}` is returned

#### Scenario: Merge with unknown partner
- **WHEN** `POST /songs/{id}/lifecycle` is called with `merged_with` containing an unknown song ID
- **THEN** a 404 response is returned and neither manifest is modified

#### Scenario: List scrapped songs
- **WHEN** `GET /songs/scrapped` is called in album mode
- **THEN** a JSON array of songs with `lifecycle_status: scrapped` is returned

#### Scenario: Update uses-parts-from
- **WHEN** `PATCH /songs/{id}/uses-parts-from` is called with a list of scrapped song IDs
- **THEN** the song's `manifest_bootstrap.yml` is updated with `uses_parts_from: [...]`
- **AND** `{"ok": true}` is returned

---

### Requirement: Song Index Lifecycle Filter Pills
The song index filter bar SHALL include three new pills: `merged`, `abandoned`, `scrapped`.
The `all` pill SHALL exclude songs whose `stage` is `"merged"`, `"abandoned"`, or
`"scrapped"`. Songs with these stages are only visible when their respective pill is active.
Each pill SHALL display the count of songs in that state.

`SongEntry["stage"]` SHALL be extended with `"merged" | "abandoned" | "scrapped"`.
`STAGE_LABELS` and `STAGE_BADGE_CLS` SHALL include entries for all three new values.

#### Scenario: All pill excludes lifecycle-terminal songs
- **WHEN** the `all` pill is active and the song list includes merged, abandoned, and scrapped songs
- **THEN** those songs are NOT shown in the list
- **AND** the `all` pill count reflects only non-terminal songs

#### Scenario: Merged pill shows merged songs
- **WHEN** the `merged` pill is active
- **THEN** only songs with `stage === "merged"` are shown

#### Scenario: Abandoned pill
- **WHEN** the `abandoned` pill is active
- **THEN** only songs with `stage === "abandoned"` are shown

#### Scenario: Scrapped pill
- **WHEN** the `scrapped` pill is active
- **THEN** only songs with `stage === "scrapped"` are shown

#### Scenario: Pill ordering
- **WHEN** the filter bar renders
- **THEN** the pill order is: all · [existing production stages] · stub · merged · abandoned · scrapped · invalid

---

### Requirement: Board Page Song Lifecycle Panel
The `/board` page SHALL include a collapsible **Song Lifecycle** panel containing three
action buttons: **Merge**, **Abandon**, and **Scrap**.

**Merge button**: opens a modal with a searchable dropdown of all non-terminal songs
(excluding the active song itself). The user selects one song and confirms. On confirm,
`POST /songs/{active_id}/lifecycle` is called with `{status: "merged", merged_with: [chosen_id]}`.
After success, the board displays a confirmation and the song's status badge updates.

**Abandon button**: opens a confirmation modal with a whimsical, empathetic plea — the
copy MUST have a playful or humorous tone distinct from a generic "are you sure?". On
confirm, `POST /songs/{active_id}/lifecycle` is called with `{status: "abandoned"}`.

**Scrap button**: opens a confirmation modal noting that scrapped material can still be
referenced by other songs. On confirm, `POST /songs/{active_id}/lifecycle` is called with
`{status: "scrapped"}`.

Songs already in a terminal lifecycle state SHALL display their status prominently in the
panel and NOT show the action buttons (the action is already done).

#### Scenario: Merge opens song picker
- **WHEN** the Merge button is clicked
- **THEN** a modal opens showing a dropdown of all active (non-terminal) songs except the current one

#### Scenario: Merge completes
- **WHEN** the user selects a partner song and confirms the merge
- **THEN** `POST /songs/{id}/lifecycle` is called with `{status: "merged", merged_with: [partner_id]}`
- **AND** the board updates to reflect the merged status

#### Scenario: Abandon confirmation has personality
- **WHEN** the Abandon button is clicked
- **THEN** a confirmation modal appears with playful copy pleading for the song's survival
- **AND** the modal has distinct Confirm and Cancel actions

#### Scenario: Scrap confirmation notes reuse
- **WHEN** the Scrap button is clicked
- **THEN** a confirmation modal appears noting that scrapped song material can still be
  referenced in other productions

#### Scenario: Terminal song shows status, not actions
- **WHEN** the board is loaded for a song with `stage === "merged"`, `"abandoned"`, or `"scrapped"`
- **THEN** the lifecycle panel displays the status label and terminal date/note
- **AND** the Merge / Abandon / Scrap action buttons are NOT rendered

---

### Requirement: Board Page "Uses Parts From" Widget
The `/board` page Song Lifecycle panel SHALL include a **"Uses parts from"** disclosure
section. This is a reference link — it records which scrapped songs donated material to
the active song for the producer's own bookkeeping. When expanded, it displays a
multi-select list populated by `GET /songs/scrapped`. Selected songs are persisted via
`PATCH /songs/{active_id}/uses-parts-from`. Already-selected songs SHALL be pre-checked
when the widget opens. The widget is available regardless of the active song's lifecycle
status (an active song may reference scrapped material).

#### Scenario: Widget shows scrapped songs
- **WHEN** the "Uses parts from" section is expanded on the board
- **THEN** `GET /songs/scrapped` is called and the results populate the list

#### Scenario: Selection persisted
- **WHEN** the user selects one or more scrapped songs and saves
- **THEN** `PATCH /songs/{active_id}/uses-parts-from` is called with the selected IDs
- **AND** a success indicator is shown inline

#### Scenario: Existing selections pre-populated
- **WHEN** the active song's `manifest_bootstrap.yml` already contains `uses_parts_from: [id1]`
- **AND** the "Uses parts from" widget is opened
- **THEN** `id1` is pre-selected in the list

#### Scenario: No scrapped songs
- **WHEN** `GET /songs/scrapped` returns an empty list
- **THEN** the widget shows a message: "No scrapped songs available"
