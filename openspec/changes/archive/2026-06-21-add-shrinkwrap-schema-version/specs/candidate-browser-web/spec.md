## ADDED Requirements

### Requirement: Schema Version Badge on Song Cards
Each song card on the index page SHALL display a `schema_version` badge when the
song's schema version is not `"2.0.0"`. Cards with `schema_version: "1.x"` or
`"1"` SHALL show a muted `v1.x` pill. Cards with `stub: true` SHALL show a `Stub`
pill and replace the key/BPM metadata line with a muted "Incomplete metadata" label.
Cards at `schema_version: "2.0.0"` with `stub: false` display no version badge.

#### Scenario: Legacy card shows version pill
- **WHEN** a song card renders with `schema_version: "1.x"`
- **THEN** a muted `v1.x` pill is visible on the card

#### Scenario: Stub card shows stub pill and masked metadata
- **WHEN** a song card renders with `stub: true`
- **THEN** a `Stub` pill is shown and the key/BPM row reads "Incomplete metadata"

#### Scenario: Current-schema card shows no version pill
- **WHEN** a song card renders with `schema_version: "2.0.0"` and `stub: false`
- **THEN** no version or stub pill is displayed

---

### Requirement: Phase Regression with Diary Confirmation Modal
The board page SHALL allow moving a song's MixStage **backward** via a "←" button
adjacent to the current stage indicator. Backward movement always requires
confirmation via a modal. Destructive regressions (where files exist that were
written at or after the target stage) list those files in the modal. All modals
offer an optional diary-entry textarea so the reason can be recorded inline.

The backend SHALL expose `POST /composition/regress` with body
`{ target_stage, confirmed, diary_entry }`. When `confirmed: false` it returns
`{ destructive, files_to_delete }` without making changes. When `confirmed: true`
it deletes the listed files, sets the stage, and (if `diary_entry` is non-empty)
writes a diary entry tagged with the song slug and the regression action.

The "←" button SHALL be absent when the current stage is `structure` (nothing to
regress to).

#### Scenario: Non-destructive regression confirmed
- **WHEN** the user clicks "←" from `mix_candidate` (target: `rough_mix`)
- **AND** no files exist in the `REGRESSION_FILE_MAP` for stages passed through
- **THEN** the modal shows "Move back to Rough Mix?" with no file list but with a diary textarea
- **AND** on Confirm the stage is set to `rough_mix` and an optional diary entry is written

#### Scenario: Destructive regression shows file list
- **WHEN** the user clicks "←" from `vocal_placeholders` (target: `lyrics`)
- **AND** vocal placeholder MIDI files exist in the Logic song dir
- **THEN** the modal shows those file paths in a scrollable list
- **AND** the Confirm button is styled destructively (red)

#### Scenario: Diary entry written on regression
- **WHEN** the user types a note in the diary textarea and confirms the regression
- **THEN** a diary entry is created with the note text, tagged with the song slug
  and a `phase_regression` metadata field noting the before/after stages

#### Scenario: Back button absent at structure
- **WHEN** the current stage is `structure`
- **THEN** the "←" back button is not rendered

#### Scenario: Modal cancel leaves stage unchanged
- **WHEN** the user opens the regression modal and clicks Cancel
- **THEN** no stage change occurs and no files are deleted

## MODIFIED Requirements

### Requirement: FastAPI Backend
The candidate browser SHALL expose a REST API at `GET /songs` (album mode) that returns
song entries sourced from `manifest_bootstrap.yml` files discovered under the shrink-wrapped
directory.

Each song entry SHALL include a `schema_version` field and a `stub` field. When
`manifest_bootstrap.yml` does not contain a `schema_version` field (legacy /
pre-uv-workspace files), the entry SHALL report `schema_version: "1.x"`. When the file
contains `stub: true` (written by the migration for pre-scaffold threads), the entry SHALL
surface `stub: true` so the UI can indicate that the song has incomplete metadata. The API
SHALL NOT crash on legacy or stub files; missing fields are surfaced as `null` or their
documented defaults.

#### Scenario: Song list in album mode
- **WHEN** `GET /songs` is called and the server was launched with `--shrink-wrapped-dir`
- **THEN** a JSON array is returned with one object per song found under `*/production/*/manifest_bootstrap.yml`, each containing: `id` (`{thread_slug}__{production_slug}`), `thread_slug`, `production_slug`, `title`, `key`, `bpm`, `rainbow_color`, `singer` (null if absent), and `schema_version` (`"1.x"` if absent from file)

#### Scenario: Song list in single-song mode
- **WHEN** `GET /songs` is called and the server was launched with `--production-dir`
- **THEN** a 503 response is returned

#### Scenario: Activate song
- **WHEN** `POST /songs/activate` is called with a valid song `id`
- **THEN** `_production_dir` is set to the resolved production path and `{"ok": true, "production_dir": "..."}` is returned

#### Scenario: Activate unknown song
- **WHEN** `POST /songs/activate` is called with an `id` that does not match any scanned song
- **THEN** a 404 response is returned

#### Scenario: Candidate endpoint before activation
- **WHEN** the server is in album mode AND no song has been activated
- **AND** `GET /candidates` (or any candidate mutation endpoint) is called
- **THEN** a 503 response is returned with `{"detail": "No song selected — POST /songs/activate first"}`

#### Scenario: Legacy manifest does not crash scan
- **WHEN** `scan_songs()` encounters a `manifest_bootstrap.yml` with no `schema_version` field
- **THEN** the song entry is returned with `schema_version: "1.x"`
- **AND** no exception is raised

#### Scenario: Stub manifest surfaces in song list
- **WHEN** `scan_songs()` encounters a `manifest_bootstrap.yml` with `stub: true`
- **THEN** the song entry is returned with `stub: true`
- **AND** the song appears in the list (stubs are not filtered out)

---

### Requirement: Song Stage Routing
Each song card SHALL display a stage badge indicating the song's current production
stage. Valid stage labels and their routing behaviour are:

| Stage label | `stage` value | Click behaviour |
|---|---|---|
| Ideation | `ideation` | Activate → init → `/candidates` |
| Generation | `generation` | Activate → `/candidates` |
| Composition | `composition` | Activate → `/board` |
| Production | `production` | Activate → `/board` |
| Mixing | `mixing` | Activate → `/board` |
| Complete | `complete` | Activate → `/board` |
| Invalid | `invalid` | Activate → `/` (no navigation; toast shown) |

#### Scenario: Ideation song selected
- **WHEN** the user clicks a song card with `stage: "ideation"`
- **THEN** `POST /songs/activate` and `POST /songs/init` are called in sequence
- **AND** the user is navigated to `/candidates`

#### Scenario: Generation song selected
- **WHEN** the user clicks a song card with `stage: "generation"`
- **THEN** `POST /songs/activate` is called
- **AND** the user is navigated to `/candidates`

#### Scenario: Composition song selected
- **WHEN** the user clicks a song card with `stage: "composition"`
- **THEN** `POST /songs/activate` is called
- **AND** the user is navigated to `/board`

#### Scenario: Invalid song selected
- **WHEN** the user clicks a song card with `stage: "invalid"`
- **THEN** no navigation occurs
- **AND** a toast is shown: "Song metadata is invalid — run migration to repair"

#### Scenario: Stage badge on card
- **WHEN** a song card renders
- **THEN** exactly one stage label is visible on the card

---

### Requirement: Song Stage Field
`scan_songs` in `candidate_server.py` SHALL include a `stage` field on every returned
song entry. The value SHALL be one of: `"ideation"`, `"generation"`, `"composition"`,
`"production"`, `"mixing"`, `"complete"`, `"invalid"`.

Computation rules (evaluated in order):
1. **`invalid`** — `manifest_bootstrap.yml` cannot be parsed, is missing both `title`
   and `rainbow_color`, or carries an unrecognised `schema_version` prefix (not absent,
   not starting with `"1"`, not starting with `"2"`)
2. **`ideation`** — `song_context.yml` is absent from the production dir
3. **`composition/production/mixing/complete`** — `LOGIC_OUTPUT_DIR` env var is set AND
   `composition.yml` exists; `current_stage` value maps to the song stage via
   `_MIX_STAGE_TO_SONG_STAGE`
4. **`generation`** — all other cases

The TypeScript `SongEntry` type SHALL include
`stage: "ideation" | "generation" | "composition" | "production" | "mixing" | "complete" | "invalid"`.
The stage filter on the song index SHALL include all seven values;
"Invalid" SHALL appear last in the filter order.

#### Scenario: Ideation stage
- **WHEN** a production dir lacks `song_context.yml`
- **THEN** `scan_songs` returns `stage: "ideation"` for that entry

#### Scenario: Invalid stage — corrupt manifest
- **WHEN** `manifest_bootstrap.yml` cannot be parsed by `yaml.safe_load()`
- **THEN** `scan_songs` returns `stage: "invalid"` for that entry

#### Scenario: Invalid stage — unrecognised schema_version
- **WHEN** `manifest_bootstrap.yml` has a `schema_version` value not starting with
  `"1"` or `"2"` (e.g. `"3.0.0"`, `"beta"`)
- **THEN** `scan_songs` returns `stage: "invalid"` for that entry

#### Scenario: Generation stage
- **WHEN** a production dir has `song_context.yml` and no `composition.yml`
- **THEN** `scan_songs` returns `stage: "generation"`

---

### Requirement: Shrinkwrap Production Scaffolding
`app/util/shrinkwrap_chain_artifacts.py` SHALL scaffold a `production/<slug>/` directory for every song proposal found in a thread's `yml/` directory when shrinkwrapping. A file is treated as a song proposal if it contains all three of `bpm`, `key`, and `rainbow_color` fields. Known non-proposal files (`evp.yml`, `all_song_proposals.yml`) are always skipped.

Each scaffolded directory SHALL contain a `manifest_bootstrap.yml` with the following fields (in this order):
- `schema_version` — always `"2.0.0"`
- `title` — from the proposal YAML (or the slug if absent)
- `key` — from the proposal YAML
- `bpm` — from the proposal YAML
- `rainbow_color` — from the proposal YAML
- `singer` — from the proposal YAML, or `null` if absent

The scaffolding SHALL be idempotent: if `manifest_bootstrap.yml` already exists in the target directory, it is not overwritten.

#### Scenario: Proposals detected during shrinkwrap
- **GIVEN** a thread's `yml/` directory contains `coral_fever_requiem_v1.yml` (with bpm, key, rainbow_color) and `evp.yml`
- **WHEN** `shrinkwrap_thread()` runs
- **THEN** `production/coral_fever_requiem_v1/manifest_bootstrap.yml` is created
- **AND** no directory is created for `evp.yml`

#### Scenario: schema_version first in bootstrap
- **WHEN** `scaffold_song_productions()` writes a new `manifest_bootstrap.yml`
- **THEN** the first YAML field is `schema_version: "2.0.0"`

#### Scenario: Idempotent scaffolding
- **WHEN** `shrinkwrap_thread()` runs a second time on the same thread
- **THEN** existing `manifest_bootstrap.yml` files are not overwritten

