## ADDED Requirements

### Requirement: Shrinkwrap Production Scaffolding
`app/util/shrinkwrap_chain_artifacts.py` SHALL scaffold a `production/<slug>/` directory for every song proposal found in a thread's `yml/` directory when shrinkwrapping. A file is treated as a song proposal if it contains all three of `bpm`, `key`, and `rainbow_color` fields. Known non-proposal files (`evp.yml`, `all_song_proposals.yml`) are always skipped.

Each scaffolded directory SHALL contain a `manifest_bootstrap.yml` with the following fields:
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

#### Scenario: Idempotent scaffolding
- **WHEN** `shrinkwrap_thread()` runs a second time on the same thread
- **THEN** existing `manifest_bootstrap.yml` files are not overwritten

### Requirement: Song Index Breadcrumb
The candidate browser at `/candidates` SHALL display a breadcrumb navigation element
above the page heading when the active song title is available. The breadcrumb SHALL
contain a "← Songs" link that navigates to `/`. The breadcrumb SHALL be hidden when the
server is in single-song mode (i.e., `GET /songs/active` returns `{"active": null}` or
a 503).

#### Scenario: Breadcrumb shown in album mode
- **WHEN** `/candidates` is loaded after a song has been activated
- **THEN** a breadcrumb reads `← Songs  /  <song title>` above the "Candidate Browser" heading
- **AND** clicking "← Songs" navigates back to `/`

#### Scenario: Breadcrumb hidden in single-song mode
- **WHEN** the server was launched with `--production-dir`
- **AND** `/candidates` is loaded
- **THEN** no breadcrumb is rendered

## MODIFIED Requirements

### Requirement: FastAPI Backend
`app/tools/candidate_server.py` SHALL expose a FastAPI application with the following
endpoints and CORS enabled for `http://localhost:3000`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/candidates` | Return all candidates as JSON; accepts `?phase=` and `?section=` query params |
| POST | `/candidates/{id}/approve` | Write `status: approved` to the matching `review.yml` entry |
| POST | `/candidates/{id}/reject` | Write `status: rejected` to the matching `review.yml` entry |
| GET | `/midi/{id}` | Stream the candidate's `.mid` file with `Content-Type: audio/midi` |
| GET | `/songs` | Return all songs found in the shrink_wrapped directory; 503 if not in album mode |
| POST | `/songs/activate` | Set the active production dir for the session; body: `{"id": "<thread_slug>__<production_slug>"}` |
| GET | `/songs/active` | Return the active song entry, or `{"active": null}` if none selected |

The server SHALL be launched via one of two modes:
- **Single-song mode**: `python -m app.tools.candidate_server --production-dir <path>` (existing behaviour, unchanged)
- **Album mode**: `python -m app.tools.candidate_server --shrink-wrapped-dir <path>`

Exactly one of `--production-dir` or `--shrink-wrapped-dir` MUST be supplied; supplying neither or both SHALL be a startup error.

#### Scenario: Candidate list endpoint
- **WHEN** `GET /candidates` is called
- **THEN** a JSON array is returned with one object per candidate containing: `id`,
  `phase`, `section`, `template`, `status`, `rank`, `composite_score`, `midi_url`,
  and `scores` dict

#### Scenario: Approve endpoint
- **WHEN** `POST /candidates/{id}/approve` is called
- **THEN** the matching entry in `review.yml` is updated to `status: approved` and
  `{"ok": true}` is returned

#### Scenario: Unknown candidate
- **WHEN** `POST /candidates/{unknown-id}/approve` is called
- **THEN** a 404 response is returned

#### Scenario: MIDI streaming
- **WHEN** `GET /midi/{id}` is called for a candidate that has a `.mid` file
- **THEN** the file bytes are returned with `Content-Type: audio/midi`
- **WHEN** the MIDI file does not exist
- **THEN** a 404 response is returned

#### Scenario: Song list in album mode
- **WHEN** `GET /songs` is called and the server was launched with `--shrink-wrapped-dir`
- **THEN** a JSON array is returned with one object per song found under `*/production/*/manifest_bootstrap.yml`, each containing: `id` (`{thread_slug}__{production_slug}`), `thread_slug`, `production_slug`, `title`, `key`, `bpm`, `rainbow_color`, and `singer` (null if absent)

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

### Requirement: Next.js Frontend
A Next.js 15 app (App Router, TypeScript, Tailwind CSS) SHALL live in `web/` and
consume the FastAPI backend at `http://localhost:8000`. The app SHALL include two
primary routes: a song index at `/` and a candidate browser at `/candidates`.

The candidate browser SHALL include a Promote button in the phase filter toolbar that
is disabled when no single phase is selected and enabled when exactly one phase is selected.

#### Scenario: Song index loads
- **WHEN** the app is opened and the server is in album mode
- **THEN** the song index page at `/` is shown with a card for every song returned by `GET /songs`

#### Scenario: Song card click
- **WHEN** a song card is clicked
- **THEN** `POST /songs/activate` is called with that song's `id`
- **AND** on success the browser navigates to `/candidates`
- **AND** a spinner is shown on the clicked card while the request is in-flight

#### Scenario: Single-song mode redirect
- **WHEN** the app is opened and `GET /songs` returns 503 (single-song mode)
- **THEN** the song index page immediately redirects to `/candidates` without showing an error

#### Scenario: Candidate table
- **WHEN** `/candidates` loads
- **THEN** all candidates are fetched from `/candidates` and displayed in a table with columns: phase, section, ID, template, composite score (bar), status, actions

#### Scenario: Approve and reject
- **WHEN** the Approve or Reject button for a row is clicked
- **THEN** a POST is sent to the backend and the row's status badge updates in place without a full page reload

#### Scenario: MIDI playback
- **WHEN** a row's Play button is clicked
- **THEN** the candidate's MIDI plays inline; any previously playing candidate is stopped

#### Scenario: Promote button disabled without phase filter
- **WHEN** the phase filter is set to "all" or is unset
- **THEN** the Promote button is disabled with tooltip "Select a phase to enable promote"

#### Scenario: Promote button enabled with phase filter
- **WHEN** a single phase (chords, drums, bass, melody, or quartet) is selected
- **THEN** the Promote button is enabled

#### Scenario: Promote action
- **WHEN** the Promote button is clicked with a phase selected
- **THEN** `POST /promote` is called with `{production_dir, phase}`
- **AND** a success toast shows the number of files promoted
- **AND** the candidate list refreshes to reflect updated statuses
- **AND** the button shows a spinner while the request is in-flight

### Requirement: No Breaking Changes
The existing terminal browser (`app/tools/candidate_browser.py`) SHALL remain unchanged.
The server's data layer SHALL import `load_all_candidates`, `approve_candidate`, and
`reject_candidate` directly from `app/tools/candidate_browser.py`.

#### Scenario: Terminal browser unaffected
- **GIVEN** the FastAPI server is installed
- **WHEN** `candidate_browser.py` is imported or run directly
- **THEN** it operates exactly as before with no changes to its public API

#### Scenario: Single-song launch unaffected
- **WHEN** the server is launched with `--production-dir <path>`
- **THEN** it behaves identically to before this change, including opening the browser at `/candidates`
