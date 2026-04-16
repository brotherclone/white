# candidate-browser-web Specification

## Purpose
TBD - created by archiving change add-candidate-browser-web. Update Purpose after archive.
## Requirements
### Requirement: FastAPI Backend
`app/tools/candidate_server.py` SHALL expose a FastAPI application with the following
endpoints and CORS enabled for `http://localhost:3000`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/candidates` | Return all candidates as JSON; accepts `?phase=` and `?section=` query params |
| POST | `/candidates/{id}/approve` | Write `status: approved` to the matching `review.yml` entry |
| POST | `/candidates/{id}/reject` | Write `status: rejected` to the matching `review.yml` entry |
| GET | `/midi/{id}` | Stream the candidate's `.mid` file with `Content-Type: audio/midi` |

The server SHALL be launched via `python -m app.tools.candidate_server --production-dir <path>`.

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

### Requirement: Next.js Frontend
A Next.js 15 app (App Router, TypeScript, Tailwind CSS) SHALL live in `web/` and consume the FastAPI backend at `http://localhost:8000`. The UI SHALL include a Promote button in the phase filter toolbar that is disabled when no single phase is selected and enabled when exactly one phase is selected.

#### Scenario: Candidate table
- **WHEN** the page loads
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

### Requirement: Evolve Candidates
The UI SHALL allow the user to generate evolved pattern candidates for drums, bass, or melody phases by clicking an "Evolve" button in the phase toolbar. The button SHALL only appear when a phase that supports evolution (drums, bass, melody) is selected. Evolved candidates join the existing candidate list with an "evolved" badge and are reviewed through the same approve/reject flow.

#### Scenario: Evolve button visible for supported phases
- **WHEN** the phase filter is set to drums, bass, or melody
- **THEN** an Evolve button appears in the toolbar alongside the Promote button

#### Scenario: Evolve button absent for unsupported phases
- **WHEN** the phase filter is set to chords, quartet, or "all"
- **THEN** no Evolve button is shown

#### Scenario: Evolve action
- **WHEN** the Evolve button is clicked
- **THEN** `POST /evolve` is called with `{production_dir, phase}`
- **AND** a spinner shows while generation runs (evolution takes 10–30s)
- **AND** on completion the candidate list refreshes and new evolved candidates appear with an "evolved" badge
- **AND** a toast reports how many evolved candidates were added

#### Scenario: Evolved badge
- **WHEN** a candidate has `is_evolved: true` in the review
- **THEN** an "evolved" badge is shown on that row so it's visually distinct from template candidates

### Requirement: ACE Studio Integration
The UI SHALL surface the ACE Studio vocal synthesis handoff as two action buttons in the melody phase toolbar: "Export to ACE Studio" (after melody is promoted) and "Import Render" (after export). Both SHALL call FastAPI endpoints that wrap the existing `ace_studio_export` and `ace_studio_import` logic.

#### Scenario: Export button visible after melody promoted
- **WHEN** the phase filter is set to melody AND melody phase status is "promoted"
- **THEN** an "Export to ACE Studio" button appears in the toolbar

#### Scenario: Export to ACE Studio
- **WHEN** "Export to ACE Studio" is clicked
- **THEN** `POST /ace/export` is called with `{production_dir}`
- **AND** a spinner shows while the export runs
- **AND** on success a toast shows the singer name and number of sections exported
- **AND** the button changes to "Exported ✓" with the singer name

#### Scenario: ACE Studio not running
- **WHEN** `POST /ace/export` is called and ACE Studio is not reachable
- **THEN** a 503 error toast is shown: "ACE Studio not running — launch it first"

#### Scenario: Import render
- **WHEN** "Import Render" is clicked
- **THEN** `POST /ace/import` is called with `{production_dir}`
- **AND** on success a toast confirms the WAV path ingested
- **AND** the button changes to "Render imported ✓"

### Requirement: Evolve Endpoint
The FastAPI backend SHALL expose a `POST /evolve` endpoint that runs evolutionary pattern breeding for a given production directory and phase, returning the count of new evolved candidates generated.

#### Scenario: Valid evolve request
- **WHEN** `POST /evolve` is called with a valid `production_dir` and `phase` in `[drums, bass, melody]`
- **THEN** the evolutionary pipeline runs and new candidates are written to the phase's candidates directory
- **AND** `{"ok": true, "evolved_count": N}` is returned

#### Scenario: Unsupported phase for evolution
- **WHEN** `POST /evolve` is called with `phase` in `[chords, quartet]`
- **THEN** a 400 response is returned

### Requirement: ACE Studio Endpoints
The FastAPI backend SHALL expose `POST /ace/export` and `POST /ace/import` endpoints wrapping the existing `ace_studio_export` and `ace_studio_import` logic.

#### Scenario: Export succeeds
- **WHEN** `POST /ace/export` is called with a valid `production_dir`
- **AND** ACE Studio is running
- **THEN** `{"ok": true, "singer": "...", "sections": [...]}` is returned

#### Scenario: ACE Studio unreachable
- **WHEN** `POST /ace/export` is called and the MCP server is not responding
- **THEN** a 503 response is returned with message "ACE Studio not running"

#### Scenario: Import succeeds
- **WHEN** `POST /ace/import` is called with a valid `production_dir`
- **AND** a WAV render exists in the expected location
- **THEN** `{"ok": true, "render_path": "..."}` is returned

#### Scenario: No render found
- **WHEN** `POST /ace/import` is called and no VocalSynth WAV exists
- **THEN** a 404 response is returned

### Requirement: Promote Endpoint
The FastAPI backend SHALL expose a `POST /promote` endpoint that runs phase promotion for a given production directory and phase. The endpoint SHALL validate the phase value and return a structured result.

#### Scenario: Valid promote request
- **WHEN** `POST /promote` is called with a valid `production_dir` and `phase`
- **THEN** `pipeline_runner promote` runs for that phase
- **AND** `{"ok": true, "promoted_count": N}` is returned

#### Scenario: Invalid phase value
- **WHEN** `POST /promote` is called with a `phase` not in `[chords, drums, bass, melody, quartet]`
- **THEN** a 400 response is returned with a descriptive error

#### Scenario: Promotion failure
- **WHEN** `pipeline_runner promote` raises an exception
- **THEN** a 500 response is returned with the error detail
- **AND** no partial state is left (promote_part is atomic)

