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
The candidate browser SHALL display only the generation phases relevant to the
MIDI production pipeline. The `lyrics`, `decisions`, and `quartet` phases SHALL be
removed from the phase filter dropdown and the pipeline status strip.

The pipeline status strip SHALL show phases in this order:
`chords → drums → bass → melody`

Backend support for `lyrics`, `decisions`, and `quartet` (API endpoints, pipeline runner)
is preserved; only the web UI omits them.

The `← Songs` breadcrumb on `/candidates` SHALL link to `/songs` (not `/`).

#### Scenario: Phase filter shows generation phases only
- **WHEN** the user opens the phase filter dropdown on `/candidates`
- **THEN** the options are: All phases, chords, drums, bass, melody
- **AND** lyrics, decisions, and quartet are not listed

#### Scenario: Pipeline strip stops at melody
- **WHEN** the pipeline status strip renders
- **THEN** it shows status indicators for: chords, drums, bass, melody only

#### Scenario: Songs breadcrumb links to /songs
- **WHEN** the user is on `/candidates`
- **THEN** the `← Songs` breadcrumb links to `/songs`, not `/`

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

### Requirement: Generate Endpoint
The FastAPI backend SHALL expose `POST /generate` and `GET /generate/status` endpoints
allowing a client to start an agent run (workflow + shrinkwrap) and poll for its
completion.

Only one generate job may run at a time per server process. The server SHALL maintain a
module-level job state (`idle`, `running`, `done`, or `error`) that persists for the
lifetime of the process.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Start an agent workflow + shrinkwrap job in the background |
| GET | `/generate/status` | Return the current job state |

`POST /generate` SHALL:
- Return 409 if a job is already running
- Start a background thread that calls `run_white_agent_workflow()` then `shrinkwrap()` with the configured `shrink_wrapped_dir`
- Return `{"status": "running", "started_at": "<ISO timestamp>"}` immediately

`GET /generate/status` SHALL return:
```json
{
  "status": "idle | running | done | error",
  "started_at": "<ISO timestamp or null>",
  "finished_at": "<ISO timestamp or null>",
  "error": "<message or null>"
}
```

After a job completes (success or error), subsequent `GET /songs` calls SHALL reflect
any new songs written to `shrink_wrapped/` without a server restart.

#### Scenario: Generate starts successfully
- **WHEN** `POST /generate` is called and no job is running
- **THEN** a 200 response is returned with `{"status": "running", "started_at": "..."}`
- **AND** the agent workflow begins in a background thread

#### Scenario: Generate rejected while running
- **WHEN** `POST /generate` is called while a job is already running
- **THEN** a 409 response is returned with `{"detail": "A generate job is already running"}`

#### Scenario: Status while running
- **WHEN** `GET /generate/status` is called while a job is in progress
- **THEN** `{"status": "running", "started_at": "...", "finished_at": null, "error": null}` is returned

#### Scenario: Status after completion
- **WHEN** `GET /generate/status` is called after a job finishes successfully
- **THEN** `{"status": "done", "started_at": "...", "finished_at": "...", "error": null}` is returned

#### Scenario: Status after error
- **WHEN** `GET /generate/status` is called after a job failed
- **THEN** `{"status": "error", "started_at": "...", "finished_at": "...", "error": "<message>"}` is returned

#### Scenario: Status with no prior job
- **WHEN** `GET /generate/status` is called and no job has been started this session
- **THEN** `{"status": "idle", "started_at": null, "finished_at": null, "error": null}` is returned

### Requirement: Generate Button on Song Index
The song index page (`/`) SHALL display a "Generate New Song" button in the page header.
The button SHALL trigger the agent workflow via `POST /generate` and poll
`GET /generate/status` every 5 seconds until the job reaches `done` or `error`.

While the job is running:
- The button is replaced by a spinner and "Generating…" label
- The button is disabled and cannot be clicked again

On success:
- `GET /songs` is re-fetched
- A toast shows how many new songs appeared (e.g. "1 new song generated")
- If no new songs appeared, a neutral toast confirms completion

On error:
- An error toast shows the message from the status response

#### Scenario: Generate button starts a job
- **WHEN** the Generate button is clicked on the song index
- **THEN** `POST /generate` is called
- **AND** the button changes to a spinner with "Generating…" label
- **AND** polling of `GET /generate/status` begins every 5 seconds

#### Scenario: Song list refreshes on completion
- **WHEN** `GET /generate/status` returns `{"status": "done"}`
- **THEN** the song list is re-fetched from `GET /songs`
- **AND** a success toast is shown indicating how many new songs appeared

#### Scenario: Error toast on failure
- **WHEN** `GET /generate/status` returns `{"status": "error"}`
- **THEN** an error toast is shown with the error message
- **AND** the Generate button is restored to its default state

#### Scenario: Generate button absent in single-song mode
- **WHEN** the server was launched with `--production-dir`
- **THEN** the Generate button is not rendered on the index page (the index redirects to /candidates anyway)

### Requirement: Root Landing Page
The application root (`/`) SHALL display a minimal landing page with two navigation
links: **Generation** (→ `/songs`) and **Composition Board** (→ `/board`).

The current songs index page SHALL move to `/songs`. All internal links that previously
pointed to `/` as the songs list SHALL be updated to `/songs`.

#### Scenario: Landing renders two links
- **WHEN** the user navigates to `/`
- **THEN** two clearly labelled links are rendered: "Generation" and "Composition Board"
- **AND** clicking "Generation" navigates to `/songs`
- **AND** clicking "Composition Board" navigates to `/board`

#### Scenario: Songs index accessible at /songs
- **WHEN** the user navigates to `/songs`
- **THEN** the full song list renders identically to the previous `/` behaviour

