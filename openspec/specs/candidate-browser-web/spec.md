# candidate-browser-web Specification

## Purpose
TBD - created by archiving change add-candidate-browser-web. Update Purpose after archive.
## Requirements
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

### Requirement: Next.js Frontend
The candidate browser SHALL display only the generation phases relevant to the MIDI
production pipeline. The `lyrics`, `decisions`, and `quartet` phases SHALL be removed
from the phase filter dropdown and the pipeline status strip.

The pipeline status strip SHALL show phases in this order:
`chords → drums → bass → melody`

Backend support for `lyrics`, `decisions`, and `quartet` (API endpoints, pipeline runner)
is preserved; only the web UI omits them.

The `← Songs` breadcrumb on `/candidates` SHALL link to `/`.

#### Scenario: Phase filter shows generation phases only
- **WHEN** the user opens the phase filter dropdown on `/candidates`
- **THEN** the options are: All phases, chords, drums, bass, melody
- **AND** lyrics, decisions, and quartet are not listed

#### Scenario: Pipeline strip stops at melody
- **WHEN** the pipeline status strip renders
- **THEN** it shows status indicators for: chords, drums, bass, melody only

#### Scenario: Songs breadcrumb links to /
- **WHEN** the user is on `/candidates`
- **THEN** the `← Songs` breadcrumb links to `/`, not `/songs`

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
The generate workflow SHALL NOT appear on the song index page (`/`). It is hosted
exclusively on the Agent Run Screen at `/agent`. The song index page header SHALL
contain only the "Run Agent" navigation link described in the Root Landing Page
requirement.

#### Scenario: No generate button on song browser
- **WHEN** the user is on `/`
- **THEN** no "Generate New Song" button or spinner is rendered on that page

#### Scenario: Run Agent link navigates to /agent
- **WHEN** the user clicks "Run Agent" in the song browser header
- **THEN** they are navigated to `/agent`

### Requirement: Root Landing Page
The application root (`/`) SHALL display the song browser — the full list of songs found
in the shrink_wrapped directory, previously served at `/songs`. The `/songs` route SHALL
redirect to `/`. No separate two-link landing page is rendered.

A "Run Agent" link in the page header SHALL navigate to `/agent`.

#### Scenario: Song browser at root
- **WHEN** the user navigates to `/`
- **THEN** the full song list renders
- **AND** a "Run Agent" link is visible in the page header

#### Scenario: /songs redirects to /
- **WHEN** the user navigates to `/songs`
- **THEN** they are redirected to `/`

### Requirement: Plan Drift Report API

The candidate server SHALL expose three endpoints for generating and retrieving the plan
drift report for the active song:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/drift-report` | Return the current `plan_drift_report.yml` as JSON; 404 if absent |
| POST | `/drift-report` | Start a background job to generate (or regenerate) the drift report |
| GET | `/drift-report/status` | Return the current job state for the drift report background job |

`POST /drift-report` accepts an optional JSON body `{"use_claude": bool}` (default `true`).
It requires both `production_plan.yml` and `arrangement.txt` to exist in the production
directory; missing either returns 422.

The background job state follows the same shape as `/handoff/status`:
`{status, started_at, finished_at, error}` where `status` is one of `idle`, `running`,
`done`, or `error`.

#### Scenario: GET with report absent

- **WHEN** `GET /drift-report` is called and `plan_drift_report.yml` does not exist
- **THEN** a 404 response is returned

#### Scenario: GET with report present

- **WHEN** `GET /drift-report` is called and `plan_drift_report.yml` exists
- **THEN** a 200 response is returned with the report fields as JSON, including
  `song_title`, `proposed_sections`, `actual_sections`, `drift`, `bar_deltas`,
  `energy_arc_correlation`, and `summary`

#### Scenario: POST starts background job

- **WHEN** `POST /drift-report` is called and both `production_plan.yml` and
  `arrangement.txt` exist in the production directory
- **THEN** a background job is started and `{"status": "running", "started_at": "..."}` is returned

#### Scenario: POST missing arrangement

- **WHEN** `POST /drift-report` is called but `arrangement.txt` is absent
- **THEN** a 422 response is returned

#### Scenario: POST missing production plan

- **WHEN** `POST /drift-report` is called but `production_plan.yml` is absent
- **THEN** a 422 response is returned

#### Scenario: POST duplicate job

- **WHEN** `POST /drift-report` is called while a drift report job is already running
- **THEN** a 409 response is returned

#### Scenario: Status endpoint

- **WHEN** `GET /drift-report/status` is called
- **THEN** the current job state is returned with `status`, `started_at`, `finished_at`,
  and `error` fields

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

### Requirement: Agent Run Screen
A page at `/agent` SHALL host the agent workflow (LangChain song generation). The page
SHALL display a "Generate New Song" button that triggers `POST /generate` and polls
`GET /generate/status` every five seconds until the job reaches `done` or `error`.

While a job is running, a spinner and "Generating…" label SHALL replace the button,
and the button SHALL be non-interactive.

A `← Songs` navigation link SHALL appear at the top of the page and navigate to `/`.

#### Scenario: Generate button starts workflow
- **WHEN** the user clicks "Generate New Song" on `/agent`
- **THEN** `POST /generate` is called
- **AND** the button shows "Generating…" with a spinner
- **AND** polling of `GET /generate/status` begins every five seconds

#### Scenario: Success toast on completion
- **WHEN** `GET /generate/status` returns `{"status": "done"}`
- **THEN** a success toast is shown
- **AND** the button is restored to its default state

#### Scenario: Error toast on failure
- **WHEN** `GET /generate/status` returns `{"status": "error"}`
- **THEN** an error toast is shown containing the error message
- **AND** the button is restored to its default state

#### Scenario: Navigation back to songs
- **WHEN** the user clicks `← Songs` on `/agent`
- **THEN** they are navigated to `/`

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

### Requirement: Unified Dev Launch
A shell script `dev.sh` at the repository root SHALL start both the FastAPI server
(album mode, port 8000) and the Next.js dev server (port 3000) with a single command.
Both processes SHALL run concurrently; a `SIGINT` (Ctrl+C) sent to the script SHALL
terminate both.

The script SHALL read `SHRINK_WRAPPED_DIR` from the environment or from a `.env` file
at the repo root, and pass the value to the FastAPI server as `--shrink-wrapped-dir`.

#### Scenario: Single command launch
- **WHEN** `./dev.sh` is run with `SHRINK_WRAPPED_DIR` set in the environment or `.env`
- **THEN** the FastAPI server starts on port 8000 and the Next.js dev server starts on
  port 3000 within a few seconds

#### Scenario: Ctrl-C stops both servers
- **WHEN** the user sends SIGINT to the `dev.sh` process
- **THEN** both the FastAPI and Next.js processes are terminated

#### Scenario: Missing SHRINK_WRAPPED_DIR
- **WHEN** `./dev.sh` is run and `SHRINK_WRAPPED_DIR` is not set
- **THEN** an error message is printed to stderr and the script exits non-zero

### Requirement: Song Concept Field
The song entry returned by `GET /songs` and `GET /songs/active` SHALL include a
`concept` field. The value SHALL be read from `song_context.yml` in the production
directory if that file exists and contains a non-empty `concept` key; otherwise the
field SHALL be `null`.

#### Scenario: Concept present in song_context.yml
- **WHEN** `GET /songs` is called and a production directory has a `song_context.yml`
  with a non-empty `concept` field
- **THEN** the song entry for that production includes `"concept": "<text>"`

#### Scenario: Concept absent
- **WHEN** `song_context.yml` is absent or `concept` is empty/missing
- **THEN** the song entry includes `"concept": null`

### Requirement: Concept Display in Candidate Browser
The `/candidates` page SHALL render a concept block between the breadcrumb and the phase
toolbar when `activeSong.concept` is non-null and non-empty.

The block SHALL default to a 3-line clamp with a "Show more" / "Show less" toggle.

#### Scenario: Concept block visible
- **WHEN** `/candidates` is loaded and the active song has a non-null concept
- **THEN** the concept text is displayed above the toolbar, clamped to 3 lines by default

#### Scenario: Toggle expands concept
- **WHEN** the user clicks "Show more"
- **THEN** the full concept text is revealed and the toggle label changes to "Show less"

#### Scenario: Concept block hidden when absent
- **WHEN** `activeSong.concept` is null
- **THEN** no concept block is rendered

### Requirement: Samples Retrieval Endpoints
The FastAPI backend SHALL expose three endpoints for chromatic sample retrieval and export:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/samples` | Return top-N CLAP-scored segments for the active song's color |
| GET | `/audio/{segment_id}` | Stream the pre-extracted WAV for a segment |
| POST | `/samples/{segment_id}/export` | Copy the segment WAV to the Logic Samples folder |

`GET /samples` SHALL accept an optional `?top_n=N` query parameter (default 20). It SHALL
call `white_composition.retrieve_samples.retrieve_by_color` using the active song's
`rainbow_color` field and return a JSON array of objects with fields:
`segment_id`, `song_slug`, `color`, `match` (float 0–1), `audio_url`.

`GET /audio/{segment_id}` SHALL return 404 if the WAV is not present on disk.

`POST /samples/{segment_id}/export` SHALL copy the WAV to
`$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/Samples/<segment_id>.wav`; returns 503 if
`LOGIC_OUTPUT_DIR` is not set, 404 if WAV absent.

#### Scenario: Samples list for active song
- **WHEN** `GET /samples` is called with an active song whose color is Orange
- **THEN** a JSON array is returned with up to 20 objects sorted descending by `match`

#### Scenario: Export to Logic
- **WHEN** `POST /samples/{segment_id}/export` is called with a valid segment
- **THEN** the WAV is copied to `$LOGIC_OUTPUT_DIR/<thread_slug>/<title>/Samples/<segment_id>.wav`
- **AND** `{"ok": true, "dest": "<path>"}` is returned

### Requirement: Sample Browser Panel
The `/candidates` page SHALL display a collapsible **Chromatic Samples** panel below the
candidate table. The panel SHALL always be expandable (regardless of pipeline stage). Each
row SHALL show: rank, segment_id, song slug, color chip, match score, inline `<audio>`
player, and an **Export** button.

The Export button SHALL be disabled (with tooltip) when the active song's stage is not
`"composition"` — exporting requires a Logic project folder to exist.

#### Scenario: Panel always expandable
- **WHEN** the user clicks the Chromatic Samples header at any pipeline stage
- **THEN** the panel expands and shows available samples

#### Scenario: Export requires composition stage
- **WHEN** the active song's stage is not `"composition"`
- **THEN** the Export button is disabled with tooltip "Handoff to Logic first"

#### Scenario: Export marks row as exported
- **WHEN** Export is clicked and the server returns 200
- **THEN** the button changes to "Exported ✓" and is disabled for the session

### Requirement: Quartet Button Alongside Handoff
The **Generate Quartet** button SHALL appear in the pipeline status strip alongside the
"Handoff to Logic" button whenever melody is promoted and the quartet phase has not yet
been generated.

The button SHALL be shown when `quartetStatus` is absent (`undefined`), `null`, or
`"pending"`. It SHALL be hidden once quartet reaches `"in_progress"`, `"generated"`, or
`"promoted"`.

#### Scenario: Button shown when melody promoted and quartet not started
- **WHEN** melody phase is promoted AND quartet status is absent or "pending"
- **THEN** a "Generate Quartet" button appears in the pipeline strip

#### Scenario: Button hidden after generation
- **WHEN** quartet status is "generated" or "promoted"
- **THEN** no Generate Quartet button is shown

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

### Requirement: Sides Navigation Entry
The client SHALL expose a "Sides" navigation link alongside the existing board,
candidates, songs, and collaborators links, routing to the LP-side sequencing page.

#### Scenario: Nav link present
- **WHEN** any client page renders the shared navigation
- **THEN** a "Sides" link is present and navigates to `/sides`

### Requirement: LP Consideration Status
Each song SHALL carry an `lp_consideration` status (`not_considered`, `candidate`,
`placed`) tracked in `manifest_bootstrap.yml`, independent of `lifecycle_status` and
mix stage.

#### Scenario: Default status
- **WHEN** a song's `manifest_bootstrap.yml` has no `lp_consideration` field
- **THEN** `scan_songs()` reports it as `not_considered`

#### Scenario: Manual status set
- **WHEN** `POST /songs/{id}/lp-consideration` is called with `{status: "candidate"}`
- **THEN** the song's `manifest_bootstrap.yml` is updated with
  `lp_consideration: candidate`
- **AND** `{"ok": true, "status": "candidate"}` is returned

#### Scenario: Auto-set to placed on side assignment
- **WHEN** a song is assigned to a side via the `add-lp-side-sequencing` assign/move
  endpoints
- **THEN** the song's `lp_consideration` is automatically set to `placed`

#### Scenario: Auto-revert on removal
- **WHEN** a song is removed from all sides via the remove endpoint
- **THEN** `lp_consideration` reverts to `candidate` if the song still has a mix file,
  or `not_considered` if it does not

#### Scenario: Filter pill and badge
- **WHEN** the song list filter bar renders
- **THEN** `lp: candidate` and `lp: placed` pills are available
- **AND** each song row shows a badge reflecting its current `lp_consideration` value

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

