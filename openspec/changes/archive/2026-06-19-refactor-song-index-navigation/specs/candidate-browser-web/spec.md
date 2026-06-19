## MODIFIED Requirements

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

---

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

---

## ADDED Requirements

### Requirement: Song Stage Routing
Each song card SHALL display a stage badge indicating the song's current production
stage. The badge SHALL replace the former "not initialized" label. Valid stage labels
and their routing behaviour are:

| Stage label | `stage` value | Click behaviour |
|---|---|---|
| Ideation | `ideation` | Activate → init → `/candidates` |
| Generation | `generation` | Activate → `/candidates` |
| Composition | `composition` | Activate → `/board` |

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

#### Scenario: Stage badge on card
- **WHEN** a song card renders
- **THEN** exactly one of the labels "Ideation", "Generation", or "Composition" is
  visible on the card

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

---

### Requirement: Song Stage Field
`scan_songs` in `candidate_server.py` SHALL include a `stage` field on every returned
song entry. The value SHALL be one of: `"ideation"`, `"generation"`, `"composition"`.

Computation rules (evaluated in order):
1. **`ideation`** — `song_context.yml` is absent from the production dir
2. **`composition`** — `LOGIC_OUTPUT_DIR` env var is set AND `composition.yml` exists
   in the Logic output dir for this song (path resolved via `_song_dir` from
   `white_composition.logic_handoff`); if `LOGIC_OUTPUT_DIR` is unset, the import
   raises, or the file is absent, this rule is skipped
3. **`generation`** — all other cases

The TypeScript `SongEntry` type SHALL include `stage: "ideation" | "generation" | "composition"`.

The `GET /songs` shape SHALL include `stage` alongside the existing fields.

#### Scenario: Ideation stage
- **WHEN** a production dir lacks `song_context.yml`
- **THEN** `scan_songs` returns `stage: "ideation"` for that entry

#### Scenario: Generation stage
- **WHEN** a production dir has `song_context.yml` and no `composition.yml` in the
  Logic dir (or LOGIC_OUTPUT_DIR is unset)
- **THEN** `scan_songs` returns `stage: "generation"`

#### Scenario: Composition stage
- **WHEN** `LOGIC_OUTPUT_DIR` is set and `composition.yml` exists in the Logic song dir
- **THEN** `scan_songs` returns `stage: "composition"`

#### Scenario: Composition check skipped when LOGIC_OUTPUT_DIR absent
- **WHEN** `LOGIC_OUTPUT_DIR` is not set
- **THEN** `scan_songs` never returns `stage: "composition"` and returns `generation`
  for all initialized songs

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
- **WHEN** `./dev.sh` is run and `SHRINK_WRAPPED_DIR` is not set in the environment or `.env`
- **THEN** an error message is printed to stderr and the script exits with a non-zero
  status without starting either server
