# logic-handoff Specification

## Purpose
TBD - created by archiving change add-logic-handoff. Update Purpose after archive.
## Requirements
### Requirement: Logic Project Scaffold
`white_composition.logic_handoff` SHALL create a Logic Pro project folder on the fast
drive when `handoff(production_dir)` is called.

The folder SHALL be created at:
`$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/`

The seed Logic project at `packages/composition/logic/seed/seed.logicx` SHALL be
copied (full directory copy) into that folder and renamed to `<song_title>.logicx`.

If the destination folder already exists, the function SHALL skip the copy and log
a warning rather than raising an error.

`LOGIC_OUTPUT_DIR` SHALL be read from the environment. If unset, the function SHALL
raise `EnvironmentError` with a descriptive message.

#### Scenario: Successful scaffold
- **WHEN** `handoff(production_dir)` is called with `LOGIC_OUTPUT_DIR` set and a
  valid production dir containing `song_context.yml`
- **THEN** `$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/<song_title>.logicx` exists
  as a copy of the seed bundle
- **AND** `composition.yml` is created in the same folder (see Composition File requirement)

#### Scenario: Destination already exists
- **WHEN** the Logic song folder already exists at the target path
- **THEN** the copy is skipped, a warning is printed, and the function continues to
  update `composition.yml`

#### Scenario: LOGIC_OUTPUT_DIR not set
- **WHEN** `LOGIC_OUTPUT_DIR` is not set in the environment
- **THEN** `EnvironmentError` is raised with the message
  `"LOGIC_OUTPUT_DIR is not set — add it to .env"`

---

### Requirement: Approved MIDI Copy
`logic_handoff` SHALL copy each phase's approved `.mid` files into phase-specific
subfolders under `MIDI/` within the Logic song folder.

Subfolders created: `MIDI/chords/`, `MIDI/drums/`, `MIDI/bass/`, `MIDI/melody/`

For each phase, all `.mid` files under `<production_dir>/<phase>/approved/` SHALL be
copied. Phases with no approved files are skipped without error.

#### Scenario: Approved MIDI files present
- **WHEN** `<production_dir>/chords/approved/` contains one or more `.mid` files
- **THEN** those files are copied to `$LOGIC_OUTPUT_DIR/.../MIDI/chords/`
- **AND** file names are preserved

#### Scenario: No approved files for a phase
- **WHEN** `<production_dir>/drums/approved/` is empty or does not exist
- **THEN** `MIDI/drums/` is created but left empty, and no error is raised

---

### Requirement: Source Text File Move
`logic_handoff` SHALL move `arrangement.txt` and any `lyrics*.txt` or `*.lrc` files
found in the production dir into the Logic song folder as siblings of the `.logicx`
file.

If no such files are found, the function proceeds without error.

#### Scenario: arrangement.txt present in production dir
- **WHEN** `<production_dir>/arrangement.txt` exists
- **THEN** it is moved to `$LOGIC_OUTPUT_DIR/<thread>/<title>/arrangement.txt`
- **AND** the original file is removed from the production dir

#### Scenario: No text files found
- **WHEN** no `arrangement.txt` or lyrics files exist in the production dir
- **THEN** the handoff completes without error and without creating placeholder files

---

### Requirement: Composition File
`logic_handoff` SHALL create or update `composition.yml` in the Logic song folder.

Schema:
```yaml
song_title: <str>
thread_slug: <str>
production_slug: <str>
logic_project_path: <absolute path str>
current_version: <int>
current_stage: <MixStage>
versions:
  - version: <int>
    created: <ISO date str>
    stage: <MixStage>
    notes: <str>
```

`MixStage` SHALL be one of (in order):
`structure`, `lyrics`, `recording`, `vocal_placeholders`, `augmentation`,
`cleaning`, `rough_mix`, `mix_candidate`, `final_mix`

On first handoff, `current_stage` SHALL be `structure` and one version entry at
`structure` SHALL be written.

If `composition.yml` already exists, it SHALL NOT be overwritten; the handoff updates
only the MIDI and text files.

#### Scenario: First handoff creates composition.yml
- **WHEN** `handoff()` is called and no `composition.yml` exists
- **THEN** `composition.yml` is written with `current_stage: structure`,
  `current_version: 1`, and one entry in `versions`

#### Scenario: Re-handoff preserves existing composition.yml
- **WHEN** `handoff()` is called and `composition.yml` already exists
- **THEN** `composition.yml` is not modified
- **AND** MIDI files are re-copied (overwriting if already present)

---

### Requirement: Handoff API Endpoint
`candidate_server.py` SHALL expose a `POST /handoff` endpoint that runs the Logic
handoff for the active song in a background thread, following the same job-state
pattern as `POST /pipeline/run`.

A `GET /handoff/status` endpoint SHALL return the current job state.

A `GET /composition` endpoint SHALL return the parsed `composition.yml` for the active
song, or `{"status": "not_initialized"}` if the file does not exist.

A `PATCH /composition/stage` endpoint SHALL accept `{"stage": "<MixStage>"}` and update
`current_stage` in `composition.yml`.

A `POST /composition/version` endpoint SHALL append a new version entry to `composition.yml`,
incrementing `current_version`, and set `current_stage` to `structure` for the new version.

#### Scenario: POST /handoff starts background job
- **WHEN** `POST /handoff` is called with an active song
- **THEN** the handoff runs in a background thread and `{"status": "running"}` is returned immediately
- **AND** `GET /handoff/status` returns `{"status": "running"}` until complete

#### Scenario: GET /composition before handoff
- **WHEN** `GET /composition` is called and no `composition.yml` exists
- **THEN** `{"status": "not_initialized"}` is returned with HTTP 200

#### Scenario: PATCH /composition/stage advances stage
- **WHEN** `PATCH /composition/stage` is called with `{"stage": "lyrics"}`
- **THEN** `composition.yml` is updated with `current_stage: lyrics`
- **AND** `{"ok": true, "stage": "lyrics"}` is returned

---

### Requirement: Composition Board UI
A new Next.js page at `/board` SHALL display all songs from `GET /songs` that have
been handed off (i.e. `composition.yml` exists) as cards arranged in a horizontal
swimlane — one column per mix stage.

Each card SHALL show the song title, thread slug, color dot, and current version.

Clicking a card's stage SHALL advance it to the next stage by calling
`PATCH /composition/stage`.

A "Handoff" button on each song card on the `/` songs index page SHALL trigger
`POST /handoff` for that song.

#### Scenario: Board renders songs by stage
- **WHEN** the user navigates to `/board`
- **THEN** each song with a `composition.yml` appears in the column matching its
  `current_stage`
- **AND** songs without a `composition.yml` are not shown

#### Scenario: Advancing a stage
- **WHEN** the user clicks the advance arrow on a card
- **THEN** the card moves to the next column
- **AND** the board re-fetches to reflect the updated state

