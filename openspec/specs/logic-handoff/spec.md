# logic-handoff Specification

## Purpose
TBD - created by archiving change add-logic-handoff. Update Purpose after archive.
## Requirements
### Requirement: Logic Project Scaffold
`white_composition.logic_handoff` SHALL create a Logic Pro project folder on the fast
drive when `handoff(production_dir)` is called.

The folder SHALL be created at:
`$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/`

The seed Logic project at `packages/composition/logic/seed.logicx` SHALL be
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

### Requirement: Samples Sweep During Handoff
`logic_handoff.handoff()` SHALL copy any WAV files found in
`<production_dir>/samples/` into `$LOGIC_OUTPUT_DIR/<thread>/<title>/Samples/`,
creating the `Samples/` subdirectory if it does not exist.

If `<production_dir>/samples/` is absent or empty, the step SHALL be silently skipped.
File names SHALL be preserved. This sweep runs after the approved MIDI copy step so
that a full re-handoff keeps the Logic project in sync with any samples previously
exported via `POST /samples/{segment_id}/export`.

#### Scenario: Samples present during handoff
- **WHEN** `handoff(production_dir)` is called and `<production_dir>/samples/` contains
  one or more `.wav` files
- **THEN** those files are copied to `$LOGIC_OUTPUT_DIR/<thread>/<title>/Samples/`
- **AND** file names are preserved

#### Scenario: No samples directory
- **WHEN** `handoff(production_dir)` is called and `<production_dir>/samples/` does not exist
- **THEN** the handoff completes normally with no error and no `Samples/` folder is created

#### Scenario: Samples directory empty
- **WHEN** `handoff(production_dir)` is called and `<production_dir>/samples/` exists
  but contains no WAV files
- **THEN** the handoff completes normally; `Samples/` is created but left empty

### Requirement: Stage Regression Info
`white_composition.logic_handoff` SHALL expose a `regression_info(logic_song_dir, current, target)` function that computes the set of files that would be deleted if the MixStage were moved backward from `current` to `target`, and whether any such files actually exist.

The function SHALL raise `ValueError` when `target` is not strictly earlier than `current` in `_STAGE_ORDER`, or when either stage name is not a valid `MixStage` value.

`REGRESSION_FILE_MAP` SHALL be a module-level constant mapping each `MixStage` value to a list of glob patterns (relative to the Logic song dir) that are deleted when moving backward past that stage:

```
"lyrics":             ["lyrics*.txt", "*.lrc"]
"vocal_placeholders": ["MIDI/melody/vocal_placeholder*.mid", "MIDI/melody/assembled*.mid"]
"recording":          ["Recordings/*"]
"augmentation":       ["Augmented/*"]
"cleaning":           ["Cleaned/*"]
```

Stages not in the map (`rough_mix`, `mix_candidate`, `final_mix`) produce no file deletions.

The returned dict has the shape `{ "destructive": bool, "files_to_delete": list[str] }` where `files_to_delete` contains paths **relative to `logic_song_dir`** for matching files that actually exist. Relative paths avoid leaking local filesystem structure to callers.

#### Scenario: Destructive regression identifies files
- **WHEN** `regression_info(song_dir, "vocal_placeholders", "lyrics")` is called
- **AND** `song_dir/MIDI/melody/vocal_placeholder_verse.mid` exists
- **THEN** returns `{ "destructive": True, "files_to_delete": ["MIDI/melody/vocal_placeholder_verse.mid"] }`

#### Scenario: Non-destructive regression
- **WHEN** `regression_info(song_dir, "mix_candidate", "rough_mix")` is called
- **THEN** returns `{ "destructive": False, "files_to_delete": [] }`
  because neither stage is in `REGRESSION_FILE_MAP`

#### Scenario: Multi-stage regression collects all passed-through patterns
- **WHEN** `regression_info(song_dir, "recording", "lyrics")` is called
- **THEN** patterns for both `vocal_placeholders` and `recording` are resolved
- **AND** `files_to_delete` contains all matching files for both stages

#### Scenario: Forward movement raises ValueError
- **WHEN** `regression_info(song_dir, "lyrics", "vocal_placeholders")` is called
  (target is later than current)
- **THEN** `ValueError` is raised with a descriptive message

#### Scenario: Invalid stage name raises ValueError
- **WHEN** either `current` or `target` is not a valid `MixStage` value
- **THEN** `ValueError` is raised

