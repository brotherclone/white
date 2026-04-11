# ace-studio-mcp Specification

## Purpose
TBD - created by archiving change add-ace-studio-mcp. Update Purpose after archive.
## Requirements
### Requirement: ACE Studio MCP Probe

A probe script SHALL discover and validate the ACE Studio MCP server's capabilities
before any pipeline integration is attempted.

#### Scenario: Probe succeeds — all required capabilities present

- **WHEN** `python -m app.reference.mcp.ace_studio.probe` is run
- **AND** ACE Studio 2.0 is running locally at `http://localhost:21572/mcp`
- **AND** the server exposes tools covering project, track, MIDI, lyric, and singer
  management
- **THEN** `tool_manifest.json` is written to `app/reference/mcp/ace_studio/`
- **AND** the script exits with code 0
- **AND** a summary of discovered tools is printed to stdout

#### Scenario: Probe fails — required capability missing

- **WHEN** the probe runs and one or more required capability keywords are absent from
  the tool manifest
- **THEN** `FEASIBILITY.md` is written documenting which capabilities are missing
- **AND** the script exits with code 1
- **AND** no Phase 2 or Phase 3 code is implemented

#### Scenario: Probe fails — server unreachable

- **WHEN** ACE Studio is not running or not reachable at localhost:21572
- **THEN** the probe prints a clear error message
- **AND** exits with code 1 (not a silent failure)

---

### Requirement: ACE Studio MCP Client

A Python client wrapper SHALL provide named, typed methods over the ACE Studio MCP
Streamable HTTP transport.

#### Scenario: Client initialises from tool manifest

- **WHEN** `AceStudioClient()` is instantiated
- **AND** `tool_manifest.json` exists in `app/reference/mcp/ace_studio/`
- **THEN** the client loads the manifest and resolves tool names
- **AND** is ready to make calls without hardcoded tool name strings

#### Scenario: Client call succeeds

- **WHEN** a client method (e.g. `create_project`) is called
- **THEN** a POST is made to `http://localhost:21572/mcp` with the correct JSON-RPC body
- **AND** the response dict is returned

#### Scenario: Server unreachable at call time

- **WHEN** a client method is called and the server is not responding
- **THEN** a `ConnectionError` is raised (not swallowed)
- **AND** callers are responsible for handling the error gracefully

---

### Requirement: ACE Studio Pipeline Export

A pipeline export step SHALL push assembled melody MIDI and approved lyrics to ACE
Studio via the MCP client.

#### Scenario: Export happy path

- **WHEN** `export_to_ace_studio(production_dir)` is called
- **AND** `assembled/assembled_melody.mid` exists
- **AND** `melody/lyrics.txt` exists
- **AND** `production_plan.yml` contains singer, BPM, key, and song title
- **AND** ACE Studio MCP is reachable
- **THEN** a new ACE Studio project is created with the correct metadata
- **AND** the assembled MIDI is imported to a vocal track with the correct singer
- **AND** the approved lyrics are assigned to the track
- **AND** a result dict with `project_id` and `track_id` is returned

#### Scenario: Export skipped — server unreachable

- **WHEN** `export_to_ace_studio` is called and the MCP server is not reachable
- **THEN** a warning is logged
- **AND** `None` is returned
- **AND** the pipeline continues without error

#### Scenario: Export skipped — missing assembled MIDI or lyrics

- **WHEN** `assembled/assembled_melody.mid` or `melody/lyrics.txt` does not exist
- **THEN** a warning is logged naming the missing file
- **AND** `None` is returned

### Requirement: ACE Studio MIDI Import

The pipeline SHALL parse ACE Studio vocal export MIDI files back into structured,
word-level note events so that downstream tools can work with the actual performed
vocal data rather than estimates from source MIDI loops.

The importer SHALL locate the export MIDI by globbing `VocalSynthvN/VocalSynthvN_*.mid`
within the production directory and selecting the highest-versioned file found.

Lyric meta messages in the export use ACE Studio's syllable-split convention
(`word#1`, `word#2`, …) for polysyllabic words. The importer SHALL merge fragments
into single word tokens, carrying the start position of the first fragment and the
end position of the last.

Each emitted word event SHALL include:
- `word` — reconstructed word string (no `#N` suffix)
- `syllable_count` — number of ACE syllable fragments that formed this word
- `start_beat` — fractional beat number (relative to start of file)
- `end_beat` — fractional beat number
- `pitch` — MIDI note number of the note sounding at word onset
- `velocity` — MIDI velocity at onset

#### Scenario: Happy path — standard ACE export

- **WHEN** `load_ace_export(production_dir)` is called
- **AND** a `VocalSynthvN/VocalSynthvN_*.mid` file exists
- **THEN** a list of word event dicts is returned, one per unique word token
- **AND** polysyllabic fragments are merged (`iron#1 iron#2` → `{word: "iron", syllable_count: 2}`)
- **AND** single-syllable words are passed through unchanged (`{word: "rust", syllable_count: 1}`)

#### Scenario: No export MIDI found

- **WHEN** `load_ace_export(production_dir)` is called
- **AND** no `VocalSynthvN/` folder or MIDI file exists
- **THEN** `None` is returned
- **AND** a warning is logged naming the production directory

#### Scenario: Multiple versioned exports

- **WHEN** both `VocalSynthv0/` and `VocalSynthv1/` folders exist
- **THEN** the highest-versioned folder's MIDI is used
- **AND** a log message notes which file was selected

---

### Requirement: LRC File Export

The importer SHALL derive word-level absolute timestamps from the MIDI tempo and
ticks-per-beat, and write a standard LRC subtitle file.

Timestamp format: `[MM:SS.cc]` where `cc` is hundredths of a second.
One LRC line per word. Lines ordered chronologically. File written as UTF-8.

#### Scenario: LRC generated from parse

- **WHEN** `export_lrc(word_events, tempo, output_path)` is called
- **THEN** an LRC file is written at `output_path`
- **AND** each line has the form `[MM:SS.cc] word`
- **AND** timestamps reflect the actual onset of each word in wall-clock time

#### Scenario: CLI default output path

- **WHEN** `python -m app.generators.midi.ace_studio_import --production-dir <dir>` is run
- **AND** `--lrc-out` is not supplied
- **THEN** the LRC file is written to `<production_dir>/vocal_alignment.lrc`

#### Scenario: 60 BPM round-trip

- **WHEN** the source MIDI has tempo 1000000 µs/beat (60 BPM) and ticks-per-beat 480
- **AND** a word event starts at beat 36.0
- **THEN** its LRC timestamp is `[00:36.00]`

### Requirement: Auto Track Selection
The ACE Studio client SHALL expose a `find_available_track()` method that returns
the index of the first track with no clips. If all tracks have clips, it SHALL
return 0 with a warning.

#### Scenario: Empty track found
- **WHEN** `find_available_track()` is called
- **AND** at least one track has no clips
- **THEN** the index of the first empty track is returned

#### Scenario: All tracks occupied
- **WHEN** `find_available_track()` is called
- **AND** all tracks have one or more clips
- **THEN** 0 is returned with a logged warning

---

### Requirement: Section-Aware Clip Placement
The ACE Studio client SHALL expose an `add_section_clips()` method that places one
clip per section, each with pre-loaded notes and lyrics, based on tick-accurate
section boundaries derived from the song BPM and bar counts.

#### Scenario: Multiple sections exported as separate clips
- **WHEN** `add_section_clips()` is called with a list of section dicts
- **THEN** one clip is created per section at the correct tick position
- **AND** each clip contains its section's notes and lyrics
- **AND** clip names match the section label

#### Scenario: Single-section song
- **WHEN** `add_section_clips()` is called with one section
- **THEN** a single named clip is placed at tick 0

