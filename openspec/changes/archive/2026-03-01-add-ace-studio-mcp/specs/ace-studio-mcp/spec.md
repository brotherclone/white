## ADDED Requirements

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
