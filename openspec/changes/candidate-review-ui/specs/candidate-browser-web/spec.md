## MODIFIED Requirements

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

## ADDED Requirements

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
