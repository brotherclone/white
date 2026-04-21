## ADDED Requirements

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
