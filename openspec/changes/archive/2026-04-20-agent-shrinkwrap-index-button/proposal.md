# Change: Agent + Shrinkwrap + Generate Button

## Why

The LangChain agent (`WhiteAgent.start_workflow()`) currently calls `shrinkwrap()` at the
**start** of each run to pick up threads from previous runs before loading negative
constraints. It does **not** call shrinkwrap at the **end**, so the thread it just created
is never scaffolded until the next invocation. This means the song index page can't see
newly generated songs until the user manually reruns the process.

Additionally, there is no way to trigger a new agent run from the web UI — users must
drop to the terminal and run the CLI directly. The song index page (`/`) should surface a
"Generate New Song" button that starts the agent, waits for it, runs shrinkwrap
(including production directory scaffolding), and refreshes the song list automatically.

## What Changes

### 1. Post-run shrinkwrap in `white_agent.py`
After `workflow.invoke()` returns, `start_workflow()` SHALL call `shrinkwrap()` with
`thread_filter=<new thread_id>` so the newly created thread is immediately cleaned,
manifested, and scaffolded into `shrink_wrapped/`. The pre-run shrinkwrap (for loading
constraints) is unchanged.

### 2. Generate endpoint on `candidate_server.py`
- `POST /generate` — starts the agent workflow + shrinkwrap as a background thread.
  Returns immediately with `{"job_id": "...", "status": "running"}`. Only one job may
  run at a time; a second POST while one is running returns 409.
- `GET /generate/status` — returns `{"status": "idle|running|done|error",
  "started_at": ..., "finished_at": ..., "error": null|"..."}`.

The background task calls `run_white_agent_workflow()` then `shrinkwrap()`. On
completion (success or error) the status transitions accordingly. The next `GET /songs`
call will reflect any new songs without a server restart.

### 3. Generate button on the song index page (`web/app/page.tsx`)
A "Generate New Song" button appears in the header of the song index. On click it calls
`POST /generate`, then polls `GET /generate/status` every 5 seconds until the status
is `done` or `error`. While running: the button is replaced by a spinner and a status
label ("Generating…"). On success: `GET /songs` is refreshed and a toast confirms how
many new songs appeared. On error: an error toast is shown.

## Impact
- Affected specs: `chain-artifacts`, `candidate-browser-web`
- Affected code:
  - `app/agents/white_agent.py` — add post-workflow shrinkwrap
  - `app/tools/candidate_server.py` — two new endpoints
  - `web/app/page.tsx` — Generate button + polling logic
  - `web/lib/api.ts` — two new fetch helpers (`startGenerate`, `getGenerateStatus`)
