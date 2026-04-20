# Tasks: agent-shrinkwrap-index-button

- [ ] 1. Add post-workflow shrinkwrap to `start_workflow()` in `app/agents/white_agent.py`:
       after `workflow.invoke()` returns, call `shrinkwrap(artifacts_dir, output_dir, thread_filter=thread_id, scaffold=True)`.
       Wrap in try/except and log a warning on failure (never raise).

- [ ] 2. Add `POST /generate` endpoint to `app/tools/candidate_server.py`:
       - Module-level `_generate_job: dict | None` state (status, started_at, finished_at, error)
       - Returns 409 if a job is already running
       - Background thread calls `run_white_agent_workflow()` then `shrinkwrap()` on the shrink_wrapped dir
       - On completion updates job status to `done` or `error`

- [ ] 3. Add `GET /generate/status` endpoint to `app/tools/candidate_server.py`:
       - Returns `{"status": "idle|running|done|error", "started_at": ..., "finished_at": ..., "error": null|"..."}`
       - Returns `{"status": "idle"}` when no job has been started this session

- [ ] 4. Add `startGenerate()` and `getGenerateStatus()` fetch helpers to `web/lib/api.ts`

- [ ] 5. Add Generate button to `web/app/page.tsx`:
       - Placed in page header alongside the title
       - On click: calls `startGenerate()`, then polls `getGenerateStatus()` every 5s
       - While running: shows spinner + "Generating…" label; button disabled
       - On success: refreshes song list via `GET /songs`, shows success toast with new song count
       - On error: shows error toast with message from status response

- [ ] 6. Add tests for the two new endpoints:
       - `tests/tools/test_candidate_server_generate.py`
       - Test: POST /generate returns 200 with job_id and status=running
       - Test: POST /generate while running returns 409
       - Test: GET /generate/status returns idle when no job started
       - Test: GET /generate/status returns done after mocked job completes
