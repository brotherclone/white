## 1. Dependencies
- [ ] 1.1 Add `fastapi>=0.115.0` and `uvicorn>=0.32.0` to `pyproject.toml`

## 2. Backend — `app/tools/candidate_server.py`
- [ ] 2.1 FastAPI app with `production_dir` set at startup via CLI arg
- [ ] 2.2 `GET /candidates` — calls `load_all_candidates()` from `candidate_browser.py`,
       returns JSON; accepts `?phase=` and `?section=` query params
- [ ] 2.3 `POST /candidates/{id}/approve` and `/candidates/{id}/reject` — finds entry
       by id and calls `approve_candidate` / `reject_candidate`
- [ ] 2.4 `GET /midi/{id}` — streams `.mid` bytes with `Content-Type: audio/midi`;
       returns 404 if file missing
- [ ] 2.5 `GET /` — serves `app/tools/static/browser.html`
- [ ] 2.6 CLI: `--production-dir`, `--port` (default 8000), `--no-open` flag;
       auto-opens browser via `webbrowser.open()`

## 3. Frontend — `app/tools/static/browser.html`
- [ ] 3.1 Fetch `/candidates` on load; render table with phase/section/ID/template/score/status columns
- [ ] 3.2 Composite score displayed as an inline bar (`<progress>` or CSS width)
- [ ] 3.3 Status badge colour-coded (green=approved, yellow=pending, red=rejected)
- [ ] 3.4 Approve / Reject buttons per row — POST to backend, update status badge in place
- [ ] 3.5 Play button per row — loads `<midi-player>` web component pointing at `/midi/{id}`;
       stops any currently playing player first
- [ ] 3.6 Score breakdown panel — click a row to expand; shows composite + theory sub-scores
       + chromatic match/confidence
- [ ] 3.7 Phase dropdown filter (fetches from backend) + status filter (client-side)
- [ ] 3.8 Column sort (click header, toggle asc/desc)
- [ ] 3.9 Keyboard shortcuts: `a` approve focused row, `r` reject, `p` play

## 4. Tests
- [ ] 4.1 Unit tests for all API endpoints using FastAPI `TestClient`
       (candidates list, approve, reject, midi 404)
