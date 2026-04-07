## 1. Dependencies
- [ ] 1.1 Add `fastapi>=0.115.0` and `uvicorn>=0.32.0` to `pyproject.toml`
- [ ] 1.2 Scaffold Next.js 15 app in `web/` with TypeScript and Tailwind CSS

## 2. Backend — `app/tools/candidate_server.py`
- [ ] 2.1 FastAPI app; `production_dir` set at startup via CLI arg; CORS for localhost:3000
- [ ] 2.2 `GET /candidates` — calls `load_all_candidates()` from `candidate_browser.py`;
       serialises to JSON; accepts `?phase=` and `?section=` query params
- [ ] 2.3 `POST /candidates/{id}/approve` and `/candidates/{id}/reject` — finds entry
       by id, calls `approve_candidate` / `reject_candidate`; returns 404 on unknown id
- [ ] 2.4 `GET /midi/{id}` — streams `.mid` bytes with `Content-Type: audio/midi`;
       returns 404 if file missing
- [ ] 2.5 CLI: `--production-dir`, `--port` (default 8000), `--no-open` flag

## 3. Frontend — `web/`
- [ ] 3.1 `app/page.tsx` — fetches `/candidates` on load; renders `<CandidateTable>`
- [ ] 3.2 `CandidateTable` component — sortable columns, phase/status filter controls
- [ ] 3.3 Composite score bar (Tailwind width-based, colour-coded green/yellow/red)
- [ ] 3.4 Status badge colour-coded (green=approved, yellow=pending, red=rejected)
- [ ] 3.5 Approve / Reject buttons — optimistic status update, POST to backend
- [ ] 3.6 Play button — loads `html-midi-player` web component; stops others first
- [ ] 3.7 `ScorePanel` component — expandable row detail with theory + chromatic breakdown
- [ ] 3.8 Keyboard shortcuts: `a` approve focused row, `r` reject, `p` play

## 4. Tests
- [ ] 4.1 Unit tests for all API endpoints using FastAPI `TestClient`
       (candidates list with phase filter, approve, reject, midi 404, unknown-id 404)
