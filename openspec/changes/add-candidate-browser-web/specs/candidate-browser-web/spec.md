## ADDED Requirements

### Requirement: FastAPI Backend
`app/tools/candidate_server.py` SHALL expose a FastAPI application with the following
endpoints and CORS enabled for `http://localhost:3000`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/candidates` | Return all candidates as JSON; accepts `?phase=` and `?section=` query params |
| POST | `/candidates/{id}/approve` | Write `status: approved` to the matching `review.yml` entry |
| POST | `/candidates/{id}/reject` | Write `status: rejected` to the matching `review.yml` entry |
| GET | `/midi/{id}` | Stream the candidate's `.mid` file with `Content-Type: audio/midi` |

The server SHALL be launched via `python -m app.tools.candidate_server --production-dir <path>`.

#### Scenario: Candidate list endpoint
- **WHEN** `GET /candidates` is called
- **THEN** a JSON array is returned with one object per candidate containing: `id`,
  `phase`, `section`, `template`, `status`, `rank`, `composite_score`, `midi_url`,
  and `scores` dict

#### Scenario: Approve endpoint
- **WHEN** `POST /candidates/{id}/approve` is called
- **THEN** the matching entry in `review.yml` is updated to `status: approved` and
  `{"ok": true}` is returned

#### Scenario: Unknown candidate
- **WHEN** `POST /candidates/{unknown-id}/approve` is called
- **THEN** a 404 response is returned

#### Scenario: MIDI streaming
- **WHEN** `GET /midi/{id}` is called for a candidate that has a `.mid` file
- **THEN** the file bytes are returned with `Content-Type: audio/midi`
- **WHEN** the MIDI file does not exist
- **THEN** a 404 response is returned

### Requirement: Next.js Frontend
A Next.js 15 app (App Router, TypeScript, Tailwind CSS) SHALL live in `web/` and
consume the FastAPI backend at `http://localhost:8000`.

#### Scenario: Candidate table
- **WHEN** the page loads
- **THEN** all candidates are fetched from `/candidates` and displayed in a table with
  columns: phase, section, ID, template, composite score (bar), status, actions
- **WHEN** a column header is clicked
- **THEN** the table sorts by that column (toggle asc/desc)

#### Scenario: Approve and reject
- **WHEN** the Approve or Reject button for a row is clicked
- **THEN** a POST is sent to the backend and the row's status badge updates in place
  without a full page reload; optimistic update applied immediately

#### Scenario: MIDI playback
- **WHEN** a row's Play button is clicked
- **THEN** an `@magenta/midi-player` (or `html-midi-player`) component loads and plays
  the candidate's MIDI inline; any previously playing candidate is stopped first

#### Scenario: Score breakdown
- **WHEN** a row is clicked (outside the action buttons)
- **THEN** a detail panel expands below that row showing the full score breakdown:
  composite, theory sub-scores, chromatic match and confidence

#### Scenario: Filter controls
- **WHEN** the phase dropdown is changed
- **THEN** the table re-fetches from `/candidates?phase=<value>`
- **WHEN** the status filter is changed (all / pending / approved / rejected)
- **THEN** the table filters client-side without a network request

#### Scenario: Keyboard shortcuts
- **WHEN** a row is focused and the user presses `a`
- **THEN** the approve action fires for that row
- **WHEN** the user presses `r`
- **THEN** the reject action fires
- **WHEN** the user presses `p`
- **THEN** playback toggles for that row

### Requirement: No Breaking Changes
The existing terminal browser (`app/tools/candidate_browser.py`) SHALL remain unchanged.
The server's data layer SHALL import `load_all_candidates`, `approve_candidate`, and
`reject_candidate` directly from `app/tools/candidate_browser.py`.
