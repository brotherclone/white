## ADDED Requirements

### Requirement: FastAPI Backend
`app/tools/candidate_server.py` SHALL expose a FastAPI application with the following
endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve `browser.html` |
| GET | `/candidates` | Return all candidates as JSON; accepts `?phase=` and `?section=` query params |
| POST | `/candidates/{id}/approve` | Write `status: approved` to the matching `review.yml` entry |
| POST | `/candidates/{id}/reject` | Write `status: rejected` to the matching `review.yml` entry |
| GET | `/midi/{id}` | Stream the candidate's `.mid` file with `Content-Type: audio/midi` |

The server SHALL be launched via `python -m app.tools.candidate_server --production-dir <path>`
and SHALL open `http://localhost:8000` in the default browser automatically on startup.

#### Scenario: Candidate list endpoint
- **WHEN** `GET /candidates` is called with a valid production dir configured
- **THEN** a JSON array is returned with one object per candidate containing: `id`,
  `phase`, `section`, `template`, `status`, `rank`, `composite_score`, `midi_url`,
  and `scores` dict

#### Scenario: Approve endpoint
- **WHEN** `POST /candidates/{id}/approve` is called
- **THEN** the matching entry in `review.yml` is updated to `status: approved` and
  `{"ok": true}` is returned

#### Scenario: MIDI streaming
- **WHEN** `GET /midi/{id}` is called for a candidate that has a `.mid` file
- **THEN** the file bytes are returned with `Content-Type: audio/midi`
- **WHEN** the MIDI file does not exist
- **THEN** a 404 response is returned

### Requirement: Browser HTML Frontend
`app/tools/static/browser.html` SHALL be a self-contained single-file page with no
build step. All JS and CSS SHALL be inline or loaded from a CDN.

#### Scenario: Candidate table
- **WHEN** the page loads
- **THEN** all candidates are displayed in a sortable table with columns:
  phase, section, ID, template, composite score (with bar), status
- **WHEN** a column header is clicked
- **THEN** the table sorts by that column (toggle asc/desc)

#### Scenario: Approve and reject
- **WHEN** the Approve or Reject button for a row is clicked
- **THEN** a POST is sent to the backend and the row's status cell updates in place
  without a page reload

#### Scenario: MIDI playback
- **WHEN** a row's Play button is clicked
- **THEN** a `<midi-player>` web component (from the `html-midi-player` CDN) loads
  and plays the candidate's MIDI inline in the page; any previously playing candidate
  is stopped first

#### Scenario: Score breakdown
- **WHEN** a row is clicked (anywhere other than a button)
- **THEN** a detail panel below the table expands showing the full score breakdown:
  composite, theory sub-scores, chromatic match and confidence

#### Scenario: Filter controls
- **WHEN** the phase dropdown is changed
- **THEN** the table re-fetches and shows only that phase's candidates
- **WHEN** the status filter is changed (all / pending / approved / rejected)
- **THEN** the table filters client-side without re-fetching

#### Scenario: Keyboard shortcuts
- **WHEN** a row is focused and the user presses `a`
- **THEN** the approve action fires for that row
- **WHEN** the user presses `r`
- **THEN** the reject action fires
- **WHEN** the user presses `p`
- **THEN** playback starts for that row

### Requirement: No Breaking Changes
The existing terminal browser (`app/tools/candidate_browser.py`) SHALL remain unchanged.
Both tools share the same `review.yml` format and data layer helpers from
`app/tools/candidate_browser.py` (`load_all_candidates`, `approve_candidate`,
`reject_candidate`).
