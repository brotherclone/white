# Change: Add Song Index Screen to Web Dashboard

## Why
The web dashboard currently requires knowing a specific `--production-dir` path before
launch. There is no way to browse available songs in the UI — users must restart the
server with a different path to switch songs. This makes multi-song workflows slow and
error-prone.

## What Changes
- **New launch mode**: `--shrink-wrapped-dir <path>` launches the server in album mode
  instead of pointing at a single production dir. `--production-dir` remains fully
  supported for single-song workflows.
- **New backend endpoints**: `GET /songs` scans all thread directories for songs with
  a `manifest_bootstrap.yml` and returns metadata. `POST /songs/activate` sets the
  active production dir for the current session.
- **Song index page** (`/`): cards showing each song's title, key, BPM, color, and
  thread. Clicking a card activates that song and navigates to the candidate browser.
- **Candidate browser moved to `/candidates`**: existing UI is unchanged except for a
  breadcrumb at the top ("← Songs") that navigates back to `/`.
- When launched with `--production-dir` (legacy mode), the breadcrumb is hidden and
  `/` redirects to `/candidates` — no behaviour change for existing users.

## Impact
- Affected specs: `candidate-browser-web`
- Affected code:
  - `app/tools/candidate_server.py` — new CLI arg, two new endpoints, active-song state
  - `web/app/page.tsx` — becomes song index
  - `web/app/candidates/page.tsx` — existing candidate browser, moved here
  - `web/lib/api.ts` — two new fetch helpers (`fetchSongs`, `activateSong`)
