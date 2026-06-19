# Change: Refactor song index to primary screen with stage-aware routing

## Why
The two-step landing page (Generation / Composition Board links) adds an unnecessary click
before reaching songs. The "not initialized" badge on song cards carries no actionable
meaning, and the generate workflow is buried as a header button on the song list. The
flow should be: open the app → see songs → click one → land in the right place.

## What Changes
- The song browser moves from `/songs` to `/` (root); `/songs` redirects to `/`
- The old two-link landing page is removed
- Song cards show a **stage badge** (Ideation / Generation / Composition) instead of
  "not initialized", and clicking routes directly to the correct screen:
  - **Ideation** → init then `/candidates`
  - **Generation** → `/candidates`
  - **Composition** → `/board`
- Generate workflow moves off the song browser header to a dedicated `/agent` page
- A "Run Agent" link in the song browser header navigates to `/agent`
- Backend `scan_songs` gains a `stage` field (`ideation | generation | composition`)
  derived from `song_context.yml` and `composition.yml` presence
- A `dev.sh` script at the repo root launches both the FastAPI server and Next.js dev
  server with a single command

## Impact
- Affected specs: `candidate-browser-web`
- Affected code:
  - `packages/api/src/white_api/candidate_server.py` — `scan_songs` adds `stage`
  - `packages/api/tests/test_candidate_server.py` — shape test updated
  - `packages/client/app/page.tsx` — replaced by song browser
  - `packages/client/app/songs/page.tsx` — becomes redirect to `/`
  - `packages/client/app/agent/page.tsx` — new page (generate workflow)
  - `packages/client/lib/types.ts` — `SongEntry` gains `stage`
  - `dev.sh` — new file at repo root
