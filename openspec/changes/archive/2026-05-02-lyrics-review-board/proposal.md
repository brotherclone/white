# Change: Lyrics Review UI on Composition Board

## Why

After `lyric_pipeline.py` runs, three candidate `.txt` files land in
`melody/candidates/` and a `lyrics_review.yml` tracks their status. Currently
there is no web UI for reviewing and promoting them — the user must hand-edit
`lyrics_review.yml` and run `promote_part` via the CLI. The Composition Board's
lyrics stage column is the natural home for this review step.

## What Changes

When the board's active song is at the **Lyrics** mix stage, the lyrics column
card shows a row of small version buttons — one per lyric candidate (`[v1]`,
`[v2]`, `[v3]`). Clicking a button opens a modal popup displaying the full lyric
text for that candidate along with its chromatic scores. A **Promote** button at
the bottom of the popup marks that candidate as approved and calls the existing
promote endpoint. After promotion:

- The `[v1] [v2] [v3]` buttons are replaced by a single **[See Lyrics]** button.
- Clicking **[See Lyrics]** reopens the modal in read-only mode showing the
  promoted lyrics.
- All modals have a standard **[×]** close button.

## Decisions

- **No new backend routes for reading lyrics** — a new `GET /lyrics` endpoint
  reads `lyrics_review.yml` and returns candidates + their text. Promotion reuses
  the existing `POST /promote` endpoint with `phase: "lyrics"` after patching the
  chosen candidate's status to `approved` via a new
  `POST /lyrics/<id>/approve` endpoint.
- **Scope** — lyrics column only, no changes to other board columns or the
  candidate browser.
- **Fitting scores** — shown in the popup as a summary (per-section worst-case
  verdict), not the full phrase breakdown, to keep the popup readable.

## Impact

- Affected specs: `candidate-browser-web` (new endpoints), new
  `lyrics-review-board`
- Affected code:
  - `packages/api/src/white_api/candidate_server.py` — new endpoints
  - `packages/client/app/board/page.tsx` — lyrics column UI + modal
  - `packages/client/lib/api.ts` — new API calls
  - `packages/client/lib/types.ts` — new types
- No changes to `lyric_pipeline.py`, `promote_part.py`, or `lyrics_review.yml`
  schema
