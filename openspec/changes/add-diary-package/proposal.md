# Change: Add song lifecycle diary package

## Why
Songs pass through multiple pipeline phases (proposal, composition, production, release),
each generating creative decisions authored by different actors: color agents (Prism,
ThreadKeepr), automated pipeline hooks, Claude Code sessions, and Gabriel. There is
currently no structured way to capture or retrieve this narrative per song.

## What Changes
- New `white_diary` package (`packages/diary/`) with a lite `DiaryEntry` Pydantic model
  and four filesystem store functions callable directly from pipeline hooks
- New `/diary/{song_slug}` CRUD routes on `white_api`
- `white_api` gains a `white-diary` dependency

## Impact
- Affected specs: (new) `song-diary`
- Affected code:
  - `packages/diary/` — all new
  - `packages/api/pyproject.toml` — add `white-diary` dependency
  - `packages/api/src/white_api/routes/diary.py` — new
  - `packages/api/src/white_api/candidate_server.py` — register diary router
