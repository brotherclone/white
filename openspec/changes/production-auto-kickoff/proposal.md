# Change: Production auto-kickoff after chain run

## Why

After a chain run completes, getting to a reviewable state currently requires
several manual steps: wait for shrinkwrap, run init_production, open the server,
navigate to the right song. The infrastructure for all of this already exists —
`_invoke_chord_pipeline_safe`, `init_production`, `candidate_server.py`, the Next.js
app — it just isn't wired together into a single automatic flow.

Additionally, the current auto-kickoff only inits one production dir (for the White
synthesis). But Black, Red, Yellow etc. are all standalone songs that can be produced
independently. Every `is_final=True` proposal should get its own production dir.

Finally, `initial_proposal.yml` is a strict subset of `song_context.yml` — every
field it contains is already in `song_context.yml`, which also adds `phases`, `title`,
`genres`, `mood`. Writing both files is redundant. `song_context.yml` is the one file.

## What Changes

**Post-finalize kickoff (white_agent.py)**
- After `finalize_song_proposal`, for each `is_final=True` proposal in
  `state.song_proposals.iterations` (requires `add-final-proposal-flag`):
  1. Run `init_production` → writes `production/<slug>/song_context.yml`
  2. Run `_invoke_chord_pipeline_safe` → writes `production/<slug>/chords/`
- White's proposal is included but processed last (it synthesises from the others)

**init_production no longer writes initial_proposal.yml**
- `init_production.py` writes only `song_context.yml`
- All fields previously in `initial_proposal.yml` are already in `song_context.yml`
- Existing dirs that have `initial_proposal.yml` are unaffected (no migration needed)

**Browser auto-launch**
- After all chord gens complete, Prism checks whether `candidate_server.py` is
  listening on port 8000 — if not, launches it (non-blocking subprocess)
- Checks whether Next.js dev server is listening on port 3000 — if not, launches
  `npm run dev` in `web/` (non-blocking subprocess)
- Opens the browser to the chord review of the first non-White `is_final` proposal:
  `http://localhost:3000?production-dir=<path>&phase=chords`
- Launch is gated behind `AUTO_BROWSER_LAUNCH=true` env var (off by default so CI
  and headless runs are unaffected); `run_white_agent start` sets it automatically
  unless `--no-browser` is passed

## Dependencies

- `add-final-proposal-flag` must be implemented first (provides `is_final` field)

## Impact

- Affected code: `app/agents/white_agent.py` (post-finalize wiring),
  `app/generators/midi/production/init_production.py` (drop initial_proposal.yml)
- Affected specs: `prism-auto-chord-generation` (MODIFIED), `pipeline-orchestration` (MODIFIED)
- Non-breaking: existing production dirs with `initial_proposal.yml` continue to work
- `run_white_agent start` gets a `--no-browser` flag to suppress launch
