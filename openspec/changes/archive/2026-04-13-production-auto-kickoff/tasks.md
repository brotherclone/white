## Prerequisites

- [x] 0.1 Confirm `add-final-proposal-flag` is implemented (SongProposalIteration has
      `is_final` field and agents mark final proposals correctly)

## 1. init_production — drop initial_proposal.yml

- [x] 1.1 In `app/generators/midi/production/init_production.py`, remove the code
      that writes `initial_proposal.yml`; `write_initial_proposal` now returns
      the path to `song_context.yml`; idempotency check updated to `song_context.yml`
- [x] 1.2 Confirmed — `song_context.yml` is a strict superset of `initial_proposal.yml`
      fields; `load_initial_proposal` retained as backward-compat loader
- [x] 1.3 Updated `init_production` tests — all assertions updated to `song_context.yml`
      (51 tests passing)

## 2. white_agent — init all final proposals

- [x] 2.1 Replaced single-proposal block with loop over all `is_final=True` iterations
- [x] 2.2 `_is_white_proposal` helper; sorted White last via `sort(key=...)`
- [x] 2.3 Each iteration: `_run_init_production` then `_invoke_chord_pipeline_safe`
- [x] 2.4 Guards: `MOCK_MODE=true` skips; empty `is_final` list skips

## 3. Browser auto-launch

- [x] 3.1 `_launch_review_browser(production_dirs)` added to `WhiteAgent`: port checks
      via `socket`, non-blocking `subprocess.Popen`, 5s poll for port 8000,
      `webbrowser.open` to first non-White dir
- [x] 3.2 Called after all chord gens complete when `AUTO_BROWSER_LAUNCH=true`
- [x] 3.3 Gated behind `AUTO_BROWSER_LAUNCH` env var (default `false`)
- [x] 3.4 `--no-browser` flag added to `run_white_agent.py start`; omitting it sets
      `AUTO_BROWSER_LAUNCH=true` automatically

## 4. Tests

- [x] 4.1 `TestMultiFinalKickoff` — two `is_final=True` iterations, White last,
      non-final skipped (3 tests)
- [x] 4.2 `TestLaunchReviewBrowser` — both ports open → no Popen, browser opened
- [x] 4.3 `TestLaunchReviewBrowser` — port 8000 closed → `candidate_server` Popen
- [x] `TestIsWhiteProposal` — string and RainbowTableColor variants (4 tests)
      21/21 passing
