## Prerequisites

- [ ] 0.1 Confirm `add-final-proposal-flag` is implemented (SongProposalIteration has
      `is_final` field and agents mark final proposals correctly)

## 1. init_production — drop initial_proposal.yml

- [ ] 1.1 In `app/generators/midi/production/init_production.py`, remove the code
      that writes `initial_proposal.yml`
- [ ] 1.2 Confirm `song_context.yml` already contains all fields that were in
      `initial_proposal.yml`; add any that are missing
- [ ] 1.3 Update `init_production` tests — remove assertions about `initial_proposal.yml`

## 2. white_agent — init all final proposals

- [ ] 2.1 In `finalize_song_proposal` (after `state.run_finished = True`), replace the
      single-proposal auto-chord-gen block with a loop over all `is_final=True`
      iterations in `state.song_proposals.iterations`
- [ ] 2.2 Sort iterations so White is processed last (White's `rainbow_color` color name
      is `"White"` — check `final.rainbow_color`)
- [ ] 2.3 For each non-skipped iteration, call `init_production` (in-process) then
      `_invoke_chord_pipeline_safe`
- [ ] 2.4 Guard: skip if `MOCK_MODE=true`, skip if no `is_final` iterations exist

## 3. Browser auto-launch

- [ ] 3.1 Add `_launch_review_browser(production_dirs: list[Path])` private method to
      `WhiteAgent`:
      - Check port 8000 (FastAPI): if not listening, launch
        `python -m app.tools.candidate_server --production-dir <first_non_white_dir>`
        as a non-blocking `subprocess.Popen`
      - Check port 3000 (Next.js): if not listening, launch `npm run dev` in `web/`
        as a non-blocking `subprocess.Popen`
      - Wait up to 5s for port 8000 to respond (poll every 0.5s)
      - Open `http://localhost:3000?production-dir=<first_non_white_dir>&phase=chords`
        via `webbrowser.open()`
- [ ] 3.2 Call `_launch_review_browser` after all chord gens complete, passing the
      list of new production dirs ordered non-White first
- [ ] 3.3 Gate behind `AUTO_BROWSER_LAUNCH` env var (default `false`);
      `run_white_agent.py start` sets `AUTO_BROWSER_LAUNCH=true` unless
      `--no-browser` flag is passed
- [ ] 3.4 Add `--no-browser` flag to `run_white_agent.py start`

## 4. Tests

- [ ] 4.1 Unit test: `finalize_song_proposal` with two `is_final=True` iterations
      (one non-White, one White) — assert chord pipeline called twice, White last
- [ ] 4.2 Unit test: `_launch_review_browser` with both ports already listening —
      assert no subprocess launched, browser opened to first non-White dir
- [ ] 4.3 Unit test: port 8000 not listening — assert `candidate_server` subprocess
      launched
